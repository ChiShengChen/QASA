#!/usr/bin/env python3
"""
Multi-Channel NISQ Noise Robustness (Referee 2, comment 2)
==========================================================
The published noise study used only the depolarizing channel. Referee 2 asks for
stability under *other* noise conditions. We extend the analysis to three physical
channels at the same reduced 4-qubit / 2-layer configuration used in the paper:

  * depolarizing    (isotropic gate error; the original channel, for reference)
  * amplitude_damping  (T1 relaxation / energy loss)
  * bit_flip        (a readout-error proxy)

For each channel we sweep noise levels p in {0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1},
retraining from scratch (chaotic logistic, the quantum-advantage task), and report
mean +- std over seeds. This shows whether the noise-as-regulariser effect observed
for depolarizing noise persists, is channel-specific, or reverses.

Usage:
  python experiments/run_noise_multichannel.py --channels amplitude_damping bit_flip --seeds 3
  python experiments/run_noise_multichannel.py --dry-run
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task
from experiments.run_baseline_comparison import (
    PositionalEncoding, train_and_evaluate, TASK_DISPLAY_NAMES,
)
from experiments.resume_utils import load_done, append_row, aggregate

N_QUBITS = 4         # reduced circuit (density-matrix simulation cost)
N_QLAYERS = 2


def _apply_noise(noise_type, p, wire):
    if noise_type == 'depolarizing':
        qml.DepolarizingChannel(p, wires=wire)
    elif noise_type == 'amplitude_damping':
        qml.AmplitudeDamping(p, wires=wire)
    elif noise_type == 'bit_flip':
        qml.BitFlip(p, wires=wire)
    elif noise_type == 'phase_damping':
        qml.PhaseDamping(p, wires=wire)
    else:
        raise ValueError(noise_type)


def make_noisy_circuit(noise_type, noise_level):
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        for i in range(N_QUBITS):
            qml.RX(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        for i in range(N_QUBITS):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        if noise_level > 0:
            for i in range(N_QUBITS):
                _apply_noise(noise_type, noise_level, i)
        for l in range(1, N_QLAYERS):
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)
            if noise_level > 0:
                for i in range(N_QUBITS):
                    _apply_noise(noise_type, noise_level, i)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return circuit


class NoisyQuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim, noise_type, noise_level):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.qlayer = qml.qnn.TorchLayer(make_noisy_circuit(noise_type, noise_level),
                                         {'weights': (N_QLAYERS, N_QUBITS)})
        self.input_proj = nn.Linear(input_dim, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)
        self.output_proj = nn.Linear(N_QUBITS, output_dim)
        for m in (self.input_proj, self.output_proj):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x, timestep=0.0):
        x_proj = self.norm(torch.tanh(self.input_proj(x)))
        ts = torch.tensor(float(timestep), device=x.device)
        outs = [self.qlayer((x_proj[i] + ts).cpu()).to(x.device) for i in range(x.size(0))]
        out = self.output_proj(torch.stack(outs))
        return x + out if self.input_dim == self.output_dim else out


class NoisyQuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, noise_type, noise_level, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = NoisyQuantumLayer(hidden_dim, hidden_dim, noise_type, noise_level)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout_rate))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        B, L, F = x.shape
        q = self.v_quantum(x.reshape(B * L, F), float(L)).view(B, L, F)
        return self.norm2(q + self.ffn(q))


class NoisyQASAModel(nn.Module):
    def __init__(self, noise_type, noise_level, input_dim=1, output_dim=1,
                 hidden_dim=64, num_layers=4, seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate))
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        layers = [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True,
                                             dropout=dropout_rate, dim_feedforward=4 * hidden_dim)
                  for _ in range(num_layers - 1)]
        layers.append(NoisyQuantumEncoderLayer(hidden_dim, noise_type, noise_level, dropout_rate))
        self.encoder = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.embedding[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.embedding[0].bias, 0)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', nargs='+',
                    default=['amplitude_damping', 'bit_flip'],
                    help="noise channels (depolarizing already in paper)")
    ap.add_argument('--levels', nargs='+', type=float,
                    default=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
    ap.add_argument('--task', default='classical_chaotic_logistic')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.seeds = 2, 1
        args.channels, args.levels = ['amplitude_damping'], [0.0, 0.01]

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    config = {'lr': 5e-4, 'weight_decay': 1e-4, 'epochs': args.epochs,
              'seq_len_train': 20, 'print_every': max(1, args.epochs // 4)}
    device = torch.device('cpu')
    task = get_task(args.task)
    disp = TASK_DISPLAY_NAMES.get(args.task, args.task)

    # Resumable per-(channel, level, seed) progress CSV.
    prog_csv = os.path.join(results_dir, f"noise_multichannel_progress_{args.task}.csv")
    done = load_done(prog_csv, key_cols=('Channel', 'NoiseLevel', 'Seed'))
    header = ['Channel', 'NoiseLevel', 'Seed', 'MAE', 'MSE', 'Epochs']

    for ch in args.channels:
        print(f"\n{'='*60}\nChannel: {ch}  ({disp})  [resume: {len(done)} runs done]\n{'='*60}")
        for p in args.levels:
            for s in range(args.seeds):
                seed = 42 + s
                if (ch, str(p), str(seed)) in done:
                    print(f"  {ch} p={p:<6} seed{seed}: SKIP (already done)")
                    continue
                torch.manual_seed(seed); np.random.seed(seed)
                model = NoisyQASAModel(ch, p, hidden_dim=64, num_layers=4, seq_len=20).to(device)
                try:
                    t0 = time.time()
                    mae, mse, _ = train_and_evaluate(model, task, args.task, device, config)
                    print(f"  {ch} p={p:<6} seed{seed}: MAE {mae:.4f} MSE {mse:.4f} ({time.time()-t0:.0f}s)")
                    append_row(prog_csv, header, [ch, p, seed, f"{mae:.6f}", f"{mse:.6f}", args.epochs])
                except Exception as e:
                    print(f"  {ch} p={p} ERROR: {e}")
                del model

    # Aggregate per (channel, level) from the progress CSV.
    rows = aggregate(prog_csv, group_col='NoiseLevel', val_cols=('MAE', 'MSE'),
                     extra_cols=('Channel',))
    # regroup by channel for display + relative degradation vs p=0
    print(f"\n{'='*60}\nMULTI-CHANNEL NOISE (MAE, Δ vs p=0)\n{'='*60}")
    by_ch = {}
    with open(prog_csv, newline='') as f:
        import csv as _csv
        for r in _csv.DictReader(f):
            by_ch.setdefault(r['Channel'], {}).setdefault(r['NoiseLevel'], []).append(float(r['MAE']))
    for ch, levels in by_ch.items():
        base = np.mean(levels.get('0.0', levels.get('0', [np.nan])))
        for p in sorted(levels, key=float):
            m = float(np.mean(levels[p]))
            delta = f"{100*(m-base)/base:+.1f}%" if np.isfinite(base) and base else "---"
            print(f"  {ch:<18} p={p:<6} MAE {m:.4f}  ({delta})  n={len(levels[p])}")
    print(f"\nProgress CSV (resumable): {prog_csv}")


if __name__ == '__main__':
    main()
