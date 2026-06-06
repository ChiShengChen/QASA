#!/usr/bin/env python3
"""
Encoding-Scheme Sensitivity (Referee 1, major 4 + minor 3)
==========================================================
Two linked referee questions:
  (M4) Why does QASA fail on clean periodic signals -- is it the RX/RZ *angle
       encoding* or an inherent limitation of the circuit's *expressivity*?
  (m3) Have other encodings (e.g. amplitude encoding) been tested? Report the
       sensitivity of the results to the encoding scheme.

We swap ONLY the data-encoding block of QASA's quantum layer, keeping the
variational ansatz, classical backbone, and training identical, and compare:

  * angle      : RX(x_i), RZ(x_i)            (the paper's default; 1 upload)
  * amplitude  : AmplitudeEmbedding of the (normalised) feature vector
  * reupload   : angle encoding re-applied before every entangling layer
                 (data re-uploading -> richer accessible Fourier spectrum)

If amplitude/re-uploading *recovers* performance on the clean periodic tasks,
the limitation is encoding-related; if not, it is an expressivity / inductive-
bias limitation (the circuit is already expressive, KL=0.029). Re-uploading
directly probes the Fourier-frequency argument of Schuld et al.

Cost-aware defaults: clean-periodic failure tasks (damped osc, waveform) +
chaotic-logistic control; reduced epochs/seeds. Override via CLI.

Usage:
  python experiments/run_encoding_sensitivity.py --epochs 100 --seeds 2
  python experiments/run_encoding_sensitivity.py --dry-run
"""

import os
import sys
import csv
import time
import math
import argparse
import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import task_registry, get_task
from experiments.run_baseline_comparison import (
    PositionalEncoding, train_and_evaluate, count_parameters, TASK_DISPLAY_NAMES,
)

N_QLAYERS = 4


# ============================================================
# Encoding-specific circuits (variational ansatz held fixed)
# ============================================================

def circuit_width(encoding, n_feat):
    """Number of data qubits the circuit uses for a given encoding of n_feat features.
    angle/reupload encode one feature per qubit (n_feat qubits); amplitude packs
    n_feat features into ceil(log2 n_feat) qubits -- the natural, dense use of
    amplitude encoding (also far cheaper to simulate than 2^n_feat amplitudes)."""
    if encoding == 'amplitude':
        return max(1, int(math.ceil(math.log2(n_feat))))
    return n_feat


def make_circuit(encoding, n_feat):
    """QNode: chosen encoding of the same n_feat features + the QASA ansatz.
    Returns <Z_i> for each of the circuit's data qubits (cq of them)."""
    cq = circuit_width(encoding, n_feat)
    dev = qml.device("lightning.qubit", wires=cq + 1)

    def ansatz(weights):
        for i in range(cq):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        for l in range(1, N_QLAYERS):
            for i in range(cq):
                qml.CNOT(wires=[i, (i + 1) % cq])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)
            qml.CNOT(wires=[cq - 1, cq])
            qml.RY(weights[l, cq], wires=cq)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        if encoding == 'angle':
            for i in range(cq):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            ansatz(weights)
        elif encoding == 'amplitude':
            # pack all n_feat features into cq qubits (2^cq amplitudes)
            qml.AmplitudeEmbedding(inputs, wires=range(cq), normalize=True, pad_with=0.0)
            ansatz(weights)
        elif encoding == 'reupload':
            for i in range(cq):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            for i in range(cq):
                qml.RX(weights[0, i], wires=i)
                qml.RZ(weights[1, i], wires=i)
            for l in range(1, N_QLAYERS):
                for i in range(cq):                # re-upload before each layer
                    qml.RX(inputs[i], wires=i)
                for i in range(cq):
                    qml.CNOT(wires=[i, (i + 1) % cq])
                    qml.RY(weights[l, i], wires=i)
                    qml.RZ(weights[l, i], wires=i)
                qml.CNOT(wires=[cq - 1, cq])
                qml.RY(weights[l, cq], wires=cq)
        else:
            raise ValueError(encoding)
        return [qml.expval(qml.PauliZ(i)) for i in range(cq)]

    return circuit


class EncQuantumLayer(nn.Module):
    """QASA quantum value layer with a swappable encoding.

    For amplitude encoding the feature vector has length 2^n_qubits amplitudes;
    we project to that dimension. For angle/reupload we project to n_qubits angles.
    """

    def __init__(self, input_dim, output_dim, encoding='angle', n_qubits=8):
        super().__init__()
        self.encoding = encoding
        self.n_feat = n_qubits                  # number of features encoded (matched across encodings)
        self.cq = circuit_width(encoding, self.n_feat)   # actual data-qubit count
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_shape = (N_QLAYERS, self.cq + 1)
        self.qlayer = qml.qnn.TorchLayer(make_circuit(encoding, self.n_feat),
                                         {'weights': self.weight_shape})
        self.input_proj = nn.Linear(input_dim, self.n_feat)
        self.norm = nn.LayerNorm(self.n_feat)
        self.output_proj = nn.Linear(self.cq, output_dim)
        for m in (self.input_proj, self.output_proj):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x, timestep=0.0):
        x_proj = torch.tanh(self.input_proj(x))
        x_proj = self.norm(x_proj)
        ts = torch.tensor(float(timestep), device=x.device)
        outputs = [self.qlayer((x_proj[i] + ts).cpu()).to(x.device) for i in range(x.size(0))]
        q = torch.stack(outputs)
        out = self.output_proj(q)
        return x + out if self.input_dim == self.output_dim else out


class EncQuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, encoding='angle', n_qubits=8, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = EncQuantumLayer(hidden_dim, hidden_dim, encoding, n_qubits)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout_rate),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        B, L, F = x.shape
        q = self.v_quantum(x.reshape(B * L, F), float(L)).view(B, L, F)
        return self.norm2(q + self.ffn(q))


class EncQASAModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1, encoding='angle', n_qubits=8):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate))
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        layers = [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True,
                                             dropout=dropout_rate, dim_feedforward=4 * hidden_dim)
                  for _ in range(num_layers - 1)]
        layers.append(EncQuantumEncoderLayer(hidden_dim, encoding, n_qubits, dropout_rate))
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


# Clean-periodic failure tasks (where QASA loses) + chaotic control (where it wins)
FOCUS_TASKS = [
    'classical_damped_oscillation',
    'classical_waveform',
    'classical_chaotic_logistic',
]
ENCODINGS = ['angle', 'amplitude', 'reupload']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=None)
    ap.add_argument('--encodings', nargs='+', default=ENCODINGS)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--n-qubits', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.seeds = 2, 1
        args.tasks = ['damped_oscillation']

    tasks = args.tasks or FOCUS_TASKS
    tasks = [f'classical_{t}' if not t.startswith('classical_') else t for t in tasks]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    ckpt_dir = os.path.join(results_dir, "checkpoints", "encoding_sensitivity")
    config = {'lr': 5e-4, 'weight_decay': 1e-4, 'epochs': args.epochs,
              'seq_len_train': 20, 'print_every': max(1, args.epochs // 5)}
    device = torch.device('cpu')

    rows = []
    for task_key in tasks:
        if task_key not in task_registry:
            print(f"skip {task_key}"); continue
        disp = TASK_DISPLAY_NAMES.get(task_key, task_key)
        task = get_task(task_key)
        print(f"\n{'='*60}\nTask: {disp}\n{'='*60}")
        for enc in args.encodings:
            maes, mses = [], []
            for s in range(args.seeds):
                seed = 42 + s
                torch.manual_seed(seed); np.random.seed(seed)
                model = EncQASAModel(hidden_dim=64, num_layers=4, seq_len=20,
                                     encoding=enc, n_qubits=args.n_qubits).to(device)
                save_dir = os.path.join(ckpt_dir, enc, task_key, f'seed{seed}')
                try:
                    t0 = time.time()
                    mae, mse, _ = train_and_evaluate(model, task, task_key, device, config, save_dir)
                    print(f"  [{enc:9s}] seed{seed}: MAE {mae:.4f} MSE {mse:.4f} ({time.time()-t0:.0f}s)")
                    maes.append(mae); mses.append(mse)
                except Exception as e:
                    print(f"  [{enc}] ERROR: {e}"); traceback.print_exc()
                del model
            if maes:
                rows.append((disp, task_key, enc, np.mean(maes), np.std(maes),
                             np.mean(mses), np.std(mses), len(maes)))

    out_csv = os.path.join(results_dir, f"encoding_sensitivity_{ts}.csv")
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'TaskKey', 'Encoding', 'MAE', 'MAE_std', 'MSE', 'MSE_std', 'N'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", f"{r[4]:.6f}",
                        f"{r[5]:.6f}", f"{r[6]:.6f}", r[7]])
    print(f"\n{'='*60}\nENCODING SENSITIVITY (MAE / MSE)\n{'='*60}")
    for r in rows:
        print(f"  {r[0]:<18}{r[2]:<11}MAE {r[3]:.4f}  MSE {r[5]:.4f}")
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
