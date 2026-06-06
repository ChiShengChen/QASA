#!/usr/bin/env python3
"""
Parameter-Matched Classical Bottleneck Baseline (Referee 1, Major Comment 1)
============================================================================
Goal: isolate the *quantum* contribution from mere parameter compactness.

Referee 1 asks whether QASA's edge comes from quantum feature mapping or just
from a more compact parameterization. To answer this, we build a baseline that
is IDENTICAL to QASA's quantum encoder layer except that the parameterized
quantum circuit (PQC, 36 quantum params) is replaced by a *classical* value map
with a comparable trainable-parameter budget (~36 params).

QASA QuantumLayer  = input_proj(d->8) + LayerNorm + [PQC: 8->8, 36 q-params] + output_proj(8->d)
Bottleneck baseline= input_proj(d->8) + LayerNorm + [Classical 8->8, ~36 params] + output_proj(8->d)

Everything else (3 classical Transformer encoder layers, attention, FFN,
embedding, positional encoding, output head, optimizer, schedule, seeds) is held
fixed. The ONLY difference vs QASA is PQC vs classical map => any performance gap
is attributable to the quantum feature map, not to parameter count.

Usage:
  python experiments/run_bottleneck_baseline.py --epochs 200 --seeds 5
  python experiments/run_bottleneck_baseline.py --dry-run
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the exact training/eval loop, tasks, and shared sub-modules from the
# main baseline runner so the comparison is apples-to-apples.
from quantum_benchmark.tasks import task_registry, get_task
from experiments.run_baseline_comparison import (
    N_QUBITS, PositionalEncoding, ClassicalModel,
    train_and_evaluate, count_parameters,
    TASK_DISPLAY_NAMES,
)


# ============================================================
# Classical value map: drop-in replacement for the 36-param PQC
# ============================================================

class ClassicalValueMap(nn.Module):
    """A classical analog of the PQC value map (8 -> 8).

    Implemented as a low-rank bilinear map with a bounded (tanh) nonlinearity,
    mirroring the PQC's bounded Pauli-Z expectation outputs in [-1, 1].

    Trainable params: U (N_QUBITS x rank) + V (rank x N_QUBITS) + bias (N_QUBITS).
    rank=2 -> 8*2 + 2*8 + 8 = 40 params, comparable to the PQC's 36 quantum params.
    Use --rank to tune; param count is printed at startup.
    """

    def __init__(self, width=N_QUBITS, rank=2, use_bias=True):
        super().__init__()
        self.U = nn.Parameter(torch.empty(width, rank))
        self.V = nn.Parameter(torch.empty(rank, width))
        self.bias = nn.Parameter(torch.zeros(width)) if use_bias else None
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, x):
        out = (x @ self.U) @ self.V
        if self.bias is not None:
            out = out + self.bias
        return torch.tanh(out)


class ClassicalBottleneckLayer(nn.Module):
    """Mirror of QASA's QuantumLayer, with the PQC swapped for ClassicalValueMap.

    Identical wrapper: input_proj(d->8) + LayerNorm + value map + output_proj(8->d)
    + residual. This isolates the quantum vs classical contribution at the value
    projection bottleneck.
    """

    def __init__(self, input_dim, output_dim, rank=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_map = ClassicalValueMap(width=N_QUBITS, rank=rank)
        self.input_proj = nn.Linear(input_dim, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)
        self.output_proj = nn.Linear(N_QUBITS, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.kaiming_uniform_(self.output_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x, timestep=0.0):
        x_proj = torch.tanh(self.input_proj(x))
        x_proj = self.norm(x_proj)
        ts = torch.tensor(float(timestep), device=x.device)
        v = self.value_map(x_proj + ts)
        out = self.output_proj(v)
        if self.input_dim == self.output_dim:
            return x + out
        return out


class ClassicalBottleneckEncoderLayer(nn.Module):
    """Mirror of QASA's QuantumEncoderLayer with the classical bottleneck value map."""

    def __init__(self, hidden_dim, dropout_rate=0.1, rank=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_bottleneck = ClassicalBottleneckLayer(hidden_dim, hidden_dim, rank=rank)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size * seq_len, features)
        v_out = self.v_bottleneck(x_flat, float(seq_len))
        v_out = v_out.view(batch_size, seq_len, features)
        x = self.norm2(v_out + self.ffn(v_out))
        return x


class ClassicalBottleneckModel(nn.Module):
    """QASA architecture with the quantum value layer replaced by a param-matched
    classical bottleneck. 3 classical Transformer layers + 1 bottleneck layer."""

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1, rank=2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        encoder_layers = []
        for _ in range(num_layers - 1):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, batch_first=True,
                    dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
                )
            )
        encoder_layers.append(
            ClassicalBottleneckEncoderLayer(hidden_dim, dropout_rate=dropout_rate, rank=rank)
        )
        self.encoder = nn.ModuleList(encoder_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embedding[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.embedding[0].bias, 0)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x)


# ============================================================
# Tasks: where QASA wins (+ one control where it loses)
# ============================================================

FOCUS_TASKS = [
    'classical_chaotic_logistic',       # QASA wins (MAE+MSE)
    'classical_trend_seasonality_noise',# QASA wins (MAE+MSE)
    'classical_square_triangle_wave',   # QASA wins MAE
    'classical_damped_oscillation',     # control: classical wins
]


def main():
    parser = argparse.ArgumentParser(description="Parameter-matched classical bottleneck baseline")
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--rank', type=int, default=2,
                        help="Rank of the classical value map (controls param count)")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 2
        args.seeds = 1
        args.tasks = ['chaotic_logistic']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    ckpt_dir = os.path.join(results_dir, "checkpoints", "bottleneck_baseline")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cpu")
    hidden_dim, num_layers, seq_len_train = 64, 4, 20
    config = {
        'lr': 5e-4, 'weight_decay': 1e-4, 'epochs': args.epochs,
        'seq_len_train': seq_len_train, 'print_every': max(1, args.epochs // 10),
    }

    if args.tasks:
        task_names = [f'classical_{t}' if not t.startswith('classical_') else t for t in args.tasks]
    else:
        task_names = FOCUS_TASKS

    # Report parameter counts so the "~36 param" match is auditable.
    bottleneck = ClassicalBottleneckModel(hidden_dim=hidden_dim, num_layers=num_layers,
                                          seq_len=seq_len_train, rank=args.rank)
    value_params = sum(p.numel() for p in bottleneck.encoder[-1].v_bottleneck.value_map.parameters())
    total_bn, _ = count_parameters(bottleneck)
    classical = ClassicalModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)
    total_cl, _ = count_parameters(classical)
    del bottleneck, classical

    print("=" * 75)
    print("PARAMETER-MATCHED CLASSICAL BOTTLENECK BASELINE (Referee 1, M1)")
    print("=" * 75)
    print(f"Classical value map params (vs PQC's 36 quantum params): {value_params}")
    print(f"Bottleneck model total params: {total_bn:,}")
    print(f"Classical baseline total params: {total_cl:,}")
    print(f"Rank: {args.rank} | epochs: {args.epochs} | seeds: {args.seeds}")
    print("=" * 75 + "\n")

    models = {
        'bottleneck': ClassicalBottleneckModel,
        'classical': ClassicalModel,
    }

    all_results = []
    for task_name in task_names:
        if task_name not in task_registry:
            print(f"WARNING: Task '{task_name}' not in registry, skipping.")
            continue
        display = TASK_DISPLAY_NAMES.get(task_name, task_name)
        print(f"\n{'='*60}\nTask: {display} ({task_name})\n{'='*60}")
        task = get_task(task_name)

        for mk, mclass in models.items():
            seed_maes, seed_mses = [], []
            for seed_idx in range(args.seeds):
                seed = 42 + seed_idx
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"\n  [{mk}] seed {seed_idx+1}/{args.seeds}")
                if mk == 'bottleneck':
                    model = mclass(hidden_dim=hidden_dim, num_layers=num_layers,
                                   seq_len=seq_len_train, rank=args.rank).to(device)
                else:
                    model = mclass(hidden_dim=hidden_dim, num_layers=num_layers,
                                   seq_len=seq_len_train).to(device)
                save_dir = os.path.join(ckpt_dir, mk, task_name, f'seed{seed}')
                try:
                    t0 = time.time()
                    mae, mse, _ = train_and_evaluate(model, task, task_name, device,
                                                     config, save_dir=save_dir)
                    print(f"  => MAE {mae:.6f}  MSE {mse:.6f}  ({time.time()-t0:.0f}s)")
                    seed_maes.append(mae); seed_mses.append(mse)
                except Exception as e:
                    print(f"  ERROR: {e}"); traceback.print_exc()
                del model

            valid_maes = [m for m in seed_maes if np.isfinite(m)]
            valid_mses = [m for m in seed_mses if np.isfinite(m)]
            all_results.append({
                'task': display, 'task_key': task_name, 'model': mk,
                'mae': np.mean(valid_maes) if valid_maes else np.inf,
                'mse': np.mean(valid_mses) if valid_mses else np.inf,
                'mae_std': np.std(valid_maes) if len(valid_maes) > 1 else 0.0,
                'mse_std': np.std(valid_mses) if len(valid_mses) > 1 else 0.0,
                'n_seeds': len(valid_maes),
            })

    out_csv = os.path.join(results_dir, f"bottleneck_baseline_{timestamp}.csv")
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Model', 'MAE', 'MSE', 'MAE_Std', 'MSE_Std', 'N_Seeds', 'ValueParams'])
        for r in all_results:
            w.writerow([r['task'], r['model'], f"{r['mae']:.6f}", f"{r['mse']:.6f}",
                        f"{r['mae_std']:.6f}", f"{r['mse_std']:.6f}", r['n_seeds'], value_params])

    print(f"\n\n{'='*72}\nBOTTLENECK BASELINE RESULTS (compare vs QASA in Table tab:ts_mae_mse)\n{'='*72}")
    print(f"{'Task':<22}{'Model':<14}{'MAE':>10}{'MSE':>12}{'MAE Std':>10}{'MSE Std':>10}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['task']:<22}{r['model']:<14}{r['mae']:>10.4f}{r['mse']:>12.4f}"
              f"{r['mae_std']:>10.4f}{r['mse_std']:>10.4f}")
    print(f"\nSaved: {out_csv}")
    print(f"Value-map params: {value_params} (PQC quantum params: 36)")


if __name__ == "__main__":
    main()
