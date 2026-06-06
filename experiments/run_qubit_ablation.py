#!/usr/bin/env python3
"""
Qubit-Mapping Ablation (Referee 1, minor 2)
===========================================
Referee 1 asks for a justification/ablation of the 8-data-qubit + 1-ancilla choice
for L=20. We sweep the data-qubit count n in {4, 6, 8, 10} (each with one ancilla),
holding everything else fixed, and report task MAE/MSE on the chaotic-logistic
benchmark (where QASA shows quantum advantage). This complements the existing
gradient-variance scaling analysis (Figure: qubit_scaling) with a *performance*
view, and clarifies that the qubit count is a property of the per-token feature
width (input_proj: d -> n_qubits), independent of the sequence length L.

Usage:
  python experiments/run_qubit_ablation.py --qubits 4 6 8 10 --seeds 3 --epochs 200
  python experiments/run_qubit_ablation.py --dry-run
"""

import os
import sys
import csv
import time
import argparse
import datetime
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task
from experiments.run_baseline_comparison import train_and_evaluate, count_parameters, TASK_DISPLAY_NAMES
from experiments.run_encoding_sensitivity import EncQASAModel   # supports variable n_qubits
from experiments.resume_utils import load_done, append_row, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', nargs='+', type=int, default=[4, 6, 8, 10])
    ap.add_argument('--task', default='classical_chaotic_logistic')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.seeds, args.qubits = 2, 1, [4, 8]

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    config = {'lr': 5e-4, 'weight_decay': 1e-4, 'epochs': args.epochs,
              'seq_len_train': 20, 'print_every': max(1, args.epochs // 5)}
    device = torch.device('cpu')
    task = get_task(args.task)
    disp = TASK_DISPLAY_NAMES.get(args.task, args.task)

    # Resumable per-seed progress CSV (stable name): skip runs already recorded.
    prog_csv = os.path.join(results_dir, f"qubit_ablation_progress_{args.task}.csv")
    done = load_done(prog_csv, key_cols=('N_Qubits', 'Seed'))
    header = ['N_Qubits', 'Q_Params', 'Seed', 'MAE', 'MSE', 'Epochs']

    print(f"{'='*60}\nQUBIT-MAPPING ABLATION  ({disp})  [resume: {len(done)} runs done]\n{'='*60}")
    for nq in args.qubits:
        for s in range(args.seeds):
            seed = 42 + s
            if (str(nq), str(seed)) in done:
                print(f"  n_qubits={nq:<3} seed{seed}: SKIP (already done)")
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            model = EncQASAModel(hidden_dim=64, num_layers=4, seq_len=20,
                                 encoding='angle', n_qubits=nq).to(device)
            q_params = int(model.encoder[-1].v_quantum.qlayer.weights.numel())
            try:
                t0 = time.time()
                mae, mse, _ = train_and_evaluate(model, task, args.task, device, config)
                print(f"  n_qubits={nq:<3} seed{seed}: MAE {mae:.4f} MSE {mse:.4f} "
                      f"(q_params={q_params}, {time.time()-t0:.0f}s)")
                append_row(prog_csv, header,
                           [nq, q_params, seed, f"{mae:.6f}", f"{mse:.6f}", args.epochs])
            except Exception as e:
                print(f"  n_qubits={nq} ERROR: {e}")
            del model

    # Aggregate per qubit count from the progress CSV.
    rows = aggregate(prog_csv, group_col='N_Qubits', val_cols=('MAE', 'MSE'),
                     extra_cols=('Q_Params',))
    print(f"\n{'='*60}\nQUBIT ABLATION RESULTS\n{'='*60}")
    print(f"{'n_qubits':<10}{'q_params':<10}{'MAE':>10}{'MSE':>12}{'N':>5}")
    for r in sorted(rows, key=lambda x: int(x['group'])):
        print(f"{r['group']:<10}{r['Q_Params']:<10}{r['MAE_mean']:>10.4f}{r['MSE_mean']:>12.4f}{r['N']:>5}")
    print(f"\nProgress CSV (resumable): {prog_csv}")


if __name__ == '__main__':
    main()
