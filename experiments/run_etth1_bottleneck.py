#!/usr/bin/env python3
"""
ETTh1 capacity-matched classical bottleneck (Referee 1, M1 — extends the control
to the real-world dataset). Trains the 40-parameter classical low-rank value
bottleneck on ETTh1 OT prediction with the SAME protocol as the published
QASA/Classical ETTh1 run (seq_len=20, hidden_dim=64, 200 epochs, seed 42),
so the three numbers (Classical / bottleneck / QASA) are directly comparable.

Usage:
  python experiments/run_etth1_bottleneck.py --epochs 200
  python experiments/run_etth1_bottleneck.py --dry-run
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_etth1_experiment import (
    download_etth1, load_etth1, train_and_evaluate_etth1, count_parameters,
)
from experiments.run_bottleneck_baseline import ClassicalBottleneckModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seq-len', type=int, default=20)
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        args.epochs = 2

    device = torch.device('cpu')
    hidden_dim, num_layers, seq_len = 64, 4, args.seq_len
    config = {'lr': 5e-4, 'weight_decay': 1e-4, 'epochs': args.epochs,
              'seq_len': seq_len, 'batch_size': 16, 'print_every': max(1, args.epochs // 10)}

    data_dir = os.path.join(PROJECT_ROOT, "experiments", "data")
    csv_path = download_etth1(data_dir)
    X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std = load_etth1(csv_path, seq_len)

    torch.manual_seed(42); np.random.seed(42)
    model = ClassicalBottleneckModel(hidden_dim=hidden_dim, num_layers=num_layers,
                                     seq_len=seq_len, rank=args.rank).to(device)
    vparams = sum(p.numel() for p in model.encoder[-1].v_bottleneck.value_map.parameters())
    total, _ = count_parameters(model)
    print(f"ETTh1 classical bottleneck: value-map params={vparams} (vs PQC 36), total={total:,}")

    save_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", "etth1", "bottleneck")
    mae, mse, best = train_and_evaluate_etth1(model, X_train, Y_train, X_test, Y_test,
                                              device, config, save_dir=save_dir)
    print(f"\n=> ETTh1 classical bottleneck — MAE: {mae:.6f}, MSE: {mse:.6f}")
    print("   (compare: published Classical 0.0718/0.0078, QASA 0.0675/0.0079)")

    out = os.path.join(PROJECT_ROOT, "experiments", "results", "etth1_bottleneck_results.csv")
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Model', 'MAE', 'MSE', 'ValueParams'])
        w.writerow(['ETTh1', 'ClassicalBottleneck', f"{mae:.6f}", f"{mse:.6f}", vparams])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
