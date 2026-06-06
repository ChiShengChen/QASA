#!/usr/bin/env python3
"""
Extended Evaluation Metrics (Referee 2, comment 1)
==================================================
Referee 2 notes that MSE/MAE alone can mask overfitting and asks for more
convincing, complementary indicators (e.g. prediction "accuracy").

For every trained checkpoint (model x task x seed) we recompute, WITHOUT
retraining, a richer set of metrics:

  * R^2          - coefficient of determination (variance explained)
  * DirAcc       - directional accuracy: fraction of steps whose predicted change
                   sign matches the ground-truth change sign (the "accuracy" metric)
  * SMAPE        - symmetric mean absolute percentage error (scale-free)
  * Train/Test one-step MSE gap - teacher-forced MSE on the train vs test segment;
                   a large ratio flags overfitting (directly answers the concern)

The autoregressive MAE/MSE reproduce the paper's Tables for consistency.

Usage:
  python experiments/metrics_extended.py --models classical qasa qlstm qnnformer
  python experiments/metrics_extended.py --models classical            # fast subset
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task
from experiments.run_baseline_comparison import (
    QASAModel, ClassicalModel, BENCHMARK_TASKS, TASK_DISPLAY_NAMES,
)
from baselines.qlstm_model import QLSTMModel
from baselines.qnnformer_model import QnnFormerModel

CKPT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", "baseline_comparison")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

MODEL_CLASSES = {
    'classical': ClassicalModel,
    'qasa': QASAModel,
    'qlstm': QLSTMModel,
    'qnnformer': QnnFormerModel,
}


# ----------------------------- metrics -----------------------------

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-8
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]))


def directional_accuracy(y_true, y_pred):
    """Fraction of steps where sign(Δy_pred) == sign(Δy_true)."""
    dt = np.diff(y_true)
    dp = np.diff(y_pred)
    mask = np.abs(dt) > 1e-8
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.sign(dt[mask]) == np.sign(dp[mask])))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')


# ------------------------ data reconstruction ----------------------

def full_series(task, seq_len=20):
    """Reconstruct the full 500-point series and the train/test boundary index."""
    X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()
    # series[:400] = X_train inputs ; series[400] = Y_train[-1] ; series[401:] = Y_test_true
    head = X_train[0, :, 0].numpy()                 # series[0:400]
    mid = Y_train[0, -1:, 0].numpy()                # series[400]
    tail = Y_test_true[0, :, 0].numpy()             # series[401:500]
    series = np.concatenate([head, mid, tail])
    boundary = head.shape[0]                          # 400
    return series.astype(np.float32), boundary, X_test_seed, Y_test_true


def teacher_forced_mse(model, series, lo, hi, seq_len=20, device='cpu'):
    """One-step teacher-forced MSE over windows whose target index lies in [lo,hi)."""
    xs, ys = [], []
    for i in range(0, len(series) - seq_len):
        tgt = i + seq_len
        if lo <= tgt < hi:
            xs.append(series[i:i + seq_len])
            ys.append(series[i + 1:i + seq_len + 1])
    if not xs:
        return float('nan')
    X = torch.from_numpy(np.stack(xs)).float().unsqueeze(-1).to(device)
    Y = torch.from_numpy(np.stack(ys)).float().unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(X)
    return float(nn.functional.mse_loss(pred, Y).item())


def autoregressive_predict(model, X_test_seed, n_steps, device='cpu'):
    model.eval()
    cur = X_test_seed.clone().to(device)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(cur)
            nxt = pred[:, -1:, :].clone()
            out.append(nxt.squeeze().cpu().numpy())
            cur = torch.cat([cur[:, 1:, :], nxt], dim=1)
    return np.array(out).flatten()


def evaluate_checkpoint(model_key, task_key, seed, device='cpu'):
    ckpt_path = os.path.join(CKPT_DIR, model_key, task_key, f'seed{seed}', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        return None
    task = get_task(task_key)
    series, boundary, X_test_seed, Y_test_true = full_series(task)

    model = MODEL_CLASSES[model_key](hidden_dim=64, num_layers=4, seq_len=20).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # autoregressive test metrics
    y_true = Y_test_true.squeeze().cpu().numpy().flatten()
    y_pred = autoregressive_predict(model, X_test_seed, len(y_true), device)
    m = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:m], y_pred[:m]

    # teacher-forced overfitting gap
    tf_train = teacher_forced_mse(model, series, 0, boundary, device=device)
    tf_test = teacher_forced_mse(model, series, boundary, len(series), device=device)
    gap = float(tf_test / tf_train) if tf_train and tf_train > 1e-12 else float('nan')

    return {
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'mse': float(np.mean((y_true - y_pred) ** 2)),
        'r2': r2(y_true, y_pred),
        'dir_acc': directional_accuracy(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'tf_train_mse': tf_train,
        'tf_test_mse': tf_test,
        'overfit_gap': gap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['classical', 'qasa', 'qlstm', 'qnnformer'])
    ap.add_argument('--tasks', nargs='+', default=BENCHMARK_TASKS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = ap.parse_args()

    rows = []
    for task_key in args.tasks:
        if not task_key.startswith('classical_'):
            task_key = f'classical_{task_key}'
        disp = TASK_DISPLAY_NAMES.get(task_key, task_key)
        for mk in args.models:
            accum = []
            for seed in args.seeds:
                r = evaluate_checkpoint(mk, task_key, seed)
                if r is not None:
                    accum.append(r)
            if not accum:
                print(f"  {disp:<18} {mk:<10} (no checkpoints)")
                continue
            agg = {k: np.nanmean([a[k] for a in accum]) for k in accum[0]}
            agg_std = {k: np.nanstd([a[k] for a in accum]) for k in accum[0]}
            rows.append((disp, task_key, mk, agg, agg_std, len(accum)))
            print(f"  {disp:<18} {mk:<10} R2={agg['r2']:.3f} DirAcc={agg['dir_acc']:.3f} "
                  f"SMAPE={agg['smape']:6.1f} gap(test/train)={agg['overfit_gap']:.2f}")

    out_csv = os.path.join(RESULTS_DIR, 'extended_metrics.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'TaskKey', 'Model', 'N',
                    'MAE', 'MSE', 'R2', 'DirAcc', 'SMAPE',
                    'TF_Train_MSE', 'TF_Test_MSE', 'Overfit_Gap',
                    'R2_std', 'DirAcc_std', 'SMAPE_std'])
        for disp, tk, mk, agg, std, n in rows:
            w.writerow([disp, tk, mk, n,
                        f"{agg['mae']:.6f}", f"{agg['mse']:.6f}", f"{agg['r2']:.4f}",
                        f"{agg['dir_acc']:.4f}", f"{agg['smape']:.3f}",
                        f"{agg['tf_train_mse']:.6f}", f"{agg['tf_test_mse']:.6f}",
                        f"{agg['overfit_gap']:.4f}",
                        f"{std['r2']:.4f}", f"{std['dir_acc']:.4f}", f"{std['smape']:.3f}"])
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
