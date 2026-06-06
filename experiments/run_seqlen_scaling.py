#!/usr/bin/env python3
"""
Sequence-Length Scalability (Referee 1, major 2)
================================================
Referee 1 notes L=20 is short vs modern horizons (96-720) and asks whether the
parsimony principle holds as L grows, or whether the 8-qubit block becomes a
ceiling.

Key architectural point this study makes concrete: the PQC width is INDEPENDENT
of L. The quantum layer acts on the per-token hidden projection (d -> 8 qubits),
so increasing L enlarges only the classical O(L^2) self-attention, not the quantum
register. We sweep L in {20, 48, 96, 192} (cap 192; L=720 is left to an analytic
extrapolation, as full-state-vector simulation at that horizon is intractable on
CPU) and report MAE for QASA vs the classical Transformer.

Windowing is self-contained (reconstructs the full 500-point series and builds
length-L windows) so it works for every task regardless of its generator's
hard-coded 20-point test seed.

Usage:
  python experiments/run_seqlen_scaling.py --models classical qasa --seq-lens 20 48 96
  python experiments/run_seqlen_scaling.py --models classical --seq-lens 20 48 96 192  # cheap sweep
  python experiments/run_seqlen_scaling.py --dry-run
"""

import os
import sys
import csv
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task
from experiments.run_baseline_comparison import QASAModel, ClassicalModel, TASK_DISPLAY_NAMES
from experiments.resume_utils import load_done, append_row, aggregate

MODEL_CLASSES = {'classical': ClassicalModel, 'qasa': QASAModel}


def reconstruct_series(task):
    """Full 500-point series + train/test boundary index (matches the 80/20 split)."""
    X_train, Y_train, _, Y_test_true = task.generate_data()
    head = X_train[0, :, 0].numpy()
    mid = Y_train[0, -1:, 0].numpy()
    tail = Y_test_true[0, :, 0].numpy()
    series = np.concatenate([head, mid, tail]).astype(np.float32)
    return series, head.shape[0]


def make_windows(series, lo, hi, L):
    """Length-L windows whose target index lies in [lo, hi)."""
    xs, ys = [], []
    for i in range(0, len(series) - L):
        if lo <= i + L < hi:
            xs.append(series[i:i + L]); ys.append(series[i + 1:i + L + 1])
    X = torch.from_numpy(np.stack(xs)).float().unsqueeze(-1)
    Y = torch.from_numpy(np.stack(ys)).float().unsqueeze(-1)
    return X, Y


def train_eval_L(model, series, boundary, L, epochs, device, batch=64):
    Xtr, Ytr = make_windows(series, 0, boundary, L)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    n = Xtr.shape[0]
    model.train()
    best, best_state = float('inf'), None
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for j in range(0, n, batch):
            idx = perm[j:j + batch]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        avg = tot / max(1, (n + batch - 1) // batch)
        if avg < best:
            best = avg; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # autoregressive test rollout over the test segment
    model.eval()
    seed_seq = torch.from_numpy(series[boundary - L:boundary]).float().view(1, L, 1).to(device)
    y_true = series[boundary:]
    preds, cur = [], seed_seq.clone()
    with torch.no_grad():
        for _ in range(len(y_true)):
            nxt = model(cur)[:, -1:, :].clone()
            preds.append(float(nxt.squeeze().cpu()))
            cur = torch.cat([cur[:, 1:, :], nxt], dim=1)
    yp = np.array(preds[:len(y_true)]); yt = y_true[:len(yp)]
    mae = float(np.mean(np.abs(yt - yp))); mse = float(np.mean((yt - yp) ** 2))
    return mae, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['classical', 'qasa'])
    ap.add_argument('--tasks', nargs='+', default=['classical_chaotic_logistic',
                                                   'classical_trend_seasonality_noise'])
    ap.add_argument('--seq-lens', nargs='+', type=int, default=[20, 48, 96, 192])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.seeds = 2, 1
        args.seq_lens = [20, 48]
        args.tasks = ['classical_chaotic_logistic']

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    device = torch.device('cpu')

    # Resumable per-(task, seqlen, model, seed) progress CSV.
    prog_csv = os.path.join(results_dir, "seqlen_scaling_progress.csv")
    done = load_done(prog_csv, key_cols=('TaskKey', 'SeqLen', 'Model', 'Seed'))
    header = ['Task', 'TaskKey', 'SeqLen', 'Model', 'Seed', 'MAE', 'MSE', 'Epochs']
    print(f"[resume: {len(done)} runs done]")

    for task_key in args.tasks:
        task = get_task(task_key)
        disp = TASK_DISPLAY_NAMES.get(task_key, task_key)
        series, boundary = reconstruct_series(task)
        print(f"\n{'='*60}\nTask: {disp}  (series={len(series)}, boundary={boundary})\n{'='*60}")
        for L in args.seq_lens:
            if L >= boundary:
                print(f"  L={L} >= train length {boundary}, skip"); continue
            for mk in args.models:
                for s in range(args.seeds):
                    seed = 42 + s
                    if (task_key, str(L), mk, str(seed)) in done:
                        print(f"  L={L:<4}{mk:<10} seed{seed}: SKIP (already done)")
                        continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = MODEL_CLASSES[mk](hidden_dim=64, num_layers=4, seq_len=L).to(device)
                    try:
                        t0 = time.time()
                        mae, mse = train_eval_L(model, series, boundary, L, args.epochs, device)
                        print(f"  L={L:<4}{mk:<10} seed{seed}: MAE {mae:.4f} MSE {mse:.4f} ({time.time()-t0:.0f}s)")
                        append_row(prog_csv, header,
                                   [disp, task_key, L, mk, seed, f"{mae:.6f}", f"{mse:.6f}", args.epochs])
                    except Exception as e:
                        print(f"  L={L} {mk} ERROR: {e}")
                    del model

    print(f"\n{'='*60}\nSEQUENCE-LENGTH SCALING (MAE, mean over seeds)\n{'='*60}")
    if os.path.exists(prog_csv):
        import csv as _csv
        agg = {}
        with open(prog_csv, newline='') as f:
            for r in _csv.DictReader(f):
                agg.setdefault((r['Task'], r['SeqLen'], r['Model']), []).append(float(r['MAE']))
        for (tk, L, mk) in sorted(agg, key=lambda x: (x[0], int(x[1]), x[2])):
            vals = agg[(tk, L, mk)]
            print(f"  {tk:<16} L={L:<4} {mk:<10} MAE {np.mean(vals):.4f}  (n={len(vals)})")
    print(f"\nProgress CSV (resumable): {prog_csv}")


if __name__ == '__main__':
    main()
