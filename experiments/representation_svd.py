#!/usr/bin/env python3
"""
Singular-Value Distribution & Meaningful-Compression Analysis (Referee 1, minor 1)
==================================================================================
Referee 1 asks whether the 49% effective-rank reduction induced by the quantum
layer is *meaningful compression* or *rank collapse* (information loss).

We distinguish the two by analysing the FULL singular-value spectrum of the
feature matrices before vs after the quantum layer (and the classical baseline's
final layer), plus two diagnostics that separate compression from collapse:

  1. Stable rank  r_s = ||X||_F^2 / ||X||_2^2 = (sum s_i^2) / (max s_i)^2.
     A smooth spectrum -> high stable rank; collapse to one direction -> r_s -> 1.
  2. Linear-probe R^2: fit a ridge regressor from the features to the one-step
     prediction target. If the post-quantum features retain (or improve) probe
     R^2 despite lower rank, the discarded directions were redundant ->
     compression is *meaningful*, not lossy collapse.

Uses the already-trained checkpoints (no retraining).

Usage:
  python experiments/representation_svd.py
  python experiments/representation_svd.py --tasks classical_chaotic_logistic --seeds 42 43 44
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from quantum_benchmark.tasks import get_task
from experiments.quantum_advantage_analysis import (
    QASAModelAnalysis, ClassicalModelAnalysis, load_models, RESULTS_DIR,
)

PLOT_DIR = os.path.join(RESULTS_DIR, "plots", "representation_svd")
os.makedirs(PLOT_DIR, exist_ok=True)


def single_window(task):
    """The single test-seed window (1, L, 1), matching the paper's effective-rank
    protocol (Table effective_rank reproduces from this)."""
    _, _, X_test_seed, _ = task.generate_data()
    return X_test_seed.float()


def build_windows(task, seq_len=20, max_windows=64):
    """Many sliding windows from the training series, for the linear probe."""
    X_train, Y_train, _, _ = task.generate_data()
    series_x = X_train[0, :, 0].numpy()
    series_y = Y_train[0, :, 0].numpy()
    n = len(series_x) - seq_len
    idx = np.linspace(0, n - 1, min(max_windows, n)).astype(int)
    xs = np.stack([series_x[i:i + seq_len] for i in idx])
    ys = np.stack([series_y[i:i + seq_len] for i in idx])
    X = torch.from_numpy(xs).float().unsqueeze(-1)
    return X, ys


def window_features(model, X, key):
    """Per-window token features for the first batch element (L, hidden),
    matching the paper's single-window analysis."""
    with torch.no_grad():
        _ = model(X)
    return model.features[key][0].numpy()      # (L, hidden)


def batch_features(model, X, key):
    """Stacked token features (M*L, hidden) across all windows (for the probe)."""
    with torch.no_grad():
        _ = model(X)
    feats = model.features[key].numpy()
    M, L, H = feats.shape
    return feats.reshape(M * L, H)


def spectrum(X):
    """Paper-consistent metrics on an (L, hidden) feature matrix.

    effective_rank: entropy of normalised singular values s/sum(s) (uncentered) —
                    identical to quantum_advantage_analysis.effective_rank, so the
                    reported numbers match Table~\\ref{tab:effective_rank}.
    stable_rank:    (sum s_i^2)/(max s_i)^2; ->1 means collapse onto one direction.
    """
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-10]
    s_sum_norm = s / s.sum()
    eff_rank = float(np.exp(-np.sum(s_sum_norm * np.log(s_sum_norm))))
    stable_rank = float((s ** 2).sum() / (s[0] ** 2))
    return s_sum_norm, eff_rank, stable_rank


def probe_r2(X, y, n_train=None):
    """Ridge linear-probe R^2 from features to the one-step target (held-out split).
    R^2 ~ 1 despite a lower rank => task-relevant information is fully retained."""
    N = X.shape[0]
    if n_train is None:
        n_train = int(0.7 * N)
    Xc = X - X.mean(axis=0, keepdims=True)
    reg = Ridge(alpha=1.0).fit(Xc[:n_train], y[:n_train])
    return float(r2_score(y[n_train:], reg.predict(Xc[n_train:])))


def analyze(task_key, seeds):
    task = get_task(task_key)
    Xwin = single_window(task)                 # paper protocol: eff_rank + spectrum
    Xbatch, ys = build_windows(task)           # many windows: linear probe
    y_flat = ys.reshape(-1)

    spectra = {'before': [], 'after': [], 'classical': []}
    rows = []
    for seed in seeds:
        try:
            qasa, classical = load_models(task_key=task_key, seed=seed)
        except Exception as e:
            print(f"  [seed {seed}] checkpoint missing, skip: {e}")
            continue
        feats = {
            'before': (window_features(qasa, Xwin, 'before_quantum'),
                       batch_features(qasa, Xbatch, 'before_quantum')),
            'after': (window_features(qasa, Xwin, 'after_quantum'),
                      batch_features(qasa, Xbatch, 'after_quantum')),
            'classical': (window_features(classical, Xwin, 'after_layer_3'),
                          batch_features(classical, Xbatch, 'after_layer_3')),
        }
        for name, (Fwin, Fbatch) in feats.items():
            s_norm, er, sr = spectrum(Fwin)
            r2 = probe_r2(Fbatch, y_flat)
            spectra[name].append(s_norm)
            rows.append({'task': task_key, 'seed': seed, 'stage': name,
                         'eff_rank': er, 'stable_rank': sr, 'probe_r2': r2})
            print(f"  [seed {seed}] {name:9s} eff_rank={er:5.2f} stable_rank={sr:5.2f} probe_R2={r2:6.3f}")
    return spectra, rows


def plot_spectra(task_key, spectra):
    """Mean +- std normalised singular value spectra: before/after/classical."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'before': 'tab:blue', 'after': 'tab:red', 'classical': 'tab:green'}
    labels = {'before': 'QASA: before quantum', 'after': 'QASA: after quantum',
              'classical': 'Classical: final layer'}

    for name, runs in spectra.items():
        if not runs:
            continue
        K = min(len(r) for r in runs)
        arr = np.stack([r[:K] for r in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(1, K + 1)
        # (a) singular value spectrum (log scale)
        axes[0].plot(x, mean, '-o', ms=3, color=colors[name], label=labels[name])
        axes[0].fill_between(x, np.clip(mean - std, 1e-6, None), mean + std,
                             color=colors[name], alpha=0.2)
        # (b) cumulative energy
        cum = np.cumsum(arr ** 2, axis=1)
        cum = cum / cum[:, -1:].clip(1e-12)
        axes[1].plot(x, cum.mean(0), '-o', ms=3, color=colors[name], label=labels[name])

    axes[0].set_yscale('log')
    axes[0].set_xlabel('Singular value index')
    axes[0].set_ylabel('Normalised singular value')
    axes[0].set_title('(a) Singular value spectrum')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative energy fraction')
    axes[1].set_title('(b) Cumulative spectral energy')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle(f'Singular-value distribution: {task_key}', fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'svd_spectrum_{task_key}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=[
        'classical_chaotic_logistic',
        'classical_damped_oscillation',
        'classical_square_triangle_wave',
    ])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = ap.parse_args()

    all_rows = []
    for task_key in args.tasks:
        print(f"\n=== {task_key} ===")
        spectra, rows = analyze(task_key, args.seeds)
        all_rows.extend(rows)
        if any(spectra.values()):
            plot_spectra(task_key, spectra)

    # Aggregate table (mean +- std across seeds)
    out_csv = os.path.join(RESULTS_DIR, 'representation_svd_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Stage', 'EffRank_mean', 'EffRank_std',
                    'StableRank_mean', 'StableRank_std', 'ProbeR2_mean', 'ProbeR2_std'])
        for task_key in args.tasks:
            for stage in ['before', 'after', 'classical']:
                sub = [r for r in all_rows if r['task'] == task_key and r['stage'] == stage]
                if not sub:
                    continue
                er = [r['eff_rank'] for r in sub]
                sr = [r['stable_rank'] for r in sub]
                r2 = [r['probe_r2'] for r in sub]
                w.writerow([task_key, stage,
                            f"{np.mean(er):.3f}", f"{np.std(er):.3f}",
                            f"{np.mean(sr):.3f}", f"{np.std(sr):.3f}",
                            f"{np.mean(r2):.4f}", f"{np.std(r2):.4f}"])
    print(f"\nSummary saved: {out_csv}")


if __name__ == '__main__':
    main()
