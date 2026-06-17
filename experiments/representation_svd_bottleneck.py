#!/usr/bin/env python3
"""
Does the capacity-matched CLASSICAL BOTTLENECK compress like QASA? (Referee 1, minor 1 follow-up)
================================================================================================
The representation analysis (Table~\\ref{tab:effective_rank}, Fig~\\ref{fig:svd_spectrum})
currently compares QASA before/after the quantum layer against the *full-capacity*
classical Transformer (rank ~5.7). It does NOT include the 40-parameter
capacity-matched classical bottleneck. Since that bottleneck matches QASA on the
error metrics, a reviewer will ask whether it ALSO compresses the representation
to ~rank 4 -- in which case the "nonlinear attractor-geometry compression" is a
property of the low-rank *bottleneck*, not of quantumness.

This script answers that directly. It loads the trained classical-bottleneck
checkpoints (no retraining), hooks the input/output of the bottleneck encoder
layer (exactly parallel to QASA's before_quantum/after_quantum hooks), and
computes the SAME diagnostics as representation_svd.py:
  - effective rank (entropy of normalised singular values; matches Table 8)
  - stable rank  r_s = ||X||_F^2/||X||_2^2
  - linear-probe R^2 to the one-step target
and overlays its singular-value spectrum on QASA-after and the full classical.

Outcome to report:
  * If bottleneck-after rank ~ QASA-after rank (~4)  -> compression is a BOTTLENECK
    property; soften the "distinct quantum mechanism" claim (keep my added caveat).
  * If bottleneck-after rank differs markedly        -> the compression IS specific
    to the quantum map; state this as the distinct quantum mechanism (removes caveat).

Usage:
  python experiments/representation_svd_bottleneck.py
  python experiments/representation_svd_bottleneck.py --task classical_chaotic_logistic --seeds 42 43 44
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

from quantum_benchmark.tasks import get_task
from experiments.representation_svd import (
    single_window, build_windows, spectrum, probe_r2, PLOT_DIR,
)
from experiments.run_bottleneck_baseline import ClassicalBottleneckModel
from experiments.quantum_advantage_analysis import load_models, RESULTS_DIR

BN_CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints", "bottleneck_baseline", "bottleneck")


def load_bottleneck(task_key, seed, rank=2):
    """Load a trained classical-bottleneck model and attach before/after hooks
    on the bottleneck encoder layer (parallel to QASA before/after_quantum)."""
    model = ClassicalBottleneckModel(hidden_dim=64, num_layers=4, seq_len=20, rank=rank)
    ckpt_path = os.path.join(BN_CKPT_DIR, task_key, f'seed{seed}', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    store = {}
    last = model.encoder[-1]
    last.register_forward_pre_hook(lambda m, inp: store.__setitem__('before', inp[0].detach().clone()))
    last.register_forward_hook(lambda m, inp, out: store.__setitem__('after', out.detach().clone()))
    return model, store


def bn_window_features(model, store, X, which):
    with torch.no_grad():
        _ = model(X)
    return store[which][0].numpy()                       # (L, hidden)


def bn_batch_features(model, store, X, which):
    with torch.no_grad():
        _ = model(X)
    feats = store[which].numpy()
    M, L, H = feats.shape
    return feats.reshape(M * L, H)


def analyze(task_key, seeds, rank=2):
    task = get_task(task_key)
    Xwin = single_window(task)
    Xbatch, ys = build_windows(task)
    y_flat = ys.reshape(-1)

    rows = []
    bn_after_spectra = []
    qasa_after_spectra = []
    classical_spectra = []

    for seed in seeds:
        # --- classical bottleneck ---
        try:
            model, store = load_bottleneck(task_key, seed, rank=rank)
        except FileNotFoundError as e:
            print(f"  [seed {seed}] bottleneck checkpoint missing, skip: {e}")
            continue
        for which, label in [('before', 'bottleneck_before'), ('after', 'bottleneck_after')]:
            Fwin = bn_window_features(model, store, Xwin, which)
            s_norm, er, sr = spectrum(Fwin)
            r2 = probe_r2(bn_batch_features(model, store, Xbatch, which), y_flat)
            rows.append({'task': task_key, 'seed': seed, 'stage': label,
                         'eff_rank': er, 'stable_rank': sr, 'probe_r2': r2})
            if which == 'after':
                bn_after_spectra.append(s_norm)
            print(f"  [seed {seed}] {label:18s} eff_rank={er:5.2f} stable_rank={sr:5.2f} probe_R2={r2:6.3f}")

        # --- QASA after + full classical (for the overlay / direct comparison) ---
        try:
            qasa, classical = load_models(task_key=task_key, seed=seed)
            with torch.no_grad():
                _ = qasa(Xwin)
                _ = classical(Xwin)
            qa = qasa.features['after_quantum'][0].numpy()
            cl = classical.features['after_layer_3'][0].numpy()
            s_qa, er_qa, sr_qa = spectrum(qa)
            s_cl, er_cl, sr_cl = spectrum(cl)
            qasa_after_spectra.append(s_qa)
            classical_spectra.append(s_cl)
            rows.append({'task': task_key, 'seed': seed, 'stage': 'qasa_after',
                         'eff_rank': er_qa, 'stable_rank': sr_qa, 'probe_r2': np.nan})
            rows.append({'task': task_key, 'seed': seed, 'stage': 'classical_full',
                         'eff_rank': er_cl, 'stable_rank': sr_cl, 'probe_r2': np.nan})
            print(f"  [seed {seed}] {'qasa_after':18s} eff_rank={er_qa:5.2f} stable_rank={sr_qa:5.2f}")
            print(f"  [seed {seed}] {'classical_full':18s} eff_rank={er_cl:5.2f} stable_rank={sr_cl:5.2f}")
        except Exception as e:
            print(f"  [seed {seed}] QASA/classical checkpoint comparison skipped: {e}")

    return rows, bn_after_spectra, qasa_after_spectra, classical_spectra


def plot_overlay(task_key, bn, qa, cl):
    series = [('Classical bottleneck (after)', bn, 'tab:green'),
              ('QASA (after quantum)', qa, 'tab:red'),
              ('Classical full (final layer)', cl, 'tab:blue')]
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, runs, color in series:
        if not runs:
            continue
        K = min(len(r) for r in runs)
        arr = np.stack([r[:K] for r in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(1, K + 1)
        ax.plot(x, mean, '-o', ms=4, color=color, label=label)
        ax.fill_between(x, np.clip(mean - std, 1e-6, None), mean + std, color=color, alpha=0.2)
    ax.set_yscale('log')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalised singular value')
    ax.set_title(f'Bottleneck vs QASA vs full-classical spectrum: {task_key}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(PLOT_DIR, f'svd_bottleneck_{task_key}.pdf')
    plt.tight_layout()
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
    ap.add_argument('--rank', type=int, default=2)
    args = ap.parse_args()

    all_rows = []
    for task in args.tasks:
        print(f"\n=== {task} (classical bottleneck representation analysis) ===")
        rows, bn, qa, cl = analyze(task, args.seeds, rank=args.rank)
        all_rows.extend(rows)
        if bn:
            plot_overlay(task, bn, qa, cl)

    out_csv = os.path.join(RESULTS_DIR, 'representation_svd_bottleneck_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Stage', 'EffRank_mean', 'EffRank_std',
                    'StableRank_mean', 'StableRank_std', 'ProbeR2_mean', 'ProbeR2_std', 'N'])
        for task in args.tasks:
            for stage in ['bottleneck_before', 'bottleneck_after', 'qasa_after', 'classical_full']:
                sub = [r for r in all_rows if r['task'] == task and r['stage'] == stage]
                if not sub:
                    continue
                er = [r['eff_rank'] for r in sub]
                sr = [r['stable_rank'] for r in sub]
                r2 = [r['probe_r2'] for r in sub if not np.isnan(r['probe_r2'])]
                w.writerow([task, stage,
                            f"{np.mean(er):.3f}", f"{np.std(er):.3f}",
                            f"{np.mean(sr):.3f}", f"{np.std(sr):.3f}",
                            f"{np.mean(r2):.4f}" if r2 else "nan",
                            f"{np.std(r2):.4f}" if r2 else "nan", len(sub)])

    print(f"\nSummary saved: {out_csv}")
    print("\nTable-8 bottleneck_after effective rank vs QASA-after decides the framing:")
    print("  - if ~equal/lower -> compression is a bottleneck property (caveat stays)")
    print("  - if much higher  -> distinct quantum mechanism (would remove the caveat)")


if __name__ == "__main__":
    main()
