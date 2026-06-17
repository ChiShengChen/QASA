#!/usr/bin/env python3
"""
Regenerate Fig.~\\ref{fig:svd_spectrum} WITH the classical-bottleneck curve.
============================================================================
The published svd_spectrum figure shows QASA before/after the quantum layer and
the full classical model. Following the bottleneck control (Referee 1), we add a
fourth curve: the capacity-matched classical bottleneck after its value layer.
It concentrates spectral energy at least as fast as QASA, visually confirming
that the compression is a bottleneck property, not a quantum one.

Writes directly to the manuscript's img/svd_spectrum.pdf (same 2-panel layout
and caption structure as before, plus the new purple bottleneck curve).

Usage:
  python experiments/make_svd_figure.py --task classical_chaotic_logistic --seeds 42 43 44
"""

import os
import sys
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2.0, 'lines.markersize': 6,
})

from quantum_benchmark.tasks import get_task
from experiments.representation_svd import single_window, spectrum
from experiments.representation_svd_bottleneck import load_bottleneck, bn_window_features
from experiments.quantum_advantage_analysis import load_models

IMG_DIR = os.path.join(PROJECT_ROOT, "IOP_QST_journel_QASA", "img")


def collect(task_key, seeds):
    task = get_task(task_key)
    Xwin = single_window(task)
    out = {'before': [], 'after': [], 'classical': [], 'bottleneck': []}
    for seed in seeds:
        try:
            qasa, classical = load_models(task_key=task_key, seed=seed)
            with torch.no_grad():
                _ = qasa(Xwin); _ = classical(Xwin)
            out['before'].append(spectrum(qasa.features['before_quantum'][0].numpy())[0])
            out['after'].append(spectrum(qasa.features['after_quantum'][0].numpy())[0])
            out['classical'].append(spectrum(classical.features['after_layer_3'][0].numpy())[0])
        except Exception as e:
            print(f"  [seed {seed}] QASA/classical skip: {e}")
        try:
            model, store = load_bottleneck(task_key, seed)
            out['bottleneck'].append(spectrum(bn_window_features(model, store, Xwin, 'after'))[0])
        except Exception as e:
            print(f"  [seed {seed}] bottleneck skip: {e}")
    return out


def plot(out, task_key):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    style = [
        ('before', 'QASA: before quantum', 'tab:blue', '-o'),
        ('after', 'QASA: after quantum', 'tab:red', '-o'),
        ('bottleneck', 'Classical bottleneck: after', 'tab:purple', '-s'),
        ('classical', 'Classical (full): final layer', 'tab:green', '-^'),
    ]
    for key, label, color, mk in style:
        runs = out[key]
        if not runs:
            continue
        K = min(len(r) for r in runs)
        arr = np.stack([r[:K] for r in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(1, K + 1)
        axes[0].plot(x, mean, mk, color=color, label=label)
        axes[0].fill_between(x, np.clip(mean - std, 1e-6, None), mean + std, color=color, alpha=0.18)
        cum = np.cumsum(arr ** 2, axis=1)
        cum = cum / cum[:, -1:].clip(1e-12)
        axes[1].plot(x, cum.mean(0), mk, color=color, label=label)

    axes[0].set_yscale('log')
    axes[0].set_xlabel('Singular value index')
    axes[0].set_ylabel('Normalised singular value')
    axes[0].set_title('(a) Singular value spectrum')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative energy fraction')
    axes[1].set_title('(b) Cumulative spectral energy')
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out_pdf = os.path.join(IMG_DIR, 'svd_spectrum.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_pdf.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='classical_chaotic_logistic')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = ap.parse_args()
    print(f"=== regenerating svd_spectrum.pdf for {args.task} (with bottleneck curve) ===")
    out = collect(args.task, args.seeds)
    plot(out, args.task)


if __name__ == "__main__":
    main()
