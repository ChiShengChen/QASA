#!/usr/bin/env python3
"""Make the MAE-vs-sequence-length figure (Referee 1, major 2) from the
seqlen_scaling progress CSV. Two panels (chaotic logistic, seasonal trend),
QASA vs classical, MAE on the y-axis, L on the x-axis."""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
CSV = os.path.join(RESULTS_DIR, "seqlen_scaling_progress.csv")
OUT = os.path.join(RESULTS_DIR, "plots", "seqlen_scaling.pdf")

TASKS = [('classical_chaotic_logistic', 'Chaotic Logistic'),
         ('classical_trend_seasonality_noise', 'Seasonal Trend')]
COLORS = {'qasa': 'tab:red', 'classical': 'tab:blue'}
LABELS = {'qasa': 'QASA (8-qubit PQC)', 'classical': 'Classical Transformer'}


def load():
    data = {}  # (taskkey, model) -> {L: [maes]}
    with open(CSV, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['TaskKey'], r['Model'])
            data.setdefault(key, {}).setdefault(int(r['SeqLen']), []).append(float(r['MAE']))
    return data


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = load()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (tk, disp) in zip(axes, TASKS):
        for model in ('classical', 'qasa'):
            d = data.get((tk, model))
            if not d:
                continue
            Ls = sorted(d)
            means = [np.mean(d[L]) for L in Ls]
            stds = [np.std(d[L]) if len(d[L]) > 1 else 0 for L in Ls]
            ax.errorbar(Ls, means, yerr=stds, marker='o', ms=6, capsize=3,
                        color=COLORS[model], label=LABELS[model])
        ax.set_title(disp)
        ax.set_xlabel('Input sequence length $L$')
        ax.set_ylabel('Test MAE')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('Scalability to longer sequences: QASA vs classical', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches='tight')
    plt.savefig(OUT.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"saved {OUT}")
    # print table for the manuscript
    print("\nMAE by L:")
    for tk, disp in TASKS:
        for model in ('classical', 'qasa'):
            d = data.get((tk, model), {})
            row = "  ".join(f"L{L}={np.mean(v):.4f}" for L, v in sorted(d.items()))
            print(f"  {disp:<16} {model:<10} {row}")


if __name__ == '__main__':
    main()
