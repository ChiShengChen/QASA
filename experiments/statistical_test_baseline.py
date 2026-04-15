#!/usr/bin/env python3
"""
Statistical significance tests for baseline comparison (4 models × 9 tasks × 3 seeds).
Recovers per-seed results from checkpoint files and runs:
  1. Pairwise paired t-tests (QASA vs each baseline)
  2. Friedman test across all 4 models per task
  3. Cohen's d effect sizes
  4. Cross-task win counts with binomial test
"""

import os, sys
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

CKPT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", "baseline_comparison")

MODELS = ["classical", "qasa", "qlstm", "qnnformer"]
MODEL_NAMES = {"classical": "Classical", "qasa": "QASA", "qlstm": "QLSTM", "qnnformer": "QnnFormer"}

TASKS = [
    ("classical_arma", "ARMA"),
    ("classical_chaotic_logistic", "Chaotic Logistic"),
    ("classical_damped_oscillation", "Damped Oscillator"),
    ("classical_noisy_damped_oscillation", "Noisy Damped Osc"),
    ("classical_piecewise_regime", "Piecewise Regime"),
    ("classical_sawtooth_wave", "Sawtooth"),
    ("classical_square_triangle_wave", "Square Wave"),
    ("classical_trend_seasonality_noise", "Seasonal Trend"),
    ("classical_waveform", "Waveform"),
]

SEEDS = [42, 43, 44]


def load_results():
    """Load per-seed MAE/MSE from checkpoints."""
    results = {}  # {(task_key, model): {"mae": [...], "mse": [...]}}
    for tk, tname in TASKS:
        for mk in MODELS:
            maes, mses = [], []
            for seed in SEEDS:
                path = os.path.join(CKPT_DIR, mk, tk, f"seed{seed}", "best_model.pth")
                if os.path.exists(path):
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    if ckpt.get("epochs", 0) == 200:
                        maes.append(ckpt["mae"])
                        mses.append(ckpt["mse"])
            results[(tk, mk)] = {"mae": np.array(maes), "mse": np.array(mses)}
    return results


def cohens_d(a, b):
    """Compute Cohen's d for paired samples."""
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def main():
    results = load_results()

    # ============================================================
    # 1. Pairwise: QASA vs each other model (paired t-test)
    # ============================================================
    print("=" * 110)
    print("PAIRWISE COMPARISON: QASA vs Each Baseline (paired t-test, 3 seeds)")
    print("=" * 110)

    for opponent in ["classical", "qlstm", "qnnformer"]:
        opp_name = MODEL_NAMES[opponent]
        print(f"\n--- QASA vs {opp_name} ---")
        print(f"{'Task':<22} {'Metric':<6} {'QASA':>14} {opp_name:>14} {'Cohen d':>8} {'t-stat':>8} {'p-value':>8} {'Better'}")
        print("-" * 110)

        for tk, tname in TASKS:
            qasa = results[(tk, "qasa")]
            opp = results[(tk, opponent)]

            for metric in ["mae", "mse"]:
                qv = qasa[metric]
                ov = opp[metric]

                if len(qv) < 2 or len(ov) < 2 or len(qv) != len(ov):
                    prefix = tname if metric == "mae" else ""
                    print(f"{prefix:<22} {metric.upper():<6} {'insufficient data':>40}")
                    continue

                t_stat, p_val = stats.ttest_rel(qv, ov)
                d = cohens_d(ov, qv)  # positive d = QASA better
                q_mean, q_std = qv.mean(), qv.std(ddof=1)
                o_mean, o_std = ov.mean(), ov.std(ddof=1)
                better = "QASA" if q_mean < o_mean else opp_name
                sig = "*" if p_val < 0.05 else " "

                prefix = tname if metric == "mae" else ""
                print(f"{prefix:<22} {metric.upper():<6} "
                      f"{q_mean:>7.4f}±{q_std:<6.4f} "
                      f"{o_mean:>7.4f}±{o_std:<6.4f} "
                      f"{d:>8.3f} {t_stat:>8.3f} {p_val:>7.4f}{sig} {better}")

            print("-" * 110)

    # ============================================================
    # 2. Friedman test per task (non-parametric, all 4 models)
    # ============================================================
    print("\n\n" + "=" * 80)
    print("FRIEDMAN TEST (non-parametric, all 4 models)")
    print("=" * 80)
    print(f"{'Task':<22} {'Metric':<6} {'Chi2':>8} {'p-value':>8} {'Sig':>4}")
    print("-" * 80)

    for tk, tname in TASKS:
        for metric in ["mae", "mse"]:
            arrays = []
            valid = True
            for mk in MODELS:
                arr = results[(tk, mk)][metric]
                if len(arr) != 3:
                    valid = False
                    break
                arrays.append(arr)

            if not valid:
                prefix = tname if metric == "mae" else ""
                print(f"{prefix:<22} {metric.upper():<6} {'incomplete data':>20}")
                continue

            chi2, p_val = stats.friedmanchisquare(*arrays)
            sig = "*" if p_val < 0.05 else " "
            prefix = tname if metric == "mae" else ""
            print(f"{prefix:<22} {metric.upper():<6} {chi2:>8.3f} {p_val:>7.4f} {sig}")
        print("-" * 80)

    # ============================================================
    # 3. Win count summary
    # ============================================================
    print("\n\n" + "=" * 80)
    print("WIN COUNTS (best mean per task)")
    print("=" * 80)

    for metric in ["mae", "mse"]:
        print(f"\n--- {metric.upper()} ---")
        wins = {mk: 0 for mk in MODELS}
        for tk, tname in TASKS:
            best_model = None
            best_val = np.inf
            for mk in MODELS:
                arr = results[(tk, mk)][metric]
                if len(arr) > 0 and arr.mean() < best_val:
                    best_val = arr.mean()
                    best_model = mk
            if best_model:
                wins[best_model] += 1
                print(f"  {tname:<22} → {MODEL_NAMES[best_model]} ({best_val:.4f})")

        print(f"\n  Totals: ", end="")
        print("  ".join(f"{MODEL_NAMES[mk]}: {wins[mk]}" for mk in MODELS))

        # Binomial: QASA wins vs rest
        qasa_wins = wins["qasa"]
        binom_p = stats.binomtest(qasa_wins, 9, 0.25).pvalue  # chance = 1/4 models
        print(f"  QASA binomial test (wins vs 1/4 chance): p = {binom_p:.4f}")

    # ============================================================
    # 4. Effect size summary (QASA vs Classical only)
    # ============================================================
    print("\n\n" + "=" * 80)
    print("EFFECT SIZE SUMMARY: QASA vs Classical (Cohen's d)")
    print("Positive d = QASA better, Negative d = Classical better")
    print("=" * 80)
    print(f"{'Task':<22} {'MAE d':>8} {'MAE interp':<12} {'MSE d':>8} {'MSE interp':<12}")
    print("-" * 80)

    for tk, tname in TASKS:
        qasa_mae = results[(tk, "qasa")]["mae"]
        class_mae = results[(tk, "classical")]["mae"]
        qasa_mse = results[(tk, "qasa")]["mse"]
        class_mse = results[(tk, "classical")]["mse"]

        if len(qasa_mae) == 3 and len(class_mae) == 3:
            d_mae = cohens_d(class_mae, qasa_mae)
            d_mse = cohens_d(class_mse, qasa_mse)

            def interp(d):
                ad = abs(d)
                if ad < 0.2: return "negligible"
                if ad < 0.5: return "small"
                if ad < 0.8: return "medium"
                return "large"

            print(f"{tname:<22} {d_mae:>8.3f} {interp(d_mae):<12} {d_mse:>8.3f} {interp(d_mse):<12}")

    print("\n\nNote: With only n=3 seeds, statistical power is limited.")
    print("p-values and effect sizes should be interpreted with caution.")
    print("* denotes p < 0.05")


if __name__ == "__main__":
    main()
