"""Statistical significance tests for QASA vs Classical benchmark (3 seeds)."""
import numpy as np
from scipy import stats

# Per-seed raw data: {task: {"classical": [(mae, mse), ...], "qasa": [(mae, mse), ...]}}
data = {
    "ARMA": {
        "classical": [(2.107764, 6.812728), (2.289957, 8.020205), (1.992934, 5.776975)],
        "qasa":      [(1.852903, 5.107881), (2.762005, 11.645108), (2.224012, 7.396634)],
    },
    "Chaotic Logistic": {
        "classical": [(0.393912, 0.219696), (0.318828, 0.168966), (0.386817, 0.209900)],
        "qasa":      [(0.295834, 0.152598), (0.405082, 0.224471), (0.279690, 0.141602)],
    },
    "Damped Oscillator": {
        "classical": [(0.065301, 0.004660), (0.018227, 0.000530), (0.043891, 0.002556)],
        "qasa":      [(0.161333, 0.039294), (0.052774, 0.004016), (0.231880, 0.084319)],
    },
    "Noisy Damped Osc": {
        "classical": [(0.044636, 0.003062), (0.042575, 0.002808), (0.042638, 0.002789)],
        "qasa":      [(0.042282, 0.002732), (0.042673, 0.002792), (0.043172, 0.002828)],
    },
    "Piecewise Regime": {
        "classical": [(22.129765, 490.460693), (22.045149, 486.721985), (22.261993, 496.337250)],
        "qasa":      [(22.589861, 511.035980), (22.291166, 497.630157), (22.267536, 496.577179)],
    },
    "Sawtooth": {
        "classical": [(0.125237, 0.067285), (0.041698, 0.011296), (0.082150, 0.041857)],
        "qasa":      [(0.162905, 0.089310), (0.156215, 0.082342), (0.557353, 0.681005)],
    },
    "Square Wave": {
        "classical": [(0.881917, 0.896536), (0.720447, 1.312857), (0.916982, 0.895987)],
        "qasa":      [(0.713913, 1.355523), (0.712814, 1.204711), (0.890853, 0.890559)],
    },
    "Seasonal Trend": {
        "classical": [(0.670059, 0.559724), (0.754669, 0.682144), (0.623035, 0.517084)],
        "qasa":      [(0.666267, 0.552182), (0.665147, 0.550930), (0.666365, 0.552293)],
    },
    "Waveform": {
        "classical": [(0.051649, 0.003538), (0.038436, 0.001791), (0.092790, 0.010297)],
        "qasa":      [(0.089617, 0.010503), (0.115348, 0.016531), (0.054261, 0.003767)],
    },
}

print("=" * 90)
print(f"{'Task':<22} {'Metric':<6} {'Classical':>14} {'QASA':>14} {'t-stat':>8} {'p-value':>8} {'Winner':<10}")
print("=" * 90)

for task, d in data.items():
    c = d["classical"]
    q = d["qasa"]
    c_mae = np.array([x[0] for x in c])
    c_mse = np.array([x[1] for x in c])
    q_mae = np.array([x[0] for x in q])
    q_mse = np.array([x[1] for x in q])

    for metric, cv, qv in [("MAE", c_mae, q_mae), ("MSE", c_mse, q_mse)]:
        t_stat, p_val = stats.ttest_rel(cv, qv)
        c_mean, c_std = cv.mean(), cv.std(ddof=1)
        q_mean, q_std = qv.mean(), qv.std(ddof=1)
        winner = "QASA" if q_mean < c_mean else "Classical"
        sig = "*" if p_val < 0.05 else ""
        prefix = task if metric == "MAE" else ""
        print(f"{prefix:<22} {metric:<6} {c_mean:>7.4f}±{c_std:<6.4f} {q_mean:>7.4f}±{q_std:<6.4f} {t_stat:>8.3f} {p_val:>7.4f}{sig} {winner}")
    print("-" * 90)

# Cross-task aggregate: paired sign test across all 9 tasks
print("\n\n=== Cross-Task Summary (MAE) ===")
qasa_wins_mae = 0
classical_wins_mae = 0
for task, d in data.items():
    c_mae = np.mean([x[0] for x in d["classical"]])
    q_mae = np.mean([x[0] for x in d["qasa"]])
    w = "QASA" if q_mae < c_mae else "Classical"
    if w == "QASA":
        qasa_wins_mae += 1
    else:
        classical_wins_mae += 1
    print(f"  {task:<22} {w}")

print(f"\nQASA wins: {qasa_wins_mae}/9, Classical wins: {classical_wins_mae}/9")

# Binomial test: is 4/9 significantly different from chance (0.5)?
binom_p = stats.binomtest(qasa_wins_mae, 9, 0.5).pvalue
print(f"Binomial test (QASA wins vs chance): p = {binom_p:.4f}")

print("\n=== Cross-Task Summary (MSE) ===")
qasa_wins_mse = 0
classical_wins_mse = 0
for task, d in data.items():
    c_mse = np.mean([x[1] for x in d["classical"]])
    q_mse = np.mean([x[1] for x in d["qasa"]])
    w = "QASA" if q_mse < c_mse else "Classical"
    if w == "QASA":
        qasa_wins_mse += 1
    else:
        classical_wins_mse += 1
    print(f"  {task:<22} {w}")

print(f"\nQASA wins: {qasa_wins_mse}/9, Classical wins: {classical_wins_mse}/9")

print("\n\nNote: With only n=3 seeds per task, paired t-tests have very limited")
print("statistical power. p-values should be interpreted with caution.")
print("* denotes p < 0.05")
