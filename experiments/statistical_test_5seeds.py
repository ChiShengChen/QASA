"""Statistical significance tests for QASA vs Classical benchmark (5 seeds)."""
import numpy as np
from scipy import stats
import csv

# Per-seed raw data: {task: {"classical": [(mae, mse), ...], "qasa": [(mae, mse), ...]}}
# Seeds 1-3 from benchmark_3seeds.log, Seeds 4-5 from benchmark_seeds45.log
data = {
    "ARMA": {
        "classical": [
            (2.107764, 6.812728), (2.289957, 8.020205), (1.992934, 5.776975),  # seeds 1-3
            (2.060741, 6.587148), (2.309791, 8.167183),  # seeds 4-5
        ],
        "qasa": [
            (1.852903, 5.107881), (2.762005, 11.645108), (2.224012, 7.396634),  # seeds 1-3
            (1.835434, 5.466956), (2.084029, 7.473607),  # seeds 4-5
        ],
    },
    "Chaotic Logistic": {
        "classical": [
            (0.393912, 0.219696), (0.318828, 0.168966), (0.386817, 0.209900),
            (0.352205, 0.189391), (0.412636, 0.234114),
        ],
        "qasa": [
            (0.295834, 0.152598), (0.405082, 0.224471), (0.279690, 0.141602),
            (0.365853, 0.201534), (0.356292, 0.191551),
        ],
    },
    "Damped Oscillator": {
        "classical": [
            (0.065301, 0.004660), (0.018227, 0.000530), (0.043891, 0.002556),
            (0.043684, 0.002181), (0.016420, 0.000351),
        ],
        "qasa": [
            (0.161333, 0.039294), (0.052774, 0.004016), (0.231880, 0.084319),
            (0.104458, 0.014694), (0.034566, 0.001580),
        ],
    },
    "Noisy Damped Osc": {
        "classical": [
            (0.044636, 0.003062), (0.042575, 0.002808), (0.042638, 0.002789),
            (0.042796, 0.002801), (0.044118, 0.002991),
        ],
        "qasa": [
            (0.042282, 0.002732), (0.042673, 0.002792), (0.043172, 0.002828),
            (0.042638, 0.002789), (0.046110, 0.003337),
        ],
    },
    "Piecewise Regime": {
        "classical": [
            (22.129765, 490.460693), (22.045149, 486.721985), (22.261993, 496.337250),
            (22.032030, 486.144287), (22.184925, 492.905273),
        ],
        "qasa": [
            (22.589861, 511.035980), (22.291166, 497.630157), (22.267536, 496.577179),
            (22.397842, 502.397369), (23.382023, 547.453064),
        ],
    },
    "Sawtooth": {
        "classical": [
            (0.125237, 0.067285), (0.041698, 0.011296), (0.082150, 0.041857),
            (0.146082, 0.168526), (0.073714, 0.010092),
        ],
        "qasa": [
            (0.162905, 0.089310), (0.156215, 0.082342), (0.557353, 0.681005),
            (0.151102, 0.076342), (0.106462, 0.051642),
        ],
    },
    "Square Wave": {
        "classical": [
            (0.881917, 0.896536), (0.720447, 1.312857), (0.916982, 0.895987),
            (0.865849, 0.896581), (0.957157, 0.930834),
        ],
        "qasa": [
            (0.713913, 1.355523), (0.712814, 1.204711), (0.890853, 0.890559),
            (0.945284, 0.919521), (0.710021, 1.222265),
        ],
    },
    "Seasonal Trend": {
        "classical": [
            (0.670059, 0.559724), (0.754669, 0.682144), (0.623035, 0.517084),
            (0.669500, 0.555964), (0.652257, 0.537637),
        ],
        "qasa": [
            (0.666267, 0.552182), (0.665147, 0.550930), (0.666365, 0.552293),
            (0.684516, 0.574396), (0.655908, 0.541104),
        ],
    },
    "Waveform": {
        "classical": [
            (0.051649, 0.003538), (0.038436, 0.001791), (0.092790, 0.010297),
            (0.108850, 0.014000), (0.045753, 0.003105),
        ],
        "qasa": [
            (0.089617, 0.010503), (0.115348, 0.016531), (0.054261, 0.003767),
            (0.117122, 0.017928), (0.070861, 0.006229),
        ],
    },
}

print("=" * 100)
print(f"{'Task':<22} {'Metric':<6} {'Classical':>14} {'QASA':>14} {'t-stat':>8} {'p-value':>8} {'Sig':>4} {'Winner':<10}")
print("=" * 100)

results_for_csv = []

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
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
        prefix = task if metric == "MAE" else ""
        print(f"{prefix:<22} {metric:<6} {c_mean:>7.4f}±{c_std:<6.4f} {q_mean:>7.4f}±{q_std:<6.4f} {t_stat:>8.3f} {p_val:>7.4f} {sig:>4} {winner}")

        if metric == "MAE":
            results_for_csv.append({
                "Task": task,
                "Model_Classical_MAE": f"{c_mean:.4f}",
                "Model_Classical_MAE_Std": f"{c_std:.4f}",
                "Model_QASA_MAE": f"{q_mean:.4f}",
                "Model_QASA_MAE_Std": f"{q_std:.4f}",
            })
        else:
            results_for_csv[-1].update({
                "Model_Classical_MSE": f"{c_mean:.4f}",
                "Model_Classical_MSE_Std": f"{c_std:.4f}",
                "Model_QASA_MSE": f"{q_mean:.4f}",
                "Model_QASA_MSE_Std": f"{q_std:.4f}",
                "MAE_t": f"{stats.ttest_rel(c_mae, q_mae)[0]:.3f}",
                "MAE_p": f"{stats.ttest_rel(c_mae, q_mae)[1]:.4f}",
                "MSE_t": f"{t_stat:.3f}",
                "MSE_p": f"{p_val:.4f}",
            })
    print("-" * 100)

# Cross-task aggregate
print("\n\n=== Cross-Task Summary (MAE, 5 seeds) ===")
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
binom_p = stats.binomtest(qasa_wins_mae, 9, 0.5).pvalue
print(f"Binomial test (QASA wins vs chance): p = {binom_p:.4f}")

print("\n=== Cross-Task Summary (MSE, 5 seeds) ===")
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

# Effect sizes (Cohen's d for paired samples)
print("\n\n=== Effect Sizes (Cohen's d, paired) ===")
print(f"{'Task':<22} {'MAE d':>8} {'MSE d':>8} {'MAE interp':<12} {'MSE interp':<12}")
print("-" * 70)
for task, d in data.items():
    c = d["classical"]
    q = d["qasa"]
    c_mae = np.array([x[0] for x in c])
    c_mse = np.array([x[1] for x in c])
    q_mae = np.array([x[0] for x in q])
    q_mse = np.array([x[1] for x in q])

    diff_mae = c_mae - q_mae
    d_mae = diff_mae.mean() / diff_mae.std(ddof=1) if diff_mae.std(ddof=1) > 0 else 0
    diff_mse = c_mse - q_mse
    d_mse = diff_mse.mean() / diff_mse.std(ddof=1) if diff_mse.std(ddof=1) > 0 else 0

    def interp(d_val):
        d_abs = abs(d_val)
        if d_abs < 0.2: return "negligible"
        elif d_abs < 0.5: return "small"
        elif d_abs < 0.8: return "medium"
        else: return "large"

    print(f"  {task:<22} {d_mae:>8.3f} {d_mse:>8.3f} {interp(d_mae):<12} {interp(d_mse):<12}")

# Save combined 5-seed CSV
with open("experiments/results/qasa_benchmark_5seeds_combined.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Task", "Model", "MAE", "MSE", "MAE_Std", "MSE_Std"])
    for task, d in data.items():
        c_mae = np.array([x[0] for x in d["classical"]])
        c_mse = np.array([x[1] for x in d["classical"]])
        q_mae = np.array([x[0] for x in d["qasa"]])
        q_mse = np.array([x[1] for x in d["qasa"]])
        writer.writerow([task, "Classical", f"{c_mae.mean():.6f}", f"{c_mse.mean():.6f}",
                         f"{c_mae.std(ddof=1):.6f}", f"{c_mse.std(ddof=1):.6f}"])
        writer.writerow([task, "QASA", f"{q_mae.mean():.6f}", f"{q_mse.mean():.6f}",
                         f"{q_mae.std(ddof=1):.6f}", f"{q_mse.std(ddof=1):.6f}"])

print("\n\nSaved: experiments/results/qasa_benchmark_5seeds_combined.csv")
print(f"\nNote: With n=5 seeds, paired t-tests have moderate power.")
print("* denotes p < 0.05, ** denotes p < 0.01")
