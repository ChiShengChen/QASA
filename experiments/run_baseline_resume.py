#!/usr/bin/env python3
"""
Resume baseline comparison from where it stopped.

Recovers completed results from checkpoint files, then runs remaining tasks.
"""

import os
import sys
import csv
import time
import datetime
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import everything from the original script
from run_baseline_comparison import (
    MODEL_REGISTRY, BENCHMARK_TASKS, TASK_DISPLAY_NAMES,
    train_and_evaluate, count_parameters, get_task,
)

CHECKPOINTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments", "results", "checkpoints", "baseline_comparison"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def recover_completed_results(model_keys, task_names, seeds, epochs):
    """Recover results from checkpoint files for runs that completed with full epochs."""
    completed = {}  # (model_key, task_name, seed) -> {mae, mse, train_loss, time}

    for mk in model_keys:
        for tn in task_names:
            for seed in seeds:
                ckpt_path = os.path.join(
                    CHECKPOINTS_DIR, mk, tn, f"seed{seed}", "best_model.pth"
                )
                if not os.path.exists(ckpt_path):
                    continue
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    # Only count as completed if trained with full epochs
                    if ckpt.get("epochs", 0) == epochs:
                        completed[(mk, tn, seed)] = {
                            "mae": ckpt["mae"],
                            "mse": ckpt["mse"],
                            "best_loss": ckpt.get("best_loss", 0),
                        }
                        display = TASK_DISPLAY_NAMES.get(tn, tn)
                        short = MODEL_REGISTRY[mk]["short"]
                        print(f"  [recovered] {short} | {display} | seed{seed} "
                              f"=> MAE: {ckpt['mae']:.6f}, MSE: {ckpt['mse']:.6f}")
                except Exception as e:
                    print(f"  [skip] {mk}/{tn}/seed{seed}: {e}")

    return completed


def main():
    device = torch.device("cpu")
    hidden_dim = 64
    num_layers = 4
    seq_len_train = 20
    epochs = 200
    n_seeds = 3
    seeds = [42 + i for i in range(n_seeds)]

    config = {
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "seq_len_train": seq_len_train,
        "print_every": max(1, epochs // 10),
    }

    model_keys = list(MODEL_REGISTRY.keys())
    task_names = BENCHMARK_TASKS

    # Skip combinations that repeatedly hang (system-level I/O issue)
    SKIP_COMBOS = {
        ("qlstm", "classical_trend_seasonality_noise"),  # hangs on U state
        ("qnnformer", "classical_trend_seasonality_noise"),  # also hangs
    }

    print("=" * 60)
    print("RESUMING BASELINE COMPARISON")
    print(f"Skipping {len(SKIP_COMBOS)} known problematic combos")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Recovering completed runs from checkpoints...\n")

    completed = recover_completed_results(model_keys, task_names, seeds, epochs)

    total_runs = len(model_keys) * len(task_names) * n_seeds
    print(f"\nRecovered: {len(completed)}/{total_runs} runs")
    print(f"Remaining: {total_runs - len(completed)} runs\n")

    # Collect all results
    all_results = []  # per (model, task) aggregated
    run_idx = 0

    for task_name in task_names:
        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)

        for mk in model_keys:
            info = MODEL_REGISTRY[mk]
            seed_maes = []
            seed_mses = []

            for seed in seeds:
                run_idx += 1
                key = (mk, task_name, seed)

                if key in completed:
                    seed_maes.append(completed[key]["mae"])
                    seed_mses.append(completed[key]["mse"])
                    continue

                # Skip known problematic combos
                if (mk, task_name) in SKIP_COMBOS:
                    print(f"  [SKIP] {info['short']} | {display_name} | Seed {seed} (known hang)")
                    seed_maes.append(np.inf)
                    seed_mses.append(np.inf)
                    continue

                # Need to run this one
                print(f"\n{'='*60}")
                print(f"[{run_idx}/{total_runs}] {info['short']} | {display_name} | Seed {seed}")
                print(f"{'='*60}")

                torch.manual_seed(seed)
                np.random.seed(seed)

                task = get_task(task_name)
                model = info["class"](
                    hidden_dim=hidden_dim, num_layers=num_layers,
                    seq_len=seq_len_train,
                ).to(device)

                save_dir = os.path.join(CHECKPOINTS_DIR, mk, task_name, f"seed{seed}")

                try:
                    start_time = time.time()
                    mae, mse, train_loss = train_and_evaluate(
                        model, task, task_name, device, config, save_dir=save_dir
                    )
                    elapsed = time.time() - start_time
                    print(f"  => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {elapsed:.0f}s")
                    seed_maes.append(mae)
                    seed_mses.append(mse)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    seed_maes.append(np.inf)
                    seed_mses.append(np.inf)

                del model

            # Aggregate
            valid_maes = [m for m in seed_maes if m != np.inf]
            valid_mses = [m for m in seed_mses if m != np.inf]

            total_params, _ = count_parameters(
                info["class"](hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)
            )

            result = {
                "task": display_name,
                "task_key": task_name,
                "model": info["short"],
                "model_key": mk,
                "mae": np.mean(valid_maes) if valid_maes else np.inf,
                "mse": np.mean(valid_mses) if valid_mses else np.inf,
                "mae_std": np.std(valid_maes) if len(valid_maes) > 1 else 0.0,
                "mse_std": np.std(valid_mses) if len(valid_mses) > 1 else 0.0,
                "total_params": total_params,
                "n_seeds": len(valid_maes),
            }
            all_results.append(result)

    # Save CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(RESULTS_DIR, f"baseline_comparison_{timestamp}.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Task", "Model", "MAE", "MSE", "MAE_Std", "MSE_Std",
            "Total_Params", "N_Seeds",
        ])
        for r in all_results:
            writer.writerow([
                r["task"], r["model"],
                f"{r['mae']:.6f}", f"{r['mse']:.6f}",
                f"{r['mae_std']:.6f}", f"{r['mse_std']:.6f}",
                r["total_params"], r["n_seeds"],
            ])

    # Print summary
    print(f"\n\n{'='*100}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"{'Task':<22} {'Model':<14} {'MAE':>10} {'MSE':>12} {'MAE Std':>10} {'MSE Std':>10} {'Params':>10}")
    print("-" * 100)

    tasks_seen = set()
    for r in all_results:
        sep = r["task"] not in tasks_seen
        if sep and tasks_seen:
            print("-" * 100)
        tasks_seen.add(r["task"])

        task_results = [x for x in all_results if x["task"] == r["task"]]
        best_mae = min(x["mae"] for x in task_results)
        best_mse = min(x["mse"] for x in task_results)

        mae_str = f"{r['mae']:.4f}"
        mse_str = f"{r['mse']:.4f}"
        if r["mae"] == best_mae:
            mae_str = f"*{mae_str}*"
        if r["mse"] == best_mse:
            mse_str = f"*{mse_str}*"

        task_col = r["task"] if sep else ""
        print(f"{task_col:<22} {r['model']:<14} {mae_str:>10} {mse_str:>12} "
              f"{r['mae_std']:>10.4f} {r['mse_std']:>10.4f} {r['total_params']:>10,}")

    print("-" * 100)

    # Win counts
    print(f"\n{'='*60}")
    print("WIN COUNTS (MAE / MSE)")
    print(f"{'='*60}")
    for mk in model_keys:
        info = MODEL_REGISTRY[mk]
        mae_wins = 0
        mse_wins = 0
        for task_name in set(r["task"] for r in all_results):
            task_results = [x for x in all_results if x["task"] == task_name]
            best_mae = min(x["mae"] for x in task_results)
            best_mse = min(x["mse"] for x in task_results)
            my = [x for x in task_results if x["model_key"] == mk]
            if my and my[0]["mae"] == best_mae:
                mae_wins += 1
            if my and my[0]["mse"] == best_mse:
                mse_wins += 1
        print(f"  {info['short']:<14} MAE wins: {mae_wins}, MSE wins: {mse_wins}")

    print(f"\nResults saved to: {results_csv}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, epochs={epochs}, "
          f"seeds={n_seeds}, lr={config['lr']}, seq_len={seq_len_train}")


if __name__ == "__main__":
    main()
