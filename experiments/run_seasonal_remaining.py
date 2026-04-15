#!/usr/bin/env python3
"""Run remaining Seasonal Trend experiments one at a time."""

import os, sys, time, csv, datetime
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_baseline_comparison import (
    MODEL_REGISTRY, train_and_evaluate, count_parameters, get_task,
)

CKPT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", "baseline_comparison")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

TASK = "classical_trend_seasonality_noise"
TASK_DISPLAY = "Seasonal Trend"

RUNS = [
    ("qlstm", 43),
    ("qlstm", 44),
    ("qnnformer", 42),
    ("qnnformer", 43),
    ("qnnformer", 44),
]

config = {
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 200,
    "seq_len_train": 20,
    "print_every": 20,
}

device = torch.device("cpu")

for mk, seed in RUNS:
    # Check if already done
    ckpt_path = os.path.join(CKPT_DIR, mk, TASK, f"seed{seed}", "best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("epochs", 0) == 200:
            print(f"[SKIP] {mk} seed{seed} already done (MAE={ckpt['mae']:.4f})")
            continue

    info = MODEL_REGISTRY[mk]
    print(f"\n{'='*60}")
    print(f"{info['short']} | {TASK_DISPLAY} | Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    task = get_task(TASK)
    model = info["class"](hidden_dim=64, num_layers=4, seq_len=20).to(device)

    save_dir = os.path.join(CKPT_DIR, mk, TASK, f"seed{seed}")

    try:
        start = time.time()
        mae, mse, train_loss = train_and_evaluate(
            model, task, TASK, device, config, save_dir=save_dir
        )
        elapsed = time.time() - start
        print(f"  => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {elapsed:.0f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    del model

print("\nAll remaining Seasonal Trend runs completed.")
