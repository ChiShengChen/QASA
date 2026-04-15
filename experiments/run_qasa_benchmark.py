#!/usr/bin/env python3
"""
QASA Unified Benchmark Runner
==============================
Uses the QASA (Quantum Adaptive Self-Attention) architecture consistently
across ALL benchmark tasks. This replaces the RWKV-based benchmark runner
to ensure fair comparison and eliminate inconsistencies between experiments.

Outputs:
  - experiments/results/qasa_benchmark_results_<timestamp>.csv
  - experiments/results/model_param_counts.csv
  - experiments/results/plots/<task_name>_*.png

Usage:
  python experiments/run_qasa_benchmark.py                    # Run all tasks
  python experiments/run_qasa_benchmark.py --tasks arma chaotic_logistic
  python experiments/run_qasa_benchmark.py --dry-run          # 1 epoch sanity check
  python experiments/run_qasa_benchmark.py --seeds 3          # Multiple seed runs
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import task_registry, get_task


# ============================================================
# Model Definitions (from QASA codebase, adapted for benchmark)
# ============================================================

# --- Quantum Components ---

N_QUBITS = 8
N_QLAYERS = 4
qml_dev = qml.device("lightning.qubit", wires=N_QUBITS + 1)


@qml.qnode(qml_dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(N_QUBITS):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, N_QLAYERS):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[N_QUBITS - 1, N_QUBITS])
        qml.RY(weights[l, -1], wires=N_QUBITS)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_shape = (N_QLAYERS, N_QUBITS + 1)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {'weights': self.weight_shape})
        self.input_proj = nn.Linear(input_dim, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)
        self.output_proj = nn.Linear(N_QUBITS, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.kaiming_uniform_(self.output_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x, timestep=0.0):
        x_proj = torch.tanh(self.input_proj(x))
        x_proj = self.norm(x_proj)
        ts = torch.tensor(float(timestep), device=x.device)
        outputs = [self.qlayer((x_proj[i] + ts).cpu()).to(x.device) for i in range(x.size(0))]
        quantum_output = torch.stack(outputs)
        out = self.output_proj(quantum_output)
        if self.input_dim == self.output_dim:
            return x + out
        return out


class QuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = QuantumLayer(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size * seq_len, features)
        q_out = self.v_quantum(x_flat, float(seq_len))
        q_out = q_out.view(batch_size, seq_len, features)
        x = self.norm2(q_out + self.ffn(q_out))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# --- QASA Model (Quantum-Enhanced) ---

class QASAModel(nn.Module):
    """QASA v2: Hybrid Quantum-Classical Transformer for time series."""
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        encoder_layers = []
        for _ in range(num_layers - 1):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, batch_first=True,
                    dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
                )
            )
        encoder_layers.append(QuantumEncoderLayer(hidden_dim, dropout_rate=dropout_rate))
        self.encoder = nn.ModuleList(encoder_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embedding[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.embedding[0].bias, 0)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x)


# --- Classical Baseline (same architecture, no quantum layer) ---

class ClassicalModel(nn.Module):
    """Classical Transformer baseline with identical architecture (no quantum layer)."""
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True,
            dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.output_layer(x)


# ============================================================
# Utilities
# ============================================================

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_param_table(models_info):
    """Print a formatted parameter comparison table."""
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'Total Params':>15} {'Trainable':>15}")
    print("-" * 65)
    for name, total, trainable in models_info:
        print(f"{name:<25} {total:>15,} {trainable:>15,}")
    print("=" * 65 + "\n")


# ============================================================
# Training & Evaluation
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_and_evaluate(model, task, task_name, device, config, save_dir=None):
    """Train a model on a benchmark task and return metrics.

    If save_dir is provided, saves:
      - best_model.pth (best checkpoint by training loss)
      - loss_curve.csv (epoch-by-epoch training loss)
      - prediction.png (predicted vs ground truth plot)
    """
    # Generate data
    is_learning = "learning" in task_name.lower()
    if is_learning:
        vocab_size = getattr(task, 'vocab_size', 50)
        X_train, Y_train, X_test_seed, Y_test_true = task.generate_data(vocab_size=vocab_size)
    else:
        X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test_seed = X_test_seed.to(device)
    Y_test_true = Y_test_true.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()

    seq_len_train = config['seq_len_train']
    num_total = X_train.shape[1]

    # Training loop
    model.train()
    best_loss = float('inf')
    best_state = None
    loss_history = []

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        n_windows = 0

        if num_total > seq_len_train:
            for i in range(num_total - seq_len_train + 1):
                optimizer.zero_grad()
                x_window = X_train[:, i:i + seq_len_train, :]
                y_window = Y_train[:, i:i + seq_len_train, :]
                pred = model(x_window)
                loss = criterion(pred, y_window)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_windows += 1
        else:
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, Y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss = loss.item()
            n_windows = 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_windows, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % config['print_every'] == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch [{epoch+1}/{config['epochs']}] Loss: {avg_loss:.6f}")

    # Restore best model for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation via autoregressive generation
    model.eval()
    generated = []
    current_input = X_test_seed.clone()
    num_to_predict = Y_test_true.shape[1]

    with torch.no_grad():
        for _ in range(num_to_predict):
            if current_input.shape[1] == 0:
                break
            pred = model(current_input)
            next_point = pred[:, -1:, :].clone()
            generated.append(next_point.squeeze().cpu().numpy())
            current_input = torch.cat([current_input[:, 1:, :], next_point], dim=1)
            if len(generated) >= num_to_predict:
                break

    if len(generated) == 0:
        return np.inf, np.inf, best_loss

    y_pred = np.array(generated).flatten()
    y_true = Y_test_true.squeeze().cpu().numpy().flatten()

    # Align lengths
    min_len = min(len(y_pred), len(y_true))
    if min_len == 0:
        return np.inf, np.inf, best_loss

    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # --- Save artifacts ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 1. Best model checkpoint
        ckpt_path = os.path.join(save_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': best_state,
            'best_loss': best_loss,
            'mae': mae,
            'mse': mse,
            'epochs': config['epochs'],
            'config': config,
        }, ckpt_path)

        # 2. Loss curve CSV
        loss_csv = os.path.join(save_dir, 'loss_curve.csv')
        with open(loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for i, l in enumerate(loss_history):
                writer.writerow([i + 1, f'{l:.6f}'])

        # 3. Prediction vs ground truth plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(y_true, 'b-', label='Ground Truth', alpha=0.8)
        axes[0].plot(y_pred, 'r--', label='Predicted', alpha=0.8)
        axes[0].set_title('Prediction vs Ground Truth')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(loss_history, 'g-', linewidth=0.8)
        axes[1].set_title(f'Training Loss (best={best_loss:.6f})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'MAE={mae:.4f}  MSE={mse:.4f}', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction.png'), dpi=150)
        plt.close(fig)

    return mae, mse, best_loss

# ============================================================
# Main Runner
# ============================================================

BENCHMARK_TASKS = [
    'classical_arma',
    'classical_chaotic_logistic',
    'classical_damped_oscillation',
    'classical_noisy_damped_oscillation',
    'classical_piecewise_regime',
    'classical_sawtooth_wave',
    'classical_square_triangle_wave',
    'classical_trend_seasonality_noise',
    'classical_waveform',
]

TASK_DISPLAY_NAMES = {
    'classical_arma': 'ARMA',
    'classical_chaotic_logistic': 'Chaotic Logistic',
    'classical_damped_oscillation': 'Damped Oscillator',
    'classical_noisy_damped_oscillation': 'Noisy Damped Osc',
    'classical_piecewise_regime': 'Piecewise Regime',
    'classical_sawtooth_wave': 'Sawtooth',
    'classical_square_triangle_wave': 'Square Wave',
    'classical_trend_seasonality_noise': 'Seasonal Trend',
    'classical_waveform': 'Waveform',
}


def run_benchmark(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    plots_dir = os.path.join(results_dir, "plots")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device("cpu")  # Quantum circuits require CPU
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PennyLane: {qml.__version__}\n")

    # Config
    hidden_dim = 64  # Reduced from 256 for benchmark tractability
    num_layers = 4
    seq_len_train = 20
    config = {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'seq_len_train': seq_len_train,
        'print_every': max(1, args.epochs // 10),
    }

    # Determine tasks to run
    if args.tasks:
        task_names = [f'classical_{t}' if not t.startswith('classical_') else t for t in args.tasks]
    else:
        task_names = BENCHMARK_TASKS

    # Parameter count (do once)
    qasa_model = QASAModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)
    classical_model = ClassicalModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)

    qasa_total, qasa_train = count_parameters(qasa_model)
    classical_total, classical_train = count_parameters(classical_model)

    print_param_table([
        ("QASA (Quantum)", qasa_total, qasa_train),
        ("Classical Transformer", classical_total, classical_train),
    ])

    # Save param counts
    param_csv = os.path.join(results_dir, "model_param_counts.csv")
    with open(param_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Total_Params', 'Trainable_Params'])
        writer.writerow(['QASA', qasa_total, qasa_train])
        writer.writerow(['Classical', classical_total, classical_train])
    print(f"Parameter counts saved to {param_csv}\n")

    del qasa_model, classical_model  # Free memory

    # Run experiments
    all_results = []
    n_seeds = args.seeds

    for task_name in task_names:
        if task_name not in task_registry:
            print(f"WARNING: Task '{task_name}' not in registry, skipping.")
            continue

        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        print(f"\n{'='*60}")
        print(f"Task: {display_name} ({task_name})")
        print(f"{'='*60}")

        task = get_task(task_name)
        seed_results = {'classical': [], 'quantum': []}

        for seed in range(args.start_seed if hasattr(args, 'start_seed') else 0, n_seeds):
            actual_seed = 42 + seed
            torch.manual_seed(actual_seed)
            np.random.seed(actual_seed)

            # --- Classical ---
            print(f"\n  [Seed {seed+1}/{n_seeds}] Classical Transformer:")
            c_model = ClassicalModel(
                hidden_dim=hidden_dim, num_layers=num_layers,
                seq_len=seq_len_train,
            ).to(device)
            c_save_dir = os.path.join(checkpoints_dir, task_name, f'classical_seed{actual_seed}')
            try:
                c_mae, c_mse, c_best_loss = train_and_evaluate(
                    c_model, task, task_name, device, config, save_dir=c_save_dir
                )
                print(f"  => MAE: {c_mae:.6f}, MSE: {c_mse:.6f}")
                seed_results['classical'].append((c_mae, c_mse))
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                seed_results['classical'].append((np.inf, np.inf))
            del c_model

            # --- QASA (Quantum) ---
            print(f"\n  [Seed {seed+1}/{n_seeds}] QASA (Quantum):")
            q_model = QASAModel(
                hidden_dim=hidden_dim, num_layers=num_layers,
                seq_len=seq_len_train,
            ).to(device)
            q_save_dir = os.path.join(checkpoints_dir, task_name, f'qasa_seed{actual_seed}')
            try:
                q_mae, q_mse, q_best_loss = train_and_evaluate(
                    q_model, task, task_name, device, config, save_dir=q_save_dir
                )
                print(f"  => MAE: {q_mae:.6f}, MSE: {q_mse:.6f}")
                seed_results['quantum'].append((q_mae, q_mse))
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                seed_results['quantum'].append((np.inf, np.inf))
            del q_model

        # Aggregate results
        c_maes = [r[0] for r in seed_results['classical'] if r[0] != np.inf]
        c_mses = [r[1] for r in seed_results['classical'] if r[1] != np.inf]
        q_maes = [r[0] for r in seed_results['quantum'] if r[0] != np.inf]
        q_mses = [r[1] for r in seed_results['quantum'] if r[1] != np.inf]

        result = {
            'task': display_name,
            'c_mae': np.mean(c_maes) if c_maes else np.inf,
            'c_mse': np.mean(c_mses) if c_mses else np.inf,
            'c_mae_std': np.std(c_maes) if len(c_maes) > 1 else 0.0,
            'c_mse_std': np.std(c_mses) if len(c_mses) > 1 else 0.0,
            'q_mae': np.mean(q_maes) if q_maes else np.inf,
            'q_mse': np.mean(q_mses) if q_mses else np.inf,
            'q_mae_std': np.std(q_maes) if len(q_maes) > 1 else 0.0,
            'q_mse_std': np.std(q_mses) if len(q_mses) > 1 else 0.0,
        }
        all_results.append(result)

        # Print summary
        winner_mae = "QASA" if result['q_mae'] < result['c_mae'] else "Classical"
        winner_mse = "QASA" if result['q_mse'] < result['c_mse'] else "Classical"
        print(f"\n  Summary for {display_name}:")
        print(f"    Classical - MAE: {result['c_mae']:.4f}±{result['c_mae_std']:.4f}, MSE: {result['c_mse']:.4f}±{result['c_mse_std']:.4f}")
        print(f"    QASA      - MAE: {result['q_mae']:.4f}±{result['q_mae_std']:.4f}, MSE: {result['q_mse']:.4f}±{result['q_mse_std']:.4f}")
        print(f"    Winner: MAE→{winner_mae}, MSE→{winner_mse}")

    # Save results CSV
    results_csv = os.path.join(results_dir, f"qasa_benchmark_{timestamp}.csv")
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task', 'Model',
            'MAE', 'MSE', 'MAE_Std', 'MSE_Std',
        ])
        for r in all_results:
            writer.writerow([r['task'], 'Classical',
                             f"{r['c_mae']:.6f}", f"{r['c_mse']:.6f}",
                             f"{r['c_mae_std']:.6f}", f"{r['c_mse_std']:.6f}"])
            writer.writerow([r['task'], 'QASA',
                             f"{r['q_mae']:.6f}", f"{r['q_mse']:.6f}",
                             f"{r['q_mae_std']:.6f}", f"{r['q_mse_std']:.6f}"])

    # Print final table (LaTeX-ready)
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS (LaTeX-ready)")
    print(f"{'='*80}")
    print(f"{'Task':<22} {'Model':<12} {'MAE':>10} {'MSE':>12} {'MAE Std':>10} {'MSE Std':>10}")
    print("-" * 80)
    for r in all_results:
        # Bold the winner
        c_mae_str = f"{r['c_mae']:.4f}"
        c_mse_str = f"{r['c_mse']:.4f}"
        q_mae_str = f"{r['q_mae']:.4f}"
        q_mse_str = f"{r['q_mse']:.4f}"

        if r['c_mae'] < r['q_mae']:
            c_mae_str = f"*{c_mae_str}*"
        else:
            q_mae_str = f"*{q_mae_str}*"
        if r['c_mse'] < r['q_mse']:
            c_mse_str = f"*{c_mse_str}*"
        else:
            q_mse_str = f"*{q_mse_str}*"

        print(f"{r['task']:<22} {'Classical':<12} {c_mae_str:>10} {c_mse_str:>12} {r['c_mae_std']:>10.4f} {r['c_mse_std']:>10.4f}")
        print(f"{'':<22} {'QASA':<12} {q_mae_str:>10} {q_mse_str:>12} {r['q_mae_std']:>10.4f} {r['q_mse_std']:>10.4f}")
        print("-" * 80)

    print(f"\nResults saved to: {results_csv}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, epochs={config['epochs']}, "
          f"seeds={n_seeds}, lr={config['lr']}, seq_len={seq_len_train}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="QASA Unified Benchmark")
    parser.add_argument('--tasks', nargs='+', default=None,
                        help="Specific tasks to run (e.g., arma chaotic_logistic)")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Training epochs per task (default: 200)")
    parser.add_argument('--seeds', type=int, default=3,
                        help="Number of random seeds for each experiment (default: 3)")
    parser.add_argument('--start-seed', type=int, default=0,
                        help="Starting seed index (default: 0). Use to resume/extend seed runs.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Quick sanity check with 1 epoch, 1 seed")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        args.seeds = 1
        if args.tasks is None:
            args.tasks = ['damped_oscillation']
        print("=== DRY RUN MODE (1 epoch, 1 seed) ===\n")

    run_benchmark(args)


if __name__ == "__main__":
    main()
