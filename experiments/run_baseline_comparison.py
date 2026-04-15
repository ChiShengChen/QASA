#!/usr/bin/env python3
"""
Quantum Baseline Comparison Runner
====================================
Compares QASA against other quantum time-series methods:
  1. QASA (Quantum Adaptive Self-Attention) — our method
  2. Classical Transformer — identical architecture, no quantum
  3. QLSTM — Quantum Long Short-Term Memory (Chen et al. 2022)
  4. QnnFormer — Quantum Neural Network Transformer (Cai et al. 2024)

Runs all 4 models on the 9 benchmark tasks with matched hyperparameters.

Usage:
  python experiments/run_baseline_comparison.py                    # All tasks
  python experiments/run_baseline_comparison.py --tasks chaotic_logistic damped_oscillation
  python experiments/run_baseline_comparison.py --models qasa qlstm  # Specific models
  python experiments/run_baseline_comparison.py --dry-run           # 1 epoch sanity check
  python experiments/run_baseline_comparison.py --seeds 3           # Multi-seed
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

# Import baseline models
from baselines.qlstm_model import QLSTMModel
from baselines.qnnformer_model import QnnFormerModel


# ============================================================
# QASA & Classical Models (from run_qasa_benchmark.py)
# ============================================================

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


class QASAModel(nn.Module):
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


class ClassicalModel(nn.Module):
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
# Model Registry
# ============================================================

MODEL_REGISTRY = {
    'classical': {
        'name': 'Classical Transformer',
        'class': ClassicalModel,
        'short': 'Classical',
    },
    'qasa': {
        'name': 'QASA (Ours)',
        'class': QASAModel,
        'short': 'QASA',
    },
    'qlstm': {
        'name': 'QLSTM (Chen 2022)',
        'class': QLSTMModel,
        'short': 'QLSTM',
    },
    'qnnformer': {
        'name': 'QnnFormer (Cai 2024)',
        'class': QnnFormerModel,
        'short': 'QnnFormer',
    },
}


# ============================================================
# Utilities & Training
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_and_evaluate(model, task, task_name, device, config, save_dir=None):
    """Train a model on a benchmark task and return metrics."""
    is_learning = "learning" in task_name.lower()
    if is_learning:
        vocab_size = getattr(task, 'vocab_size', 50)
        X_train, Y_train, X_test_seed, Y_test_true = task.generate_data(vocab_size=vocab_size)
    else:
        X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test_seed = X_test_seed.to(device)
    Y_test_true = Y_test_true.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()

    seq_len_train = config['seq_len_train']
    num_total = X_train.shape[1]

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

    if best_state is not None:
        model.load_state_dict(best_state)

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

    min_len = min(len(y_pred), len(y_true))
    if min_len == 0:
        return np.inf, np.inf, best_loss

    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'best_loss': best_loss,
            'mae': mae,
            'mse': mse,
            'epochs': config['epochs'],
            'config': config,
        }, os.path.join(save_dir, 'best_model.pth'))

        with open(os.path.join(save_dir, 'loss_curve.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for i, l in enumerate(loss_history):
                writer.writerow([i + 1, f'{l:.6f}'])

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
# Task List
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


# ============================================================
# Main Runner
# ============================================================

def run_comparison(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    checkpoints_dir = os.path.join(results_dir, "checkpoints", "baseline_comparison")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PennyLane: {qml.__version__}\n")

    hidden_dim = 64
    num_layers = 4
    seq_len_train = 20
    config = {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'seq_len_train': seq_len_train,
        'print_every': max(1, args.epochs // 10),
    }

    # Determine models to run
    if args.models:
        model_keys = [k for k in args.models if k in MODEL_REGISTRY]
    else:
        model_keys = list(MODEL_REGISTRY.keys())

    # Determine tasks to run
    if args.tasks:
        task_names = [f'classical_{t}' if not t.startswith('classical_') else t for t in args.tasks]
    else:
        task_names = BENCHMARK_TASKS

    # Print model info
    print("=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)
    print(f"{'Model':<28} {'Total Params':>14} {'Trainable':>14}")
    print("-" * 75)
    for mk in model_keys:
        info = MODEL_REGISTRY[mk]
        m = info['class'](hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)
        total, trainable = count_parameters(m)
        print(f"{info['name']:<28} {total:>14,} {trainable:>14,}")
        del m
    print("=" * 75 + "\n")

    # Run experiments
    all_results = []
    n_seeds = args.seeds
    total_runs = len(model_keys) * len(task_names) * n_seeds
    run_idx = 0

    for task_name in task_names:
        if task_name not in task_registry:
            print(f"WARNING: Task '{task_name}' not in registry, skipping.")
            continue

        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        print(f"\n{'='*60}")
        print(f"Task: {display_name} ({task_name})")
        print(f"{'='*60}")

        task = get_task(task_name)

        for mk in model_keys:
            info = MODEL_REGISTRY[mk]
            seed_maes = []
            seed_mses = []

            for seed_idx in range(n_seeds):
                run_idx += 1
                actual_seed = 42 + seed_idx
                torch.manual_seed(actual_seed)
                np.random.seed(actual_seed)

                print(f"\n  [{run_idx}/{total_runs}] {info['short']} | Seed {seed_idx+1}/{n_seeds}")

                model = info['class'](
                    hidden_dim=hidden_dim, num_layers=num_layers,
                    seq_len=seq_len_train,
                ).to(device)

                save_dir = os.path.join(
                    checkpoints_dir, mk, task_name, f'seed{actual_seed}'
                )

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
                    traceback.print_exc()
                    seed_maes.append(np.inf)
                    seed_mses.append(np.inf)

                del model

            # Aggregate across seeds
            valid_maes = [m for m in seed_maes if m != np.inf]
            valid_mses = [m for m in seed_mses if m != np.inf]

            total_params, _ = count_parameters(
                info['class'](hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len_train)
            )

            result = {
                'task': display_name,
                'task_key': task_name,
                'model': info['short'],
                'model_key': mk,
                'mae': np.mean(valid_maes) if valid_maes else np.inf,
                'mse': np.mean(valid_mses) if valid_mses else np.inf,
                'mae_std': np.std(valid_maes) if len(valid_maes) > 1 else 0.0,
                'mse_std': np.std(valid_mses) if len(valid_mses) > 1 else 0.0,
                'total_params': total_params,
                'n_seeds': len(valid_maes),
            }
            all_results.append(result)

    # Save CSV
    results_csv = os.path.join(results_dir, f"baseline_comparison_{timestamp}.csv")
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task', 'Model', 'MAE', 'MSE', 'MAE_Std', 'MSE_Std',
            'Total_Params', 'N_Seeds',
        ])
        for r in all_results:
            writer.writerow([
                r['task'], r['model'],
                f"{r['mae']:.6f}", f"{r['mse']:.6f}",
                f"{r['mae_std']:.6f}", f"{r['mse_std']:.6f}",
                r['total_params'], r['n_seeds'],
            ])

    # Print summary table
    print(f"\n\n{'='*100}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"{'Task':<22} {'Model':<14} {'MAE':>10} {'MSE':>12} {'MAE Std':>10} {'MSE Std':>10} {'Params':>10}")
    print("-" * 100)

    tasks_seen = set()
    for r in all_results:
        sep = r['task'] not in tasks_seen
        if sep and tasks_seen:
            print("-" * 100)
        tasks_seen.add(r['task'])

        # Bold winner per task
        task_results = [x for x in all_results if x['task'] == r['task']]
        best_mae = min(x['mae'] for x in task_results)
        best_mse = min(x['mse'] for x in task_results)

        mae_str = f"{r['mae']:.4f}"
        mse_str = f"{r['mse']:.4f}"
        if r['mae'] == best_mae:
            mae_str = f"*{mae_str}*"
        if r['mse'] == best_mse:
            mse_str = f"*{mse_str}*"

        task_col = r['task'] if sep else ''
        print(f"{task_col:<22} {r['model']:<14} {mae_str:>10} {mse_str:>12} "
              f"{r['mae_std']:>10.4f} {r['mse_std']:>10.4f} {r['total_params']:>10,}")

    print("-" * 100)

    # Print win counts
    print(f"\n{'='*60}")
    print("WIN COUNTS (MAE / MSE)")
    print(f"{'='*60}")
    for mk in model_keys:
        info = MODEL_REGISTRY[mk]
        mae_wins = 0
        mse_wins = 0
        for task_name in set(r['task'] for r in all_results):
            task_results = [x for x in all_results if x['task'] == task_name]
            best_mae = min(x['mae'] for x in task_results)
            best_mse = min(x['mse'] for x in task_results)
            my = [x for x in task_results if x['model_key'] == mk]
            if my and my[0]['mae'] == best_mae:
                mae_wins += 1
            if my and my[0]['mse'] == best_mse:
                mse_wins += 1
        print(f"  {info['short']:<14} MAE wins: {mae_wins}, MSE wins: {mse_wins}")

    print(f"\nResults saved to: {results_csv}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, epochs={config['epochs']}, "
          f"seeds={n_seeds}, lr={config['lr']}, seq_len={seq_len_train}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Quantum Baseline Comparison")
    parser.add_argument('--tasks', nargs='+', default=None,
                        help="Specific tasks to run")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Specific models to run (classical, qasa, qlstm, qnnformer)")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Training epochs (default: 200)")
    parser.add_argument('--seeds', type=int, default=1,
                        help="Number of random seeds (default: 1)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Quick sanity check with 1 epoch")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        args.seeds = 1
        if args.tasks is None:
            args.tasks = ['damped_oscillation']
        if args.models is None:
            args.models = ['classical', 'qasa', 'qlstm', 'qnnformer']
        print("=== DRY RUN MODE (1 epoch, 1 seed) ===\n")

    run_comparison(args)


if __name__ == "__main__":
    main()
