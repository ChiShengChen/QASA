#!/usr/bin/env python3
"""
QASA Ablation Study Runner
============================
Tests the effect of quantum layer POSITION and COUNT in the QASA architecture.

Ablation A (Position): Where to place 1 quantum layer among 4 total layers
  [Q,C,C,C] / [C,Q,C,C] / [C,C,Q,C] / [C,C,C,Q]

Ablation B (Count): How many quantum layers (0 to 4)
  0Q / 1Q / 2Q / 4Q

Representative tasks:
  - classical_chaotic_logistic    (QASA wins in benchmark)
  - classical_damped_oscillation  (Classical wins in benchmark)
  - classical_square_triangle_wave (Mixed results)

Usage:
  python experiments/run_ablation.py                  # Run all configs
  python experiments/run_ablation.py --dry-run         # 1 epoch sanity check
  python experiments/run_ablation.py --configs pos_Q_first count_0Q
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
# Quantum Components (identical to run_qasa_benchmark.py)
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


# ============================================================
# Ablation Model
# ============================================================

class AblationQASAModel(nn.Module):
    """
    QASA model with configurable quantum layer placement.

    Args:
        quantum_layer_indices: List of layer indices (0-based) that should be
            QuantumEncoderLayer. All other layers are classical TransformerEncoderLayer.
            E.g., [3] = default QASA, [] = pure classical, [0,1,2,3] = all quantum.
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1, quantum_layer_indices=None):
        super().__init__()
        if quantum_layer_indices is None:
            quantum_layer_indices = [num_layers - 1]  # Default: last layer only

        self.quantum_layer_indices = quantum_layer_indices
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)

        encoder_layers = []
        for i in range(num_layers):
            if i in quantum_layer_indices:
                encoder_layers.append(QuantumEncoderLayer(hidden_dim, dropout_rate=dropout_rate))
            else:
                encoder_layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim, nhead=4, batch_first=True,
                        dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
                    )
                )
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


# ============================================================
# Training & Evaluation (identical to run_qasa_benchmark.py)
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

    # Restore best model
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

    min_len = min(len(y_pred), len(y_true))
    if min_len == 0:
        return np.inf, np.inf, best_loss

    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # Save artifacts
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Checkpoint
        torch.save({
            'model_state_dict': best_state,
            'best_loss': best_loss,
            'mae': mae,
            'mse': mse,
            'epochs': config['epochs'],
            'config': config,
        }, os.path.join(save_dir, 'best_model.pth'))

        # Loss curve
        with open(os.path.join(save_dir, 'loss_curve.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for i, l in enumerate(loss_history):
                writer.writerow([i + 1, f'{l:.6f}'])

        # Plot
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
# Ablation Configurations
# ============================================================

ABLATION_CONFIGS = [
    # Position ablation: where to place 1 quantum layer
    {"name": "pos_Q_first",  "type": "position", "quantum_indices": [0]},
    {"name": "pos_Q_second", "type": "position", "quantum_indices": [1]},
    {"name": "pos_Q_third",  "type": "position", "quantum_indices": [2]},
    {"name": "pos_Q_last",   "type": "position", "quantum_indices": [3]},
    # Count ablation: how many quantum layers
    # Note: count_1Q is omitted — identical to pos_Q_last ([3]).
    # Use pos_Q_last results for the 1Q data point in count analysis.
    {"name": "count_0Q", "type": "count", "quantum_indices": []},
    {"name": "count_2Q", "type": "count", "quantum_indices": [2, 3]},
    {"name": "count_4Q", "type": "count", "quantum_indices": [0, 1, 2, 3]},
]

ABLATION_TASKS = [
    'classical_chaotic_logistic',
    'classical_damped_oscillation',
    'classical_square_triangle_wave',
]

TASK_DISPLAY_NAMES = {
    'classical_chaotic_logistic': 'Chaotic Logistic',
    'classical_damped_oscillation': 'Damped Oscillator',
    'classical_square_triangle_wave': 'Square Wave',
}


# ============================================================
# Main Runner
# ============================================================

def run_ablation(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    checkpoints_dir = os.path.join(results_dir, "checkpoints", "ablation")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PennyLane: {qml.__version__}\n")

    # Config (identical to benchmark)
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

    # Filter configs if specified
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c['name'] in args.configs]
        if not configs:
            print(f"ERROR: No matching configs found for {args.configs}")
            print(f"Available: {[c['name'] for c in ABLATION_CONFIGS]}")
            return
    else:
        configs = ABLATION_CONFIGS

    # Print parameter counts for each config
    print("=" * 70)
    print("ABLATION CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Config':<20} {'Quantum Indices':<20} {'Total Params':>12} {'Q Layers':>10}")
    print("-" * 70)
    for cfg in configs:
        model = AblationQASAModel(
            hidden_dim=hidden_dim, num_layers=num_layers,
            seq_len=seq_len_train, quantum_layer_indices=cfg['quantum_indices']
        )
        total, _ = count_parameters(model)
        n_q = len(cfg['quantum_indices'])
        print(f"{cfg['name']:<20} {str(cfg['quantum_indices']):<20} {total:>12,} {n_q:>10}")
        del model
    print("=" * 70 + "\n")

    # Run experiments
    all_results = []
    total_runs = len(configs) * len(ABLATION_TASKS)
    run_idx = 0

    for cfg in configs:
        for task_name in ABLATION_TASKS:
            run_idx += 1
            display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)

            print(f"\n[{run_idx}/{total_runs}] Config: {cfg['name']} | Task: {display_name}")
            print("-" * 50)

            # Set seed
            torch.manual_seed(42)
            np.random.seed(42)

            task = get_task(task_name)
            model = AblationQASAModel(
                hidden_dim=hidden_dim, num_layers=num_layers,
                seq_len=seq_len_train,
                quantum_layer_indices=cfg['quantum_indices'],
            ).to(device)

            total_params, _ = count_parameters(model)
            save_dir = os.path.join(checkpoints_dir, cfg['name'], task_name)

            try:
                start_time = time.time()
                mae, mse, train_loss = train_and_evaluate(
                    model, task, task_name, device, config, save_dir=save_dir
                )
                elapsed = time.time() - start_time

                print(f"  => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {elapsed:.0f}s")

                all_results.append({
                    'ablation_type': cfg['type'],
                    'config_name': cfg['name'],
                    'quantum_indices': str(cfg['quantum_indices']),
                    'n_quantum_layers': len(cfg['quantum_indices']),
                    'task': display_name,
                    'task_key': task_name,
                    'mae': mae,
                    'mse': mse,
                    'train_loss': train_loss,
                    'total_params': total_params,
                    'time_seconds': elapsed,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    'ablation_type': cfg['type'],
                    'config_name': cfg['name'],
                    'quantum_indices': str(cfg['quantum_indices']),
                    'n_quantum_layers': len(cfg['quantum_indices']),
                    'task': display_name,
                    'task_key': task_name,
                    'mae': np.inf,
                    'mse': np.inf,
                    'train_loss': np.inf,
                    'total_params': total_params,
                    'time_seconds': 0,
                })

            del model

    # Save results CSV
    results_csv = os.path.join(results_dir, f"ablation_results_{timestamp}.csv")
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ablation_type', 'config_name', 'quantum_indices', 'n_quantum_layers',
            'task', 'mae', 'mse', 'train_loss', 'total_params', 'time_seconds',
        ])
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in writer.fieldnames}
            for k in ['mae', 'mse', 'train_loss']:
                row[k] = f"{r[k]:.6f}"
            writer.writerow(row)

    # Print summary tables
    print(f"\n\n{'='*90}")
    print("ABLATION RESULTS — POSITION (1 quantum layer)")
    print(f"{'='*90}")
    print(f"{'Config':<18} {'Task':<22} {'MAE':>10} {'MSE':>12} {'Params':>10} {'Time(s)':>8}")
    print("-" * 90)
    for r in all_results:
        if r['ablation_type'] == 'position':
            print(f"{r['config_name']:<18} {r['task']:<22} {r['mae']:>10.4f} {r['mse']:>12.4f} "
                  f"{r['total_params']:>10,} {r['time_seconds']:>8.0f}")
    print("-" * 90)

    print(f"\n{'='*90}")
    print("ABLATION RESULTS — COUNT (varying quantum layers)")
    print(f"{'='*90}")
    print(f"{'Config':<18} {'Task':<22} {'MAE':>10} {'MSE':>12} {'Params':>10} {'Time(s)':>8}")
    print("-" * 90)
    for r in all_results:
        if r['ablation_type'] == 'count':
            print(f"{r['config_name']:<18} {r['task']:<22} {r['mae']:>10.4f} {r['mse']:>12.4f} "
                  f"{r['total_params']:>10,} {r['time_seconds']:>8.0f}")
    print("-" * 90)

    print(f"\nResults saved to: {results_csv}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, epochs={config['epochs']}, "
          f"seed=42, lr={config['lr']}, seq_len={seq_len_train}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="QASA Ablation Study")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Training epochs (default: 200)")
    parser.add_argument('--configs', nargs='+', default=None,
                        help="Specific configs to run (e.g., pos_Q_first count_0Q)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Quick sanity check with 1 epoch")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        if args.configs is None:
            args.configs = ['pos_Q_last', 'count_0Q']
        print("=== DRY RUN MODE (1 epoch) ===\n")

    run_ablation(args)


if __name__ == "__main__":
    main()
