#!/usr/bin/env python3
"""
ETTh1 Real-World Dataset Experiment
=====================================
Runs QASA vs Classical Transformer on the ETTh1 (Electricity Transformer
Temperature) dataset — univariate OT (Oil Temperature) prediction.

Outputs:
  - experiments/results/etth1_results.csv
  - experiments/results/plots/etth1_*.png

Usage:
  python experiments/run_etth1_experiment.py                # Full run
  python experiments/run_etth1_experiment.py --dry-run      # 1 epoch sanity check
  python experiments/run_etth1_experiment.py --epochs 100
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
import traceback
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Model Definitions (identical to run_qasa_benchmark.py)
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
                 seq_len=96, dropout_rate=0.1):
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
                 seq_len=96, dropout_rate=0.1):
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
# Data Loading
# ============================================================

def download_etth1(data_dir):
    """Download ETTh1.csv if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "ETTh1.csv")
    if os.path.exists(filepath):
        print(f"ETTh1.csv already exists at {filepath}")
        return filepath
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    print(f"Downloading ETTh1.csv from {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to {filepath}")
    return filepath


def load_etth1(filepath, seq_len=96):
    """Load ETTh1 and prepare univariate OT prediction data.

    Returns train/val/test as (X, Y) tensors where:
      X: [1, num_windows, seq_len, 1]  (input windows)
      Y: [1, num_windows, 1]           (next-step targets)
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    ot = df['OT'].values.astype(np.float32)

    # Standard split
    train_end = 8640
    val_end = 11520
    test_end = 14400

    # Z-score normalize using training set only
    train_data = ot[:train_end]
    mean = train_data.mean()
    std = train_data.std()
    ot_norm = (ot - mean) / (std + 1e-8)

    def make_windows(data, seq_len):
        X, Y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len])
        X = np.array(X)[:, :, np.newaxis]  # [N, seq_len, 1]
        Y = np.array(Y)[:, np.newaxis]      # [N, 1]
        return torch.FloatTensor(X), torch.FloatTensor(Y)

    X_train_full, Y_train_full = make_windows(ot_norm[:train_end], seq_len)
    X_val, Y_val = make_windows(ot_norm[train_end - seq_len:val_end], seq_len)
    X_test, Y_test = make_windows(ot_norm[val_end - seq_len:test_end], seq_len)

    # Subsample training windows for quantum circuit tractability
    max_train = 500
    if X_train_full.shape[0] > max_train:
        idx = np.linspace(0, X_train_full.shape[0] - 1, max_train, dtype=int)
        X_train = X_train_full[idx]
        Y_train = Y_train_full[idx]
    else:
        X_train = X_train_full
        Y_train = Y_train_full

    # Also subsample test to 500 windows
    max_test = 500
    if X_test.shape[0] > max_test:
        idx = np.linspace(0, X_test.shape[0] - 1, max_test, dtype=int)
        X_test = X_test[idx]
        Y_test = Y_test[idx]

    print(f"ETTh1 loaded: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} windows (subsampled)")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std


# ============================================================
# Training & Evaluation
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_and_evaluate_etth1(model, X_train, Y_train, X_test, Y_test,
                              device, config, save_dir=None):
    """Train on ETTh1 with mini-batch windowed training, evaluate on test set."""
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()

    seq_len = config['seq_len']
    batch_size = config.get('batch_size', 32)
    n_train = X_train.shape[0]

    model.train()
    best_loss = float('inf')
    best_state = None
    loss_history = []

    for epoch in range(config['epochs']):
        # Shuffle training data
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            x_batch = X_train[idx]  # [B, seq_len, 1]
            y_batch = Y_train[idx]  # [B, 1]

            optimizer.zero_grad()
            pred = model(x_batch)         # [B, seq_len, 1]
            pred_last = pred[:, -1, :]    # [B, 1] — predict last step
            loss = criterion(pred_last, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % config['print_every'] == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch [{epoch+1}/{config['epochs']}] Loss: {avg_loss:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    n_test = X_test.shape[0]
    all_preds = []

    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_batch = X_test[start:end]
            pred = model(x_batch)
            pred_last = pred[:, -1, :]
            all_preds.append(pred_last.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = Y_test.cpu().numpy().flatten()

    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # Save artifacts
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            'model_state_dict': best_state,
            'best_loss': best_loss,
            'mae': mae, 'mse': mse,
            'epochs': config['epochs'],
        }, os.path.join(save_dir, 'best_model.pth'))

        with open(os.path.join(save_dir, 'loss_curve.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for i, l in enumerate(loss_history):
                writer.writerow([i + 1, f'{l:.6f}'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Show first 200 test points for clarity
        show_n = min(200, min_len)
        axes[0].plot(y_true[:show_n], 'b-', label='Ground Truth', alpha=0.8)
        axes[0].plot(y_pred[:show_n], 'r--', label='Predicted', alpha=0.8)
        axes[0].set_title('ETTh1 OT Prediction (Test Set)')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Normalized OT')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(loss_history, 'g-', linewidth=0.8)
        axes[1].set_title(f'Training Loss (best={best_loss:.6f})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'ETTh1 — MAE={mae:.4f}  MSE={mse:.4f}', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction.png'), dpi=150)
        plt.close(fig)

    return mae, mse, best_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ETTh1 Real-World Dataset Experiment")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seq-len', type=int, default=20,
                        help='Input sequence length (default: 20, matching benchmark)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        print("=== DRY RUN MODE (1 epoch) ===\n")

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PennyLane: {qml.__version__}\n")

    # Config
    hidden_dim = 64
    num_layers = 4
    seq_len = args.seq_len
    config = {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'seq_len': seq_len,
        'batch_size': 16,
        'print_every': max(1, args.epochs // 10),
    }

    # Download and load data
    data_dir = os.path.join(PROJECT_ROOT, "experiments", "data")
    csv_path = download_etth1(data_dir)
    X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std = load_etth1(csv_path, seq_len)

    # Results directory
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Parameter count
    qasa_model = QASAModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len)
    classical_model = ClassicalModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len)
    qt, qtr = count_parameters(qasa_model)
    ct, ctr = count_parameters(classical_model)
    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'Total':>12} {'Trainable':>12}")
    print(f"{'-'*60}")
    print(f"{'QASA (Quantum)':<25} {qt:>12,} {qtr:>12,}")
    print(f"{'Classical Transformer':<25} {ct:>12,} {ctr:>12,}")
    print(f"{'='*60}\n")
    del qasa_model, classical_model

    results = {}
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Classical ---
    print("=" * 60)
    print("Training Classical Transformer on ETTh1")
    print("=" * 60)
    c_model = ClassicalModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len).to(device)
    c_save = os.path.join(results_dir, "checkpoints", "etth1", "classical")
    c_mae, c_mse, c_best = train_and_evaluate_etth1(
        c_model, X_train, Y_train, X_test, Y_test, device, config, save_dir=c_save
    )
    print(f"=> Classical — MAE: {c_mae:.6f}, MSE: {c_mse:.6f}\n")
    results['classical'] = (c_mae, c_mse)
    del c_model

    # --- QASA ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("=" * 60)
    print("Training QASA (Quantum) on ETTh1")
    print("=" * 60)
    q_model = QASAModel(hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len).to(device)
    q_save = os.path.join(results_dir, "checkpoints", "etth1", "qasa")
    q_mae, q_mse, q_best = train_and_evaluate_etth1(
        q_model, X_train, Y_train, X_test, Y_test, device, config, save_dir=q_save
    )
    print(f"=> QASA — MAE: {q_mae:.6f}, MSE: {q_mse:.6f}\n")
    results['qasa'] = (q_mae, q_mse)
    del q_model

    # Save results
    results_csv = os.path.join(results_dir, "etth1_results.csv")
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Model', 'MAE', 'MSE'])
        writer.writerow(['ETTh1', 'Classical', f"{c_mae:.6f}", f"{c_mse:.6f}"])
        writer.writerow(['ETTh1', 'QASA', f"{q_mae:.6f}", f"{q_mse:.6f}"])

    # Summary
    winner_mae = "QASA" if q_mae < c_mae else "Classical"
    winner_mse = "QASA" if q_mse < c_mse else "Classical"
    print(f"\n{'='*60}")
    print("ETTh1 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Classical — MAE: {c_mae:.4f}, MSE: {c_mse:.4f}")
    print(f"QASA      — MAE: {q_mae:.4f}, MSE: {q_mse:.4f}")
    print(f"Winner: MAE→{winner_mae}, MSE→{winner_mse}")
    print(f"\nResults saved to: {results_csv}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, epochs={config['epochs']}, "
          f"seq_len={seq_len}, lr={config['lr']}, batch_size={config['batch_size']}")


if __name__ == "__main__":
    main()
