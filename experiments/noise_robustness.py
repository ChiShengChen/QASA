#!/usr/bin/env python3
"""
Quantum Noise Robustness Analysis for QASA
=============================================
Tests QASA performance under realistic NISQ noise models:
  - Depolarizing noise (gate errors)
  - Bit-flip noise (readout errors)
  - Amplitude damping (energy relaxation)

Runs on 2 representative tasks:
  - Chaotic Logistic (QASA wins) — tests if advantage survives noise
  - Damped Oscillator (Classical wins) — tests if gap widens under noise

Output: CSV + plots for the paper.
"""

import os, sys, csv, time, datetime
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
N_QUBITS = 8
N_QLAYERS = 4
HIDDEN_DIM = 64
NUM_LAYERS = 4
SEQ_LEN = 20
EPOCHS = 200
SEED = 42

NOISE_TYPES = ['depolarizing', 'bit_flip', 'amplitude_damping']
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

TASKS = [
    ('classical_chaotic_logistic', 'Chaotic Logistic'),
    ('classical_damped_oscillation', 'Damped Oscillator'),
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


# ============================================================
# Noisy quantum circuit
# ============================================================
def make_noisy_circuit(noise_type, noise_level):
    """Create a quantum circuit with specified noise channel."""
    dev = qml.device("default.mixed", wires=N_QUBITS + 1)

    @qml.qnode(dev, interface="torch")
    def noisy_circuit(inputs, weights):
        # Input encoding
        for i in range(N_QUBITS):
            qml.RX(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)

        # First variational layer
        for i in range(N_QUBITS):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)

        # Apply noise after encoding
        if noise_level > 0:
            for i in range(N_QUBITS):
                _apply_noise(noise_type, noise_level, i)

        # Subsequent variational layers with entanglement
        for l in range(1, N_QLAYERS):
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)

            # Apply noise after each layer
            if noise_level > 0:
                for i in range(N_QUBITS):
                    _apply_noise(noise_type, noise_level, i)

            qml.CNOT(wires=[N_QUBITS - 1, N_QUBITS])
            qml.RY(weights[l, -1], wires=N_QUBITS)

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return noisy_circuit


def _apply_noise(noise_type, p, wire):
    """Apply a noise channel to a single wire."""
    if noise_type == 'depolarizing':
        qml.DepolarizingChannel(p, wires=wire)
    elif noise_type == 'bit_flip':
        qml.BitFlip(p, wires=wire)
    elif noise_type == 'amplitude_damping':
        qml.AmplitudeDamping(p, wires=wire)


# ============================================================
# Noisy QASA Model
# ============================================================
import math

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


class NoisyQuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim, noise_type, noise_level):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_shape = (N_QLAYERS, N_QUBITS + 1)

        noisy_circuit = make_noisy_circuit(noise_type, noise_level)
        self.qlayer = qml.qnn.TorchLayer(noisy_circuit, {'weights': self.weight_shape})
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


class NoisyQuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, noise_type, noise_level, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = NoisyQuantumLayer(hidden_dim, hidden_dim, noise_type, noise_level)
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


class NoisyQASAModel(nn.Module):
    def __init__(self, noise_type='depolarizing', noise_level=0.0,
                 input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
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
        encoder_layers.append(NoisyQuantumEncoderLayer(hidden_dim, noise_type, noise_level, dropout_rate))
        self.encoder = nn.ModuleList(encoder_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x)


# ============================================================
# Training & Evaluation (reuse from benchmark)
# ============================================================
def train_and_evaluate(model, task, task_name, config):
    device = torch.device("cpu")
    X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test_seed, Y_test_true = X_test_seed.to(device), Y_test_true.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()

    seq_len_train = config['seq_len_train']
    num_total = X_train.shape[1]

    model.train()
    best_loss = float('inf')
    best_state = None

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

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % config['print_every'] == 0:
            print(f"    Epoch [{epoch+1}/{config['epochs']}] Loss: {avg_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
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

    if len(generated) == 0:
        return np.inf, np.inf, best_loss

    y_pred = np.array(generated).flatten()
    y_true = Y_test_true.squeeze().cpu().numpy().flatten()
    min_len = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:min_len], y_true[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse, best_loss


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config = {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': EPOCHS,
        'seq_len_train': SEQ_LEN,
        'print_every': 50,
    }

    all_results = []

    for noise_type in NOISE_TYPES:
        print(f"\n{'='*70}")
        print(f"NOISE TYPE: {noise_type}")
        print(f"{'='*70}")

        for noise_level in NOISE_LEVELS:
            for task_key, task_name in TASKS:
                print(f"\n  [{noise_type} p={noise_level}] {task_name}")

                torch.manual_seed(SEED)
                np.random.seed(SEED)

                task = get_task(task_key)

                model = NoisyQASAModel(
                    noise_type=noise_type,
                    noise_level=noise_level,
                    hidden_dim=HIDDEN_DIM,
                    num_layers=NUM_LAYERS,
                    seq_len=SEQ_LEN,
                )

                start = time.time()
                mae, mse, train_loss = train_and_evaluate(model, task, task_key, config)
                elapsed = time.time() - start

                print(f"    => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {elapsed:.0f}s")

                all_results.append({
                    'noise_type': noise_type,
                    'noise_level': noise_level,
                    'task': task_name,
                    'mae': mae,
                    'mse': mse,
                    'train_loss': train_loss,
                    'time': elapsed,
                })

                del model

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"noise_robustness_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {csv_path}")

    # Plot
    plot_noise_results(all_results, timestamp)


def plot_noise_results(results, timestamp):
    """Generate noise robustness plots."""
    fig, axes = plt.subplots(len(NOISE_TYPES), 2, figsize=(14, 4 * len(NOISE_TYPES)))

    for row, noise_type in enumerate(NOISE_TYPES):
        for col, (task_key, task_name) in enumerate(TASKS):
            ax = axes[row, col]
            subset = [r for r in results if r['noise_type'] == noise_type and r['task'] == task_name]
            levels = [r['noise_level'] for r in subset]
            maes = [r['mae'] for r in subset]
            mses = [r['mse'] for r in subset]

            ax.plot(levels, maes, 'o-', color='tab:blue', label='MAE', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(levels, mses, 's--', color='tab:red', label='MSE', linewidth=2)

            ax.set_xlabel('Noise Level (p)')
            ax.set_ylabel('MAE', color='tab:blue')
            ax2.set_ylabel('MSE', color='tab:red')
            ax.set_title(f'{noise_type}\n{task_name}')
            ax.set_xscale('symlog', linthresh=0.001)
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plots", f"noise_robustness_{timestamp}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
