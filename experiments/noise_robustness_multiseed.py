#!/usr/bin/env python3
"""
Noise robustness with 3 seeds (42, 43, 44) for statistical confidence.
4 qubits, 2 layers, depolarizing noise, Chaotic Logistic, 100 epochs.
"""

import os, sys, csv, time, datetime, math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task

N_QUBITS = 4
N_QLAYERS = 2
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
SEEDS = [43, 44]  # seed 42 already done in lite run
TASK_KEY = 'classical_chaotic_logistic'
EPOCHS = 100
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def make_noisy_circuit(noise_level):
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def noisy_circuit(inputs, weights):
        for i in range(N_QUBITS):
            qml.RX(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        for i in range(N_QUBITS):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        if noise_level > 0:
            for i in range(N_QUBITS):
                qml.DepolarizingChannel(noise_level, wires=i)
        for l in range(1, N_QLAYERS):
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)
            if noise_level > 0:
                for i in range(N_QUBITS):
                    qml.DepolarizingChannel(noise_level, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return noisy_circuit


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
    def __init__(self, input_dim, output_dim, noise_level):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        circuit = make_noisy_circuit(noise_level)
        self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': (N_QLAYERS, N_QUBITS)})
        self.input_proj = nn.Linear(input_dim, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)
        self.output_proj = nn.Linear(N_QUBITS, output_dim)
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
    def __init__(self, hidden_dim, noise_level, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = NoisyQuantumLayer(hidden_dim, hidden_dim, noise_level)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout_rate),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        b, s, f = x.shape
        x_flat = x.reshape(b * s, f)
        q_out = self.v_quantum(x_flat, float(s))
        q_out = q_out.view(b, s, f)
        x = self.norm2(q_out + self.ffn(q_out))
        return x


class NoisyQASAModel(nn.Module):
    def __init__(self, noise_level=0.0, hidden_dim=64, num_layers=4, seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        encoder_layers = []
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, batch_first=True,
                dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
            ))
        encoder_layers.append(NoisyQuantumEncoderLayer(hidden_dim, noise_level, dropout_rate))
        self.encoder = nn.ModuleList(encoder_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x)


def train_and_evaluate(model, task):
    device = torch.device("cpu")
    X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test_seed, Y_test_true = X_test_seed.to(device), Y_test_true.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    seq_len = 20
    num_total = X_train.shape[1]

    model.train()
    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_windows = 0
        if num_total > seq_len:
            for i in range(num_total - seq_len + 1):
                optimizer.zero_grad()
                pred = model(X_train[:, i:i + seq_len, :])
                loss = criterion(pred, Y_train[:, i:i + seq_len, :])
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
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

    if best_state:
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

    if not generated:
        return np.inf, np.inf, best_loss

    y_pred = np.array(generated).flatten()
    y_true = Y_test_true.squeeze().cpu().numpy().flatten()
    min_len = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:min_len], y_true[:min_len]
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), best_loss


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    task = get_task(TASK_KEY)

    # Seed 42 results from lite run
    seed42 = {
        0.0:   (0.388263, 0.214685),
        0.001: (0.367554, 0.198395),
        0.005: (0.430811, 0.244445),
        0.01:  (0.347158, 0.189453),
        0.05:  (0.422747, 0.242207),
        0.1:   (0.322017, 0.177836),
    }

    all_results = []

    print("=" * 70)
    print("NOISE ROBUSTNESS (Multi-seed): Seeds 43, 44")
    print(f"Levels: {NOISE_LEVELS}, Epochs: {EPOCHS}")
    print("=" * 70)

    for seed in SEEDS:
        for p in NOISE_LEVELS:
            print(f"\n  [p={p}] Seed {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = NoisyQASAModel(noise_level=p, hidden_dim=64, num_layers=4, seq_len=20)
            start = time.time()
            mae, mse, _ = train_and_evaluate(model, task)
            elapsed = time.time() - start
            print(f"    => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {elapsed:.0f}s")

            all_results.append({'seed': seed, 'noise_level': p, 'mae': mae, 'mse': mse})
            del model

    # Combine with seed 42
    for p in NOISE_LEVELS:
        all_results.append({'seed': 42, 'noise_level': p, 'mae': seed42[p][0], 'mse': seed42[p][1]})

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"noise_robustness_3seeds_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'noise_level', 'mae', 'mse'])
        writer.writeheader()
        writer.writerows(sorted(all_results, key=lambda x: (x['noise_level'], x['seed'])))

    # Print mean±std summary
    print(f"\n\n{'='*70}")
    print("3-SEED SUMMARY (mean ± std)")
    print(f"{'='*70}")
    print(f"{'Noise Level':<12} {'MAE':>18} {'MSE':>18}")
    print("-" * 70)

    base_mae = None
    for p in NOISE_LEVELS:
        maes = [r['mae'] for r in all_results if r['noise_level'] == p]
        mses = [r['mse'] for r in all_results if r['noise_level'] == p]
        m_mae, s_mae = np.mean(maes), np.std(maes)
        m_mse, s_mse = np.mean(mses), np.std(mses)
        if base_mae is None:
            base_mae = m_mae
        deg = (m_mae - base_mae) / base_mae * 100
        print(f"  p={p:<8} {m_mae:.4f} ± {s_mae:.4f}    {m_mse:.4f} ± {s_mse:.4f}    ({deg:+.1f}%)")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
