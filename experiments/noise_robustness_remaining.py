#!/usr/bin/env python3
"""
Run remaining noise robustness experiments.
Seed 42: all done. Seed 43: p=0.0-0.01 done, need p=0.05, 0.1. Seed 44: all needed.
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
TASK_KEY = 'classical_chaotic_logistic'
EPOCHS = 100
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

# Already completed results
DONE = {
    (42, 0.0): (0.388263, 0.214685), (42, 0.001): (0.367554, 0.198395),
    (42, 0.005): (0.430811, 0.244445), (42, 0.01): (0.347158, 0.189453),
    (42, 0.05): (0.422747, 0.242207), (42, 0.1): (0.322017, 0.177836),
    (43, 0.0): (0.358436, 0.197282), (43, 0.001): (0.331503, 0.171968),
    (43, 0.005): (0.369972, 0.206616), (43, 0.01): (0.332747, 0.180177),
}

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
SEEDS = [42, 43, 44]


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
        self.input_dim, self.output_dim = input_dim, output_dim
        self.qlayer = qml.qnn.TorchLayer(make_noisy_circuit(noise_level), {'weights': (N_QLAYERS, N_QUBITS)})
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
        out = self.output_proj(torch.stack(outputs))
        return x + out if self.input_dim == self.output_dim else out


class NoisyQuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, noise_level, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = NoisyQuantumLayer(hidden_dim, hidden_dim, noise_level)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(4*hidden_dim, hidden_dim), nn.Dropout(dropout_rate))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        b, s, f = x.shape
        q_out = self.v_quantum(x.reshape(b*s, f), float(s)).view(b, s, f)
        return self.norm2(q_out + self.ffn(q_out))


class NoisyQASAModel(nn.Module):
    def __init__(self, noise_level=0.0, hidden_dim=64, num_layers=4, seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate))
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len+100)
        layers = [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=dropout_rate, dim_feedforward=4*hidden_dim) for _ in range(num_layers-1)]
        layers.append(NoisyQuantumEncoderLayer(hidden_dim, noise_level, dropout_rate))
        self.encoder = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))
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
    seq_len, num_total = 20, X_train.shape[1]
    model.train()
    best_loss, best_state = float('inf'), None
    for epoch in range(EPOCHS):
        epoch_loss, n = 0.0, 0
        if num_total > seq_len:
            for i in range(num_total - seq_len + 1):
                optimizer.zero_grad()
                loss = criterion(model(X_train[:, i:i+seq_len, :]), Y_train[:, i:i+seq_len, :])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                epoch_loss += loss.item(); n += 1
        else:
            optimizer.zero_grad()
            loss = criterion(model(X_train), Y_train)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            epoch_loss, n = loss.item(), 1
        scheduler.step()
        avg = epoch_loss / max(n, 1)
        if avg < best_loss: best_loss = avg; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0: print(f"    Epoch [{epoch+1}/{EPOCHS}] Loss: {avg:.6f}")
    if best_state: model.load_state_dict(best_state)
    model.eval(); gen = []; inp = X_test_seed.clone()
    with torch.no_grad():
        for _ in range(Y_test_true.shape[1]):
            if inp.shape[1] == 0: break
            p = model(inp); nxt = p[:, -1:, :].clone(); gen.append(nxt.squeeze().cpu().numpy())
            inp = torch.cat([inp[:, 1:, :], nxt], dim=1)
    if not gen: return np.inf, np.inf
    yp = np.array(gen).flatten(); yt = Y_test_true.squeeze().cpu().numpy().flatten()
    ml = min(len(yp), len(yt)); yp, yt = yp[:ml], yt[:ml]
    return mean_absolute_error(yt, yp), mean_squared_error(yt, yp)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task = get_task(TASK_KEY)
    all_results = []

    # Add completed results
    for (seed, p), (mae, mse) in DONE.items():
        all_results.append({'seed': seed, 'noise_level': p, 'mae': mae, 'mse': mse})
        print(f"  [DONE] seed={seed} p={p} MAE={mae:.4f}")

    # Run remaining
    remaining = [(s, p) for s in SEEDS for p in NOISE_LEVELS if (s, p) not in DONE]
    print(f"\nRemaining: {len(remaining)} runs\n")

    for seed, p in remaining:
        print(f"\n  [p={p}] Seed {seed}")
        torch.manual_seed(seed); np.random.seed(seed)
        model = NoisyQASAModel(noise_level=p)
        start = time.time()
        mae, mse = train_and_evaluate(model, task)
        print(f"    => MAE: {mae:.6f}, MSE: {mse:.6f}, Time: {time.time()-start:.0f}s")
        all_results.append({'seed': seed, 'noise_level': p, 'mae': mae, 'mse': mse})
        del model

    # Save & print summary
    csv_path = os.path.join(RESULTS_DIR, f"noise_robustness_3seeds_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'noise_level', 'mae', 'mse'])
        w.writeheader(); w.writerows(sorted(all_results, key=lambda x: (x['noise_level'], x['seed'])))

    print(f"\n{'='*70}\n3-SEED SUMMARY (mean ± std)\n{'='*70}")
    print(f"{'Noise Level':<12} {'MAE':>18} {'MSE':>18}")
    print("-" * 70)
    base = None
    for p in NOISE_LEVELS:
        maes = [r['mae'] for r in all_results if r['noise_level'] == p]
        mses = [r['mse'] for r in all_results if r['noise_level'] == p]
        mm, sm = np.mean(maes), np.std(maes)
        if base is None: base = mm
        print(f"  p={p:<8} {mm:.4f} ± {sm:.4f}    {np.mean(mses):.4f} ± {np.std(mses):.4f}    ({(mm-base)/base*100:+.1f}%)")
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
