#!/usr/bin/env python3
"""
QnnFormer Baseline Model
=========================
Adapted from Cai et al. "QNNformer" (2024)
Reference: https://github.com/caizhongqi/QNNformer

QnnFormer replaces the Q/K/V linear projections in multi-head attention with
Variational Quantum Circuits that incorporate Grover Diffusion and QAOA layers,
plus multi-basis measurement (PauliX, PauliY, PauliZ).

This implementation adapts the quantum attention mechanism for our benchmark's
time-series forecasting setup (input_dim=1, seq_len=20, hidden_dim=64).

Architecture:
  - Embedding: Linear(1→hidden_dim) + LayerNorm + Dropout
  - PositionalEncoding
  - num_layers × Encoder: [QnnFormerAttention + FFN]
    (1 quantum attention layer + (num_layers-1) classical TransformerEncoderLayers)
  - Output: Linear(hidden_dim→1)

Quantum Attention:
  - Input split into n_qubit chunks
  - Each chunk processed by VQC: AngleEmbedding → [RX/RY/RZ + Entanglement +
    GroverDiffusion + QAOA] × n_layers → Multi-basis measurement (X,Y,Z)
  - 3× output features fused via linear projection
  - Separate VQC sets for Q, K, V

Interface: QnnFormerModel(input_dim=1, output_dim=1, hidden_dim=64, num_layers=4, seq_len=20)
  forward(x): (batch, seq_len, input_dim) → (batch, seq_len, output_dim)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# Quantum Components
# ============================================================

N_QUBITS = 4  # QnnFormer uses 4 qubits (smaller than QASA's 8)
N_CIRCUIT_LAYERS = 3  # Variational depth

# Device for quantum circuits
qnnformer_dev = qml.device("lightning.qubit", wires=N_QUBITS)


def entangling_layer(wires, weights):
    """Advanced entanglement: CRX for long-range coupling."""
    n = len(wires)
    for i in range(n):
        qml.CRX(weights[i], wires=[wires[i], wires[(i + 1) % n]])


def grover_diffusion(wires):
    """Grover diffusion operator for amplitude amplification."""
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for w in wires:
        qml.PauliX(wires=w)
    # Multi-controlled Z via Hadamard sandwich on target
    if n >= 2:
        qml.Hadamard(wires=wires[-1])
        if n == 2:
            qml.CNOT(wires=[wires[0], wires[1]])
        elif n == 3:
            qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
        else:
            # For 4+ qubits use decomposed multi-controlled X
            qml.ctrl(qml.PauliX(wires=wires[-1]), control=list(wires[:-1]))
        qml.Hadamard(wires=wires[-1])
    for w in wires:
        qml.PauliX(wires=w)
    for w in wires:
        qml.Hadamard(wires=w)


def qaoa_layer(wires, gamma, beta):
    """QAOA-inspired cost + mixer layer."""
    n = len(wires)
    # Cost layer: ZZ interactions
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(gamma, wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    # Mixer layer: X rotations
    for w in wires:
        qml.RX(beta, wires=w)


@qml.qnode(qnnformer_dev, interface="torch")
def qnnformer_circuit(inputs, weights, entangle_weights, gammas, betas):
    """
    QnnFormer VQC for Q/K/V generation.

    Architecture:
      1. AngleEmbedding (RY)
      2. For each layer:
         a. RX, RY, RZ trainable rotations
         b. CRX entangling layer
         c. Grover diffusion operator
         d. QAOA cost + mixer layer
      3. Multi-basis measurement: PauliZ + PauliX + PauliY on all qubits

    Args:
        inputs: (N_QUBITS,) — input features for this chunk
        weights: (N_CIRCUIT_LAYERS, N_QUBITS) — rotation weights
        entangle_weights: (N_CIRCUIT_LAYERS, N_QUBITS) — entanglement weights
        gammas: (N_CIRCUIT_LAYERS,) — QAOA cost parameter
        betas: (N_CIRCUIT_LAYERS,) — QAOA mixer parameter

    Returns:
        3*N_QUBITS expectation values (Z, X, Y measurements)
    """
    wires = list(range(N_QUBITS))

    # Angle encoding
    qml.AngleEmbedding(inputs, wires=wires, rotation='Y')

    for layer in range(N_CIRCUIT_LAYERS):
        # Trainable rotations (3 axes)
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)

        # Entanglement
        entangling_layer(wires, entangle_weights[layer])

        # Grover diffusion
        grover_diffusion(wires)

        # QAOA layer
        qaoa_layer(wires, gammas[layer], betas[layer])

    # Multi-basis measurement
    measurements = []
    for i in range(N_QUBITS):
        measurements.append(qml.expval(qml.PauliZ(i)))
    for i in range(N_QUBITS):
        measurements.append(qml.expval(qml.PauliX(i)))
    for i in range(N_QUBITS):
        measurements.append(qml.expval(qml.PauliY(i)))

    return measurements


class QnnFormerVQC(nn.Module):
    """Single VQC module for one input chunk."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(N_CIRCUIT_LAYERS, N_QUBITS))
        self.entangle_weights = nn.Parameter(0.01 * torch.randn(N_CIRCUIT_LAYERS, N_QUBITS))
        self.gammas = nn.Parameter(0.01 * torch.randn(N_CIRCUIT_LAYERS))
        self.betas = nn.Parameter(0.01 * torch.randn(N_CIRCUIT_LAYERS))

    def forward(self, x):
        """
        Args:
            x: (batch, N_QUBITS) — one chunk of input features

        Returns:
            (batch, 3*N_QUBITS) — multi-basis measurements
        """
        outputs = []
        for i in range(x.size(0)):
            result = qnnformer_circuit(
                x[i], self.weights, self.entangle_weights, self.gammas, self.betas
            )
            outputs.append(torch.stack(result))
        return torch.stack(outputs)


# ============================================================
# Quantum Attention
# ============================================================

class QnnFormerAttention(nn.Module):
    """
    Quantum attention layer that generates Q, K, V using VQCs.

    Uses projection approach (like QASA): hidden_dim → N_QUBITS via linear,
    process through VQC, multi-basis measurement gives 3*N_QUBITS features,
    then project back to hidden_dim. One VQC each for Q, K, V (3 total).
    """

    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Input projection: hidden_dim → N_QUBITS
        self.input_proj = nn.Linear(hidden_dim, N_QUBITS)
        self.input_norm = nn.LayerNorm(N_QUBITS)

        # One VQC per Q, K, V
        self.vqc_q = QnnFormerVQC()
        self.vqc_k = QnnFormerVQC()
        self.vqc_v = QnnFormerVQC()

        # Output projection: 3*N_QUBITS → hidden_dim for each of Q, K, V
        fused_dim = 3 * N_QUBITS  # Multi-basis measurement triples output
        self.proj_q = nn.Linear(fused_dim, hidden_dim)
        self.proj_k = nn.Linear(fused_dim, hidden_dim)
        self.proj_v = nn.Linear(fused_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.input_proj, self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.kaiming_uniform_(proj.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(proj.bias, 0)

    def _quantum_transform(self, x, vqc):
        """Project to qubit space, run VQC, project back."""
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, self.hidden_dim)

        # Project to N_QUBITS
        projected = torch.tanh(self.input_proj(x_flat))  # (batch*seq, N_QUBITS)
        projected = self.input_norm(projected)

        # Run VQC
        proj_cpu = projected.cpu() if projected.device.type != 'cpu' else projected
        vqc_out = vqc(proj_cpu).to(x.device)  # (batch*seq, 3*N_QUBITS)

        return vqc_out.reshape(batch_size, seq_len, -1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim) — attention output
        """
        batch_size, seq_len, _ = x.shape

        # Generate Q, K, V via quantum circuits
        q = self.proj_q(self._quantum_transform(x, self.vqc_q))
        k = self.proj_k(self._quantum_transform(x, self.vqc_k))
        v = self.proj_v(self._quantum_transform(x, self.vqc_v))

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.proj_out(out)


# ============================================================
# QnnFormer Encoder Layer
# ============================================================

class QnnFormerEncoderLayer(nn.Module):
    """Transformer encoder layer with quantum attention."""

    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = QnnFormerAttention(hidden_dim, num_heads=4, dropout_rate=dropout_rate)
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
        # Self-attention with residual
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


# ============================================================
# QnnFormer Model (Top-Level)
# ============================================================

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


class QnnFormerModel(nn.Module):
    """
    QnnFormer model for time-series forecasting.

    Architecture:
      Embedding(input_dim → hidden_dim) + LayerNorm + Dropout
      → PositionalEncoding
      → (num_layers-1) × Classical TransformerEncoderLayer
      → 1 × QnnFormerEncoderLayer (quantum attention in last layer)
      → Linear(hidden_dim → output_dim)

    This mirrors QASA's approach of placing the quantum layer last,
    ensuring a fair architectural comparison.
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)

        # (num_layers-1) classical + 1 quantum encoder layer
        encoder_layers = []
        for _ in range(num_layers - 1):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, batch_first=True,
                    dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
                )
            )
        encoder_layers.append(QnnFormerEncoderLayer(hidden_dim, dropout_rate=dropout_rate))
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
# Sanity Check
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Count quantum params: 3 VQCs (Q, K, V) × (weights + entangle_weights + gammas + betas)
    q_per_vqc = N_CIRCUIT_LAYERS * N_QUBITS + N_CIRCUIT_LAYERS * N_QUBITS + N_CIRCUIT_LAYERS + N_CIRCUIT_LAYERS
    quantum = 3 * q_per_vqc  # 3 VQCs: Q, K, V
    return total, trainable, quantum


if __name__ == "__main__":
    print("QnnFormer Baseline Model")
    print("=" * 50)

    model = QnnFormerModel(input_dim=1, output_dim=1, hidden_dim=64, num_layers=4, seq_len=20)
    total, trainable, quantum = count_parameters(model)
    print(f"Total parameters:    {total:,}")
    print(f"Trainable parameters:{trainable:,}")
    print(f"Quantum parameters:  {quantum} (3 VQCs × {N_CIRCUIT_LAYERS} layers, Grover+QAOA)")
    print(f"VQC circuit: {N_QUBITS} qubits, {N_CIRCUIT_LAYERS} layers, multi-basis (X,Y,Z)")
    print(f"Device: lightning.qubit")

    print("\nRunning forward pass...")
    x = torch.randn(1, 20, 1)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 20, 1), f"Expected (1, 20, 1), got {y.shape}"
    print("✓ Forward pass successful!")
