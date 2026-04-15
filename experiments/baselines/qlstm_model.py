#!/usr/bin/env python3
"""
Quantum LSTM (QLSTM) Baseline Model
====================================
Adapted from Chen et al. "Quantum Long Short-Term Memory" (2022)
Reference: https://github.com/ycchen1989/Quantum_Long_Short_Term_Memory

This implementation replaces the 4 classical LSTM gates (input, forget, cell, output)
with Variational Quantum Circuits (VQCs). To keep simulation tractable and ensure
a fair comparison with QASA (8 qubits), we use projection layers to map the
concatenated [x_t, h_{t-1}] to 8 qubits and back.

Architecture:
  - Embedding: Linear(1→hidden_dim) + LayerNorm + Dropout
  - QLSTM Cell: 4 VQC gates (8 qubits, 4 variational layers each)
  - Output: Linear(hidden_dim→1)

Interface: QLSTMModel(input_dim=1, output_dim=1, hidden_dim=64, num_layers=4, seq_len=20)
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

N_QUBITS = 8
N_VQC_LAYERS = 4  # Depth of variational circuit (matches QASA's N_QLAYERS)

# Single shared device for all VQCs
qlstm_dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(qlstm_dev, interface="torch")
def vqc_circuit(inputs, weights):
    """
    Variational Quantum Circuit for QLSTM gates.

    Architecture (from Chen et al.):
      1. Hadamard layer (superposition)
      2. RY angle encoding of input
      3. For each variational layer:
         a. CNOT entangling (even pairs then odd pairs)
         b. Trainable RY rotations
      4. PauliZ measurement on all qubits

    Args:
        inputs: (N_QUBITS,) — projected input vector
        weights: (N_VQC_LAYERS, N_QUBITS) — trainable parameters

    Returns:
        List of N_QUBITS PauliZ expectation values
    """
    # 1. Initial superposition
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)

    # 2. Angle encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # 3. Variational layers
    for layer in range(N_VQC_LAYERS):
        # Entangling: even pairs
        for i in range(0, N_QUBITS - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        # Entangling: odd pairs
        for i in range(1, N_QUBITS - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        # Trainable rotations
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)

    # 4. Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class VQC(nn.Module):
    """Variational Quantum Circuit module wrapping a PennyLane QNode."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(N_VQC_LAYERS, N_QUBITS))

    def forward(self, x):
        """
        Args:
            x: (batch, N_QUBITS) — projected inputs

        Returns:
            (batch, N_QUBITS) — quantum circuit outputs
        """
        # QNode doesn't support batching; iterate over batch
        outputs = []
        for i in range(x.size(0)):
            result = vqc_circuit(x[i], self.weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs)


# ============================================================
# QLSTM Cell
# ============================================================

class QLSTMCell(nn.Module):
    """
    Quantum LSTM Cell with 4 VQC gates.

    Each gate: project(input_size+hidden_size → 8) → VQC → project(8 → hidden_size)
    Then apply standard LSTM activations (sigmoid for i/f/o, tanh for g).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # Shared input projection (concatenated [x_t, h_{t-1}] → 8 qubits)
        self.input_proj = nn.Linear(combined_size, N_QUBITS)
        self.input_norm = nn.LayerNorm(N_QUBITS)

        # 4 VQC gates
        self.vqc_input = VQC()
        self.vqc_forget = VQC()
        self.vqc_cell = VQC()
        self.vqc_output = VQC()

        # Output projections (8 → hidden_size) for each gate
        self.proj_input = nn.Linear(N_QUBITS, hidden_size)
        self.proj_forget = nn.Linear(N_QUBITS, hidden_size)
        self.proj_cell = nn.Linear(N_QUBITS, hidden_size)
        self.proj_output = nn.Linear(N_QUBITS, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in [self.input_proj, self.proj_input, self.proj_forget,
                       self.proj_cell, self.proj_output]:
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, x_t, hidden):
        """
        Args:
            x_t: (batch, input_size) — input at time step t
            hidden: (h_prev, c_prev) each (batch, hidden_size)

        Returns:
            h_t: (batch, hidden_size)
            c_t: (batch, hidden_size)
        """
        h_prev, c_prev = hidden

        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)  # (batch, input_size + hidden_size)

        # Project to qubit space
        proj = torch.tanh(self.input_proj(combined))  # (batch, 8)
        proj = self.input_norm(proj)

        # Run through 4 VQC gates (on CPU)
        proj_cpu = proj.cpu() if proj.device.type != 'cpu' else proj
        vqc_i = self.vqc_input(proj_cpu).to(proj.device)
        vqc_f = self.vqc_forget(proj_cpu).to(proj.device)
        vqc_g = self.vqc_cell(proj_cpu).to(proj.device)
        vqc_o = self.vqc_output(proj_cpu).to(proj.device)

        # Project back and apply activations
        i_t = torch.sigmoid(self.proj_input(vqc_i))
        f_t = torch.sigmoid(self.proj_forget(vqc_f))
        g_t = torch.tanh(self.proj_cell(vqc_g))
        o_t = torch.sigmoid(self.proj_output(vqc_o))

        # Standard LSTM state updates
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# ============================================================
# QLSTM Model (Top-Level)
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


class QLSTMModel(nn.Module):
    """
    Quantum LSTM model for time-series forecasting.
    
    Architecture:
      Embedding(input_dim → hidden_dim) + LayerNorm + Dropout
      → PositionalEncoding
      → QLSTMCell (single cell with 4 VQC gates, iterated over sequence)
      → Linear(hidden_dim → output_dim)
    
    Uses a single QLSTM cell following the reference implementation
    (Chen et al. 2022). The VQC depth (4 layers) provides the model's
    quantum depth, rather than stacking multiple cells. The num_layers
    parameter is accepted for interface compatibility but only 1 cell is used.
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=4,
                 seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)

        # Single QLSTM cell (faithful to reference)
        self.cell = QLSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embedding[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.embedding[0].bias, 0)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Initialize hidden states
        device = x.device
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Iterate over sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, hidden_dim)
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            # Residual + norm + dropout
            out_t = self.dropout(self.output_norm(h_t + x_t))
            outputs.append(out_t.unsqueeze(1))

        # Stack and project to output
        out = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return self.output_layer(out)  # (batch, seq_len, output_dim)


# ============================================================
# Sanity Check
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = 4 * N_VQC_LAYERS * N_QUBITS  # 1 cell × 4 gates × weight shape
    return total, trainable, quantum


if __name__ == "__main__":
    print("QLSTM Baseline Model")
    print("=" * 50)

    model = QLSTMModel(input_dim=1, output_dim=1, hidden_dim=64, num_layers=4, seq_len=20)
    total, trainable, quantum = count_parameters(model)
    print(f"Total parameters:    {total:,}")
    print(f"Trainable parameters:{trainable:,}")
    print(f"Quantum parameters:  {quantum} (1 cell × 4 VQCs × {N_VQC_LAYERS} layers × {N_QUBITS} qubits)")
    print(f"VQC circuit: {N_QUBITS} qubits, {N_VQC_LAYERS} layers")
    print(f"Device: lightning.qubit")

    print("\nRunning forward pass...")
    x = torch.randn(1, 20, 1)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 20, 1), f"Expected (1, 20, 1), got {y.shape}"
    print("✓ Forward pass successful!")
