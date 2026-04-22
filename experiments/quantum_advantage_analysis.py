#!/usr/bin/env python3
"""
Why Does Quantum Help on Chaotic Tasks?
=========================================
Analyzes the mechanism behind QASA's advantage on chaotic time series:

1. Attention pattern comparison (QASA vs Classical)
   - Do quantum-enhanced attention weights capture different temporal correlations?

2. Feature space analysis (PCA/t-SNE)
   - How does the quantum layer transform the representation space?
   - Compare feature distributions before vs after quantum layer

3. Mutual information analysis
   - Does the quantum layer increase information content about future states?

4. Lyapunov-like sensitivity analysis
   - Is the quantum model more sensitive to initial conditions (important for chaos)?

Output: Figures + analysis text for the paper.
"""

import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots", "quantum_advantage")
os.makedirs(PLOT_DIR, exist_ok=True)

CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints", "baseline_comparison")

# ============================================================
# Model definitions (must match training code)
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
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {'weights': (N_QLAYERS, N_QUBITS + 1)})
        self.input_proj = nn.Linear(input_dim, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)
        self.output_proj = nn.Linear(N_QUBITS, output_dim)

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
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout_rate),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        b, s, f = x.shape
        x_flat = x.reshape(b * s, f)
        q_out = self.v_quantum(x_flat, float(s))
        q_out = q_out.view(b, s, f)
        x = self.norm2(q_out + self.ffn(q_out))
        return x, attn_weights


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


class QASAModelAnalysis(nn.Module):
    """QASA model with hooks for extracting intermediate representations."""
    def __init__(self, hidden_dim=64, num_layers=4, seq_len=20, dropout_rate=0.1):
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
        encoder_layers.append(QuantumEncoderLayer(hidden_dim, dropout_rate))
        self.encoder = nn.ModuleList(encoder_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Storage for intermediate features
        self.features = {}
        self.attn_weights = {}

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        self.features['after_pos_enc'] = x.detach().clone()

        for i, layer in enumerate(self.encoder):
            if i < len(self.encoder) - 1:
                # Classical layers - hook attention
                x_residual = x
                # Manual forward to capture attention weights
                attn_out, attn_w = layer.self_attn(x, x, x, need_weights=True)
                self.attn_weights[f'layer_{i}'] = attn_w.detach().clone()
                x = layer.norm1(x + attn_out)
                x = layer.norm2(x + layer.linear2(layer.activation(layer.linear1(x))))
                self.features[f'after_layer_{i}'] = x.detach().clone()
            else:
                # Quantum layer
                self.features['before_quantum'] = x.detach().clone()
                x, attn_w = layer(x)
                self.attn_weights[f'layer_{i}_quantum'] = attn_w.detach().clone()
                self.features['after_quantum'] = x.detach().clone()

        return self.output_layer(x)


class ClassicalModelAnalysis(nn.Module):
    """Classical model with hooks for extracting intermediate representations."""
    def __init__(self, hidden_dim=64, num_layers=4, seq_len=20, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout_rate),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True,
            dropout=dropout_rate, dim_feedforward=4 * hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.features = {}
        self.attn_weights = {}

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        self.features['after_pos_enc'] = x.detach().clone()

        for i, layer in enumerate(self.encoder.layers):
            attn_out, attn_w = layer.self_attn(x, x, x, need_weights=True)
            self.attn_weights[f'layer_{i}'] = attn_w.detach().clone()
            x = layer.norm1(x + attn_out)
            x = layer.norm2(x + layer.linear2(layer.activation(layer.linear1(x))))
            self.features[f'after_layer_{i}'] = x.detach().clone()

        return self.output_layer(x)


# ============================================================
# Analysis functions
# ============================================================

def load_models(task_key='classical_chaotic_logistic', seed=42):
    """Load trained QASA and Classical models."""
    # QASA
    qasa = QASAModelAnalysis(hidden_dim=64, num_layers=4, seq_len=20)
    qasa_ckpt = torch.load(
        os.path.join(CKPT_DIR, 'qasa', task_key, f'seed{seed}', 'best_model.pth'),
        map_location='cpu', weights_only=False
    )
    qasa.load_state_dict(qasa_ckpt['model_state_dict'], strict=False)
    qasa.eval()

    # Classical
    classical = ClassicalModelAnalysis(hidden_dim=64, num_layers=4, seq_len=20)
    classical_ckpt = torch.load(
        os.path.join(CKPT_DIR, 'classical', task_key, f'seed{seed}', 'best_model.pth'),
        map_location='cpu', weights_only=False
    )
    classical.load_state_dict(classical_ckpt['model_state_dict'], strict=False)
    classical.eval()

    return qasa, classical


def analyze_attention_patterns(qasa, classical, X_test, task_name):
    """Compare attention patterns between QASA and Classical models."""
    print("  Analyzing attention patterns...")

    with torch.no_grad():
        _ = qasa(X_test)
        _ = classical(X_test)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Attention Patterns: {task_name}', fontsize=14)

    # Classical attention (4 layers)
    for i in range(4):
        ax = axes[0, i]
        key = f'layer_{i}'
        if key in classical.attn_weights:
            attn = classical.attn_weights[key][0].numpy()  # first batch
            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_title(f'Classical Layer {i}')
            if i == 0:
                ax.set_ylabel('Query position')
            ax.set_xlabel('Key position')
            plt.colorbar(im, ax=ax, fraction=0.046)

    # QASA attention (3 classical + 1 quantum)
    for i in range(3):
        ax = axes[1, i]
        key = f'layer_{i}'
        if key in qasa.attn_weights:
            attn = qasa.attn_weights[key][0].numpy()
            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_title(f'QASA Layer {i} (classical)')
            if i == 0:
                ax.set_ylabel('Query position')
            ax.set_xlabel('Key position')
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Quantum layer attention
    ax = axes[1, 3]
    q_key = 'layer_3_quantum'
    if q_key in qasa.attn_weights:
        attn = qasa.attn_weights[q_key][0].numpy()
        im = ax.imshow(attn, cmap='magma', aspect='auto')
        ax.set_title('QASA Layer 3 (QUANTUM)', fontweight='bold')
        ax.set_xlabel('Key position')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'attention_patterns.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'attention_patterns.pdf'), bbox_inches='tight')
    plt.close()
    print("    Saved attention_patterns.png/pdf")


def analyze_feature_space(qasa, classical, X_test, task_name):
    """Compare feature representations using PCA and t-SNE."""
    print("  Analyzing feature space...")

    with torch.no_grad():
        _ = qasa(X_test)
        _ = classical(X_test)

    # Get features before and after quantum layer
    before_q = qasa.features['before_quantum'][0].numpy()  # (seq_len, hidden_dim)
    after_q = qasa.features['after_quantum'][0].numpy()

    # Get classical final layer features
    classical_final = classical.features['after_layer_3'][0].numpy()

    # Also get QASA's layer 2 output (before quantum, after 3 classical layers)
    qasa_layer2 = qasa.features.get('after_layer_2', qasa.features['before_quantum'])[0].numpy()

    # PCA analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Feature Space Analysis: {task_name}', fontsize=14)

    time_steps = np.arange(before_q.shape[0])
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(time_steps)))

    # PCA of before quantum
    pca = PCA(n_components=2)
    before_pca = pca.fit_transform(before_q)
    ax = axes[0, 0]
    scatter = ax.scatter(before_pca[:, 0], before_pca[:, 1], c=time_steps, cmap='coolwarm', s=50, edgecolors='k', linewidth=0.5)
    ax.set_title('QASA: Before Quantum Layer')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax, label='Time step')
    # Connect consecutive points
    for i in range(len(before_pca) - 1):
        ax.plot([before_pca[i, 0], before_pca[i+1, 0]], [before_pca[i, 1], before_pca[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    # PCA of after quantum
    after_pca = pca.fit_transform(after_q)
    ax = axes[0, 1]
    scatter = ax.scatter(after_pca[:, 0], after_pca[:, 1], c=time_steps, cmap='coolwarm', s=50, edgecolors='k', linewidth=0.5)
    ax.set_title('QASA: After Quantum Layer', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax, label='Time step')
    for i in range(len(after_pca) - 1):
        ax.plot([after_pca[i, 0], after_pca[i+1, 0]], [after_pca[i, 1], after_pca[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    # PCA of classical final
    classical_pca = pca.fit_transform(classical_final)
    ax = axes[0, 2]
    scatter = ax.scatter(classical_pca[:, 0], classical_pca[:, 1], c=time_steps, cmap='coolwarm', s=50, edgecolors='k', linewidth=0.5)
    ax.set_title('Classical: Final Layer')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax, label='Time step')
    for i in range(len(classical_pca) - 1):
        ax.plot([classical_pca[i, 0], classical_pca[i+1, 0]], [classical_pca[i, 1], classical_pca[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    # Variance explained comparison
    ax = axes[1, 0]
    n_components = min(20, before_q.shape[1])
    pca_full = PCA(n_components=n_components)

    pca_full.fit(before_q)
    var_before = np.cumsum(pca_full.explained_variance_ratio_)
    pca_full.fit(after_q)
    var_after = np.cumsum(pca_full.explained_variance_ratio_)
    pca_full.fit(classical_final)
    var_classical = np.cumsum(pca_full.explained_variance_ratio_)

    ax.plot(range(1, n_components + 1), var_before, 'b--o', label='QASA before Q', markersize=4)
    ax.plot(range(1, n_components + 1), var_after, 'r-s', label='QASA after Q', markersize=4)
    ax.plot(range(1, n_components + 1), var_classical, 'g-.^', label='Classical final', markersize=4)
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('Effective Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cosine similarity matrices
    ax = axes[1, 1]
    cos_sim_quantum = cosine_similarity(after_q)
    im = ax.imshow(cos_sim_quantum, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('QASA: Pairwise Cosine Similarity\n(After Quantum)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Time step')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    cos_sim_classical = cosine_similarity(classical_final)
    im = ax.imshow(cos_sim_classical, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('Classical: Pairwise Cosine Similarity\n(Final Layer)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Time step')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'feature_space.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'feature_space.pdf'), bbox_inches='tight')
    plt.close()
    print("    Saved feature_space.png/pdf")

    return before_q, after_q, classical_final


def analyze_quantum_transformation(before_q, after_q, classical_final):
    """Quantify what the quantum layer does to the features."""
    print("  Analyzing quantum transformation...")

    # 1. Feature rank (effective dimensionality)
    def effective_rank(X):
        """Compute effective rank via singular values."""
        s = np.linalg.svd(X, compute_uv=False)
        s_norm = s / s.sum()
        s_norm = s_norm[s_norm > 1e-10]
        return np.exp(-np.sum(s_norm * np.log(s_norm)))

    rank_before = effective_rank(before_q)
    rank_after = effective_rank(after_q)
    rank_classical = effective_rank(classical_final)

    print(f"    Effective rank - Before quantum: {rank_before:.2f}")
    print(f"    Effective rank - After quantum: {rank_after:.2f}")
    print(f"    Effective rank - Classical final: {rank_classical:.2f}")

    # 2. Feature norm change
    norm_before = np.linalg.norm(before_q, axis=1).mean()
    norm_after = np.linalg.norm(after_q, axis=1).mean()
    norm_classical = np.linalg.norm(classical_final, axis=1).mean()

    print(f"    Mean feature norm - Before quantum: {norm_before:.4f}")
    print(f"    Mean feature norm - After quantum: {norm_after:.4f}")
    print(f"    Mean feature norm - Classical final: {norm_classical:.4f}")

    # 3. Temporal autocorrelation structure
    def temporal_autocorr(X, max_lag=10):
        """Compute average autocorrelation across feature dimensions."""
        n = X.shape[0]
        autocorrs = []
        for lag in range(1, min(max_lag + 1, n)):
            corr = np.corrcoef(X[:-lag].flatten(), X[lag:].flatten())[0, 1]
            autocorrs.append(corr)
        return autocorrs

    autocorr_before = temporal_autocorr(before_q)
    autocorr_after = temporal_autocorr(after_q)
    autocorr_classical = temporal_autocorr(classical_final)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Autocorrelation plot
    ax = axes[0]
    lags = range(1, len(autocorr_before) + 1)
    ax.plot(lags, autocorr_before, 'b--o', label='QASA before Q', markersize=4)
    ax.plot(lags, autocorr_after, 'r-s', label='QASA after Q', markersize=4)
    ax.plot(lags, autocorr_classical, 'g-.^', label='Classical final', markersize=4)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Temporal Autocorrelation of Features')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective rank bar chart
    ax = axes[1]
    models = ['Before\nQuantum', 'After\nQuantum', 'Classical\nFinal']
    ranks = [rank_before, rank_after, rank_classical]
    bars = ax.bar(models, ranks, color=['steelblue', 'crimson', 'forestgreen'])
    ax.set_ylabel('Effective Rank')
    ax.set_title('Feature Space Dimensionality')
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rank:.1f}', ha='center', fontsize=11)

    # Feature difference analysis
    ax = axes[2]
    quantum_delta = after_q - before_q  # What the quantum layer adds
    delta_norms = np.linalg.norm(quantum_delta, axis=1)
    ax.plot(delta_norms, 'r-', linewidth=2, label='||quantum_output - input||')
    ax.set_xlabel('Time step')
    ax.set_ylabel('L2 norm of quantum correction')
    ax.set_title('Quantum Layer Correction Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'quantum_transformation.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'quantum_transformation.pdf'), bbox_inches='tight')
    plt.close()
    print("    Saved quantum_transformation.png/pdf")

    return {
        'effective_rank_before': rank_before,
        'effective_rank_after': rank_after,
        'effective_rank_classical': rank_classical,
        'norm_before': norm_before,
        'norm_after': norm_after,
        'norm_classical': norm_classical,
    }


def compare_tasks(tasks, seed=42):
    """Compare quantum layer behavior across chaotic vs non-chaotic tasks."""
    print("\n  Comparing across tasks...")

    results = {}
    for task_key, task_name in tasks:
        print(f"\n    Task: {task_name}")
        task = get_task(task_key)
        X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()

        qasa, classical = load_models(task_key, seed)

        with torch.no_grad():
            _ = qasa(X_test_seed)
            _ = classical(X_test_seed)

        before_q = qasa.features['before_quantum'][0].numpy()
        after_q = qasa.features['after_quantum'][0].numpy()
        classical_final = classical.features['after_layer_3'][0].numpy()

        # Effective rank
        def effective_rank(X):
            s = np.linalg.svd(X, compute_uv=False)
            s_norm = s / s.sum()
            s_norm = s_norm[s_norm > 1e-10]
            return np.exp(-np.sum(s_norm * np.log(s_norm)))

        # Quantum correction magnitude
        delta = after_q - before_q
        correction_mag = np.linalg.norm(delta, axis=1).mean()

        results[task_name] = {
            'rank_before': effective_rank(before_q),
            'rank_after': effective_rank(after_q),
            'rank_classical': effective_rank(classical_final),
            'rank_change': effective_rank(after_q) - effective_rank(before_q),
            'correction_magnitude': correction_mag,
        }

        print(f"      Rank: {results[task_name]['rank_before']:.1f} → {results[task_name]['rank_after']:.1f} "
              f"(Δ={results[task_name]['rank_change']:+.1f}), Classical: {results[task_name]['rank_classical']:.1f}")
        print(f"      Quantum correction: {correction_mag:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    task_names = list(results.keys())
    x = np.arange(len(task_names))
    width = 0.25

    # Rank comparison
    ax = axes[0]
    ax.bar(x - width, [results[t]['rank_before'] for t in task_names], width, label='Before Q', color='steelblue')
    ax.bar(x, [results[t]['rank_after'] for t in task_names], width, label='After Q', color='crimson')
    ax.bar(x + width, [results[t]['rank_classical'] for t in task_names], width, label='Classical', color='forestgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Dimensionality by Task')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Correction magnitude
    ax = axes[1]
    corrections = [results[t]['correction_magnitude'] for t in task_names]
    bars = ax.bar(task_names, corrections, color=['crimson' if t == 'Chaotic Logistic' else 'steelblue' for t in task_names])
    ax.set_ylabel('Mean ||quantum correction||')
    ax.set_title('Quantum Layer Correction Magnitude by Task')
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'cross_task_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'cross_task_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("    Saved cross_task_comparison.png/pdf")

    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("WHY DOES QUANTUM HELP ON CHAOTIC TASKS?")
    print("=" * 60)

    # Load chaotic logistic data
    task = get_task('classical_chaotic_logistic')
    X_train, Y_train, X_test_seed, Y_test_true = task.generate_data()

    # Load models
    print("\nLoading models...")
    qasa, classical = load_models('classical_chaotic_logistic', seed=42)

    # 1. Attention patterns
    print("\n1. Attention Pattern Analysis")
    analyze_attention_patterns(qasa, classical, X_test_seed, 'Chaotic Logistic')

    # 2. Feature space
    print("\n2. Feature Space Analysis")
    before_q, after_q, classical_final = analyze_feature_space(
        qasa, classical, X_test_seed, 'Chaotic Logistic'
    )

    # 3. Quantum transformation
    print("\n3. Quantum Transformation Analysis")
    metrics = analyze_quantum_transformation(before_q, after_q, classical_final)

    # 4. Cross-task comparison
    print("\n4. Cross-Task Comparison")
    tasks = [
        ('classical_chaotic_logistic', 'Chaotic Logistic'),
        ('classical_damped_oscillation', 'Damped Oscillator'),
        ('classical_square_triangle_wave', 'Square Wave'),
    ]
    cross_results = compare_tasks(tasks)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nQuantum layer transformation on Chaotic Logistic:")
    print(f"  Effective rank: {metrics['effective_rank_before']:.1f} → {metrics['effective_rank_after']:.1f}")
    print(f"  Feature norm: {metrics['norm_before']:.4f} → {metrics['norm_after']:.4f}")
    print(f"\nCross-task quantum correction magnitudes:")
    for task_name, r in cross_results.items():
        print(f"  {task_name}: {r['correction_magnitude']:.4f} (rank Δ={r['rank_change']:+.1f})")

    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
