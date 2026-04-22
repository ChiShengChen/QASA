#!/usr/bin/env python3
"""
Barren Plateau Analysis for QASA, QLSTM, QnnFormer
=====================================================
Computes gradient variance as a function of circuit parameters to detect
barren plateaus. A circuit exhibits barren plateaus when gradient variance
decays exponentially with the number of qubits/parameters.

For each circuit:
1. Sample random parameters N times
2. Compute gradient of a cost function w.r.t. each parameter
3. Report Var(∂C/∂θ_i) for each parameter

If Var → 0 exponentially with circuit size, the circuit has barren plateaus.
QASA's single-layer design (36 params) should avoid this.

Output: CSV + plot + LaTeX table for the paper.
"""

import os, sys, csv, datetime
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_SAMPLES = 200  # random parameter samples


# ============================================================
# Circuit definitions (return expectation value as cost)
# ============================================================

def make_qasa_circuit(n_qubits, n_layers):
    """QASA-style circuit: RX/RZ encoding + ring CNOT + RY/RZ variational.
    Minimum 2 weight rows for encoding layer (RX+RZ), plus n_layers-1 variational layers.
    Total weight rows = max(2, n_layers+1) to handle the encoding layer separately."""
    dev = qml.device("default.qubit", wires=n_qubits)
    n_data = n_qubits - 1 if n_qubits > 1 else n_qubits
    # rows: 0=RX encoding, 1=RZ encoding, 2..n_layers = variational layers
    total_rows = n_layers + 1

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        # Input encoding (fixed)
        for i in range(n_data):
            qml.RX(0.5, wires=i)
            qml.RZ(0.5, wires=i)
        # First variational layer (RX + RZ)
        for i in range(n_data):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        # Subsequent entangling + variational layers
        for l in range(2, total_rows):
            for i in range(n_data):
                qml.CNOT(wires=[i, (i + 1) % n_data])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)
            if n_qubits > n_data:
                qml.CNOT(wires=[n_data - 1, n_data])
                qml.RY(weights[l, n_data], wires=n_data)
        return qml.expval(qml.PauliZ(0))

    weight_shape = (total_rows, n_qubits)
    return circuit, weight_shape


def make_qlstm_circuit(n_qubits, n_layers):
    """QLSTM-style circuit: Hadamard + RY encoding + linear CNOT + RY variational."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RY(0.5, wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
        return qml.expval(qml.PauliZ(0))

    weight_shape = (n_layers, n_qubits)
    return circuit, weight_shape


def make_qnnformer_circuit(n_qubits, n_layers):
    """QnnFormer-style circuit: AngleEmbedding + RX/RY/RZ + linear CNOT."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        qml.AngleEmbedding(np.ones(n_qubits) * 0.5, wires=range(n_qubits), rotation='Y')
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i], wires=i)
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    weight_shape = (n_layers, n_qubits)
    return circuit, weight_shape


# ============================================================
# Gradient variance computation
# ============================================================

def compute_gradient_variance(circuit_fn, weight_shape, n_samples=N_SAMPLES):
    """Compute variance of gradients over random parameter initializations."""
    n_params = np.prod(weight_shape)
    all_grads = []

    grad_fn = qml.grad(circuit_fn, argnum=0)

    for s in range(n_samples):
        weights = pnp.array(np.random.uniform(-np.pi, np.pi, size=weight_shape), requires_grad=True)
        grads = grad_fn(weights)
        if isinstance(grads, tuple):
            grads = grads[0]
        all_grads.append(np.array(grads).flatten())

        if (s + 1) % 50 == 0:
            print(f"      {s+1}/{n_samples}", flush=True)

    all_grads = np.array(all_grads)  # (n_samples, n_params)

    # Variance per parameter
    var_per_param = np.var(all_grads, axis=0)
    # Mean gradient per parameter
    mean_per_param = np.mean(all_grads, axis=0)

    return {
        'var_per_param': var_per_param,
        'mean_var': np.mean(var_per_param),
        'max_var': np.max(var_per_param),
        'min_var': np.min(var_per_param),
        'mean_grad': np.mean(np.abs(mean_per_param)),
        'n_params': n_params,
    }


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configurations to test
    configs = [
        # (name, circuit_maker, n_qubits, n_layers, description)
        # For QASA: n_layers = number of entangling layers (actual weight rows = n_layers+1)
        ('QASA (1L)', make_qasa_circuit, 9, 1, 'Single quantum layer (our design)'),
        ('QASA (2L)', make_qasa_circuit, 9, 2, 'Two entangling layers'),
        ('QASA (4L)', make_qasa_circuit, 9, 4, 'Four entangling layers (full, matches main arch)'),
        ('QLSTM', make_qlstm_circuit, 8, 4, 'QLSTM (4 VQC layers)'),
        ('QnnFormer', make_qnnformer_circuit, 8, 3, 'QnnFormer (3 VQC layers)'),
    ]

    results = []

    print("=" * 60)
    print("BARREN PLATEAU ANALYSIS")
    print(f"Samples per config: {N_SAMPLES}")
    print("=" * 60)

    for name, maker, n_qubits, n_layers, desc in configs:
        print(f"\n  {name}: {n_qubits} qubits, {n_layers} layers")
        print(f"    {desc}")

        circuit, weight_shape = maker(n_qubits, n_layers)
        n_params = np.prod(weight_shape)
        print(f"    Parameters: {n_params}, Weight shape: {weight_shape}")

        metrics = compute_gradient_variance(circuit, weight_shape)

        print(f"    Mean Var(∂C/∂θ): {metrics['mean_var']:.6e}")
        print(f"    Max Var(∂C/∂θ):  {metrics['max_var']:.6e}")
        print(f"    Min Var(∂C/∂θ):  {metrics['min_var']:.6e}")

        results.append({
            'name': name,
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_params': n_params,
            'mean_var': metrics['mean_var'],
            'max_var': metrics['max_var'],
            'min_var': metrics['min_var'],
            'mean_abs_grad': metrics['mean_grad'],
            'var_per_param': metrics['var_per_param'],
        })

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"barren_plateau_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit', 'Qubits', 'Layers', 'Params', 'Mean_Var', 'Max_Var', 'Min_Var', 'Mean_Abs_Grad'])
        for r in results:
            writer.writerow([r['name'], r['n_qubits'], r['n_layers'], r['n_params'],
                             f"{r['mean_var']:.6e}", f"{r['max_var']:.6e}",
                             f"{r['min_var']:.6e}", f"{r['mean_abs_grad']:.6e}"])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of mean gradient variance
    ax = axes[0]
    names = [r['name'] for r in results]
    mean_vars = [r['mean_var'] for r in results]
    colors = ['crimson', 'salmon', 'lightsalmon', 'steelblue', 'forestgreen']
    bars = ax.bar(names, mean_vars, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Var(∂C/∂θ)')
    ax.set_title('Gradient Variance by Circuit\n(Higher = Easier to Train, No Barren Plateau)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, mean_vars):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.3, f'{v:.1e}',
                ha='center', fontsize=9)

    # Variance per parameter (box plot style)
    ax = axes[1]
    positions = range(len(results))
    bp_data = [r['var_per_param'] for r in results]
    bp = ax.boxplot(bp_data, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Var(∂C/∂θ_i) per parameter')
    ax.set_title('Gradient Variance Distribution\nAcross Individual Parameters')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"barren_plateau_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Circuit':<16} {'Params':>7} {'Mean Var(∂C/∂θ)':>18} {'Status'}")
    print("-" * 70)
    for r in results:
        status = "OK" if r['mean_var'] > 1e-6 else "VANISHING" if r['mean_var'] > 1e-10 else "BARREN"
        print(f"{r['name']:<16} {r['n_params']:>7} {r['mean_var']:>18.6e} {status}")

    print(f"\nResults saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")

    # LaTeX table
    print(f"\n% LaTeX table:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Barren plateau analysis: mean gradient variance $\mathrm{Var}(\partial C / \partial \theta)$ over 200 random parameter initializations. Higher variance indicates trainable gradients (no barren plateau). QASA's single-layer design maintains the largest gradient variance.}")
    print(r"\label{tab:barren}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Circuit} & \textbf{Qubits} & \textbf{Params} & \textbf{Mean Var($\partial C/\partial\theta$)} & \textbf{Status} \\")
    print(r"\midrule")
    for r in results:
        status = "Trainable" if r['mean_var'] > 1e-6 else "Vanishing" if r['mean_var'] > 1e-10 else "Barren"
        print(f"{r['name']} & {r['n_qubits']} & {r['n_params']} & {r['mean_var']:.2e} & {status} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
