#!/usr/bin/env python3
"""
Qubit Scaling Analysis
========================
How does gradient variance (barren plateau) scale with number of qubits?
Tests QASA's single-layer design at 4, 6, 8, 10, 12 qubits.

No training needed — just gradient computation over random parameters.
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

N_SAMPLES = 200
QUBIT_SIZES = [4, 6, 8, 10, 12]


def make_qasa_1l_circuit(n_qubits):
    """QASA single-layer circuit with n_qubits (n-1 data + 1 ancilla)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    n_data = n_qubits - 1

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        for i in range(n_data):
            qml.RX(0.5, wires=i)
            qml.RZ(0.5, wires=i)
        for i in range(n_data):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        return qml.expval(qml.PauliZ(0))

    weight_shape = (2, n_qubits)
    return circuit, weight_shape


def make_qasa_full_circuit(n_qubits, n_layers=4):
    """QASA full circuit (multiple entangling layers)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    n_data = n_qubits - 1
    total_rows = n_layers + 1

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        for i in range(n_data):
            qml.RX(0.5, wires=i)
            qml.RZ(0.5, wires=i)
        for i in range(n_data):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
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


def compute_gradient_variance(circuit_fn, weight_shape, n_samples=N_SAMPLES):
    """Compute mean gradient variance over random initializations."""
    grad_fn = qml.grad(circuit_fn, argnum=0)
    all_grads = []

    for s in range(n_samples):
        weights = pnp.array(np.random.uniform(-np.pi, np.pi, size=weight_shape), requires_grad=True)
        grads = grad_fn(weights)
        if isinstance(grads, tuple):
            grads = grads[0]
        all_grads.append(np.array(grads).flatten())

        if (s + 1) % 50 == 0:
            print(f"      {s+1}/{n_samples}", flush=True)

    all_grads = np.array(all_grads)
    var_per_param = np.var(all_grads, axis=0)
    return np.mean(var_per_param), np.std(var_per_param)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_1l = []
    results_4l = []

    print("=" * 60)
    print("QUBIT SCALING ANALYSIS")
    print(f"Qubit sizes: {QUBIT_SIZES}, Samples: {N_SAMPLES}")
    print("=" * 60)

    for nq in QUBIT_SIZES:
        print(f"\n  {nq} qubits:")

        # Single layer
        print(f"    QASA 1L...")
        circuit, wshape = make_qasa_1l_circuit(nq)
        mean_var, std_var = compute_gradient_variance(circuit, wshape)
        n_params = np.prod(wshape)
        print(f"      Params: {n_params}, Var: {mean_var:.6e} ± {std_var:.6e}")
        results_1l.append({'n_qubits': nq, 'n_params': n_params, 'mean_var': mean_var, 'std_var': std_var})

        # Full (4 entangling layers)
        print(f"    QASA 4L...")
        circuit, wshape = make_qasa_full_circuit(nq, n_layers=4)
        mean_var, std_var = compute_gradient_variance(circuit, wshape)
        n_params = np.prod(wshape)
        print(f"      Params: {n_params}, Var: {mean_var:.6e} ± {std_var:.6e}")
        results_4l.append({'n_qubits': nq, 'n_params': n_params, 'mean_var': mean_var, 'std_var': std_var})

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"qubit_scaling_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit', 'Qubits', 'Params', 'Mean_Var', 'Std_Var'])
        for r in results_1l:
            writer.writerow(['QASA_1L', r['n_qubits'], r['n_params'], f"{r['mean_var']:.6e}", f"{r['std_var']:.6e}"])
        for r in results_4l:
            writer.writerow(['QASA_4L', r['n_qubits'], r['n_params'], f"{r['mean_var']:.6e}", f"{r['std_var']:.6e}"])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    qubits_1l = [r['n_qubits'] for r in results_1l]
    vars_1l = [r['mean_var'] for r in results_1l]
    stds_1l = [r['std_var'] for r in results_1l]

    qubits_4l = [r['n_qubits'] for r in results_4l]
    vars_4l = [r['mean_var'] for r in results_4l]
    stds_4l = [r['std_var'] for r in results_4l]

    ax.errorbar(qubits_1l, vars_1l, yerr=stds_1l, fmt='o-', color='crimson',
                linewidth=2, markersize=8, capsize=5, label='QASA (1 layer) — our design')
    ax.errorbar(qubits_4l, vars_4l, yerr=stds_4l, fmt='s--', color='steelblue',
                linewidth=2, markersize=8, capsize=5, label='QASA (4 layers)')

    # Reference line for exponential decay
    ax.axhline(y=1e-4, color='gray', linestyle=':', alpha=0.5, label='Barren plateau threshold')

    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Mean Var(∂C/∂θ)', fontsize=12)
    ax.set_title('Gradient Variance Scaling with Qubit Count', fontsize=13)
    ax.set_yscale('log')
    ax.set_xticks(QUBIT_SIZES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"qubit_scaling_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Qubits':<8} {'QASA 1L Var':>15} {'QASA 4L Var':>15} {'Ratio':>10}")
    print("-" * 60)
    for r1, r4 in zip(results_1l, results_4l):
        ratio = r1['mean_var'] / r4['mean_var'] if r4['mean_var'] > 0 else float('inf')
        print(f"  {r1['n_qubits']:<6} {r1['mean_var']:>15.4e} {r4['mean_var']:>15.4e} {ratio:>10.1f}x")

    print(f"\nResults saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
