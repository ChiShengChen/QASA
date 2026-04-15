#!/usr/bin/env python3
"""
Quantum Circuit Expressibility & Entangling Capability Analysis
================================================================
Computes quantitative metrics for all three quantum circuits (QASA, QLSTM, QnnFormer):

1. Expressibility (Sim et al., 2019):
   - KL divergence between circuit fidelity distribution and Haar-random distribution
   - Lower = more expressive (closer to uniform coverage of Hilbert space)

2. Entangling Capability (Meyer-Wallach measure):
   - Average entanglement across random parameter samples
   - Higher = more entangling power

3. Quantum Resource Counting:
   - Qubits, circuit depth, CNOT count, single-qubit gates, trainable parameters

Output: CSV + LaTeX table for the paper.
"""

import os, sys
import numpy as np
import pennylane as qml
from scipy.special import rel_entr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_QUBITS = 8
N_QLAYERS = 4
N_SAMPLES = 5000  # number of random parameter pairs for expressibility
N_BINS = 75       # histogram bins for fidelity distribution


# ============================================================
# Circuit definitions (statevector for fidelity computation)
# ============================================================

# --- QASA circuit ---
qasa_dev = qml.device("default.qubit", wires=N_QUBITS + 1)

@qml.qnode(qasa_dev)
def qasa_circuit(inputs, weights):
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
    return qml.state()


# --- QLSTM circuit ---
qlstm_dev = qml.device("default.qubit", wires=N_QUBITS)
QLSTM_LAYERS = 4

@qml.qnode(qlstm_dev)
def qlstm_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for layer in range(QLSTM_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)
    return qml.state()


# --- QnnFormer circuit ---
qnnformer_dev = qml.device("default.qubit", wires=N_QUBITS)
QNNFORMER_LAYERS = 3

@qml.qnode(qnnformer_dev)
def qnnformer_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for layer in range(QNNFORMER_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i], wires=i)
            qml.RY(weights[layer, i], wires=i)
            qml.RZ(weights[layer, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.state()


# ============================================================
# Circuit metadata
# ============================================================
CIRCUITS = {
    'QASA': {
        'circuit': qasa_circuit,
        'n_qubits': N_QUBITS + 1,  # 8 data + 1 ancilla
        'input_shape': (N_QUBITS,),
        'weight_shape': (N_QLAYERS, N_QUBITS + 1),
        'n_quantum_params': N_QLAYERS * (N_QUBITS + 1),
        'description': 'RX/RZ encoding + ring CNOT + RY/RZ variational',
    },
    'QLSTM': {
        'circuit': qlstm_circuit,
        'n_qubits': N_QUBITS,
        'input_shape': (N_QUBITS,),
        'weight_shape': (QLSTM_LAYERS, N_QUBITS),
        'n_quantum_params': QLSTM_LAYERS * N_QUBITS,
        'description': 'Hadamard + RY encoding + linear CNOT + RY variational',
    },
    'QnnFormer': {
        'circuit': qnnformer_circuit,
        'n_qubits': N_QUBITS,
        'input_shape': (N_QUBITS,),
        'weight_shape': (QNNFORMER_LAYERS, N_QUBITS),
        'n_quantum_params': QNNFORMER_LAYERS * N_QUBITS,
        'description': 'AngleEmbedding + RX/RY/RZ + linear CNOT',
    },
}


# ============================================================
# Expressibility (Sim et al., 2019)
# ============================================================
def haar_probability(f, n_qubits, n_bins):
    """Analytical Haar-random fidelity distribution for n-qubit systems."""
    dim = 2 ** n_qubits
    # P(F) = (dim - 1) * (1 - F)^(dim - 2)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    probs = np.zeros(n_bins)
    for i in range(n_bins):
        f_low, f_high = bin_edges[i], bin_edges[i + 1]
        # Integrate: (dim-1) * (1-f)^(dim-2) from f_low to f_high
        # = [-(1-f)^(dim-1)] from f_low to f_high
        probs[i] = (1 - f_low) ** (dim - 1) - (1 - f_high) ** (dim - 1)
    return probs / (probs.sum() + 1e-12)


def compute_expressibility(circuit_fn, input_shape, weight_shape, n_qubits,
                           n_samples=N_SAMPLES, n_bins=N_BINS):
    """Compute expressibility as KL divergence from Haar distribution."""
    print(f"    Computing fidelities ({n_samples} pairs)...", flush=True)
    fidelities = []

    for i in range(n_samples):
        # Random inputs and two random weight sets
        inp = np.random.uniform(-np.pi, np.pi, size=input_shape)
        w1 = np.random.uniform(-np.pi, np.pi, size=weight_shape)
        w2 = np.random.uniform(-np.pi, np.pi, size=weight_shape)

        state1 = circuit_fn(inp, w1)
        state2 = circuit_fn(inp, w2)

        # Fidelity = |<psi1|psi2>|^2
        fid = np.abs(np.dot(np.conj(state1), state2)) ** 2
        fidelities.append(float(np.real(fid)))

        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{n_samples}", flush=True)

    fidelities = np.array(fidelities)

    # Histogram
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=False)
    pqc_dist = hist / (hist.sum() + 1e-12)

    # Haar distribution
    haar_dist = haar_probability(None, n_qubits, n_bins)

    # KL divergence: sum p * log(p/q), with smoothing
    eps = 1e-10
    pqc_smooth = pqc_dist + eps
    haar_smooth = haar_dist + eps
    pqc_smooth /= pqc_smooth.sum()
    haar_smooth /= haar_smooth.sum()

    kl_div = np.sum(rel_entr(pqc_smooth, haar_smooth))

    return kl_div, fidelities, pqc_smooth, haar_smooth


# ============================================================
# Entangling Capability (Meyer-Wallach measure)
# ============================================================
def meyer_wallach_measure(statevector, n_qubits):
    """
    Compute the Meyer-Wallach entanglement measure Q.
    Q = (2/n) * sum_k (1 - tr(rho_k^2))
    where rho_k is the reduced density matrix of qubit k.
    """
    dim = 2 ** n_qubits
    state = np.array(statevector).reshape(-1)

    if len(state) != dim:
        # Truncate or pad if ancilla qubits present
        if len(state) > dim:
            # Trace out ancilla (last qubit)
            full_dim = len(state)
            n_total = int(np.log2(full_dim))
            state_reshaped = state.reshape([2] * n_total)
            # Trace out last qubit
            rho_data = np.tensordot(state_reshaped, np.conj(state_reshaped),
                                     axes=([n_total - 1], [n_total - 1]))
            # Flatten back to density matrix
            rho_data = rho_data.reshape(dim, dim)
        else:
            return 0.0
    else:
        rho_data = np.outer(state, np.conj(state))

    Q = 0.0
    for k in range(n_qubits):
        # Partial trace to get reduced density matrix of qubit k
        rho_k = partial_trace_single(rho_data, k, n_qubits)
        purity = np.real(np.trace(rho_k @ rho_k))
        Q += 1 - purity

    Q *= 2.0 / n_qubits
    return float(Q)


def partial_trace_single(rho, keep_qubit, n_qubits):
    """Get reduced density matrix of a single qubit by tracing out all others."""
    dim = 2 ** n_qubits
    rho_reshaped = rho.reshape([2] * n_qubits + [2] * n_qubits)

    # Trace out all qubits except keep_qubit
    trace_axes = list(range(n_qubits))
    trace_axes.remove(keep_qubit)

    # Sort in reverse to avoid index shifting
    for ax in sorted(trace_axes, reverse=True):
        rho_reshaped = np.trace(rho_reshaped, axis1=ax, axis2=ax + len(rho_reshaped.shape) // 2)
        # After each trace, dimensions reduce

    # Simpler approach: direct computation
    rho_k = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for basis in range(dim // 2):
                # Construct full basis indices
                idx_i = _insert_bit(basis, keep_qubit, i, n_qubits)
                idx_j = _insert_bit(basis, keep_qubit, j, n_qubits)
                rho_k[i, j] += rho[idx_i, idx_j]

    return rho_k


def _insert_bit(number, position, bit_value, n_qubits):
    """Insert a bit at given position in a binary number."""
    mask_high = (number >> (n_qubits - 1 - position)) << (n_qubits - position)
    mask_low = number & ((1 << (n_qubits - 1 - position)) - 1)
    return mask_high | (bit_value << (n_qubits - 1 - position)) | mask_low


def compute_entangling_capability(circuit_fn, input_shape, weight_shape, n_qubits,
                                   n_samples=1000):
    """Average Meyer-Wallach entanglement over random parameters."""
    print(f"    Computing entanglement ({n_samples} samples)...", flush=True)
    measures = []

    for i in range(n_samples):
        inp = np.random.uniform(-np.pi, np.pi, size=input_shape)
        w = np.random.uniform(-np.pi, np.pi, size=weight_shape)

        state = circuit_fn(inp, w)
        Q = meyer_wallach_measure(state, n_qubits)
        measures.append(Q)

        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{n_samples}", flush=True)

    return np.mean(measures), np.std(measures), measures


# ============================================================
# Gate counting
# ============================================================
def count_gates(circuit_fn, input_shape, weight_shape):
    """Count gates using qml.specs."""
    inp = np.random.uniform(-np.pi, np.pi, size=input_shape)
    w = np.random.uniform(-np.pi, np.pi, size=weight_shape)

    specs = qml.specs(circuit_fn)(inp, w)
    resources = specs['resources']

    # Parse gate_types from resources string representation
    gate_types = {}
    gate_sizes = {}
    if hasattr(resources, 'gate_types'):
        gate_types = dict(resources.gate_types)
    if hasattr(resources, 'gate_sizes'):
        gate_sizes = dict(resources.gate_sizes)

    single_qubit = gate_sizes.get(1, 0)
    two_qubit = gate_sizes.get(2, 0)
    three_qubit = sum(v for k, v in gate_sizes.items() if k >= 3)
    depth = resources.depth if hasattr(resources, 'depth') else 'N/A'

    return {
        'single_qubit_gates': single_qubit,
        'two_qubit_gates': two_qubit,
        'three_qubit_gates': three_qubit,
        'total_gates': resources.num_gates if hasattr(resources, 'num_gates') else single_qubit + two_qubit + three_qubit,
        'gate_breakdown': gate_types,
        'depth': depth,
    }


# ============================================================
# Main
# ============================================================
def main():
    import csv, datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for name, info in CIRCUITS.items():
        print(f"\n{'='*60}")
        print(f"Circuit: {name}")
        print(f"  Qubits: {info['n_qubits']}, Quantum params: {info['n_quantum_params']}")
        print(f"{'='*60}")

        # Gate counting
        print("  Counting gates...")
        gates = count_gates(info['circuit'], info['input_shape'], info['weight_shape'])
        print(f"    Single-qubit: {gates['single_qubit_gates']}")
        print(f"    Two-qubit (CNOT): {gates['two_qubit_gates']}")
        print(f"    Three+-qubit: {gates['three_qubit_gates']}")
        print(f"    Total: {gates['total_gates']}")
        print(f"    Breakdown: {gates['gate_breakdown']}")

        # Expressibility
        print("  Computing expressibility...")
        # Use fewer qubits for Haar reference (data qubits only)
        data_qubits = min(info['n_qubits'], N_QUBITS)
        kl_div, fidelities, pqc_dist, haar_dist = compute_expressibility(
            info['circuit'], info['input_shape'], info['weight_shape'],
            data_qubits, n_samples=N_SAMPLES
        )
        print(f"    Expressibility (KL div): {kl_div:.6f}")
        print(f"    Mean fidelity: {np.mean(fidelities):.6f}")

        # Entangling capability
        print("  Computing entangling capability...")
        ent_mean, ent_std, ent_samples = compute_entangling_capability(
            info['circuit'], info['input_shape'], info['weight_shape'],
            data_qubits, n_samples=1000
        )
        print(f"    Meyer-Wallach Q: {ent_mean:.6f} ± {ent_std:.6f}")

        results[name] = {
            'n_qubits': info['n_qubits'],
            'n_quantum_params': info['n_quantum_params'],
            'single_qubit_gates': gates['single_qubit_gates'],
            'two_qubit_gates': gates['two_qubit_gates'],
            'total_gates': gates['total_gates'],
            'expressibility_kl': kl_div,
            'mean_fidelity': np.mean(fidelities),
            'entangling_capability': ent_mean,
            'entangling_std': ent_std,
        }

    # ============================================================
    # Print summary table
    # ============================================================
    print(f"\n\n{'='*90}")
    print("CIRCUIT ANALYSIS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Metric':<30} {'QASA':>15} {'QLSTM':>15} {'QnnFormer':>15}")
    print("-" * 90)

    metrics = [
        ('Qubits', 'n_qubits'),
        ('Quantum Parameters', 'n_quantum_params'),
        ('Single-qubit Gates', 'single_qubit_gates'),
        ('Two-qubit Gates (CNOT)', 'two_qubit_gates'),
        ('Total Gates', 'total_gates'),
        ('Expressibility (KL↓)', 'expressibility_kl'),
        ('Entangling Capability (Q↑)', 'entangling_capability'),
    ]

    for label, key in metrics:
        vals = []
        for name in ['QASA', 'QLSTM', 'QnnFormer']:
            v = results[name][key]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        print(f"{label:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print("-" * 90)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"circuit_expressibility_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit', 'Qubits', 'Q_Params', '1Q_Gates', '2Q_Gates',
                         'Total_Gates', 'Expressibility_KL', 'Entangling_Q', 'Entangling_Std'])
        for name in ['QASA', 'QLSTM', 'QnnFormer']:
            r = results[name]
            writer.writerow([name, r['n_qubits'], r['n_quantum_params'],
                             r['single_qubit_gates'], r['two_qubit_gates'],
                             r['total_gates'], f"{r['expressibility_kl']:.6f}",
                             f"{r['entangling_capability']:.6f}", f"{r['entangling_std']:.6f}"])

    print(f"\nResults saved to: {csv_path}")

    # Print LaTeX table
    print(f"\n\n% LaTeX table for paper:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Quantum circuit analysis: expressibility (KL divergence from Haar-random, lower is better) and entangling capability (Meyer-Wallach measure, higher is better) across all quantum models.}")
    print(r"\label{tab:circuit_analysis}")
    print(r"\resizebox{\columnwidth}{!}{")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"\textbf{Circuit} & \textbf{Qubits} & \textbf{Q-Params} & \textbf{CNOT Gates} & \textbf{Total Gates} & \textbf{Expr. (KL$\downarrow$)} & \textbf{Ent. Cap. ($Q\uparrow$)} \\")
    print(r"\midrule")
    for name in ['QASA', 'QLSTM', 'QnnFormer']:
        r = results[name]
        print(f"{name} & {r['n_qubits']} & {r['n_quantum_params']} & {r['two_qubit_gates']} & "
              f"{r['total_gates']} & {r['expressibility_kl']:.4f} & "
              f"{r['entangling_capability']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
