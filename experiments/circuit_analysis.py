#!/usr/bin/env python3
"""
QASA Circuit Analysis
=====================
Produces circuit specs, entanglement entropy, Meyer-Wallach measure,
expressibility estimate, and a publication-quality circuit diagram.

Outputs:
  - experiments/results/circuit_analysis.txt  (metrics table)
  - experiments/results/circuit_diagram.png   (figure for paper)
"""

import os
import sys
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Circuit definition (same as benchmark runner) ──────────────────────────

N_QUBITS = 8
N_QLAYERS = 4
N_WIRES = N_QUBITS + 1  # 8 data + 1 auxiliary

dev_expval = qml.device("default.qubit", wires=N_WIRES)
dev_state  = qml.device("default.qubit", wires=N_WIRES)


def qasa_ansatz(inputs, weights):
    """The QASA parameterized quantum circuit (ansatz only, no measurement)."""
    # Encoding
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    # First trainable rotation
    for i in range(N_QUBITS):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    # Variational layers with entanglement
    for l in range(1, N_QLAYERS):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[N_QUBITS - 1, N_QUBITS])
        qml.RY(weights[l, -1], wires=N_QUBITS)


@qml.qnode(dev_expval, interface="autograd")
def circuit_expval(inputs, weights):
    """Original QASA circuit returning expectation values."""
    qasa_ansatz(inputs, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev_state, interface="autograd")
def circuit_state(inputs, weights):
    """QASA circuit returning full state vector."""
    qasa_ansatz(inputs, weights)
    return qml.state()


@qml.qnode(dev_state, interface="autograd")
def circuit_vn_entropy(inputs, weights, wire_subset):
    """QASA circuit returning von Neumann entropy of a subsystem."""
    qasa_ansatz(inputs, weights)
    return qml.vn_entropy(wires=wire_subset)


@qml.qnode(dev_state, interface="autograd")
def circuit_purity(inputs, weights, wire_idx):
    """QASA circuit returning purity of a single qubit."""
    qasa_ansatz(inputs, weights)
    return qml.purity(wires=[wire_idx])


# ── 1. Circuit Specs ──────────────────────────────────────────────────────

def analyze_specs():
    """Use qml.specs to get gate counts, depth, etc."""
    dummy_inputs = pnp.zeros(N_QUBITS)
    dummy_weights = pnp.zeros((N_QLAYERS, N_QUBITS + 1))
    
    specs_fn = qml.specs(circuit_expval)
    specs = specs_fn(dummy_inputs, dummy_weights)
    
    print("=" * 60)
    print("1. CIRCUIT SPECIFICATIONS (qml.specs)")
    print("=" * 60)
    
    gate_types = specs.get("resources", specs).gate_types if hasattr(specs.get("resources", specs), "gate_types") else {}
    
    # Fallback: count manually from specs dict
    print(f"  Qubits (data):       {N_QUBITS}")
    print(f"  Qubits (total):      {N_WIRES}")
    print(f"  Circuit layers:      {N_QLAYERS}")
    print(f"  Quantum parameters:  {N_QLAYERS * (N_QUBITS + 1)}")
    
    # Print all available specs
    for key, val in specs.items():
        if key not in ("specs",):
            print(f"  {key}: {val}")
    
    return specs


# ── 2. Entanglement Entropy ──────────────────────────────────────────────

def analyze_entanglement(n_samples=200):
    """
    Compute average von Neumann entropy across random parameter samples.
    Reports entropy for the bipartition: qubits [0..3] | [4..7].
    """
    print("\n" + "=" * 60)
    print("2. ENTANGLEMENT ENTROPY (von Neumann)")
    print("=" * 60)
    
    wire_A = list(range(N_QUBITS // 2))  # qubits 0-3
    
    entropies_random = []
    entropies_zero_input = []
    
    for i in range(n_samples):
        weights = pnp.random.uniform(0, 2 * np.pi, (N_QLAYERS, N_QUBITS + 1))
        inputs_rand = pnp.random.uniform(-1, 1, N_QUBITS)
        inputs_zero = pnp.zeros(N_QUBITS)
        
        try:
            s_rand = float(circuit_vn_entropy(inputs_rand, weights, wire_A))
            entropies_random.append(s_rand)
            
            s_zero = float(circuit_vn_entropy(inputs_zero, weights, wire_A))
            entropies_zero_input.append(s_zero)
        except Exception as e:
            if i == 0:
                print(f"  Warning: vn_entropy failed ({e}), using state-based method")
            # Fallback: compute from state vector
            state = circuit_state(inputs_rand, weights)
            state_np = np.array(state)
            # Reshape to bipartite system
            n_A = len(wire_A)
            n_B = N_WIRES - n_A
            reshaped = state_np.reshape(2**n_A, 2**n_B)
            rho_A = reshaped @ reshaped.conj().T
            eigenvalues = np.linalg.eigvalsh(rho_A)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            s = -np.sum(eigenvalues * np.log2(eigenvalues))
            entropies_random.append(s)
        
        if (i + 1) % 50 == 0:
            print(f"  Sampled {i+1}/{n_samples}...")
    
    max_entropy = N_QUBITS // 2  # log2(2^4) = 4 bits for 4-qubit subsystem
    
    mean_rand = np.mean(entropies_random)
    std_rand = np.std(entropies_random)
    
    print(f"\n  Bipartition: qubits [0-3] | [4-7,aux]")
    print(f"  Max possible entropy: {max_entropy:.2f} bits")
    print(f"  Random params, random input:  {mean_rand:.4f} ± {std_rand:.4f}")
    if entropies_zero_input:
        mean_zero = np.mean(entropies_zero_input)
        std_zero = np.std(entropies_zero_input)
        print(f"  Random params, zero input:    {mean_zero:.4f} ± {std_zero:.4f}")
    print(f"  Normalized (random/max):      {mean_rand/max_entropy:.4f}")
    
    return {
        "mean_entropy_random": mean_rand,
        "std_entropy_random": std_rand,
        "max_entropy": max_entropy,
        "normalized": mean_rand / max_entropy,
    }


# ── 3. Meyer-Wallach Entangling Capability ───────────────────────────────

def analyze_meyer_wallach(n_samples=200):
    """
    Meyer-Wallach measure: Q = (2/n) * sum_j (1 - Tr(rho_j^2))
    where rho_j is the reduced density matrix of qubit j.
    Averaged over random parameter samples.
    """
    print("\n" + "=" * 60)
    print("3. MEYER-WALLACH ENTANGLING CAPABILITY")
    print("=" * 60)
    
    mw_values = []
    
    for i in range(n_samples):
        weights = pnp.random.uniform(0, 2 * np.pi, (N_QLAYERS, N_QUBITS + 1))
        inputs = pnp.random.uniform(-1, 1, N_QUBITS)
        
        purities = []
        for q in range(N_QUBITS):
            try:
                p = float(circuit_purity(inputs, weights, q))
            except Exception:
                # Fallback: compute from state
                state = circuit_state(inputs, weights)
                state_np = np.array(state)
                # Trace out all qubits except q
                full_dm = np.outer(state_np, state_np.conj())
                n_total = N_WIRES
                dim = 2 ** n_total
                # Partial trace keeping only qubit q
                rho_q = np.zeros((2, 2), dtype=complex)
                for a in range(2):
                    for b in range(2):
                        for other_state in range(dim // 2):
                            # Build indices
                            idx_a = (other_state >> (n_total - 1 - q)) << 1 | a
                            idx_a = (other_state & ((1 << (n_total - 1 - q)) - 1)) | (a << (n_total - 1 - q)) | ((other_state >> (n_total - 1 - q)) << (n_total - q))
                            # Simplified: use numpy partial trace
                            pass
                # Use simpler approach: reshape state
                state_reshaped = state_np.reshape([2] * n_total)
                axes_to_trace = list(range(n_total))
                axes_to_trace.remove(q)
                rho_q = np.tensordot(state_reshaped, state_reshaped.conj(), axes=(axes_to_trace, axes_to_trace))
                p = float(np.real(np.trace(rho_q @ rho_q)))
                purities.append(p)
                continue
            purities.append(p)
        
        # Q = (2/n) * sum(1 - purity_j)
        q_val = (2.0 / N_QUBITS) * sum(1 - p for p in purities)
        mw_values.append(q_val)
        
        if (i + 1) % 50 == 0:
            print(f"  Sampled {i+1}/{n_samples}...")
    
    mean_mw = np.mean(mw_values)
    std_mw = np.std(mw_values)
    
    print(f"\n  Meyer-Wallach Q (random params): {mean_mw:.4f} ± {std_mw:.4f}")
    print(f"  Range: [{np.min(mw_values):.4f}, {np.max(mw_values):.4f}]")
    print(f"  (Q=0: product state, Q=1: maximally entangled)")
    
    return {"mean_mw": mean_mw, "std_mw": std_mw}


# ── 4. Expressibility (KL-divergence vs Haar) ───────────────────────────

def analyze_expressibility(n_samples=500, n_bins=75):
    """
    Expressibility via KL divergence between PQC fidelity distribution
    and Haar-random distribution (Sim et al. 2019).
    """
    print("\n" + "=" * 60)
    print("4. EXPRESSIBILITY (Sim et al. 2019)")
    print("=" * 60)
    
    fidelities = []
    
    for i in range(n_samples):
        w1 = pnp.random.uniform(0, 2 * np.pi, (N_QLAYERS, N_QUBITS + 1))
        w2 = pnp.random.uniform(0, 2 * np.pi, (N_QLAYERS, N_QUBITS + 1))
        inp = pnp.random.uniform(-1, 1, N_QUBITS)
        
        state1 = np.array(circuit_state(inp, w1))
        state2 = np.array(circuit_state(inp, w2))
        
        fid = np.abs(np.vdot(state1, state2)) ** 2
        fidelities.append(fid)
        
        if (i + 1) % 100 == 0:
            print(f"  Sampled {i+1}/{n_samples} fidelity pairs...")
    
    fidelities = np.array(fidelities)
    
    # Haar-random fidelity distribution for N_WIRES qubits: P(F) = (2^n - 1)(1 - F)^(2^n - 2)
    dim = 2 ** N_WIRES
    
    # Histogram of sampled fidelities
    hist_pqc, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Haar distribution
    hist_haar = (dim - 1) * (1 - bin_centers) ** (dim - 2)
    
    # KL divergence: D_KL(PQC || Haar)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_pqc_normed = hist_pqc * bin_width + eps
    hist_haar_normed = hist_haar * bin_width + eps
    
    # Normalize to proper distributions
    hist_pqc_normed = hist_pqc_normed / hist_pqc_normed.sum()
    hist_haar_normed = hist_haar_normed / hist_haar_normed.sum()
    
    kl_div = np.sum(hist_pqc_normed * np.log(hist_pqc_normed / hist_haar_normed))
    
    print(f"\n  Hilbert space dimension: 2^{N_WIRES} = {dim}")
    print(f"  KL divergence (PQC || Haar): {kl_div:.6f}")
    print(f"  (Lower = more expressible, 0 = Haar-random)")
    print(f"  Mean fidelity: {np.mean(fidelities):.6f}")
    print(f"  Expected Haar mean fidelity: {1/dim:.6f}")
    
    return {"kl_divergence": kl_div, "mean_fidelity": np.mean(fidelities)}


# ── 5. Circuit Diagram ───────────────────────────────────────────────────

def draw_circuit():
    """Generate publication-quality circuit diagram."""
    print("\n" + "=" * 60)
    print("5. CIRCUIT DIAGRAM")
    print("=" * 60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        dummy_inputs = pnp.array([0.1 * i for i in range(N_QUBITS)])
        dummy_weights = pnp.random.uniform(0, 1, (N_QLAYERS, N_QUBITS + 1))
        
        fig, ax = qml.draw_mpl(circuit_expval, style="pennylane")(dummy_inputs, dummy_weights)
        fig.savefig(os.path.join(RESULTS_DIR, "circuit_diagram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved to {RESULTS_DIR}/circuit_diagram.png")
    except Exception as e:
        print(f"  Circuit diagram generation failed: {e}")
        # Fallback: text-based
        print(qml.draw(circuit_expval)(
            pnp.zeros(N_QUBITS),
            pnp.zeros((N_QLAYERS, N_QUBITS + 1))
        ))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("QASA Circuit Analysis")
    print("=" * 60)
    print(f"PennyLane version: {qml.__version__}")
    print(f"Configuration: {N_QUBITS} qubits + 1 aux, {N_QLAYERS} layers")
    print(f"Weight shape: ({N_QLAYERS}, {N_QUBITS + 1}) = {N_QLAYERS * (N_QUBITS + 1)} params")
    
    results = {}
    
    # 1. Specs
    specs = analyze_specs()
    
    # 2. Entanglement
    ent = analyze_entanglement(n_samples=200)
    results.update(ent)
    
    # 3. Meyer-Wallach
    mw = analyze_meyer_wallach(n_samples=200)
    results.update(mw)
    
    # 4. Expressibility
    expr = analyze_expressibility(n_samples=500)
    results.update(expr)
    
    # 5. Diagram
    draw_circuit()
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "circuit_analysis.txt")
    with open(summary_path, 'w') as f:
        f.write("QASA Circuit Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Qubits (data): {N_QUBITS}\n")
        f.write(f"Qubits (total): {N_WIRES}\n")
        f.write(f"Variational layers: {N_QLAYERS}\n")
        f.write(f"Quantum parameters: {N_QLAYERS * (N_QUBITS + 1)}\n")
        f.write(f"\nEntanglement Entropy (vN, bipartite [0-3]|[4-7+aux]):\n")
        f.write(f"  Mean: {results['mean_entropy_random']:.4f} ± {results['std_entropy_random']:.4f}\n")
        f.write(f"  Max possible: {results['max_entropy']:.2f}\n")
        f.write(f"  Normalized: {results['normalized']:.4f}\n")
        f.write(f"\nMeyer-Wallach Entangling Capability:\n")
        f.write(f"  Q: {results['mean_mw']:.4f} ± {results['std_mw']:.4f}\n")
        f.write(f"\nExpressibility (KL-div vs Haar):\n")
        f.write(f"  D_KL: {results['kl_divergence']:.6f}\n")
        f.write(f"  Mean fidelity: {results['mean_fidelity']:.6f}\n")
    
    print(f"\n\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
