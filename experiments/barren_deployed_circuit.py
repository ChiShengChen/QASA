#!/usr/bin/env python3
"""
Gradient variance of the EXACT deployed QASA PQC (Referee 1, Major comment re depth)
====================================================================================
Context. Table~\\ref{tab:barren} sweeps a *reference* circuit parameterised as
1/2/4 entangling layers (18/27/45 angles). The *deployed* QASA PQC, however, is
the circuit in run_baseline_comparison.quantum_circuit: weight tensor (N_QLAYERS,
N_QUBITS+1) = (4, 9) = 36 angles, with 3 entangling layers (the loop l=1..3).
Because of an off-by-one in how the two scripts count "layers", the deployed
36-angle circuit is NOT exactly any row of Table~\\ref{tab:barren} (it lies
between the 2-layer/27-param and 4-layer/45-param reference rows).

This script computes Var(dC/dtheta) for the *deployed* circuit itself, over the
same 200 random initialisations and the same cost (C = <Z_0>) as
barren_plateau_analysis.py, so the deployed PQC can be placed as an explicit
point on the barren-plateau plot and the "trainable vs reduced" question answered
directly rather than by analogy.

Usage:
  python experiments/barren_deployed_circuit.py
  python experiments/barren_deployed_circuit.py --samples 200 --cost mean
"""

import os
import sys
import csv
import argparse
import datetime
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Deployed configuration (must match run_baseline_comparison.py exactly)
N_QUBITS = 8
N_QLAYERS = 4
WEIGHT_SHAPE = (N_QLAYERS, N_QUBITS + 1)   # (4, 9) = 36 angles

# Reference values from Table~\ref{tab:barren} (for context in the printout)
REFERENCE_TABLE = {
    'QASA reference 1 entangling layer (18 params)': 2.62e-2,
    'QASA reference 2 entangling layers (27 params)': 8.87e-4,
    'QASA reference 4 entangling layers (45 params)': 8.34e-4,
}


def build_deployed_circuit(fixed_input=0.5, cost='z0'):
    """Return a QNode replicating the deployed PQC, with fixed data encoding.

    The deployed forward adds the (constant) sequence-length scalar t to the
    encoded angles; here we fold that into the fixed encoding value, since a
    constant shift of the *encoding* angles does not change the *variational*
    gradient statistics we are measuring. Cost:
      'z0'   -> C = <Z_0>            (matches barren_plateau_analysis.py)
      'mean' -> C = mean_i <Z_i>     (closer to the layer's vector output)
    """
    dev = qml.device("default.qubit", wires=N_QUBITS + 1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        # Data encoding (fixed, non-trainable) -- identical structure to deployed
        for i in range(N_QUBITS):
            qml.RX(fixed_input, wires=i)
            qml.RZ(fixed_input, wires=i)
        # First variational block (rows 0,1)
        for i in range(N_QUBITS):
            qml.RX(weights[0, i], wires=i)
            qml.RZ(weights[1, i], wires=i)
        # Entangling + variational blocks (l = 1,2,3 -> 3 entangling layers)
        for l in range(1, N_QLAYERS):
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
                qml.RY(weights[l, i], wires=i)
                qml.RZ(weights[l, i], wires=i)
            qml.CNOT(wires=[N_QUBITS - 1, N_QUBITS])
            qml.RY(weights[l, -1], wires=N_QUBITS)
        if cost == 'mean':
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_QUBITS)])) / N_QUBITS
        return qml.expval(qml.PauliZ(0))

    return circuit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--cost', choices=['z0', 'mean'], default='z0',
                    help="Cost function (z0 matches Table tab:barren)")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    circuit = build_deployed_circuit(cost=args.cost)
    grad_fn = qml.grad(circuit, argnum=0)
    n_params = int(np.prod(WEIGHT_SHAPE))

    print("=" * 72)
    print("BARREN-PLATEAU CHECK: EXACT DEPLOYED QASA PQC")
    print("=" * 72)
    print(f"Qubits: {N_QUBITS}+1 ancilla | weight shape {WEIGHT_SHAPE} = {n_params} angles")
    print(f"Entangling layers: 3 (loop l=1..3) | cost: C=<Z_0>"
          f"{' (mean over Z_i)' if args.cost=='mean' else ''}")
    print(f"Samples: {args.samples}\n")

    all_grads = []
    for s in range(args.samples):
        w = pnp.array(np.random.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE), requires_grad=True)
        g = grad_fn(w)
        if isinstance(g, tuple):
            g = g[0]
        all_grads.append(np.array(g).flatten())
        if (s + 1) % 50 == 0:
            print(f"  {s+1}/{args.samples}", flush=True)

    all_grads = np.array(all_grads)
    var_per_param = np.var(all_grads, axis=0)
    mean_var = float(np.mean(var_per_param))
    max_var = float(np.max(var_per_param))
    min_var = float(np.min(var_per_param))

    status = "Trainable" if mean_var > 1e-6 else ("Vanishing" if mean_var > 1e-10 else "Barren")

    print(f"\n{'-'*72}")
    print(f"DEPLOYED PQC  mean Var(dC/dtheta) = {mean_var:.6e}   [{status}]")
    print(f"              max = {max_var:.3e}   min = {min_var:.3e}")
    print(f"{'-'*72}")
    print("Context (Table tab:barren reference circuit, same 200-sample protocol):")
    for name, v in REFERENCE_TABLE.items():
        print(f"  {name:<48} {v:.2e}")
    print(f"\nInterpretation: the deployed 36-angle/3-entangling-layer PQC should be")
    print(f"reported as an explicit point; mean_var > 1e-6 means it remains trainable")
    print(f"(it is what we deploy and it trains), even though it sits below the")
    print(f"shallow 1-entangling-layer reference.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"barren_deployed_{timestamp}.csv")
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Circuit', 'Qubits', 'Angles', 'EntanglingLayers',
                    'Cost', 'Samples', 'Mean_Var', 'Max_Var', 'Min_Var', 'Status'])
        w.writerow(['QASA deployed (4 variational layers)', N_QUBITS + 1, n_params, 3,
                    args.cost, args.samples, f"{mean_var:.6e}", f"{max_var:.6e}",
                    f"{min_var:.6e}", status])
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
