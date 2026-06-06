#!/usr/bin/env python3
"""
Physical Quantum Hardware Validation on IBM Quantum (Referee 1 M3 / Referee 2 q3)
================================================================================
Both referees ask for at least one key benchmark executed on a real QPU to
verify resilience to real-world cross-talk and gate errors.

Strategy (NO retraining): we load an already-trained QASA checkpoint
(chaotic logistic, 8 data qubits + 1 ancilla, 4 layers) and run *inference*
on hardware. The classical front-end (embedding + 3 Transformer layers +
attention + input projection) is run in PyTorch to produce, for each token, the
8 rotation angles fed to the PQC. These angle vectors are evaluated on the IBM
backend (one batched Sampler job), the resulting Pauli-Z expectations are pushed
through the trained classical back-end (output projection + FFN + head), and we
compare predictions/expectations against the noiseless simulator.

The circuit is identical to the trained PennyLane circuit
(quantum_circuit in run_baseline_comparison.py): RX/RZ angle encoding, an
RX/RZ variational layer, then CNOT-ring + RY/RZ layers with an ancilla CNOT.

Execution modes:
  --mode pennylane   reference expectation values (lightning.qubit), local
  --mode aer         noiseless Aer simulator via the Sampler pipeline (validates
                     the qiskit circuit matches PennyLane), local, no credentials
  --mode fake        Aer + a real IBM device NOISE MODEL (FakeBackend) -- a
                     hardware-realistic emulation, offline, no queue
  --mode hardware    real IBM QPU via QiskitRuntimeService (needs token)

Free-tier note: Open plan = 10 min QPU / 28 days, job/batch mode only. Run
  --mode pilot   first to measure real per-circuit QPU time before the full batch.

Setup for real hardware (run once):
  from qiskit_ibm_runtime import QiskitRuntimeService
  QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<TOKEN>",
                                    instance="<crn>", region="us-east", overwrite=True)

Usage:
  python experiments/run_ibm_hardware.py --mode aer --n-windows 4
  python experiments/run_ibm_hardware.py --mode fake --fake-backend FakeBrisbane --n-windows 8
  python experiments/run_ibm_hardware.py --mode pilot --backend ibm_brisbane
  python experiments/run_ibm_hardware.py --mode hardware --backend ibm_brisbane --n-windows 16 --shots 2048
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_benchmark.tasks import get_task
from experiments.run_baseline_comparison import QASAModel, N_QUBITS, N_QLAYERS

CKPT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", "baseline_comparison")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
ANCILLA = N_QUBITS            # ancilla wire index (9th qubit)
N_WIRES = N_QUBITS + 1


# ============================================================
# Qiskit circuit mirroring the trained PennyLane PQC
# ============================================================

def build_qiskit_circuit(inputs, weights):
    """Construct the QASA PQC in Qiskit for one token.
    inputs: (N_QUBITS,) angle vector;  weights: (N_QLAYERS, N_QUBITS+1).
    Measures the 8 data qubits; <Z_i> is estimated from the sampled bitstrings."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_WIRES, N_QUBITS)
    # angle encoding
    for i in range(N_QUBITS):
        qc.rx(float(inputs[i]), i)
        qc.rz(float(inputs[i]), i)
    # first variational layer
    for i in range(N_QUBITS):
        qc.rx(float(weights[0, i]), i)
        qc.rz(float(weights[1, i]), i)
    # entangling layers
    for l in range(1, N_QLAYERS):
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)
            qc.ry(float(weights[l, i]), i)
            qc.rz(float(weights[l, i]), i)
        qc.cx(N_QUBITS - 1, ANCILLA)
        qc.ry(float(weights[l, N_QUBITS]), ANCILLA)
    for i in range(N_QUBITS):
        qc.measure(i, i)
    return qc


def expvals_from_counts(counts, shots):
    """<Z_i> for i in 0..N_QUBITS-1 from a counts dict over N_QUBITS classical bits.
    Qiskit bitstrings are little-endian: rightmost char = classical bit 0."""
    z = np.zeros(N_QUBITS)
    for bitstr, c in counts.items():
        bits = bitstr.replace(" ", "")[::-1]      # index i -> bits[i]
        for i in range(N_QUBITS):
            z[i] += c * (1.0 if bits[i] == '0' else -1.0)
    return z / shots


# ============================================================
# Quantum-evaluation backends: angles (M,8) + weights -> expvals (M,8)
# ============================================================

def eval_pennylane(angles, weights):
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=N_WIRES)

    @qml.qnode(dev)
    def circ(inp, w):
        for i in range(N_QUBITS):
            qml.RX(inp[i], wires=i); qml.RZ(inp[i], wires=i)
        for i in range(N_QUBITS):
            qml.RX(w[0, i], wires=i); qml.RZ(w[1, i], wires=i)
        for l in range(1, N_QLAYERS):
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
                qml.RY(w[l, i], wires=i); qml.RZ(w[l, i], wires=i)
            qml.CNOT(wires=[N_QUBITS - 1, ANCILLA])
            qml.RY(w[l, N_QUBITS], wires=ANCILLA)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return np.stack([np.array(circ(a, weights)) for a in angles])


def _run_aer(circuits, sim, shots):
    """Run a list of measured circuits on an AerSimulator and return expvals (M,8).
    Uses AerSimulator.run() directly: the SamplerV2 primitive can deadlock when
    other CPU-bound jobs are running, whereas .run() is robust."""
    res = sim.run(circuits, shots=shots).result()
    out = []
    for i in range(len(circuits)):
        out.append(expvals_from_counts(res.get_counts(i), shots))
    return np.stack(out)


def eval_aer(angles, weights, shots=2048, threads=1):
    """Noiseless Aer simulation (validates the qiskit circuit vs PennyLane)."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method='statevector', max_parallel_threads=threads)
    circuits = [build_qiskit_circuit(a, weights) for a in angles]
    return _run_aer(circuits, sim, shots)


def eval_fake(angles, weights, fake_backend="FakeBrisbane", shots=2048, threads=1):
    """Aer + a real IBM device noise model (hardware-realistic, offline)."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    import qiskit_ibm_runtime.fake_provider as fp
    backend = getattr(fp, fake_backend)()
    sim = AerSimulator.from_backend(backend)          # inherits device noise model
    sim.set_options(max_parallel_threads=threads)
    circuits = [transpile(build_qiskit_circuit(a, weights), backend, optimization_level=1)
                for a in angles]
    return _run_aer(circuits, sim, shots)


def report_calibration(backend):
    """Median device error rates / coherence for the marked-up hardware table.
    Robust to API differences across qiskit-ibm-runtime versions."""
    info = {}

    def _safe(fn):
        """Collect a per-qubit property, skipping qubits where it is undefined."""
        vals = []
        for q in range(backend.num_qubits):
            try:
                v = fn(q)
                if v is not None and np.isfinite(v):
                    vals.append(v)
            except Exception:
                continue
        return vals

    try:
        props = backend.properties()
        ro = _safe(props.readout_error)
        t1 = _safe(props.t1)
        t2 = _safe(props.t2)
        sx, recr = [], []
        for g in props.gates:
            try:
                err = props.gate_error(g.gate, g.qubits)
            except Exception:
                continue
            if err is None or not np.isfinite(err):
                continue
            if len(g.qubits) == 1:
                sx.append(err)
            elif len(g.qubits) == 2:
                recr.append(err)
        med = lambda xs: float(np.median(xs)) if xs else float('nan')
        info = {
            'median_readout_error': med(ro),
            'median_1q_gate_error': med(sx),
            'median_2q_gate_error': med(recr),
            'median_T1_us': med(t1) * 1e6 if t1 else float('nan'),
            'median_T2_us': med(t2) * 1e6 if t2 else float('nan'),
        }
    except Exception as e:
        print(f"  (calibration unavailable: {str(e)[:80]})")
    return info


def eval_hardware(angles, weights, backend_name, shots=2048, service=None, resilience=0):
    """Real IBM QPU via QiskitRuntimeService + SamplerV2 (one batched job).
    Returns (expvals, info) where info has job_id, qpu_seconds, calibration."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    if service is None:
        service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    cal = report_calibration(backend)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = [pm.run(build_qiskit_circuit(a, weights)) for a in angles]
    sampler = SamplerV2(mode=backend)
    if resilience:
        try:
            sampler.options.twirling.enable_measure = True   # readout (TREX) mitigation
        except Exception:
            pass
    job = sampler.run(circuits, shots=shots)
    print(f"  submitted job {job.job_id()} ({len(circuits)} circuits, {shots} shots, "
          f"resilience={resilience}) -> backend {backend_name}")
    res = job.result()
    out = [expvals_from_counts(res[i].data.c.get_counts(), shots) for i in range(len(circuits))]
    info = {'job_id': job.job_id(), 'backend': backend_name, 'calibration': cal}
    try:
        info['qpu_seconds'] = float(job.usage_estimation['quantum_seconds'])
        print(f"  QPU usage estimate: {info['qpu_seconds']:.1f} s")
    except Exception:
        info['qpu_seconds'] = None
    if cal:
        print(f"  calibration (medians): readout={cal['median_readout_error']:.2e} "
              f"1q={cal['median_1q_gate_error']:.2e} 2q={cal['median_2q_gate_error']:.2e} "
              f"T1={cal['median_T1_us']:.0f}us T2={cal['median_T2_us']:.0f}us")
    return np.stack(out), info


# ============================================================
# QASA hybrid forward: classical front-end -> angles -> (HW) expvals -> head
# ============================================================

def load_qasa(task_key='classical_chaotic_logistic', seed=42):
    model = QASAModel(hidden_dim=64, num_layers=4, seq_len=20)
    ckpt = torch.load(os.path.join(CKPT_DIR, 'qasa', task_key, f'seed{seed}', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def collect_angles(model, X):
    """Run the classical front-end and the quantum layer's input projection to get
    the per-token angle vectors fed to the PQC. Returns angles (B*L, 8) and the
    cached tensors needed to finish the forward pass."""
    enc = model.encoder
    qlayer_block = enc[-1]                      # QuantumEncoderLayer
    vq = qlayer_block.v_quantum                 # QuantumLayer
    with torch.no_grad():
        x = model.embedding(X)
        x = model.pos_encoding(x)
        for layer in enc[:-1]:
            x = layer(x)
        # quantum encoder layer up to the angle inputs
        attn_out, _ = qlayer_block.attn(x, x, x)
        x = qlayer_block.norm1(x + attn_out)
        B, L, F = x.shape
        x_flat = x.reshape(B * L, F)
        x_proj = torch.tanh(vq.input_proj(x_flat))
        x_proj = vq.norm(x_proj)
        ts = float(L)
        angles = (x_proj + ts).numpy()          # (B*L, 8)
    return angles, x_flat, (B, L, F)


def finish_forward(model, expvals, x_flat, shape):
    """Given PQC expectation values, complete the QASA forward and return the
    one-step prediction at the last position of each window."""
    enc = model.encoder
    qlayer_block = enc[-1]
    vq = qlayer_block.v_quantum
    B, L, F = shape
    with torch.no_grad():
        q = torch.from_numpy(expvals).float()
        out = vq.output_proj(q)
        q_out = (x_flat + out).view(B, L, F)
        x = qlayer_block.norm2(q_out + qlayer_block.ffn(q_out))
        pred = model.output_layer(x)            # (B, L, 1)
    return pred[:, -1, 0].numpy()               # last-step prediction per window


def build_test_windows(task, n_windows, seq_len=20):
    """Teacher-forced test windows from the test split; returns X (M,L,1), y_true (M,)."""
    X_train, Y_train, _, Y_test_true = task.generate_data()
    head = X_train[0, :, 0].numpy()
    mid = Y_train[0, -1:, 0].numpy()
    tail = Y_test_true[0, :, 0].numpy()
    series = np.concatenate([head, mid, tail]).astype(np.float32)
    boundary = head.shape[0]
    idxs = [i for i in range(len(series) - seq_len) if i + seq_len >= boundary]
    idxs = idxs[:n_windows]
    X = np.stack([series[i:i + seq_len] for i in idxs])
    y = np.array([series[i + seq_len] for i in idxs])
    return torch.from_numpy(X).float().unsqueeze(-1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pennylane', 'aer', 'fake', 'hardware', 'pilot'],
                    default='aer')
    ap.add_argument('--task', default='classical_chaotic_logistic')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-windows', type=int, default=8)
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--backend', default='ibm_brisbane')
    ap.add_argument('--fake-backend', default='FakeBrisbane')
    ap.add_argument('--mitigation', action='store_true',
                    help="enable readout (TREX) error mitigation on hardware")
    args = ap.parse_args()

    task = get_task(args.task)
    model = load_qasa(args.task, args.seed)
    weights = model.encoder[-1].v_quantum.qlayer.weights.detach().numpy()  # (4,9)

    nW = 2 if args.mode == 'pilot' else args.n_windows
    X, y_true = build_test_windows(task, nW)
    angles, x_flat, shape = collect_angles(model, X)
    n_circuits = angles.shape[0]
    print(f"Mode={args.mode}  task={args.task}  windows={nW}  circuits={n_circuits}  shots={args.shots}")

    # reference (noiseless simulator)
    ref = eval_pennylane(angles, weights)
    pred_ref = finish_forward(model, ref, x_flat, shape)

    info = {}
    if args.mode == 'pennylane':
        expvals = ref
    elif args.mode == 'aer':
        expvals = eval_aer(angles, weights, shots=args.shots)
    elif args.mode == 'fake':
        expvals = eval_fake(angles, weights, fake_backend=args.fake_backend, shots=args.shots)
    elif args.mode in ('hardware', 'pilot'):
        expvals, info = eval_hardware(angles, weights, args.backend, shots=args.shots,
                                      resilience=1 if args.mitigation else 0)
    pred_hw = finish_forward(model, expvals, x_flat, shape)

    # metrics (per-window error -> mean +/- std error bars)
    exp_mae = float(np.mean(np.abs(expvals - ref)))            # expectation-value fidelity
    err_ref = np.abs(pred_ref - y_true)
    err_hw = np.abs(pred_hw - y_true)
    mae_ref, mae_ref_std = float(err_ref.mean()), float(err_ref.std())
    mae_hw, mae_hw_std = float(err_hw.mean()), float(err_hw.std())
    pred_shift = float(np.mean(np.abs(pred_hw - pred_ref)))    # sim-vs-exec prediction shift
    cal = info.get('calibration', {}) or {}
    print(f"\n  <Z> mean|hw-sim|        : {exp_mae:.4f}")
    print(f"  one-step MAE (sim)      : {mae_ref:.4f} +/- {mae_ref_std:.4f}")
    print(f"  one-step MAE ({args.mode:9s}): {mae_hw:.4f} +/- {mae_hw_std:.4f}")
    print(f"  pred shift |hw-sim|     : {pred_shift:.4f}")

    out_csv = os.path.join(RESULTS_DIR, f'ibm_hardware_{args.mode}_{args.task}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'task', 'seed', 'n_windows', 'n_circuits', 'shots', 'mitigation',
                    'backend', 'job_id', 'qpu_seconds', 'expval_mae',
                    'mae_sim', 'mae_sim_std', 'mae_exec', 'mae_exec_std', 'pred_shift',
                    'readout_err', '1q_err', '2q_err', 'T1_us', 'T2_us'])
        w.writerow([args.mode, args.task, args.seed, nW, n_circuits, args.shots, int(args.mitigation),
                    info.get('backend', args.mode), info.get('job_id', ''),
                    f"{info.get('qpu_seconds') or 0:.1f}",
                    f"{exp_mae:.5f}", f"{mae_ref:.5f}", f"{mae_ref_std:.5f}",
                    f"{mae_hw:.5f}", f"{mae_hw_std:.5f}", f"{pred_shift:.5f}",
                    f"{cal.get('median_readout_error', float('nan')):.4e}",
                    f"{cal.get('median_1q_gate_error', float('nan')):.4e}",
                    f"{cal.get('median_2q_gate_error', float('nan')):.4e}",
                    f"{cal.get('median_T1_us', float('nan')):.1f}",
                    f"{cal.get('median_T2_us', float('nan')):.1f}"])
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
