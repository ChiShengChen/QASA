# QASA: Quantum Adaptive Self-Attention for Quantum Transformer Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.05336-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.05336)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.42-green)](https://pennylane.ai/)

## Overview

QASA (Quantum Adaptive Self-Attention) is a hybrid quantum-classical Transformer built around a principle of **architectural parsimony**: place minimal quantum computation at the optimal position rather than distributing it across the model. QASA replaces only the value projection in a **single** encoder layer with a parameterized quantum circuit (PQC), using just **36 trainable quantum parameters** — the fewest of any competing quantum model.

This repository accompanies the manuscript under major revision at *Quantum Science and Technology* (QST-105282). A central commitment of the work is **honest, capacity-controlled evaluation**: we test the quantum component not only against prior *quantum* baselines but against a **capacity-matched classical control**, and report transparently where quantum does — and does **not** — help.

![image](https://github.com/user-attachments/assets/c78da79a-9325-4378-8d28-6e9bbcc7ca9b)

## Key Findings (honest, capacity-controlled)

- **The accuracy gain is the bottleneck, not quantumness.** A parameter-matched *classical* bottleneck (40 params) matches QASA on the error metrics — and is better on clean-periodic signals, and the best model on the real-world ETTh1 dataset. So the advantage over a full-capacity Transformer comes from the low-rank value-projection **bottleneck structure**, not from the quantum substrate.
- **The compression is also a bottleneck property, not quantum-specific.** The classical bottleneck compresses the chaotic-logistic representation (effective rank ≈ 2.6) at least as strongly as the quantum layer (≈ 4.2) — so the "nonlinear dimensionality reduction" is not unique to quantum.
- **One well-placed quantum layer is enough; position matters more than count.** Across 3 seeds, the deployed last-layer position (Q@3) is the best single quantum position on the quantum-favourable tasks; adding more quantum layers (2Q/4Q) does not help.
- **Quantum's distinct value is physical.** Highest entangling capability (Meyer–Wallach *Q* = 0.981) with the fewest CNOTs (27) and fewest quantum parameters (36) among quantum models; barren-plateau-aware trainability; and **NISQ deployability validated on real IBM Quantum hardware** (≤ 7.5% one-step MAE degradation vs noiseless simulation, no error mitigation).

> **Note on framing.** We deliberately do **not** claim a quantum accuracy advantage. The contribution is (i) a reusable *methodology* for attributing apparent quantum gains via capacity-matched controls, and (ii) a hardware-validated case study delineating exactly where a NISQ quantum layer is and is not the source of benefit.

## Results

### Capacity-matched classical bottleneck (5 seeds) — the core control

| Task (MAE / MSE) | Classical (full, 200k) | **Classical bottleneck (40)** | QASA (36 q) |
|---|---|---|---|
| Chaotic Logistic | 0.373 / 0.204 | 0.344 / 0.186 | 0.341 / 0.182 |
| Seasonal Trend | 0.674 / 0.571 | **0.666 / 0.553** | 0.668 / 0.554 |
| Square Wave | 0.869 / **0.987** | **0.802** / 1.127 | 0.795 / 1.119 |
| Damped Osc. (control) | **0.038 / 0.002** | 0.064 / 0.009 | 0.117 / 0.029 |

On the favourable tasks the classical bottleneck is statistically indistinguishable from QASA; on the clean-periodic control it is clearly better. On real-world **ETTh1**, the classical bottleneck is the *best* model (MAE 0.0621 vs QASA 0.0675 vs full classical 0.0718).

### Baseline comparison (4 models × 9 tasks × 3 seeds)

| Model | Params | Q-Params (model) | MAE wins | MSE wins |
|-------|--------|:----------------:|:--------:|:--------:|
| Classical | 200,257 | 0 | 3 | 4 |
| **QASA (Ours)** | 201,405 | **36** | 2 | 2 clear + 2 tie |
| QLSTM | 3,929 | 128 | 2 | 0 |
| QnnFormer | 190,631 | 90 | 2 | 1 |

Among the quantum models QASA is the most resource-efficient (fewest quantum parameters and CNOTs, strongest entanglement). Two of QASA's MSE "wins" are statistical ties (chaotic logistic, noisy damped oscillator); we report them as ties rather than clear wins.

### Ablation: quantum-layer position & count (3 seeds, mean ± std, MAE)

| Configuration | Quantum Layers | Chaotic Logistic | Damped Osc. | Square Wave |
|---|---|---|---|---|
| 0Q (Classical) | None | 0.366 ± .006 | 0.243 ± .115 | **0.705 ± .005** |
| Q@0 (First)    | {0}       | 0.396 ± .011 | 0.097 ± .022 | 0.810 ± .109 |
| Q@1 (Second)   | {1}       | 0.389 ± .037 | 0.183 ± .061 | 0.878 ± .117 |
| Q@2 (Third)    | {2}       | 0.404 ± .023 | 0.106 ± .051 | 0.821 ± .081 |
| **Q@3 (Last, deployed)** | {3} | **0.337 ± .012** | **0.086 ± .051** | 0.825 ± .070 |
| 2Q (Last two)  | {2,3}     | 0.382 ± .033 | 0.116 ± .023 | 0.731 ± .011 |
| 4Q (All)       | {0,1,2,3} | 0.373 ± .003 | 0.126 ± .020 | 0.794 ± .106 |

Q@3 (the deployed position) is the best single quantum position on both quantum-favourable tasks (chaotic robustly, small variance; damped within seed noise); more quantum layers do not help. *Note:* an earlier single-seed table suggested "Q@2 best for damped (0.039)" — that was a seed-42 artifact (3-seed mean 0.106 ± .051) and is superseded here.

### Circuit analysis

| Circuit | Qubits | Q-Params (per-circuit) | CNOT Gates | Expressibility (KL↓) | Entangling Cap. (Q↑) |
|---------|:------:|:----------------------:|:----------:|:--------------------:|:--------------------:|
| **QASA** | 9 | 36 | 27 | 0.029 | **0.981** |
| QLSTM | 8 | 32 | 56 | 0.125 | 0.710 |
| QnnFormer | 8 | 24 | 21 | **0.026** | 0.883 |

QASA attains the strongest entanglement with the fewest CNOTs. (The per-circuit Q-Param counts above differ from the model-level budgets — 36 / 128 / 90 — used for the "fewest quantum parameters" comparison.)

## Project Structure

```
QASA/
├── quantum_benchmark/              # 9 synthetic benchmark task definitions
│   └── tasks/
├── experiments/
│   ├── baselines/                  # QLSTM and QnnFormer implementations
│   ├── run_qasa_benchmark.py       # Main QASA benchmark runner
│   ├── run_baseline_comparison.py  # 4-model comparison
│   ├── run_ablation.py             # Position/count ablation (supports --seeds, resumable)
│   ├── run_etth1_experiment.py     # Real-world ETTh1
│   ├── circuit_expressibility.py   # Expressibility & entangling analysis
│   ├── barren_plateau_analysis.py  # Gradient-variance / barren-plateau sweep
│   ├── qubit_scaling_analysis.py   # Gradient variance vs qubit count
│   ├── statistical_test_baseline.py
│   │   #  --- QST revision analyses (capacity-controlled / honest reporting) ---
│   ├── run_bottleneck_baseline.py        # Parameter-matched CLASSICAL bottleneck control
│   ├── run_etth1_bottleneck.py           # Bottleneck control on ETTh1
│   ├── representation_svd.py             # Effective-rank / SVD (QASA vs full classical)
│   ├── representation_svd_bottleneck.py  # SVD of the classical bottleneck (compression is a bottleneck property)
│   ├── make_svd_figure.py                # SVD-spectrum figure incl. bottleneck curve
│   ├── barren_deployed_circuit.py        # Gradient variance of the EXACT deployed PQC
│   ├── run_encoding_sensitivity.py       # angle vs amplitude vs data re-uploading
│   ├── run_seqlen_scaling.py             # sequence-length scalability (L = 20/48/96)
│   ├── run_qubit_ablation.py             # qubit-count ablation
│   ├── run_noise_multichannel.py         # depolarizing / amplitude-damping / bit-flip
│   ├── metrics_extended.py               # R^2, directional accuracy, SMAPE, overfit gap
│   ├── run_ibm_hardware.py               # real IBM Quantum execution (ibm_fez)
│   └── results/                          # CSV results and plots
├── QASA/                           # Original model implementations
└── plot/                           # Visualization utilities
```

> The LaTeX manuscript directories are managed separately and are not tracked in git.

## Getting Started

### Prerequisites
- Python 3.8+ · PyTorch 2.0+ · PennyLane 0.42 · scikit-learn · qiskit / qiskit-ibm-runtime (for the hardware run)

### Installation
```bash
git clone https://github.com/ChiShengChen/QASA.git
cd QASA
pip install -r requirements.txt
```

### Running Experiments

```bash
# Main benchmark (9 tasks, 5 seeds) and 4-model comparison
python -u experiments/run_qasa_benchmark.py --seeds 5 --epochs 200
python -u experiments/run_baseline_comparison.py --seeds 3 --epochs 200

# Capacity-matched classical bottleneck control (the core honesty check)
python -u experiments/run_bottleneck_baseline.py --epochs 200 --seeds 5
python -u experiments/run_etth1_bottleneck.py

# Ablation (position & count) — multi-seed + resumable
python -u experiments/run_ablation.py --epochs 200 --seeds 3 --seed-start 42

# Representation / compression analysis
python experiments/representation_svd.py
python experiments/representation_svd_bottleneck.py        # bottleneck compresses >= QASA
python experiments/make_svd_figure.py

# Trainability
python experiments/barren_plateau_analysis.py
python experiments/barren_deployed_circuit.py --samples 200  # exact deployed PQC

# Robustness / scalability / encoding / metrics
python experiments/run_noise_multichannel.py
python experiments/run_seqlen_scaling.py
python experiments/run_qubit_ablation.py
python experiments/run_encoding_sensitivity.py
python experiments/metrics_extended.py

# Real quantum hardware (requires an IBM Quantum account)
python experiments/run_ibm_hardware.py
```

## News / Updates

- **[2026-06-17]**: 3-seed ablation completed (supersedes single-seed Table); deployed-circuit barren-plateau check; classical-bottleneck SVD analysis — **the representation compression is a bottleneck property, not quantum-specific**.
- **[2026-06-13]**: **Major revision submitted to QST (QST-105282).** Added a parameter-matched classical bottleneck control → reframed: the accuracy gain comes from the low-rank bottleneck, not quantumness. Also added real IBM Quantum hardware validation, multi-channel noise study, encoding-scheme sensitivity, sequence-length scalability, qubit-mapping ablation, and extended metrics.
- **[2026-04-14]**: Circuit expressibility & entangling capability analysis added (Q = 0.981 with fewest CNOTs).
- **[2026-04-11]**: Baseline comparison completed (4 models × 9 tasks × 3 seeds).
- **[2026-03-15]**: Ablation study (position vs count).
- **[2025-04-13]**: QASA v2 with improved stability (LayerNorm, Kaiming init, dropout).

## Citation

If you use this code, please cite:

```
@article{chen2025qasa,
  title={Quantum Adaptive Self-Attention for Quantum Transformer Models},
  author={Chen, Chi-Sheng and Kuo, En-Jui},
  journal={arXiv preprint arXiv:2504.05336},
  year={2025}
}
```
