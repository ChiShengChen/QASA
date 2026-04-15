# QASA: Quantum Adaptive Self-Attention for Quantum Transformer Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.05336-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.05336)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-green)](https://pennylane.ai/)

## Overview

QASA (Quantum Adaptive Self-Attention) is a hybrid quantum-classical Transformer designed around a principle of **architectural parsimony**: maximal quantum benefit through minimal, strategically placed quantum integration.

Instead of distributing quantum computation across the entire model, QASA replaces only the value projection in a **single** encoder layer with a parameterized quantum circuit (PQC), using just **36 trainable quantum parameters** — fewer than any competing quantum model.

![image](https://github.com/user-attachments/assets/c78da79a-9325-4378-8d28-6e9bbcc7ca9b)

## Key Findings

- **Less is more**: A single quantum layer outperforms architectures with 2-4x more quantum parameters. Adding more quantum layers *degrades* performance.
- **Position matters more than count**: Q@3 (last layer) is best for chaotic dynamics; Q@2 (third layer) yields 3x improvement on damped oscillation.
- **Task-conditional advantage**: QASA excels on chaotic, noisy, and trend-dominated signals; classical Transformers remain superior for clean periodic waveforms.
- **Highest entangling capability**: QASA achieves Q=0.981 (near-maximal) with only 27 CNOT gates, while QLSTM uses 56 CNOTs yet achieves only Q=0.710.

## Results

### Baseline Comparison (4 models x 9 tasks x 3 seeds)

| Model | Params | Q-Params | MAE Wins | MSE Wins |
|-------|--------|----------|:--------:|:--------:|
| Classical | 200,257 | 0 | 3 | 4 |
| **QASA (Ours)** | 201,405 | **36** | 2 | **4** |
| QLSTM | 3,929 | 128 | 2 | 0 |
| QnnFormer | 190,631 | 90 | 2 | 1 |

QASA achieves the best MSE on 4 tasks (chaotic logistic, noisy damped oscillator, square wave, seasonal trend) — all involving nonlinear dynamics or complex temporal patterns. Statistically significant: QASA vs QLSTM on seasonal trend (p=0.009, Cohen's d>6).

### Ablation Study: Quantum Layer Position & Count

| Configuration | Quantum Layers | Chaotic Logistic (MAE/MSE) | Damped Osc. (MAE/MSE) | Square Wave (MAE/MSE) |
|---|---|---|---|---|
| 0Q (Classical) | None | 0.3594 / 0.1991 | 0.1193 / 0.0198 | **0.6988 / 1.2549** |
| Q@0 (First)    | {0}     | 0.3808 / 0.2092 | 0.0966 / 0.0138 | 0.7618 / 1.0343 |
| Q@1 (Second)   | {1}     | 0.3740 / 0.2033 | 0.1924 / 0.0511 | 0.7132 / 1.3366 |
| Q@2 (Third)    | {2}     | 0.4134 / 0.2437 | **0.0386 / 0.0020** | 0.8974 / 0.8941 |
| Q@3 (Last)     | {3}     | **0.3499 / 0.1832** | 0.1244 / 0.0206 | 0.7280 / 1.2989 |
| 2Q (Last two)  | {2,3}   | 0.3738 / 0.2090 | 0.1244 / 0.0225 | 0.7389 / 1.0912 |
| 4Q (All)       | {0,1,2,3} | 0.3753 / 0.2046 | 0.1532 / 0.0285 | 0.9441 / 0.9186 |

### Circuit Analysis

| Circuit | Qubits | Q-Params | CNOT Gates | Expressibility (KL↓) | Entangling Cap. (Q↑) |
|---------|:------:|:--------:|:----------:|:--------------------:|:--------------------:|
| **QASA** | 9 | 36 | 27 | 0.029 | **0.981** |
| QLSTM | 8 | 32 | 56 | 0.125 | 0.710 |
| QnnFormer | 8 | 24 | 21 | **0.026** | 0.883 |

## Project Structure

```
QASA/
├── quantum_benchmark/        # Benchmark task definitions (9 synthetic tasks)
│   └── tasks/
├── experiments/
│   ├── baselines/            # QLSTM and QnnFormer implementations
│   ├── run_qasa_benchmark.py # Main QASA benchmark runner
│   ├── run_baseline_comparison.py  # 4-model comparison
│   ├── run_ablation.py       # Ablation study (position & count)
│   ├── run_etth1_experiment.py     # Real-world ETTh1 dataset
│   ├── circuit_expressibility.py   # Expressibility & entangling analysis
│   ├── noise_robustness.py         # NISQ noise robustness (WIP)
│   ├── statistical_test_baseline.py # Statistical significance tests
│   └── results/              # CSV results and plots
├── IEEE_QASATransformer/     # Paper source (LaTeX)
├── QASA/                     # Original model implementations
└── plot/                     # Visualization utilities
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- PennyLane 0.30+

### Installation
```bash
git clone https://github.com/ChiShengChen/QASA.git
cd QASA
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run QASA benchmark (9 tasks, 5 seeds)
python -u experiments/run_qasa_benchmark.py --seeds 5 --epochs 200

# Run baseline comparison (4 models x 9 tasks x 3 seeds)
python -u experiments/run_baseline_comparison.py --seeds 3 --epochs 200

# Run ablation study
python -u experiments/run_ablation.py --epochs 200

# Run circuit expressibility analysis
python experiments/circuit_expressibility.py

# Run statistical tests
python experiments/statistical_test_baseline.py

# Dry run (quick sanity check)
python experiments/run_baseline_comparison.py --dry-run
```

## News / Updates

- **[2026-04-14]**: Circuit expressibility & entangling capability analysis added. QASA achieves Q=0.981 with fewest CNOTs.
- **[2026-04-11]**: Baseline comparison completed (4 models x 9 tasks x 3 seeds). Cover letter for EPJ Quantum Technology prepared.
- **[2026-03-15]**: Ablation study completed. Position matters more than count.
- **[2025-05-21]**: Benchmark files added.
- **[2025-04-13]**: Released QASA v2 with improved stability, LayerNorm, Kaiming init, and dropout support.

## Citation

If you use this code for your research, please cite:

```
@article{chen2025qasa,
  title={Quantum Adaptive Self-Attention for Quantum Transformer Models},
  author={Chen, Chi-Sheng and Kuo, En-Jui},
  journal={arXiv preprint arXiv:2504.05336},
  year={2025}
}
```
