# qml-timeseries-bench
A simple and extensible benchmarking suite for quantum/classical time series tasks. It supports time series data generation, training, evaluation, and comparison for both Quantum Machine Learning (QML) and classical models.


## Features
- Built-in support for various classical and quantum time series tasks (e.g., waveform, trend + seasonality, piecewise regime, token learning, etc.)
- Decoupled task and model interfaces – tasks can be reused across different models
- Unified API: Use `get_task` to retrieve a task, `Task.generate_data` to generate data, and `Task.evaluate` to compute performance
- Supports automated training pipelines and simple examples

## Install

```bash
pip install -e .
```

## Quick Start

### 1. Generate and evaluate task data
```python
from quantum_rwkv import get_task

task = get_task('classical_waveform')
X_train, Y_train, X_test_seed, Y_test_true_full = task.generate_data()
# ...model training and prediction...
mae, mse = task.evaluate(y_true, y_pred)
```

### 2. Example: Classical waveform task + RWKV
See `quantum_rwkv/example_classical_waveform.py`

### 3. Example: Token learning task + RWKV
See `quantum_rwkv/example_classical_token.py`

### 4. Example: Quantum piecewise regime task + QuantumRWKV
See `quantum_rwkv/example_quantum_piecewise_regime.py`

## Supported Tasks
- classical_waveform
- classical_trend_seasonality_noise
- classical_square_triangle_wave
- classical_sawtooth_wave
- classical_piecewise_regime
- classical_noisy_damped_oscillation
- classical_learning (token)
- classical_damped_oscillation
- classical_chaotic_logistic
- classical_arma
- quantum_waveform
- quantum_trend_seasonality_noise
- quantum_square_triangle_wave
- quantum_sawtooth_wave
- quantum_piecewise_regime
- quantum_noisy_damped_oscillation
- quantum_learning (token)
- quantum_damped_oscillation
- quantum_chaotic_logistic
- quantum_arma

## Extending with New Tasks
1. Add a new task file under quantum_rwkv/tasks/ and implement the Task class with generate_data and evaluate methods.
2. Register the new task in task_registry inside quantum_rwkv/tasks/__init__.py.

## Naming
**qml-timeseries-bench** = Quantum Machine Learning + Time Series + Benchmark

