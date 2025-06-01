# qml-timeseries-bench

一個簡單、可擴充的量子/經典時序任務評測套件，支援 QML (Quantum Machine Learning) 與經典模型的時序資料生成、訓練、評估與比較。

## 特色
- 內建多種經典與量子時序任務（waveform、trend+seasonality、piecewise regime、token learning...）
- 任務與模型分離，任務可重用於不同模型
- 統一 API：`get_task` 取得任務，`Task.generate_data` 產生資料，`Task.evaluate` 評估
- 支援自動化訓練流程與簡單範例

## 安裝

```bash
pip install -e .
```

## 快速開始

### 1. 任務資料生成與評估
```python
from quantum_rwkv import get_task

task = get_task('classical_waveform')
X_train, Y_train, X_test_seed, Y_test_true_full = task.generate_data()
# ...模型訓練與預測...
mae, mse = task.evaluate(y_true, y_pred)
```

### 2. 範例：經典 waveform 任務 + RWKV
見 `quantum_rwkv/example_classical_waveform.py`

### 3. 範例：token 任務 + RWKV
見 `quantum_rwkv/example_classical_token.py`

### 4. 範例：量子 piecewise regime 任務 + QuantumRWKV
見 `quantum_rwkv/example_quantum_piecewise_regime.py`

## 支援的任務
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

## 擴充任務
1. 在 `quantum_rwkv/tasks/` 新增任務檔案，實作 `Task` class 的 `generate_data` 與 `evaluate`。
2. 在 `quantum_rwkv/tasks/__init__.py` 的 `task_registry` 註冊新任務。

## 命名由來
**qml-timeseries-bench** = Quantum Machine Learning + Time Series + Benchmark

