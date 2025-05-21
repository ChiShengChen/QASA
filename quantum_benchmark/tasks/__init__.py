# tasks submodule

from importlib import import_module

task_registry = {
    'classical_waveform': 'QASA.quantum_benchmark.tasks.classical_waveform',
    'classical_trend_seasonality_noise': 'QASA.quantum_benchmark.tasks.classical_trend_seasonality_noise',
    'classical_square_triangle_wave': 'QASA.quantum_benchmark.tasks.classical_square_triangle_wave',
    'classical_sawtooth_wave': 'QASA.quantum_benchmark.tasks.classical_sawtooth_wave',
    'classical_piecewise_regime': 'QASA.quantum_benchmark.tasks.classical_piecewise_regime',
    'classical_noisy_damped_oscillation': 'QASA.quantum_benchmark.tasks.classical_noisy_damped_oscillation',
    'classical_learning': 'QASA.quantum_benchmark.tasks.classical_learning',
    'classical_damped_oscillation': 'QASA.quantum_benchmark.tasks.classical_damped_oscillation',
    'classical_chaotic_logistic': 'QASA.quantum_benchmark.tasks.classical_chaotic_logistic',
    'classical_arma': 'QASA.quantum_benchmark.tasks.classical_arma',
    'quantum_waveform': 'QASA.quantum_benchmark.tasks.quantum_waveform',
    'quantum_trend_seasonality_noise': 'QASA.quantum_benchmark.tasks.quantum_trend_seasonality_noise',
    'quantum_square_triangle_wave': 'QASA.quantum_benchmark.tasks.quantum_square_triangle_wave',
    'quantum_sawtooth_wave': 'QASA.quantum_benchmark.tasks.quantum_sawtooth_wave',
    'quantum_piecewise_regime': 'QASA.quantum_benchmark.tasks.quantum_piecewise_regime',
    'quantum_noisy_damped_oscillation': 'QASA.quantum_benchmark.tasks.quantum_noisy_damped_oscillation',
    'quantum_learning': 'QASA.quantum_benchmark.tasks.quantum_learning',
    'quantum_damped_oscillation': 'QASA.quantum_benchmark.tasks.quantum_damped_oscillation',
    'quantum_chaotic_logistic': 'QASA.quantum_benchmark.tasks.quantum_chaotic_logistic',
    'quantum_arma': 'QASA.quantum_benchmark.tasks.quantum_arma',
}

def get_task(task_name):
    """
    根據任務名稱回傳對應的 Task class 實例。
    """
    if task_name not in task_registry:
        raise ValueError(f"Unknown task: {task_name}")
    module = import_module(task_registry[task_name])
    return module.Task()
