# QASA: Quantum Adaptive Self-Attention for Quantum Transformer Models 
[![arXiv](https://img.shields.io/badge/arXiv-2504.05336-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.05336)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-green)](https://pennylane.ai/)

## Overview
QASA (Quantum Adaptive Self-Attention) is an innovative hybrid quantum-classical model designed to improve time series prediction tasks. By leveraging the power of quantum computing principles, QASA demonstrates superior predictive capabilities compared to traditional classical models, particularly for complex time-varying systems such as damped oscillators.
![image](https://github.com/user-attachments/assets/e419e626-649f-4a01-b5b4-a51fc4e6961d)


## Key Advantages of QASA

- **Enhanced Prediction Accuracy**: QASA's hybrid architecture combines the representational power of quantum circuits with classical transformer models to achieve improved prediction accuracy over purely classical approaches.
  
- **Efficient Parameter Usage**: The quantum components enable more efficient parameter utilization, requiring fewer trainable parameters to achieve comparable or better results than classical counterparts.
  
- **Robust Feature Extraction**: The quantum layers provide unique feature extraction capabilities that can capture complex non-linear patterns in time series data.
  
- **Adaptability**: QASA is particularly effective at modeling systems with complex dynamics, such as the damped oscillator demonstrated in this repository.

## Project Structure

The repository contains three main implementations for comparison:

1. **QASA (Quantum-Assisted Model)**: `quantum_v4_damped.py`
   - A hybrid model that integrates quantum circuits within a transformer architecture
   - Leverages quantum principles for enhanced feature representation
   - Implements parameterized quantum circuits using PennyLane

2. **Classical Transformer V4**: `transformer_classicalv4_damped.py`
   - A purely classical transformer model with similar architecture to QASA
   - Used as a direct comparison to demonstrate quantum advantage

3. **Baseline Transformer**: `transformer_attention_damped.py`
   - A standard transformer implementation with attention mechanisms
   - Serves as the baseline for performance benchmarking

## Technical Details

### QASA Architecture
- **Hybrid Design**: Combines quantum circuit layers with classical transformer components
- **Quantum Processing**: Incorporates parameterized quantum circuits with 8 qubits and 4 layers
- **Circuit Interface**: Implemented using PennyLane with PyTorch integration

### Data
The models are trained on simulated damped oscillator data, representing a common physical system with complex dynamics:
```
x(t) = A * exp(-γt) * cos(ωt + φ)
```
where:
- A: Amplitude
- γ: Damping coefficient
- ω: Angular frequency
- φ: Initial phase

## Results

QASA demonstrates several advantages over the classical implementations:

1. **Prediction Accuracy**: Lower MSE and MAE values on test data compared to classical models
2. **Convergence Rate**: Faster convergence to optimal solutions during training
3. **Pattern Recognition**: Better ability to capture the underlying patterns of damped oscillation

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PennyLane 0.20+
- PyTorch Lightning 1.5+

### Installation
```bash
# Clone the repository
git clone https://github.com/ChiShengChen/QASA.git
cd QASA

# Install dependencies
pip install -r requirements.txt
```

### Training the Models
Each model can be trained using the respective Python scripts:

```bash
# Train the QASA model
python quantum_v4_damped.py

# Train the Classical V4 model
python transformer_classicalv4_damped.py

# Train the baseline transformer model
python transformer_attention_damped.py
```

## Conclusion

QASA represents a significant step forward in leveraging quantum computing principles for practical machine learning applications. The hybrid approach demonstrates that even with today's quantum computing capabilities, we can achieve meaningful improvements over classical methods in predicting complex time series data.

The results suggest that as quantum hardware continues to advance, hybrid quantum-classical models like QASA will likely offer even greater advantages in accuracy, efficiency, and capability for a wide range of predictive modeling tasks. 
