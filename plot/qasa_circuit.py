import torch
import pennylane as qml
from matplotlib import pyplot as plt

n_qubits = 8
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits + 1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[n_qubits - 1, n_qubits])
        qml.RY(weights[l, -1], wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 示範輸入
dummy_input = torch.ones(n_qubits)
dummy_weights = torch.ones((n_layers, n_qubits + 1))

# 畫圖
drawer = qml.draw_mpl(quantum_circuit, expansion_strategy="device")
fig, _ = drawer(dummy_input, dummy_weights)
fig.savefig("/home/aidan/EEGPT/downstream/quantum_circuit_diagram.pdf", format="pdf")
fig.savefig("/home/aidan/EEGPT/downstream/quantum_circuit_diagram.svg", format="svg")
