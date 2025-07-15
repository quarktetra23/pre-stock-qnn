import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(weights[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.q_weights = nn.Parameter(torch.randn((n_layers, n_qubits), dtype=torch.float32))

    def forward(self, x):
        batch_out = []
        for sample in x:
            reduced = sample[: self.n_qubits]
            q_out = self.circuit(reduced, self.q_weights)
            vec = torch.stack(q_out)
            vec = vec.to(dtype=x.dtype, device=x.device)
            batch_out.append(vec)
        return torch.stack(batch_out)
