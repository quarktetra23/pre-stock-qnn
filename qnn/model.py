import torch
import torch.nn as nn
from .attention import AttentionBlock
from .quantum import QuantumLayer

class QuantumAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttentionBlock(144)
        self.fc1 = nn.Linear(144, 256)
        self.q = QuantumLayer(n_qubits=4)
        self.fc2 = nn.Linear(4, 15)

    def forward(self, x):
        x = self.attn(x)
        x = torch.relu(self.fc1(x))
        x = self.q(x)
        return self.fc2(x).view(-1, 5, 3)
