import torch
import torch.nn as nn
from .attention import AttentionBlock
from .quantum import QuantumLayer

class QuantumAttentionNet(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.attn = AttentionBlock(144, device=self.device)
        self.fc1 = nn.Linear(144, 256).to(self.device)
        self.q = QuantumLayer(n_qubits=4, device=self.device)
        self.fc2 = nn.Linear(4, 15).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.attn(x)
        x = torch.relu(self.fc1(x))
        x = self.q(x)
        return self.fc2(x).view(-1, 5, 3)
