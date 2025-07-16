import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        # Select device dynamically if not provided
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x):
        # Move input to the correct device
        x = x.to(self.device)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
