import os
import numpy as np
import torch
from torch.utils.data import Dataset

def get_dataset_path(base, mode, norm, split, fold):
    norm_name = norm.split('.', 1)[-1]
    folder = f"{norm_name}_{split}"
    file_norm = norm_name.replace("Zscore", "ZScore").replace("Minmax", "MinMax").replace("Decpre", "DecPre")
    prefix = "Train" if split == "Training" else "Test"
    filename = f"{prefix}_Dst_{file_norm}_CF_{fold}.txt"
    return os.path.join(base, mode, norm, folder, filename)

class BenchmarkDataset(Dataset):
    def __init__(self, path, device=None):
        raw = np.loadtxt(path).T
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = torch.tensor(raw[:, :144], dtype=torch.float32).to(device)
        self.y = torch.tensor(raw[:, 144:149].astype(int) - 1, dtype=torch.long).to(device)
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
