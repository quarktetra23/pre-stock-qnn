"""
QNN-Based Stock Trend Prediction with Attention

This script implements a hybrid model for predicting stock price trends using
Limit Order Book (LOB) data. The model architecture combines an attention block
with a simulated quantum layer. It processes historical LOB data and predicts the
direction of future mid-price trends over 5 different horizons (k = 1, 2, 3, 5, 10).

Author: Abhishek Manhas
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# === Dataset Path Builder ===
def get_dataset_path(base, mode, norm, split, fold):
    """
    Constructs the file path for a given fold and configuration.
    """
    norm_name = norm.split(".", 1)[-1]
    folder = f"{norm_name}_{split}"
    file_norm = norm_name.replace("Zscore", "ZScore").replace("Minmax", "MinMax").replace("Decpre", "DecPre")
    prefix = "Train" if split == "Training" else "Test"
    filename = f"{prefix}_Dst_{file_norm}_CF_{fold}.txt"
    return os.path.join(base, mode, norm, folder, filename)

# === Dataset Class ===
class BenchmarkDataset(Dataset):
    """
    Loads and formats LOB data for training/testing.
    """
    def __init__(self, path):
        raw = np.loadtxt(path).T
        self.X = torch.tensor(raw[:, :144], dtype=torch.float32)
        self.y = torch.tensor(raw[:, 144:149].astype(int) - 1, dtype=torch.long)  # classes: 0 = D, 1 = S, 2 = U

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# === Attention Layer ===
class AttentionBlock(nn.Module):
    """
    Basic self-attention mechanism that helps in some way I haven't fully understood (from the theoratical point of view). 
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

# === Simulated Quantum Layer ===
class QuantumLayer(nn.Module):
    """
    Simulates a quantum layer using a non-linear transformation.
    Acts as a placeholder for future quantum circuits.
    """
    def __init__(self, n_qubits=4):
        super().__init__()
        self.linear = nn.Linear(256, n_qubits)

    def forward(self, x):
        return torch.tanh(self.linear(x))  # Emulates quantum-style non-linearity

# === Model Definition ===
class QuantumAttentionNet(nn.Module):
    """
    Full hybrid model: Attention -> Fully Connected -> Simulated Quantum -> Output
    """
    def __init__(self):
        super().__init__()
        self.attn = AttentionBlock(144)
        self.fc1 = nn.Linear(144, 256)
        self.q = QuantumLayer(n_qubits=4)
        self.fc2 = nn.Linear(4, 15)  # 5 horizons * 3 classes

    def forward(self, x):
        x = self.attn(x)
        x = torch.relu(self.fc1(x))
        x = self.q(x)
        return self.fc2(x).view(-1, 5, 3)

# === Training Function ===
def train(model, loader, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = sum(loss_fn(out[:, i], yb[:, i]) for i in range(5))  
            loss.backward()
            opt.step()

# === Evaluation Function ===
def evaluate(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb)
            p = torch.argmax(out, dim=2)
            preds.append(p.numpy())
    predictions = np.concatenate(preds, axis=0)
    print("\nModel predictions for each test sample across 5 horizons:")
    print(f"Predictions shape: {predictions.shape} (rows=samples, cols=horizons)")
    print("Sample predictions (rows are individual samples, columns are horizons k=1,2,3,5,10):")
    print(predictions[:3])
    np.save("predictions.npy", predictions)
    return predictions

# === Benchmark Comparison ===
def benchmark_comparison(avg_f1_scores):
    benchmark_scores = {1: 88.7, 2: 80.6, 3: 80.1, 5: 88.2, 10: 91.6}
    model_scores_pct = {k: avg_f1_scores[i] * 100 for i, k in enumerate([1, 2, 3, 5, 10])}
    percentage_diffs = {}

    for k in benchmark_scores:
        model_score = model_scores_pct[k]
        benchmark = benchmark_scores[k]
        diff = ((model_score - benchmark) / benchmark) * 100
        percentage_diffs[k] = diff

    avg_percentage_diff = np.mean(list(percentage_diffs.values()))

    print("\nPercentage difference from benchmark scores (positive = better, negative = worse):")
    for k in [1, 2, 3, 5, 10]:
        print(f"Horizon {k}: Our Model = {model_scores_pct[k]:.2f}% | Benchmark = {benchmark_scores[k]}% | Difference = {percentage_diffs[k]:+.2f}%")

    print(f"\nAverage % difference from benchmark across all horizons: {avg_percentage_diff:+.2f}%")
    return avg_percentage_diff

# === Main Execution ===
def run_manual():
    base = os.path.expanduser("~/Desktop/qnnprediction2/BenchmarkDatasets")
    mode = "Auction"
    norm = "1.Auction_Zscore"

    folds = int(input("How many folds do you want to run (1-9)? "))
    EPOCHS = 2 if folds < 5 else 5

    scores = []
    for fold in range(1, folds + 1):
        try:
            train_path = get_dataset_path(base, mode, norm, "Training", fold)
            test_path = get_dataset_path(base, mode, norm, "Testing", fold)
            train_ds = BenchmarkDataset(train_path)
            test_ds = BenchmarkDataset(test_path)

            model = QuantumAttentionNet()
            train(model, DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2), epochs=EPOCHS)
            predictions = evaluate(model, DataLoader(test_ds, batch_size=32))
            y_true = np.loadtxt(test_path).T[:, 144:149].astype(int) - 1
            f1s = [f1_score(y_true[:, i], predictions[:, i], average='macro') for i in range(5)]
            scores.append(f1s)
        except FileNotFoundError as e:
            print(f"Skipping missing file: {e}")
            continue

    if scores:
        avg = np.mean(scores, axis=0)
        print("\nAverage F1 Scores across horizons k = 1, 2, 3, 5, 10:")
        for k, score in zip([1, 2, 3, 5, 10], avg):
            print(f"Horizon {k}: F1 Score = {score:.4f}")
        benchmark_comparison(avg)

if __name__ == '__main__':
    run_manual()
