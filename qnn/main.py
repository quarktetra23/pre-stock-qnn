import os
import numpy as np
from .data import get_dataset_path, BenchmarkDataset
from .model import QuantumAttentionNet
from .train import train, evaluate
from .benchmark import benchmark_comparison
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

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
            train(model, DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0), epochs=EPOCHS)
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
