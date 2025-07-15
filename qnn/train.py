import torch
import torch.optim as optim
import torch.nn as nn

def train(model, loader, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            losses = [loss_fn(out[:, i], yb[:, i]) for i in range(5)]
            loss = torch.stack(losses).sum()
            loss.backward()
            opt.step()

def evaluate(model, loader):
    import numpy as np
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
