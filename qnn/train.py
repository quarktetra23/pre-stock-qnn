import torch
import torch.optim as optim
import torch.nn as nn

def train(model, loader, epochs):
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (xb, yb) in enumerate(loader):
            try:
                xb = xb.to(device)
                yb = yb.to(device)
                
                
                if xb.ndim != 2 or xb.shape[1] != 144:
                    raise ValueError(f"Input xb shape mismatch: expected (?, 144), got {xb.shape}")
                if yb.ndim != 2 or yb.shape[1] != 5:
                    raise ValueError(f"Target yb shape mismatch: expected (?, 5), got {yb.shape}")
                opt.zero_grad()
                out = model(xb)
                if out.shape[1:] != (5, 3):
                    raise ValueError(f"Model output shape mismatch: expected (?, 5, 3), got {out.shape}")
                losses = [loss_fn(out[:, i], yb[:, i]) for i in range(5)]
                loss = torch.stack(losses).sum()
                loss.backward()
                opt.step()
            except Exception as e:
                print(f"[train] Error in epoch {epoch}, batch {batch_idx}: {e}")
                raise

# Model evaluation 

def evaluate(model, loader):
    import numpy as np
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(loader):
            try:
                xb = xb.to(device)
                if xb.ndim != 2 or xb.shape[1] != 144:
                    raise ValueError(f"Input xb shape mismatch: expected (?, 144), got {xb.shape}")
                out = model(xb)
                if out.shape[1:] != (5, 3):
                    raise ValueError(f"Model output shape mismatch: expected (?, 5, 3), got {out.shape}")
                p = torch.argmax(out, dim=2)
                preds.append(p.cpu().numpy())
            except Exception as e:
                print(f"[evaluate] Error in batch {batch_idx}: {e}")
                raise
    predictions = np.concatenate(preds, axis=0)
    print("\nModel predictions for each test sample across 5 horizons:")
    print(f"Predictions shape: {predictions.shape} (rows=samples, cols=horizons)")
    print("Sample predictions (rows are individual samples, columns are horizons k=1,2,3,5,10):")
    print(predictions[:3])
    np.save("predictions.npy", predictions)
    return predictions
