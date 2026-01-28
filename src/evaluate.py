import torch
import numpy as np

def evaluate(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()

    preds = (probs > 0.5).astype(int)
    return (preds == y).mean()