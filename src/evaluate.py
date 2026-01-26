import torch
import numpy as np

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()

    preds = (probs > 0.5).astype(int)
    accuracy = (preds == y).mean()
    return accuracy
