import torch
import numpy as np

def evaluate(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X)
        # move off GPU before converting to numpy
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    preds = (probs > 0.5).astype(int)

    # ensure y is a numpy array on CPU for comparison
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    return (preds == y_np).mean()
