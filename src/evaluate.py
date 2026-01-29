import torch
import numpy as np


def evaluate(model, X, y, device: str = "cpu", batch_size: int = 1) -> float:
    """Memory-safe evaluation that streams batches to the device.

    Args:
        model: Trained torch.nn.Module.
        X: Features (NumPy array or Tensor) with shape (N, ...).
        y: Labels (NumPy array or Tensor) with shape (N,) or (N, 1).
        device: "cpu" or torch device string.
        batch_size: Number of samples per evaluation batch. Defaults to 1 to
            avoid GPU OOM on very large validation sets (e.g., 2M samples).

    Returns:
        Accuracy over the provided dataset.
    """

    model.eval()
    batch_size = max(1, int(batch_size))

    # Ensure tensors for DataLoader without holding everything on GPU.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device != "cpu"),
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).long()

            correct += (preds == yb.long()).sum().item()
            total += yb.numel()

    return correct / total if total else 0.0
