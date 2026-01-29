import torch
from torch.utils.data import DataLoader, TensorDataset


def train(model, loader, epochs=10, optimizer=None, criterion=None, device=None):
    # Allow caller to force device; default keeps auto CUDA/CPU behavior.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in loader:
            labels, inputs = labels.to(device), inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
