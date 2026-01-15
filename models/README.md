# Trained Models

This directory contains **trained model artifacts** produced by the training notebooks.

---

## What This Folder Contains

- Serialized model weights (e.g. `.pt`, `.pth`, `.h5`)
- Files saved **after training** for evaluation or reuse

Example:
```
models/
├── stock_model_v1.pt
├── stock_model_v2.pt
└── README.md
```


---

## What This Folder Does NOT Contain

- ❌ Raw training data
- ❌ Any portion of the Kaggle competition dataset
- ❌ Preprocessed data files or feature matrices

This is required to comply with **Kaggle competition data usage rules**.

---

## Usage

Trained models can be loaded in code as follows (example in PyTorch):

```python
import torch
from src.model import StockNet

model = StockNet(input_dim=INPUT_DIM)
model.load_state_dict(torch.load("models/stock_model_v1.pt"))
model.eval()

```