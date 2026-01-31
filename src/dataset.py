import pandas as pd
import numpy as np

from src.features import normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Ticker", "Date"])
    return df

def make_windows(data, window=30, threshold=0.02):
    o = data["Open"].values
    h = data["High"].values
    l = data["Low"].values
    c = data["Close"].values

    X, y = [], []

    for i in range(len(c) - window - 1):
        # Shape: (window, 4)
        w = np.stack([
            o[i:i+window-1],
            h[i:i+window-1],
            l[i:i+window-1],
            c[i:i+window-1],
        ], axis=1)

        future_ret = (c[i+window] - c[i]) / c[i]

        if future_ret > threshold:
            label = 1
        elif future_ret < -threshold:
            label = 0 
        else:
            continue
        
        X.append(w)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_data_loader(df, batch_size=32, shuffled=None, normalize =False):
    X, y = make_windows(df)
    if normalize:
        X = normalize(X)
    dataset = StocksDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffled)

class StocksDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.tensor((y > 0).astype(float), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
