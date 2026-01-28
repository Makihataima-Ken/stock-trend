import pandas as pd
import numpy as np

from src.features import normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

def make_windows(df, window=30, horizon=1):
    X, y = [], []

    prices = df["Close"].values

    for i in range(len(prices) - window - horizon):
        X.append(prices[i:i+window])
        y.append(int(prices[i+window+horizon] > prices[i+window]))

    return np.array(X), np.array(y)

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

