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

def make_windows(data, window = 30, threshold=0.003):
    prices = data["Close"].values
    X, y = [], []

    for i in range(len(prices) - window - 1):
        w = prices[i:i+window]
        ret = (prices[i+window] - prices[i+window-1]) / prices[i+window-1]

        if ret > threshold:
            label = 1   # up
        elif ret < -threshold:
            label = -1   # down
        else:
            continue    # flat

        X.append(w)
        y.append(label)

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

def prepare_data(df):
    """Prepare features and target from raw data"""
    
    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'Date'])
    
    # Calculate technical indicators
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volume_change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        df[f'volume_sma_{window}'] = df.groupby('Ticker')['Volume'].transform(
            lambda x: x.rolling(window=window).mean()
        )
    
    # Volatility (20-day rolling std)
    df['volatility'] = df.groupby('Ticker')['returns'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    # RSI (14-day)
    df['rsi'] = df.groupby('Ticker')['Close'].transform(calculate_rsi)
    
    # Price ranges
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_spread'] = (df['Close'] - df['Open']) / df['Open']
    
    # FIX 1: Calculate future return without the deprecated pct_change
    df['future_close'] = df.groupby('Ticker')['Close'].shift(-30)
    df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # Drop temporary column
    df = df.drop('future_close', axis=1)
    
    
    # Drop rows with NaN (due to rolling windows and future target)
    df = df.dropna()
    
    # FIX 2: Replace inf values with NaN, then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df


def make_windows(data, window=30, threshold=0.003):
    o = data["Open"].values
    h = data["High"].values
    l = data["Low"].values
    c = data["Close"].values

    X, y = [], []

    for i in range(len(c) - window - 1):
        # Shape: (window, 4)
        w = np.stack([
            o[i:i+window],
            h[i:i+window],
            l[i:i+window],
            c[i:i+window],
        ], axis=1)

        future_ret = (c[i+window] - c[i+window-1]) / c[i+window-1]

        if future_ret > threshold:
            label = 1
        elif future_ret < -threshold:
            label = 0   # use 0/1 for BCE
        else:
            continue
        
        X.append(w)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)