import pandas as pd
import numpy as np

from src.features import normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

DEFAULT_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]


def load_data(
    path,
    frac=1.0,
    random_state=None,
    usecols=None,
    chunksize=200_000,
    sample_strategy="sequential",
):
    """Load dataset with optional subsampling.

    - Uses pyarrow when available (2-3x faster).
    - Reads only needed columns by default.
    - When frac<1, defaults to **sequential** sampling so time windows stay
      contiguous. Pass sample_strategy="random" to keep the prior random
      chunked sampling.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")

    cols = usecols or DEFAULT_COLS

    def _read_full():
        try:
            return pd.read_csv(path, engine="pyarrow", usecols=cols)
        except Exception:
            return pd.read_csv(path, usecols=cols)

    # Fast path: full read or sequential time-ordered slice
    if frac >= 1 or sample_strategy == "sequential":
        df = _read_full()
    else:
        rng = np.random.default_rng(random_state)
        sampled_chunks = []

        # Stream through the file; keep each row with probability `frac`.
        try:
            reader = pd.read_csv(path, engine="pyarrow", usecols=cols, chunksize=chunksize)
        except Exception:
            reader = pd.read_csv(path, usecols=cols, chunksize=chunksize)

        for chunk in reader:
            mask = rng.random(len(chunk)) < frac
            if mask.any():
                sampled_chunks.append(chunk.loc[mask])

        if sampled_chunks:
            df = pd.concat(sampled_chunks, ignore_index=True)
        else:
            # Handle extremely small datasets with frac << 1
            df = pd.DataFrame(columns=cols)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if frac < 1 and sample_strategy == "sequential":
        keep = max(1, int(len(df) * frac))
        # For time-ordered evaluation we want the most recent slice, not the oldest.
        df = df.iloc[-keep:]

    return df

def make_windows(data, window = 30, threshold=0.003):
    prices = data["Close"].values
    X, y = [], []

    for i in range(len(prices) - window - 1):
        w = prices[i:i+window]
        ret = (prices[i+window] - prices[i]) / prices[i]

        if ret > threshold:
            label = 1   # up
        elif ret < -threshold:
            label = 0   # down
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
