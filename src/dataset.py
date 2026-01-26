import pandas as pd
import numpy as np

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
