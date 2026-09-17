
import os
import re
import glob
import numpy as np
import pandas as pd


def parse_train_len(path: str) -> int:
    """Parse the normal-prefix length tr from the filename."""
    m = re.search(r"tr_(\d+)", os.path.basename(path))
    if m is None:
        raise ValueError(f"could not parse tr from filename: {path}")
    return int(m.group(1))


def load_series(path: str, cap: int = None):
    """Returns (X[T,N] float64, y[T] int, tr int). cap optionally truncates
    long series to their first `cap` steps (used to bound training cost on
    the Tuning split; eval uses the full series, cap=None)."""
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    y = df["Label"].astype(int).to_numpy()
    X = df.drop(columns=["Label"]).to_numpy().astype(float)
    tr = parse_train_len(path)
    tr = min(tr, len(X) - 1)
    if cap:
        c = min(len(X), cap)
        X, y = X[:c], y[:c]
    return X, y, tr


def preprocess(X: np.ndarray, tr: int, mode: str = "zscore") -> np.ndarray:
    """Standardize using only the training-prefix statistics (semi-supervised
    protocol: never look at the test segment)."""
    Xtr = X[:tr]
    if mode == "zscore":
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
    elif mode == "minmax":
        mn = Xtr.min(0); mx = Xtr.max(0); rng = mx - mn; rng[rng < 1e-8] = 1.0
        Z = (X - mn) / rng
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")
    return Z.astype(np.float32)


def list_series(data_dir: str, pattern: str = "*.csv"):
    """List series files under a directory (optionally filtered by a glob pattern)."""
    return sorted(glob.glob(os.path.join(data_dir, pattern)))
