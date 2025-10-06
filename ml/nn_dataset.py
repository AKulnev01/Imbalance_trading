import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

FX_PREFIX = "fx__"
TH_PREFIX = "th__"
TARGET = "pnl_pct_after_cost"

def load_offline(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # заполняем пропуски нулями, чистим бесконечности
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    fx_cols = [c for c in df.columns if c.startswith(FX_PREFIX)]
    th_cols = [c for c in df.columns if c.startswith(TH_PREFIX)]
    return fx_cols, th_cols

def make_arrays(df: pd.DataFrame, fx_cols: List[str], th_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_fx = df[fx_cols].astype(float).values
    X_th = df[th_cols].astype(float).values
    y = df[TARGET].astype(float).values
    return X_fx, X_th, y

def train_val_split_idx(n: int, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    # тайм-сплит: первые 80% train, хвост — val
    cut = int(n * (1.0 - val_frac))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]