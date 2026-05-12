# predict/tp_entry/data_utils.py
import os
import pandas as pd
import numpy as np
from .label_triple_barrier import compute_atr  # одна реализация ATR

def load_m1(symbol: str, m1_dir: str) -> pd.DataFrame:
    """
    Читает минутки из <m1_dir>/<SYMBOL>_m1.parquet
    Возвращает DataFrame с index=UTC datetime и колонками: open,high,low,close,volume.
    """
    p = os.path.join(os.path.expanduser(m1_dir), f"{symbol}_m1.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    cols = ["open","high","low","close","volume"]
    return df.sort_index()[cols]

def make_4h(m1: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует 1m → 4h (right-closed/right-labeled), добавляет atr14, ema12/ema26, trend_state, vol_regime.
    """
    ohlc = m1.resample("4h", label="right", closed="right").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()
    ohlc["atr14"] = compute_atr(ohlc, 14)

    ohlc["ema12"] = ohlc["close"].ewm(span=12, adjust=False).mean()
    ohlc["ema26"] = ohlc["close"].ewm(span=26, adjust=False).mean()
    ohlc["trend_state"] = np.where(ohlc["ema12"]>ohlc["ema26"], 1,
                            np.where(ohlc["ema12"]<ohlc["ema26"], -1, 0))

    # волатильностный режим по z-score ATR
    atr = ohlc["atr14"]
    mean = atr.rolling(500, min_periods=50).mean()
    std  = atr.rolling(500, min_periods=50).std(ddof=0)
    atr_z = (atr - mean) / std
    cuts = pd.qcut(atr_z.dropna(), q=3, labels=False, duplicates="drop")  # 0/1/2
    ohlc["vol_regime"] = cuts.reindex(ohlc.index).ffill().fillna(1).astype(int)

    return ohlc