import numpy as np
import pandas as pd

# --------- UTC utils ---------
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантирует tz-aware UTC индекс и сортировку.
    """
    x = df.copy()
    ix = pd.to_datetime(x.index, utc=True, errors="coerce")
    # если уже tz-aware → приведём к UTC (на всякий)
    try:
        if getattr(ix.tz, "zone", None) is None and ix.tz is None:
            ix = ix.tz_localize("UTC")
        else:
            ix = ix.tz_convert("UTC")
    except Exception:
        # fallback на «считаем UTC»
        ix = pd.to_datetime(x.index, utc=True, errors="coerce")
    x.index = ix
    return x.sort_index()

# --------- indicators (без внешних либ) ---------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill().fillna(0.0)

# --------- resample 1m → 4h ---------
def resample_4h(df_m1: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидает df_m1 с индексом-UTC и колонками open, high, low, close, volume.
    Возвращает 4h бары с индексом = open-время 4h бара (UTC).
    Последний незакрытый бар отбрасывается.
    """
    if df_m1 is None or df_m1.empty:
        return pd.DataFrame()

    x = ensure_utc_index(df_m1)[["open","high","low","close","volume"]].astype(float)

    o = x["open"].resample("4h").first()
    h = x["high"].resample("4h").max()
    l = x["low"].resample("4h").min()
    c = x["close"].resample("4h").last()
    v = x["volume"].resample("4h").sum()

    df4 = pd.concat([o, h, l, c, v], axis=1)
    df4.columns = ["open","high","low","close","volume"]
    df4 = df4.dropna(how="any")

    # убрать незакрытую текущую свечу
    now = pd.Timestamp.now(tz="UTC")
    if not df4.empty and (df4.index[-1] + pd.Timedelta(hours=4)) > now:
        df4 = df4.iloc[:-1]

    return df4

# --------- feature builder на 4h ---------
def build_4h_features(df4h: pd.DataFrame) -> pd.DataFrame:
    """
    На входе: 4h бары (open, high, low, close, volume) с UTC индексом (open-время).
    Возвращает: df с исходными колонками + engineered features.
    """
    if df4h is None or df4h.empty:
        return pd.DataFrame()

    x = ensure_utc_index(df4h).copy()

    # доходности
    x["ret1"] = x["close"].pct_change().fillna(0.0)
    x["ret2"] = x["close"].pct_change(2).fillna(0.0)
    x["ret_to_prev_close"] = (x["close"] - x["close"].shift(1)) / (x["close"].shift(1) + 1e-12)

    # геометрия свечи
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    x["body_ratio"] = (body / rng).clip(0, 10).fillna(0.0)
    x["upper_wick_ratio"] = ((x["high"] - x[["open", "close"]].max(axis=1)) / rng).clip(0, 10).fillna(0.0)
    x["lower_wick_ratio"] = ((x[["open", "close"]].min(axis=1) - x["low"]) / rng).clip(0, 10).fillna(0.0)

    # объём
    vol_ma = x["volume"].rolling(20).mean()
    x["vol_sma20"] = vol_ma
    x["vol_z"] = (x["volume"] - vol_ma) / (vol_ma.replace(0, np.nan))
    x["vol_z"] = x["vol_z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # индикаторы
    x["rsi14"] = _rsi(x["close"], 14)
    x["ema_fast"] = _ema(x["close"], 20)
    x["ema_slow"] = _ema(x["close"], 50)
    x["ema_diff_pct"] = (x["ema_fast"] - x["ema_slow"]) / (x["ema_slow"].replace(0, np.nan))
    x["ema_diff_pct"] = x["ema_diff_pct"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x["atr14"] = _atr(x, 14)

    return x

__all__ = ["build_4h_features", "resample_4h", "ensure_utc_index"]