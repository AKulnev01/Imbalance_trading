import pandas as pd
import numpy as np

# ------------------------------
# Technical analysis helper functions (manual implementations)
# ------------------------------

def SMA(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average
    """
    return series.rolling(window=period, min_periods=period).mean()


def RSI(series: pd.Series, period: int) -> pd.Series:
    """
    Relative Strength Index (Wilder's)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing: first average is simple then exponential
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ATR(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range
    """
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


def is_bull_div(df: pd.DataFrame, timestamp) -> bool:
    """
    Бычья дивергенция RSI: цена делает более низкий минимум, а RSI — более высокий.
    """
    if timestamp not in df.index:
        return False
    window = df[df.index <= timestamp].tail(21)
    if len(window) < 21:
        return False

    lows = window['low'].values
    close = window['close']
    rsi = RSI(close, 14).values

    low1, low2 = lows[-15], lows[-1]
    rsi1, rsi2 = rsi[-15], rsi[-1]
    return (low2 < low1) and (rsi2 > rsi1)


def is_bear_div(df: pd.DataFrame, timestamp) -> bool:
    """
    Медвежья дивергенция RSI: цена делает более высокий максимум, а RSI — более низкий.
    """
    if timestamp not in df.index:
        return False
    window = df[df.index <= timestamp].tail(21)
    if len(window) < 21:
        return False

    highs = window['high'].values
    close = window['close']
    rsi = RSI(close, 14).values

    high1, high2 = highs[-15], highs[-1]
    rsi1, rsi2 = rsi[-15], rsi[-1]
    return (high2 > high1) and (rsi2 < rsi1)