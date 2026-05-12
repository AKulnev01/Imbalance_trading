# scripts/reconstruct_meta_from_m1.py
import os, re, glob
import pandas as pd
import numpy as np

IN_DIR  = "./data/m1"
OUT_DIR = "./data/m1_features_proxy"

def _infer_symbol_from_path(p):
    # пробуем достать символ из имени файла или родительской папки
    fname = os.path.basename(p)
    m = re.search(r'([A-Z0-9]+USDT)', fname)
    if m: return m.group(1)
    up = os.path.basename(os.path.dirname(p))
    m = re.search(r'([A-Z0-9]+USDT)', up)
    return m.group(1) if m else None

def load_and_normalize(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # --- детект и нормализация времени ---
    time_cols = [
        "timestamp","time","ts","open_time","start_time","startTime","t","datetime"
    ]
    tcol = next((c for c in time_cols if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"No time column in {path}. Columns={list(df.columns)}")
    s = df[tcol]
    # к datetime UTC
    if np.issubdtype(s.dtype, np.integer):
        # чаще всего миллисекунды
        # если похоже на секунды, тоже поддержим
        ser = s.copy()
        is_ms = ser.dropna().astype("int64").median() > 10_000_000_000
        df["timestamp"] = pd.to_datetime(ser, unit=("ms" if is_ms else "s"), utc=True)
    else:
        df["timestamp"] = pd.to_datetime(s, utc=True)

    # --- нормализация цен/объёма ---
    rename_map = {
        "o":"open","h":"high","l":"low","c":"close",
        "Open":"open","High":"high","Low":"low","Close":"close",
        "v":"volume","vol":"volume","Vol":"volume","turnover":"volume","quote_volume":"volume"
    }
    for k,v in list(rename_map.items()):
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)

    needed = ["open","high","low","close"]
    for c in needed:
        if c not in df.columns:
            # иногда только close присутствует — тогда дублируем
            if c != "close" and "close" in df.columns:
                df[c] = df["close"]
            else:
                raise KeyError(f"Column '{c}' not found in {path}. Columns={list(df.columns)}")

    if "volume" not in df.columns:
        # если нет — заполним нулями, чтобы фичи считались
        df["volume"] = 0.0

    # символ
    if "symbol" not in df.columns:
        sym = _infer_symbol_from_path(path) or "UNKNOWN"
        df["symbol"] = sym

    # чистим и сортируем
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","symbol","open","high","low","close","volume"]]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # базовые фичи на минутках
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)

    # ATR(14)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # RSI(14)
    delta = df["close"].diff()
    up = (delta.clip(lower=0)).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # OBV
    df["obv"] = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()

    # CMF(20)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = (mfm.fillna(0) * df["volume"])
    df["cmf_20"] = (mfv.rolling(20).sum()) / (df["volume"].rolling(20).sum().replace(0, np.nan))

    # прокси-фичи (vwap/imbalance простые)
    df["vwap_win20"] = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum().replace(0, np.nan))
    df["close_pos_in_range20"] = (df["close"] - df["low"].rolling(20).min()) / (df["high"].rolling(20).max() - df["low"].rolling(20).min()).replace(0,np.nan)

    return df

def process_one(path: str):
    df = load_and_normalize(path)
    df = add_features(df)
    sym = df["symbol"].iloc[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{sym}_m1_features_proxy.parquet")
    # дописываем партиями (по символу). Если файл существует — конкат и дедуп
    if os.path.exists(out):
        old = pd.read_parquet(out)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(out, index=False)
    return sym, len(df)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No parquet files under {IN_DIR}")
    for f in files:
        try:
            sym, n = process_one(f)
            print(f"[OK] {sym} -> {n} rows")
        except Exception as e:
            print(f"[SKIP] {f}: {e}")

if __name__ == "__main__":
    main()