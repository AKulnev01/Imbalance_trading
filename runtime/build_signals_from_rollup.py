# runtime/build_signals_from_rollup.py
import os, glob
import pandas as pd
from utils.detect_fvg import detect_fvg_imbalances
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.detect_fvg import detect_fvg_imbalances

def to_dtidx(df):
    df = df.copy()
    # поддержка ts (ms) или timestamp
    if "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        # иногда индекс уже DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Нет ts/timestamp и индекс не datetime")
        df["timestamp"] = df.index
    df = df.set_index("timestamp").sort_index()
    # привести числа
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    return df

def find_4h_files(root="./data"):
    pats = [
        "**/*_ohlcv_4h.parquet",
        "**/*ohlcv*4h*.parquet",
        "**/*_4h.parquet",
    ]
    seen = set()
    out = []
    for p in pats:
        for f in glob.glob(os.path.join(root, p), recursive=True):
            if f.lower().endswith(".parquet") and f not in seen:
                out.append(f)
                seen.add(f)
    return sorted(out)

def infer_symbol(path):
    base = os.path.basename(path)
    # варианты имён: SYMBOL_ohlcv_4h.parquet или SYMBOL_4h.parquet
    name = base.replace(".parquet", "")
    if "_ohlcv_" in name:
        return name.split("_ohlcv_")[0]
    if name.endswith("_4h"):
        return name[:-3]
    return name

def main(out="./data/signals/signals"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    files = find_4h_files("./data")
    print(f"Found {len(files)} candidate 4h files")
    if not files:
        print("No signals produced")
        return

    rows = []
    for p in files:
        sym = infer_symbol(p)
        try:
            df = pd.read_parquet(p)
            df = to_dtidx(df)
            if len(df) < 20:
                print(f"{sym}: too few rows, skip")
                continue

            imbs = detect_fvg_imbalances(
                df,
                volume_multiplier=1.5,
                max_days_to_fill=9999,
                tolerance_pct=0.0,
                min_strength_pct=0.0
            ) or []

            cnt = 0
            for imb in imbs:
                side = imb.get("type")
                if side not in ("BUY", "SELL"):
                    continue
                rows.append({
                    "symbol":   sym,
                    "type":     side,
                    "strength": float(imb.get("strength", 0.0)),
                    "imb_time": pd.to_datetime(imb["time"], utc=True),
                    "entry":    float(imb.get("next_open") or (imb["low2"] if side == "BUY" else imb["high2"])),
                    "stop":     pd.NA,
                    "tp":       pd.NA,
                    "touched":  bool(imb.get("filled", False)),
                })
                cnt += 1
            print(f"{sym}: signals={cnt}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    df_sig = pd.DataFrame(rows)
    if df_sig.empty:
        print("No signals produced")
        return

    # базовый фильтр силы
    df_sig = df_sig[df_sig["strength"].fillna(0.0) >= 0.5].copy()
    df_sig = df_sig.sort_values(["symbol", "imb_time"]).reset_index(drop=True)

    # ✅ убираем таймзону для Excel
    df_sig["imb_time"] = pd.to_datetime(df_sig["imb_time"], utc=True, errors="coerce").dt.tz_localize(None)

    df_sig.to_parquet(out + ".parquet", index=False)
    try:
        df_sig.to_excel(out + ".xlsx", index=False)
    except Exception as e:
        print(f"[WARN] Excel export failed: {e}")

    print(f"Saved: {out}.parquet ({len(df_sig)} rows)")
    if os.path.exists(out + ".xlsx"):
        print(f"Saved: {out}.xlsx")

if __name__ == "__main__":
    main()