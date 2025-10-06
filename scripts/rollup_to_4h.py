# scripts/rollup_to_4h.py
# Универсальный роллап минутных данных (OHLCV, trades, orderbook, OI/funding, liquidations)
# в любой таймфрейм pandas (например: 5T, 15T, 1H, 4H, 1D).
# Выходные файлы получают суффикс _<family>_<tf>.parquet, где <tf> — в нижнем регистре (напр. _ohlcv_4h.parquet).

import os
import glob
import argparse
import pandas as pd
import numpy as np

# -------- utils --------
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _parse_tf(tf: str) -> str:
    """Валидируем и возвращаем pandas offset alias. Примеры: 5T, 15T, 1H, 4H, 1D."""
    tf = (tf or "4H").strip()
    # допускаем формы типа '5m','15m','1h','4h','1d'
    m = tf.lower()
    x = None
    if m.endswith("m"): x = f"{int(m[:-1])}T"
    elif m.endswith("h"): x = f"{int(m[:-1])}H"
    elif m.endswith("d"): x = f"{int(m[:-1])}D"
    else:
        # пусть пользователь сам передал валидный alias ('5T','15T','1H','4H','1D', etc.)
        x = tf.upper()
    return x

def _tf_suffix(tf_alias: str) -> str:
    """Строка для имени файла — в нижнем регистре без пробелов."""
    return tf_alias.lower()

def _to_dtindex(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    d = df.copy()
    if ts_col in d.columns:
        d[ts_col] = pd.to_datetime(d[ts_col], unit="ms", utc=True)
        d = d.set_index(ts_col)
    elif not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or 'ts' column in ms.")
    return d.sort_index()

def _to_ms(df: pd.DataFrame, idx_name: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out = out.reset_index()
    if isinstance(out[idx_name].dtype, pd.DatetimeTZDtype) or np.issubdtype(out[idx_name].dtype, np.datetime64):
        out[idx_name] = (out[idx_name].astype("int64") // 10**6)
    else:
        # уже миллисекунды
        pass
    return out

# -------- агрегаторы --------
def resample_ohlcv_agg(df_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """OHLCV (и turnover, если есть): ohlc + sum(volume/turnover)."""
    d = _to_dtindex(df_m1, "ts")
    # гарантируем столбцы
    for c in ["open","high","low","close"]:
        if c not in d.columns: d[c] = pd.NA
    ohlc = d[["open","high","low","close"]].resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"})
    out = ohlc.copy()
    for c in ["volume","turnover"]:
        if c in d.columns:
            out[c] = d[c].resample(tf).sum(min_count=1)
    out = out.dropna(how="all")
    return _to_ms(out, "ts")

def _vwap_sum(df: pd.DataFrame, price_col: str, vol_col: str, tf: str) -> pd.Series:
    """VWAP по окну tf из агрегированных минуток: sum(price*vol)/sum(vol)."""
    price_x_vol = (df[price_col] * df[vol_col]).resample(tf).sum(min_count=1)
    vol_sum = df[vol_col].resample(tf).sum(min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = price_x_vol / vol_sum.replace(0, np.nan)
    return vwap.fillna(method="ffill").fillna(0.0)

def resample_trades_agg(df_tr_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    trades_1m схема: ts, trades, vwap, vol_buy, vol_sell, delta, last
    Роллап:
      • trades, vol_buy/sell, delta — sum
      • vwap — VWAP по (vwap_min * (vol_buy+vol_sell)_min)
      • last — last (последняя цена в окне)
    """
    d = _to_dtindex(df_tr_m1, "ts")
    # суммируем
    out = pd.DataFrame()
    for c in ["trades","vol_buy","vol_sell","delta"]:
        if c in d.columns:
            out[c] = d[c].resample(tf).sum(min_count=1)
    # VWAP
    if "vwap" in d.columns:
        vol_total = d.get("vol_buy", pd.Series(dtype=float)).fillna(0.0) + d.get("vol_sell", pd.Series(dtype=float)).fillna(0.0)
        tmp = d.copy()
        tmp["vol_total"] = vol_total
        out["vwap"] = _vwap_sum(tmp, "vwap", "vol_total", tf)
    # last
    if "last" in d.columns:
        out["last"] = d["last"].resample(tf).last()
    out = out.dropna(how="all")
    out = _to_ms(out, "ts")
    return out

def resample_ob_agg(df_ob_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    ob_1m схема: ts, bid_vol, ask_vol, imbalance, bid_vwap, ask_vwap, spread_bp
    Роллап: средние значения по окну (mean). Имбаланс и спред — mean.
    """
    d = _to_dtindex(df_ob_m1, "ts")
    cols = [c for c in ["bid_vol","ask_vol","imbalance","bid_vwap","ask_vwap","spread_bp"] if c in d.columns]
    if not cols: return pd.DataFrame(columns=["ts"])
    out = d[cols].resample(tf).mean()
    out = out.dropna(how="all")
    out = _to_ms(out, "ts")
    return out

def resample_oi_funding_agg(df_oi_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    oi_funding_1m схема: ts, open_interest, funding_rate
    Роллап: mean (или last — на твой выбор; по умолчанию mean).
    """
    d = _to_dtindex(df_oi_m1, "ts")
    cols = [c for c in ["open_interest","funding_rate"] if c in d.columns]
    if not cols: return pd.DataFrame(columns=["ts"])
    out = d[cols].resample(tf).mean()
    out = out.dropna(how="all")
    out = _to_ms(out, "ts")
    return out

def resample_liq_agg(df_liq_any: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    liquidations.parquet может быть тик-частоты. Делаем минутную сводку, затем суммируем по tf.
    Минутная сводка:
      • liq_buy_sz, liq_sell_sz — сумма по сторонам
      • liq_count — количество событий
      • liq_avg_px — средняя цена за минуту
    Роллап:
      • суммы/liqs — sum
      • средний px — mean (можно заменить на size-weighted mean при желании)
    """
    df = df_liq_any.copy()
    if "ts" not in df.columns:
        return pd.DataFrame(columns=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # минутная аггрегация
    m = pd.DataFrame()
    if "liq_size" in df.columns and "liq_side" in df.columns:
        m["liq_buy_sz"]  = df.loc[df["liq_side"].str.lower()=="buy", "liq_size"].resample("1T").sum()
        m["liq_sell_sz"] = df.loc[df["liq_side"].str.lower()=="sell","liq_size"].resample("1T").sum()
        m["liq_count"]   = df["liq_size"].resample("1T").count()
    else:
        # fallback: если только "size"
        if "liq_size" in df.columns:
            m["liq_buy_sz"] = df["liq_size"].resample("1T").sum()
            m["liq_sell_sz"] = 0.0
            m["liq_count"] = df["liq_size"].resample("1T").count()
    if "liq_price" in df.columns:
        m["liq_avg_px"]  = df["liq_price"].resample("1T").mean()

    m = m.fillna(0.0)

    # роллап в tf
    out = pd.DataFrame()
    for c in ["liq_buy_sz","liq_sell_sz","liq_count"]:
        if c in m.columns:
            out[c] = m[c].resample(tf).sum(min_count=1)
    if "liq_avg_px" in m.columns:
        out["liq_avg_px"] = m["liq_avg_px"].resample(tf).mean()
    out = out.dropna(how="all")
    out = _to_ms(out, "ts")
    return out

# -------- main orchestration --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", default="./data/m1", help="директория с *_m1.parquet (OHLCV)")
    ap.add_argument("--meta1m-dir", default="./data/meta1m", help="директория с *_trades_1m.parquet, *_ob_1m.parquet, *_oi_funding_1m.parquet, *_liquidations.parquet")
    ap.add_argument("--out-dir", default="./data/rollup4h")
    ap.add_argument("--tf", default="4H", help="целевой таймфрейм: 5T, 15T, 1H, 4H, 1D ... или 5m/15m/1h/4h/1d")
    args = ap.parse_args()

    tf_alias = _parse_tf(args.tf)
    suffix = _tf_suffix(tf_alias)

    _ensure_dir(args.out_dir)

    # --- OHLCV ---
    for p in sorted(glob.glob(os.path.join(args.m1_dir, "*_m1.parquet"))):
        sym = os.path.basename(p).replace("_m1.parquet","")
        try:
            df = pd.read_parquet(p)
            out = resample_ohlcv_agg(df, tf_alias)
            out.to_parquet(os.path.join(args.out_dir, f"{sym}_ohlcv_{suffix}.parquet"), index=False)
            print(f"[{args.tf}] OHLCV {sym} rows={len(out)}")
        except Exception as e:
            print(f"[{args.tf}_ERR] {sym} OHLCV {e}")

    # --- TRADES ---
    for p in sorted(glob.glob(os.path.join(args.meta1m_dir, "*_trades_1m.parquet"))):
        sym = os.path.basename(p).replace("_trades_1m.parquet","")
        try:
            df = pd.read_parquet(p)
            out = resample_trades_agg(df, tf_alias)
            out.to_parquet(os.path.join(args.out_dir, f"{sym}_trades_{suffix}.parquet"), index=False)
            print(f"[{args.tf}] TRADES {sym} rows={len(out)}")
        except Exception as e:
            print(f"[{args.tf}_ERR] {sym} TRADES {e}")

    # --- ORDERBOOK ---
    for p in sorted(glob.glob(os.path.join(args.meta1m_dir, "*_ob_1m.parquet"))):
        sym = os.path.basename(p).replace("_ob_1m.parquet","")
        try:
            df = pd.read_parquet(p)
            out = resample_ob_agg(df, tf_alias)
            out.to_parquet(os.path.join(args.out_dir, f"{sym}_ob_{suffix}.parquet"), index=False)
            print(f"[{args.tf}] OB {sym} rows={len(out)}")
        except Exception as e:
            print(f"[{args.tf}_ERR] {sym} OB {e}")

    # --- OI/FUNDING ---
    for p in sorted(glob.glob(os.path.join(args.meta1m_dir, "*_oi_funding_1m.parquet"))):
        sym = os.path.basename(p).replace("_oi_funding_1m.parquet","")
        try:
            df = pd.read_parquet(p)
            out = resample_oi_funding_agg(df, tf_alias)
            out.to_parquet(os.path.join(args.out_dir, f"{sym}_oi_funding_{suffix}.parquet"), index=False)
            print(f"[{args.tf}] OI/FUND {sym} rows={len(out)}")
        except Exception as e:
            print(f"[{args.tf}_ERR] {sym} OI/FUND {e}")

    # --- LIQUIDATIONS ---
    for p in sorted(glob.glob(os.path.join(args.meta1m_dir, "*_liquidations.parquet"))):
        sym = os.path.basename(p).replace("_liquidations.parquet","")
        try:
            df = pd.read_parquet(p)
            out = resample_liq_agg(df, tf_alias)
            out.to_parquet(os.path.join(args.out_dir, f"{sym}_liq_{suffix}.parquet"), index=False)
            print(f"[{args.tf}] LIQ {sym} rows={len(out)}")
        except Exception as e:
            print(f"[{args.tf}_ERR] {sym} LIQ {e}")

if __name__ == "__main__":
    main()