# scripts/rollup_any_tf.py
import os, sys, argparse, pandas as pd

# Verbose logging for debugging rollup
ROLLUP_VERBOSE = os.getenv("ROLLUP_VERBOSE", "0").strip().lower() in ("1","true","yes","y","on")

def _dbg(*a, **k):
    if ROLLUP_VERBOSE:
        print(*a, **k)

def _df_info(df: pd.DataFrame):
    if df is None:
        return "none"
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            left = str(df.index.min())
            right = str(df.index.max())
            return f"rows={len(df)} range=[{left}..{right}]"
        except Exception:
            return f"rows={len(df)}"
    return "empty"
from typing import Optional, List, Dict

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_dt_utc(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        df = df.set_index(ts_col).sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    else:
        raise ValueError("Нет колонки ts и индекс не DatetimeIndex")
    return df

def _resample_rule(tf: str) -> str:
    """
    Поддержка: '5m','15m','30m','1h','4h','1d' и т.п. (pandas offset alias)
    """
    return tf.lower()

def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

def _agg_trades(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # ожидаемые колонки: trades, vwap, vol_buy, vol_sell, delta, last, symbol
    df = df.copy()
    vol = (df["vol_buy"].fillna(0.0) + df["vol_sell"].fillna(0.0)).rename("vol_tot")
    vwap_num = (df["vwap"].fillna(0.0) * vol).rename("vwap_num")
    to_res = pd.concat([df[["trades","vol_buy","vol_sell","delta","last"]].fillna(0.0), vwap_num, vol], axis=1)

    g = to_res.resample(_resample_rule(tf), label="right", closed="right", origin="start_day").agg({
        "trades":"sum",
        "vol_buy":"sum",
        "vol_sell":"sum",
        "delta":"sum",
        "last":"last",
        "vwap_num":"sum",
        "vol_tot":"sum"
    })
    g["vwap"] = g.apply(lambda r: (r["vwap_num"]/r["vol_tot"]) if r["vol_tot"]>0 else None, axis=1)
    g = g.drop(columns=["vwap_num","vol_tot"])
    return g

def _agg_ob(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # ожидаемые: bid_vol, ask_vol, imbalance, bid_vwap, ask_vwap, spread_bp
    g = df[["bid_vol","ask_vol","imbalance","bid_vwap","ask_vwap","spread_bp"]].resample(
        _resample_rule(tf), label="right", closed="right", origin="start_day"
    ).mean()
    return g

def _agg_oi_fund(df: pd.DataFrame, tf: str, funding_mode: str="mean") -> pd.DataFrame:
    # ожидаемые: open_interest, funding_rate
    if funding_mode not in ("mean","last"):
        funding_mode = "mean"
    agg = {"open_interest":"last", "funding_rate":("mean" if funding_mode=="mean" else "last")}
    g = df[["open_interest","funding_rate"]].resample(
        _resample_rule(tf), label="right", closed="right", origin="start_day"
    ).agg(agg)
    return g

def _agg_liq(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # ожидаемые: liq_buy_sz, liq_sell_sz, liq_count, liq_avg_px
    # liq_avg_px_TF — простое среднее по минутам, где cnt>0; можно усложнить взвешиванием — сделаем позже при необходимости.
    df = df.copy()
    g_sum = df[["liq_buy_sz","liq_sell_sz","liq_count"]].resample(
        _resample_rule(tf), label="right", closed="right", origin="start_day"
    ).sum()
    g_avg = df["liq_avg_px"].replace(0.0, pd.NA).resample(
        _resample_rule(tf), label="right", closed="right", origin="start_day"
    ).mean()
    g = g_sum
    g["liq_avg_px"] = g_avg
    return g

def _agg_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # ожидаемые: open, high, low, close, volume, trades (если есть)
    cols = [c for c in ["open","high","low","close","volume","trades"] if c in df.columns]
    d = {}
    if "open" in cols:   d["open"]   = "first"
    if "high" in cols:   d["high"]   = "max"
    if "low" in cols:    d["low"]    = "min"
    if "close" in cols:  d["close"]  = "last"
    if "volume" in cols: d["volume"] = "sum"
    if "trades" in cols: d["trades"] = "sum"
    g = df[cols].resample(_resample_rule(tf), label="right", closed="right", origin="start_day").agg(d)
    return g

def process_symbol(sym: str, src_m1_dir: str, src_meta1m_dir: str, out_dir: str, tf: str, funding_mode: str="mean"):
    sym = sym.strip().upper()

    # --- база OHLCV (опционально, если есть свой M1 файл свечей)
    minute_candidates = [
        os.path.join(src_m1_dir, f"{sym}_1m.parquet"),
        os.path.join(src_m1_dir, f"{sym}_m1.parquet"),
        os.path.join(src_m1_dir, f"{sym}_1m.csv"),
        os.path.join(src_m1_dir, f"{sym}_m1.csv"),
    ]
    base = None
    used_path = None
    for p in minute_candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".parquet"):
                    base = _read_parquet(p)
                else:
                    base = pd.read_csv(p)
                used_path = p
                break
            except Exception:
                base = None
                used_path = p
                break
    _dbg(f"[{sym}] minute base: {used_path if used_path else 'not found'}")

    if base is not None and not base.empty:
        # ожидаем ts, open, high, low, close, volume, (trades — опц.)
        base = _to_dt_utc(base, "ts")
        ohlcv_tf = _agg_ohlcv(base, tf)
    else:
        ohlcv_tf = None

    # --- trades
    trades_path = os.path.join(src_meta1m_dir, f"{sym}_trades_1m.parquet")
    trades_tf = None
    if os.path.exists(trades_path):
        tr = _read_parquet(trades_path)
        if tr is not None and not tr.empty:
            tr = _to_dt_utc(tr, "ts")
            trades_tf = _agg_trades(tr, tf)

    # --- orderbook
    ob_path = os.path.join(src_meta1m_dir, f"{sym}_ob_1m.parquet")
    ob_tf = None
    if os.path.exists(ob_path):
        ob = _read_parquet(ob_path)
        if ob is not None and not ob.empty:
            ob = _to_dt_utc(ob, "ts")
            ob_tf = _agg_ob(ob, tf)

    # --- oi/funding
    oi_path = os.path.join(src_meta1m_dir, f"{sym}_oi_funding_1m.parquet")
    oi_tf = None
    if os.path.exists(oi_path):
        oi = _read_parquet(oi_path)
        if oi is not None and not oi.empty:
            oi = _to_dt_utc(oi, "ts")
            oi_tf = _agg_oi_fund(oi, tf, funding_mode=funding_mode)

    # --- liquidations
    liq_path = os.path.join(src_meta1m_dir, f"{sym}_liquidations.parquet")
    liq_tf = None
    if os.path.exists(liq_path):
        lq = _read_parquet(liq_path)
        if lq is not None and not lq.empty:
            lq = _to_dt_utc(lq, "ts")
            liq_tf = _agg_liq(lq, tf)

    _dbg(f"[{sym}] ohlcv_tf: {_df_info(ohlcv_tf)}")
    _dbg(f"[{sym}] trades_tf: {_df_info(trades_tf)}")
    _dbg(f"[{sym}] ob_tf: {_df_info(ob_tf)}")
    _dbg(f"[{sym}] oi_tf: {_df_info(oi_tf)}")
    _dbg(f"[{sym}] liq_tf: {_df_info(liq_tf)}")

    # --- merge (outer join по времени)
    frames = []
    if ohlcv_tf is not None: frames.append(ohlcv_tf)
    if trades_tf is not None: frames.append(trades_tf.add_prefix("tr_"))
    if ob_tf is not None: frames.append(ob_tf.add_prefix("ob_"))
    if oi_tf is not None: frames.append(oi_tf.add_prefix("oi_"))
    if liq_tf is not None: frames.append(liq_tf.add_prefix("liq_"))

    if not frames:
        print(
            f"[WARN] {sym}: нет данных для агрегации — sources: "
            f"ohlcv[{_df_info(ohlcv_tf)}], trades[{_df_info(trades_tf)}], ob[{_df_info(ob_tf)}], "
            f"oi[{_df_info(oi_tf)}], liq[{_df_info(liq_tf)}]"
        )
        return

    out = frames[0].copy()
    for f in frames[1:]:
        out = out.join(f, how="outer")

    out = out.sort_index()
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{sym}_{tf}.parquet")
    out.to_parquet(out_path)
    print(f"[OK] {sym} → {out_path} rows={len(out)}", flush=True)

def _load_universe(args_symbols: Optional[str]) -> List[str]:
    if args_symbols:
        return [s.strip().upper() for s in args_symbols.split(",") if s.strip()]
    # сначала из env
    env = os.getenv("TRADE_UNIVERSE","").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    # потом из config.py
    try:
        sys.path.insert(0, os.getcwd())
        from config import TRADE_UNIVERSE
        return [s.strip().upper() for s in (TRADE_UNIVERSE or [])]
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="CSV-список символов (иначе TRADE_UNIVERSE/.env/.envrc/config.py)")
    ap.add_argument("--tf", required=True, help="цель: 5m,15m,30m,1h,4h,1d и т.д.")
    ap.add_argument("--m1-dir", default="./data/m1", help="путь к M1 свечам (parquet/csv)")
    ap.add_argument("--meta1m-dir", default="./data/meta1m", help="путь к M1 метрикам (наши parquet)")
    ap.add_argument("--out-dir", default="./data/rollup", help="куда сохранять агрегаты")
    ap.add_argument("--funding-mode", choices=["mean","last"], default="mean")
    args = ap.parse_args()

    syms = _load_universe(args.symbols)
    if not syms:
        print("WARN: пустой список символов."); return

    out_tf_dir = os.path.join(args.out_dir, args.tf.lower())
    _ensure_dir(out_tf_dir)

    for s in syms:
        try:
            process_symbol(s, args.m1_dir, args.meta1m_dir, out_tf_dir, args.tf, funding_mode=args.funding_mode)
        except Exception as e:
            print(f"[ERR] {s} {e}", flush=True)

if __name__ == "__main__":
    main()