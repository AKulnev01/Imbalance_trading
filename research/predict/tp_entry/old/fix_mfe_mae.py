import os, sys, argparse, math
import numpy as np
import pandas as pd
from typing import Tuple, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ensure_utc(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return t
    try:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except Exception:
        return pd.Timestamp(t).tz_localize("UTC")


def apply_entry_slip(px: float, side: str, slip_pct: float) -> float:
    return px * (1 + slip_pct) if side == "BUY" else px * (1 - slip_pct)


def load_m1(symbol: str, m1_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.expanduser(m1_dir), f"{symbol}_m1.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise RuntimeError(f"{symbol}: need ts or timestamp")
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.assign(ts=ts).set_index("ts").sort_index()
    return df[["open", "high", "low", "close", "volume"]].dropna()


def compute_mfe_mae_window(m1: pd.DataFrame, side: str,
                           entry_ts: pd.Timestamp, entry_px: float,
                           ttl_hours: int) -> Tuple[Optional[float], Optional[float]]:
    """Возвращает (mfe_pct, mae_pct) в процентах относительно entry_px."""
    if m1.empty or pd.isna(entry_ts) or not np.isfinite(entry_px) or entry_px <= 0:
        return (np.nan, np.nan)

    t_end = entry_ts + pd.Timedelta(hours=int(ttl_hours))
    w = m1[(m1.index >= entry_ts) & (m1.index <= t_end)]
    if w.empty:
        return (np.nan, np.nan)

    hi = float(w["high"].max())
    lo = float(w["low"].min())

    if side == "BUY":
        mfe = (hi - entry_px) / entry_px * 100.0
        mae = (lo - entry_px) / entry_px * 100.0
    else:
        mfe = (entry_px - lo) / entry_px * 100.0
        mae = (entry_px - hi) / entry_px * 100.0
    return (mfe, mae)


def recompute_for_symbol(df_sym: pd.DataFrame, m1: pd.DataFrame,
                         ttl_hours: int, slip_pct: float) -> pd.DataFrame:
    """Пересчитывает MFE/MAE для всех строк одного символа."""
    if m1.empty or df_sym.empty:
        df_sym[["mfe_pct", "mae_pct"]] = np.nan
        return df_sym

    out_mfe, out_mae = [], []
    for _, row in df_sym.iterrows():
        side = str(row.get("side", "BUY")).upper()
        t_open = ensure_utc(row.get("time_open"))
        if pd.isna(t_open):
            out_mfe.append(np.nan); out_mae.append(np.nan); continue

        entry_ts = t_open + pd.Timedelta(hours=4)
        m_win = m1[m1.index <= entry_ts]
        if m_win.empty:
            out_mfe.append(np.nan); out_mae.append(np.nan); continue

        entry_ref = float(m_win.iloc[-1]["close"])
        entry_px = apply_entry_slip(entry_ref, side, slip_pct)
        mfe, mae = compute_mfe_mae_window(m1, side, entry_ts, entry_px, ttl_hours)
        out_mfe.append(mfe); out_mae.append(mae)

    df_sym = df_sym.copy()
    df_sym["mfe_pct"] = out_mfe
    df_sym["mae_pct"] = out_mae
    return df_sym


def main():
    ap = argparse.ArgumentParser(description="Fix/compute MFE/MAE (resume-friendly).")
    ap.add_argument("--in", dest="inp", required=True, help="input parquet with base rows")
    ap.add_argument("--out", dest="out", required=True, help="output parquet")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--slippage-pct", type=float, default=0.004)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--mfe-threshold", type=float, default=20.0,
                    help="Порог (в процентах), выше которого пересчитываем (по модулю).")
    args = ap.parse_args()

    base = pd.read_parquet(args.inp)
    base["symbol"] = base["symbol"].astype(str)
    base["time_open"] = pd.to_datetime(base["time_open"], utc=True, errors="coerce").dt.tz_localize(None)

    if "mfe_pct" not in base.columns or "mae_pct" not in base.columns:
        base["mfe_pct"] = np.nan
        base["mae_pct"] = np.nan

    thr = float(args.mfe_threshold)
    mask_nan = base["mfe_pct"].isna() | base["mae_pct"].isna()
    mask_thr = (base["mfe_pct"].abs() > thr) | (base["mae_pct"].abs() > thr)
    mask_absurd = (base["mfe_pct"].abs() > 2000) | (base["mae_pct"].abs() > 2000)
    need = base[mask_nan | mask_thr | mask_absurd].copy()

    if need.empty:
        print("Nothing to fix. Copying input → output.")
        base.to_parquet(args.out, index=False)
        return

    part_path = args.out + ".part"
    done_keys = set()

    if args.resume and os.path.exists(part_path):
        part = pd.read_parquet(part_path)
        # гарантируем нужные колонки даже у пустого part
        for col in base.columns:
            if col not in part.columns:
                part[col] = pd.Series(dtype=base[col].dtype)
        part["symbol"] = part["symbol"].astype(str)
        part["time_open"] = pd.to_datetime(part["time_open"], utc=True, errors="coerce").dt.tz_localize(None)
        done_keys = set(zip(part["symbol"], part["time_open"]))
        print(f"Resuming from part: already {len(done_keys)} rows computed.")
    else:
        # создаём пустой part с правильной схемой
        empty = pd.DataFrame(columns=base.columns)
        empty.to_parquet(part_path, index=False)

    mask_done = need.apply(lambda r: (r["symbol"], r["time_open"]) in done_keys, axis=1)
    need = need[~mask_done]
    if need.empty:
        print("All rows already computed in .part. Finalizing…")
        final = base.drop(columns=["mfe_pct", "mae_pct"], errors="ignore").merge(
            pd.read_parquet(part_path)[["symbol", "time_open", "mfe_pct", "mae_pct"]],
            on=["symbol", "time_open"], how="left")
        final.to_parquet(args.out, index=False)
        print(f"Saved: {args.out}")
        return

    from math import ceil
    total = len(need)
    batches = ceil(total / args.batch_size)
    print(f"To compute: {total} rows → {batches} batches (threshold={thr}%)")

    for bi in range(batches):
        beg, end = bi * args.batch_size, min((bi + 1) * args.batch_size, total)
        chunk = need.iloc[beg:end].copy()

        upd_list = []
        for sym, grp in chunk.groupby("symbol"):
            m1 = load_m1(sym, args.m1_dir)
            if m1.empty:
                g = grp.copy()
                g["mfe_pct"] = np.nan; g["mae_pct"] = np.nan
                upd_list.append(g); continue
            upd_list.append(recompute_for_symbol(grp, m1, args.ttl_hours, args.slippage_pct))

        upd = pd.concat(upd_list, ignore_index=True)
        upd["symbol"] = upd["symbol"].astype(str)
        upd["time_open"] = pd.to_datetime(upd["time_open"], utc=True, errors="coerce").dt.tz_localize(None)

        part = pd.read_parquet(part_path)
        for col in base.columns:
            if col not in part.columns:
                part[col] = pd.Series(dtype=base[col].dtype)
        part["symbol"] = part["symbol"].astype(str)
        part["time_open"] = pd.to_datetime(part["time_open"], utc=True, errors="coerce").dt.tz_localize(None)

        key = ["symbol", "time_open"]
        part = part[~part.set_index(key).index.isin(upd.set_index(key).index)]
        part = pd.concat([part, upd[part.columns.intersection(upd.columns)]], ignore_index=True)
        part.to_parquet(part_path, index=False)

        print(f"[{bi + 1}/{batches}] saved → {part_path} (+{len(upd)} rows)")

    part = pd.read_parquet(part_path)
    part["symbol"] = part["symbol"].astype(str)
    part["time_open"] = pd.to_datetime(part["time_open"], utc=True, errors="coerce").dt.tz_localize(None)

    final = base.drop(columns=["mfe_pct", "mae_pct"], errors="ignore").merge(
        part[["symbol", "time_open", "mfe_pct", "mae_pct"]],
        on=["symbol", "time_open"], how="left"
    )
    final.to_parquet(args.out, index=False)

    print(f"✅ Done! Saved → {args.out}")
    print(f"Rows={len(final)}, symbols={final['symbol'].nunique()}")
    print(f"Part kept at: {part_path} (resume support)")


if __name__ == "__main__":
    main()