#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Фильтр/обогащение сигналов имбалансов индикаторами 4h:
- OB/OS (RSI / Stoch / WPR / CCI)
- Свинг-диапазон (пивоты) + Fibo-сетка и целевой TP по индексу
- Мягкий OB/OS-фильтр: помечает skip_reason (не выкидывает строку)
- Ослабление фильтра по объёму: только метрики volume, без жёсткого отсечения

Пример:
  python tools/filter_signals.py \
    --in ./data/signals/signals_var.xlsx \
    --out ./data/signals/signals_var_filtered.xlsx \
    --use-obos 1 --use-fib 1 --policy-obos filter_entry \
    --fib-lookback 120 --fib-pivot-len 3 --fib-tp-index 3
"""

import os
import math
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# -------------------- попытка использовать проектные загрузчики --------------------
def _try_get_klines_4h(symbol: str, lookback_days: int = 360) -> pd.DataFrame:
    """
    Сначала пробуем utils.strategy.get_klines_4h; если не получилось —
    ресемплим локальные минутки из LTF_ROOT.
    """
    try:
        from utils.strategy import get_klines_4h  # type: ignore
        df = get_klines_4h(symbol, lookback_days=lookback_days, interval="4h")
        if df is not None and not df.empty:
            # убедимся в правильной форме
            out = df.copy()
            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
            else:
                # стратегия возвращает naive как UTC → поднимем в UTC, потом при записи урони́м
                if out.index.tz is None:
                    out.index = out.index.tz_localize("UTC")
            out = out.dropna(subset=["open","high","low","close"]).sort_index()
            return out
    except Exception:
        pass

    # Фолбэк: ресемпл из минуток
    ltf_root = os.getenv("LTF_ROOT", "./data/m1")
    p = Path(ltf_root) / f"{symbol.upper()}_m1.parquet"
    if not p.exists():
        p_alt = p.with_name(f"{symbol.upper()}.parquet")
        if not p_alt.exists():
            raise FileNotFoundError(f"no local minutes for {symbol}: {p}")
        p = p_alt

    m1 = pd.read_parquet(p)
    # нормализуем время
    candidates = ["time","timestamp","open_time","datetime","date","ts"]
    time_col = next((c for c in candidates if c in m1.columns), None)
    if time_col is None:
        raise ValueError(f"no time column in {p}")
    t = m1[time_col]
    if np.issubdtype(t.dtype, np.number):
        mx = float(t.max()) if len(t) else 0.0
        idx = pd.to_datetime(t, unit=("ms" if mx > 1e12 else "s"), utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(t, utc=True, errors="coerce")
    m1 = m1.assign(__t=idx).dropna(subset=["__t"]).set_index("__t").sort_index()
    # ensure numeric
    for c in ("open","high","low","close","volume"):
        if c in m1.columns:
            m1[c] = pd.to_numeric(m1[c], errors="coerce")
    m1 = m1.dropna(subset=["open","high","low","close"])

    # lookback
    if lookback_days and len(m1) > 0:
        cutoff = m1.index.max() - pd.Timedelta(days=int(lookback_days))
        m1 = m1[m1.index >= cutoff]

    o = m1["open"].resample("4h", label="left", closed="left").first()
    h = m1["high"].resample("4h").max()
    l = m1["low"].resample("4h").min()
    c = m1["close"].resample("4h").last()
    df4h = pd.concat([o,h,l,c], axis=1)
    if "volume" in m1.columns:
        df4h["volume"] = m1["volume"].resample("4h").sum()
    df4h = df4h.dropna().sort_index()
    df4h.index.name = "time"
    return df4h

# -------------------- индикаторы / утилиты --------------------
def _rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0.0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _stoch(df: pd.DataFrame, k=14, d=3, sma=3):
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    k_raw = (df["close"] - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k_sma = k_raw.rolling(sma, min_periods=sma).mean()
    d_sma = k_sma.rolling(d, min_periods=d).mean()
    return k_sma, d_sma

def _wpr(df: pd.DataFrame, n=14):
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return (df["close"] - hh) / (hh - ll).replace(0, np.nan) * 100.0  # ~[-100..0], OB>-20, OS<-80

def _cci(df: pd.DataFrame, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def _find_swing_range(df4h: pd.DataFrame, t0: pd.Timestamp, lookback: int, piv_len: int):
    bars = df4h[df4h.index < t0].tail(lookback).copy()
    if bars.empty:
        return (np.nan, np.nan, pd.NaT, pd.NaT)
    # простые пивоты
    win = piv_len * 2 + 1
    hi = bars["high"].rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len]==max(a) else 0.0, raw=True)
    lo = bars["low"].rolling(win, center=True).apply(lambda a: 1.0 if a[piv_len]==min(a) else 0.0, raw=True)
    pv_hi = bars[hi==1.0]
    pv_lo = bars[lo==1.0]
    if pv_hi.empty or pv_lo.empty:
        L = float(bars["low"].min()); H = float(bars["high"].max())
        return (L, H, bars["low"].idxmin(), bars["high"].idxmax())
    last_hi = pv_hi.iloc[-1]; last_lo = pv_lo.iloc[-1]
    if last_hi.name > last_lo.name:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    else:
        L, Lt = float(pv_lo["low"].iloc[-1]), pv_lo.index[-1]
        H, Ht = float(pv_hi["high"].iloc[-1]), pv_hi.index[-1]
    return (L, H, Lt, Ht)

def _build_fib_levels(L: float, H: float, fib_set: List[float], side: str):
    if not np.isfinite(L) or not np.isfinite(H) or L<=0 or H<=0 or H<=L:
        return []
    if side=="BUY":
        return [L + (H-L)*r for r in fib_set]
    else:
        return [H - (H-L)*r for r in fib_set]

def _choose_fib_tp(levels: list, idx: int) -> Tuple[float, int]:
    if not levels: return (np.nan, -1)
    i = max(0, min(len(levels)-1, idx))
    return (float(levels[i]), i)

def _calc_obos(df4h: pd.DataFrame, at: pd.Timestamp, obos_type: str,
               rsi_ob: float, rsi_os: float,
               st_k: int, st_d: int, st_sma: int,
               st_ob: float, st_os: float,
               wpr_ob: float, wpr_os: float,
               cci_ob: float, cci_os: float):
    df = df4h[df4h.index <= at].copy()
    if df.empty:
        return ("none", np.nan)
    t = str(obos_type).lower().strip()
    if t=="rsi":
        r = _rsi(df["close"], 14).iloc[-1]
        if r>=rsi_ob: return ("ob", float(r))
        if r<=rsi_os: return ("os", float(r))
        return ("none", float(r))
    if t=="stoch":
        k, d = _stoch(df, st_k, st_d, st_sma)
        v = float(k.iloc[-1])
        if v>=st_ob: return ("ob", v)
        if v<=st_os: return ("os", v)
        return ("none", v)
    if t=="wpr":
        v = float(_wpr(df, st_k).iloc[-1])  # используем st_k как окно
        if v>=wpr_ob: return ("ob", v)
        if v<=wpr_os: return ("os", v)
        return ("none", v)
    if t=="cci":
        v = float(_cci(df, 20).iloc[-1])
        if v>=cci_ob: return ("ob", v)
        if v<=cci_os: return ("os", v)
        return ("none", v)
    return ("none", np.nan)

def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY","LONG"): return "BUY"
    if s in ("SELL","SHORT"): return "SELL"
    return s

# -------------------- основной скрипт --------------------
def read_signals(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        try:
            head = pd.read_excel(path, nrows=0)
            parse_dates = [c for c in ["imb_time","entry_at"] if c in head.columns]
        except Exception:
            parse_dates = []
        df = pd.read_excel(path, sheet_name="data" if "data" in pd.ExcelFile(path).sheet_names else 0,
                           parse_dates=parse_dates or None)
    else:
        df = pd.read_csv(path, parse_dates=["imb_time"], infer_datetime_format=True)
    # нормализация
    cols = {c.lower(): c for c in df.columns}
    if "side" not in cols and "type" in cols:
        df = df.rename(columns={cols["type"]: "side"})
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df = df[df["side"].isin(["BUY","SELL"])]
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["imb_time"] = pd.to_datetime(df["imb_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol","side","imb_time"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Обогащение сигналов индикаторами 4h + мягкий OB/OS фильтр + Fib TP.")
    ap.add_argument("--in", dest="inp", required=True, help="signals.xlsx/csv (лист 'data')")
    ap.add_argument("--out", dest="out", required=True, help="куда сохранить отфильтрованные сигналы (xlsx)")
    ap.add_argument("--out-parquet", default=None, help="доп. parquet (опц.)")

    # включатели
    ap.add_argument("--use-obos", type=int, default=1, help="1/0 — считать OB/OS и метить ob/os")
    ap.add_argument("--use-fib",  type=int, default=1, help="1/0 — строить Fibo и вычислять fib_tp_4h")
    # политика OB/OS
    ap.add_argument("--policy-obos", choices=["off","filter_entry","tp_bias"], default="filter_entry",
                    help="off: только метки; filter_entry: skip_reason='filtered_obos'; tp_bias: только метка для последующей корректировки TP")

    # параметры индикаторов
    ap.add_argument("--rsi-ob", type=float, default=70.0)
    ap.add_argument("--rsi-os", type=float, default=30.0)
    ap.add_argument("--st-k", type=int, default=14)
    ap.add_argument("--st-d", type=int, default=3)
    ap.add_argument("--st-sma", type=int, default=3)
    ap.add_argument("--st-ob", type=float, default=80.0)
    ap.add_argument("--st-os", type=float, default=20.0)
    ap.add_argument("--wpr-ob", type=float, default=-20.0)
    ap.add_argument("--wpr-os", type=float, default=-80.0)
    ap.add_argument("--cci-ob", type=float, default=100.0)
    ap.add_argument("--cci-os", type=float, default=-100.0)
    ap.add_argument("--obos-type", choices=["off","rsi","stoch","wpr","cci"], default="stoch")

    # Параметры Fibo
    ap.add_argument("--fib-lookback", type=int, default=120)
    ap.add_argument("--fib-pivot-len", type=int, default=3)
    ap.add_argument("--fib-set", default="0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618")
    ap.add_argument("--fib-tp-index", type=int, default=3)

    # lookback для загрузки 4h
    ap.add_argument("--lookback-days", type=int, default=360)

    args = ap.parse_args()

    sigs = read_signals(os.path.expanduser(args.inp))
    if sigs.empty:
        print("⚠️ Входной файл не содержит валидных сигналов.")
        return

    print(f"rows in: {len(sigs)}  | symbols: {sigs['symbol'].nunique()}")

    fib_set = [float(x) for x in str(args.fib_set).split(",") if str(x).strip()]

    out_rows = []
    # заранее сгруппируем по символам и загрузим df4h один раз на символ
    by_sym = sigs.groupby("symbol")
    for sym, g in by_sym:
        try:
            df4h = _try_get_klines_4h(sym, lookback_days=int(args.lookback_days))
        except Exception as e:
            print(f"⚠️ {sym}: load 4h failed: {e}")
            df4h = pd.DataFrame()

        # подготовим объёмные метрики
        if not df4h.empty:
            if df4h.index.tz is None:
                df4h.index = df4h.index.tz_localize("UTC")
            vol_present = "volume" in df4h.columns
            vol_ma = df4h["volume"].rolling(20, min_periods=1).mean() if vol_present else None
        else:
            vol_present = False
            vol_ma = None

        for _, r in g.iterrows():
            side = _normalize_side(r["side"])
            t0 = pd.to_datetime(r["imb_time"], utc=True, errors="coerce")
            if pd.isna(t0) or df4h is None or df4h.empty:
                out = r.to_dict()
                out.update({
                    "obos_flag": "none", "obos_value": np.nan,
                    "fib_anchor_L": np.nan, "fib_anchor_H": np.nan,
                    "fib_tp_4h": np.nan, "fib_tp_index": -1,
                    "vol_avg4h": np.nan, "vol_at_entry": np.nan, "vol_rel": np.nan,
                    "skip_reason": "no_df4h",
                })
                out_rows.append(out)
                continue

            # бар входа на 4h (свеча, которая закрывается в t0)
            pos = df4h.index.searchsorted(t0)
            ix = pos - 1
            entry_bar = df4h.iloc[ix] if ix >= 0 else None

            # объемные метрики (мягкие)
            if entry_bar is not None and vol_present:
                v = float(entry_bar.get("volume", np.nan))
                vavg = float(vol_ma.iloc[ix]) if vol_ma is not None else np.nan
                vol_rel = (v / vavg) if (np.isfinite(v) and np.isfinite(vavg) and vavg>0) else np.nan
            else:
                v = np.nan; vavg = np.nan; vol_rel = np.nan

            # OB/OS
            obos_flag, obos_val = ("none", np.nan)
            if args.use_obos and args.obos_type != "off":
                obos_flag, obos_val = _calc_obos(
                    df4h, t0, args.obos_type,
                    args.rsi_ob, args.rsi_os,
                    args.st_k, args.st_d, args.st_sma,
                    args.st_ob, args.st_os,
                    args.wpr_ob, args.wpr_os,
                    args.cci_ob, args.cci_os,
                )

            # Fibo
            fib_L, fib_H, fib_tp, fib_idx = (np.nan, np.nan, np.nan, -1)
            if args.use_fib:
                L, H, Lt, Ht = _find_swing_range(df4h, t0, int(args.fib_lookback), int(args.fib_pivot_len))
                levels = _build_fib_levels(L, H, fib_set, side)
                tp, i = _choose_fib_tp(levels, int(args.fib_tp_index))
                fib_L, fib_H, fib_tp, fib_idx = (
                    float(L) if np.isfinite(L) else np.nan,
                    float(H) if np.isfinite(H) else np.nan,
                    float(tp) if np.isfinite(tp) else np.nan,
                    int(i),
                )

            # Политика OB/OS: мягкая
            skip_reason = ""
            if args.policy_obos == "filter_entry" and obos_flag in ("ob","os"):
                # жёстко против стороны:
                if (side=="BUY" and obos_flag=="ob") or (side=="SELL" and obos_flag=="os"):
                    skip_reason = "filtered_obos"

            out = r.to_dict()
            out.update({
                "obos_type": args.obos_type,
                "obos_flag": obos_flag,
                "obos_value": float(obos_val) if np.isfinite(obos_val) else np.nan,
                "fib_anchor_L": fib_L,
                "fib_anchor_H": fib_H,
                "fib_tp_4h": fib_tp,
                "fib_tp_index": fib_idx,
                "vol_avg4h": vavg,
                "vol_at_entry": v,
                "vol_rel": vol_rel,
                "skip_reason": skip_reason or "",
            })
            out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    # Excel-friendly: убрать tz
    for c in [col for col in ["imb_time","entry_at"] if col in out_df.columns]:
        s = pd.to_datetime(out_df[c], errors="coerce", utc=True)
        out_df[c] = s.dt.tz_convert(None)

    # Запись
    out_path = os.path.expanduser(args.out)
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as wr:
        out_df.to_excel(wr, index=False, sheet_name="data")
        meta = pd.DataFrame({
            "param": ["use_obos","obos_type","policy_obos","use_fib","fib_lookback","fib_pivot_len","fib_set","fib_tp_index","rows"],
            "value": [args.use_obos, args.obos_type, args.policy_obos, args.use_fib,
                      args.fib_lookback, args.fib_pivot_len, args.fib_set, args.fib_tp_index, len(out_df)]
        })
        meta.to_excel(wr, index=False, sheet_name="meta")

    if args.out_parquet:
        pq = os.path.expanduser(args.out_parquet)
        Path(Path(pq).parent).mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(pq, index=False)

    print(f"✅ saved → {out_path} (rows={len(out_df)})" + (f"  | parquet: {args.out_parquet}" if args.out_parquet else ""))

if __name__ == "__main__":
    main()