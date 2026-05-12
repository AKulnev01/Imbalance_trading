# predict/tp_entry/build_tp_training_base.py
import os, sys, argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- безопасные импорты фич/реземпла ---
try:
    from predict.tp_entry.features_shared import build_4h_features, resample_4h
except Exception:
    import importlib.util
    _CUR = os.path.abspath(os.path.dirname(__file__))
    _FEAT = os.path.join(_CUR, "features_shared.py")
    spec = importlib.util.spec_from_file_location("features_shared", _FEAT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    build_4h_features, resample_4h = m.build_4h_features, m.resample_4h

# --- простые утилиты ---
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.index = pd.to_datetime(x.index, utc=True, errors="coerce")
    return x.sort_index()

def load_m1(symbol: str, m1_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.expanduser(m1_dir), f"{symbol}_m1.parquet")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise RuntimeError(f"{symbol}: need ts or timestamp")
    cols = ["open","high","low","close","volume"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.assign(ts=ts).set_index("ts")[cols].dropna()
    return ensure_utc_index(df)

def _symbols_from_m1_dir(m1_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(m1_dir): return out
    for n in os.listdir(m1_dir):
        if n.endswith("_m1.parquet"):
            out.append(n[:-len("_m1.parquet")].upper())
    return sorted(list(dict.fromkeys(out)))

def _resolve_symbols(slist: str, m1_dir: str) -> List[str]:
    if slist and slist.strip():
        return [s.strip().upper() for s in slist.split(",") if s.strip()]
    try:
        from config import TRADE_UNIVERSE, filter_universe
        syms = filter_universe(TRADE_UNIVERSE or []); syms = [s.upper() for s in syms]
        if syms: return syms
    except Exception:
        pass
    return _symbols_from_m1_dir(m1_dir)

def side_of_bar(o: float, c: float) -> str:
    return "BUY" if c >= o else "SELL"

def apply_entry_slip(px: float, side: str, slip_pct: float) -> float:
    return px*(1.0+slip_pct) if side=="BUY" else px*(1.0-slip_pct)

# ---------- ПРЕФИЛЬТР малоинформативных баров ----------
def is_boring_bar(fr: pd.Series,
                  atr_min_pct: float,
                  ema_diff_min: float,
                  body_min: float,
                  volz_min: float) -> bool:
    """
    Считаем бар «скучным», если все признаки слабые одновременно:
    низкая ATR%, почти нет тренда (ema_diff), крошечное тело и отрицат. vol_z.
    """
    close = float(fr.get("close", np.nan))
    atr = float(fr.get("atr14", 0.0))
    atr_pct = (atr / max(close, 1e-12)) if np.isfinite(close) else 0.0

    cond_atr = (atr_pct < atr_min_pct)
    cond_trd = (abs(float(fr.get("ema_diff_pct", 0.0))) < ema_diff_min)
    cond_body = (float(fr.get("body_ratio", 0.0)) < body_min)
    cond_volz = (float(fr.get("vol_z", -1.0)) < volz_min)
    return bool(cond_atr and cond_trd and cond_body and cond_volz)

# ---------- ВЕКТОРНЫЙ РАСЧЁТ FIRST-HIT ----------
def _first_hit_indices_monotone(cum_nondec: np.ndarray,
                                thresholds: np.ndarray) -> np.ndarray:
    """
    cum_nondec: монотонно неубывающий вектор длины T (float32)
    thresholds: пороги (float32), длины K
    Возвращает индексы первого достиж. для каждого порога (K,), либо T (если не достигнут).
    Реализация через сжатие уникальных значений + searchsorted → O(T_unique + K).
    """
    # unique по возрастанию, возвращаем индексы первых появлений
    vals, first_idx = np.unique(cum_nondec, return_index=True)
    # поиск позиций для всех порогов
    pos = np.searchsorted(vals, thresholds, side="left")
    # куда не влезли (порог > max vals) — считаем "не достигнут" → индекс T
    miss = (pos >= len(vals))
    pos = np.clip(pos, 0, max(len(vals)-1, 0))
    out = first_idx[pos]
    if miss.any():
        out = out.astype(np.int32, copy=True)
        out[miss] = len(cum_nondec)  # T = «не достигнут»
    return out.astype(np.int32, copy=False)

def evaluate_grid_fast(m1: pd.DataFrame, side: str,
                       entry_ts: pd.Timestamp, entry_px: float,
                       ttl_h: int, slip_pct: float,
                       tp_grid: np.ndarray, sl_grid: np.ndarray,
                       tie_break: str = "sl") -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает пары (tp_first_idx, sl_first_idx) длины |grid| с индексом первой минуты,
    или T (длина окна), если порог не достигнут.
    """
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = m1.loc[(m1.index >= entry_ts) & (m1.index <= t_end)]
    if m.empty:
        T = 0
        return np.full(tp_grid.size, T, np.int32), np.full(sl_grid.size, T, np.int32)

    hi = m["high"].to_numpy(np.float32, copy=False)
    lo = m["low" ].to_numpy(np.float32, copy=False)
    T = hi.shape[0]

    if side == "BUY":
        # приросты относительно entry
        # max up: (high/entry - 1)  неубывающий cummax
        up  = (hi / entry_px) - 1.0
        up  = np.maximum.accumulate(up).astype(np.float32, copy=False)
        # max down в % для SL: (entry/low - 1)  неубывающий cummax
        dn  = (entry_px / np.maximum(lo, 1e-12)) - 1.0
        dn  = np.maximum.accumulate(dn).astype(np.float32, copy=False)
        tp_idx = _first_hit_indices_monotone(up, tp_grid.astype(np.float32))
        sl_idx = _first_hit_indices_monotone(dn, sl_grid.astype(np.float32))
    else:
        # SELL: прибыль растёт, если цена падает
        # upSELL = (entry/high - 1) — рост прибыли при падении high ниже entry
        up  = (entry_px / np.maximum(hi, 1e-12)) - 1.0
        up  = np.maximum.accumulate(up).astype(np.float32, copy=False)
        # dnSELL = (low/entry - 1) со знаком «вверх» для SL
        dn  = (np.maximum(lo, 1e-12) / entry_px) - 1.0
        dn  = -dn  # делаем положит. метрику падения в убыток
        dn  = np.maximum.accumulate(dn).astype(np.float32, copy=False)
        tp_idx = _first_hit_indices_monotone(up, tp_grid.astype(np.float32))
        sl_idx = _first_hit_indices_monotone(dn, sl_grid.astype(np.float32))

    return tp_idx, sl_idx

import numpy as np

def pick_best_from_indices(tp_idx, sl_idx, tp_grid, sl_grid, tie_break: str,
                           objective: str, min_rr: float):
    """
    tp_idx: array-like длиной T (минутный индекс срабатывания TP, np.inf если не сработал)
    sl_idx: array-like длиной S (минутный индекс срабатывания SL, np.inf если не сработал)
    tp_grid: shape (T,), проценты TP (напр. 0.02..0.35)
    sl_grid: shape (S,), проценты SL (напр. 0.01..0.28)
    """
    tp_idx = np.asarray(tp_idx, dtype=float)  # shape (T,)
    sl_idx = np.asarray(sl_idx, dtype=float)  # shape (S,)
    T, S = tp_idx.shape[0], sl_idx.shape[0]

    # матрицы T×S попарных сравнений
    tp_m = tp_idx[:, None]            # (T,1)
    sl_m = sl_idx[None, :]            # (1,S)

    # кто раньше?
    tp_before = tp_m < sl_m           # (T,S) True, если TP раньше SL
    sl_before = sl_m < tp_m           # (T,S) True, если SL раньше TP

    # обработка случая "оба не сработали" и "оба в один тик"
    is_tp_hit = np.isfinite(tp_m)     # (T,1)
    is_sl_hit = np.isfinite(sl_m)     # (1,S)
    both_inf  = (~is_tp_hit) & (~is_sl_hit)  # (T,S) → таймаут
    same_tick = (tp_m == sl_m) & is_tp_hit & is_sl_hit

    if tie_break == "tp":
        tp_before = tp_before | same_tick
    elif tie_break == "sl":
        sl_before = sl_before | same_tick
    # tie_break == "skip": ничего не добавляем, такие пары останутся «ни TP, ни SL»

    timeout = both_inf | (~tp_before & ~sl_before)

    # метрики по сетке
    tp_rate = tp_before.astype(np.float32)     # (T,S)
    sl_rate = sl_before.astype(np.float32)     # (T,S)
    to_rate = timeout.astype(np.float32)       # (T,S)

    # RR и фильтр по RR
    rr = tp_grid[:, None] / np.maximum(sl_grid[None, :], 1e-12)   # (T,S)
    rr_mask = rr >= float(min_rr)

    # ожидаемый PnL (в «процентах движения»; комиссии добавишь при желании)
    exp_pnl = tp_rate * tp_grid[:, None] - sl_rate * sl_grid[None, :]

    # суррогат F1: сбалансировать «TP раньше SL»
    f1_like = (2.0 * tp_rate) / (2.0 * tp_rate + sl_rate + to_rate + 1e-12)

    if objective == "exp_pnl":
        score = np.where(rr_mask, exp_pnl, -np.inf)
    elif objective == "f1_like":
        score = np.where(rr_mask, f1_like, -np.inf)
    else:  # "tp_rate_rr" — поощряем TP rate и RR одновременно
        score = np.where(rr_mask, tp_rate * np.log1p(rr), -np.inf)

    if not np.isfinite(score).any():
        return None

    ti, si = np.unravel_index(np.nanargmax(score), score.shape)
    return {
        "best_tp_pct": float(tp_grid[ti]),
        "best_sl_pct": float(sl_grid[si]),
        "rr":          float(rr[ti, si]),
        "tp_rate":     float(tp_rate[ti, si]),
        "sl_rate":     float(sl_rate[ti, si]),
        "timeout_rate":float(to_rate[ti, si]),
        "exp_pnl":     float(exp_pnl[ti, si]),
        "objective_used": objective,
        "ti": int(ti), "si": int(si),
    }

# ---------- обработка одного символа (для параллели) ----------
def process_symbol(args_tuple):
    (sym, m1_dir, lookback_days, ttl_hours, slippage_pct, min_4h_bars,
     tp_grid, sl_grid, objective, min_rr, atr_min_pct, ema_diff_min, body_min, volz_min) = args_tuple

    m1 = load_m1(sym, m1_dir)
    if m1.empty: return sym, []

    if lookback_days and lookback_days > 0:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_days))
        m1 = m1[m1.index >= cutoff]
    if m1.empty: return sym, []

    h4 = resample_4h(m1)
    if h4.empty or len(h4) < int(min_4h_bars): return sym, []

    feats = build_4h_features(h4)
    rows = []

    for t_open, r in h4.iterrows():
        try:
            fr = feats.loc[t_open]
        except KeyError:
            continue

        # префильтр «скучных» баров
        if is_boring_bar(fr, atr_min_pct, ema_diff_min, body_min, volz_min):
            continue

        side = side_of_bar(float(r["open"]), float(r["close"]))
        entry_ts = t_open + pd.Timedelta(hours=4)
        entry_ref = float(r["close"])
        entry_px = apply_entry_slip(entry_ref, side, float(slippage_pct))

        # быстрая оценка first-hit для всей сетки
        tp_idx, sl_idx = evaluate_grid_fast(
            m1, side, entry_ts, entry_px, int(ttl_hours), float(slippage_pct),
            tp_grid, sl_grid, tie_break="sl"
        )
        best = pick_best_from_indices(tp_idx, sl_idx, tp_grid, sl_grid, "sl", objective, float(min_rr))
        if best is None:
            continue

        row = {
            "symbol": sym,
            "time_open": pd.Timestamp(t_open).tz_localize("UTC") if getattr(t_open, "tzinfo", None) is None else t_open.tz_convert("UTC"),
            "time_close": entry_ts,
            "side": side,
            "best_tp_pct": best["best_tp_pct"],
            "best_sl_pct": best["best_sl_pct"],
            "best_rr": best["rr"],
            "best_tp_rate": best["tp_rate"],
            "best_sl_rate": best["sl_rate"],
            "best_exp_pnl": best.get("exp_pnl", np.nan),
            "objective_used": best["objective_used"],
        }
        for c in feats.columns:
            v = fr[c]
            row[c] = float(v) if pd.notna(v) else np.nan

        rows.append(row)

    return sym, rows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build training base with outcomes + best TP/SL per bar (FAST)")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--lookback-days", type=int, default=720)
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--slippage-pct", type=float, default=0.004)
    ap.add_argument("--min-4h-bars", type=int, default=60)
    ap.add_argument("--tp-grid", default="0.02:0.35:34")   # start:stop:steps
    ap.add_argument("--sl-grid", default="0.01:0.28:28")
    ap.add_argument("--objective", choices=["exp_pnl","f1_like","tp_rate_rr"], default="exp_pnl")
    ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count())))
    ap.add_argument("--fmt", choices=["parquet","csv","xlsx"], default="parquet")

    # префильтр пороги
    ap.add_argument("--atr-min-pct", type=float, default=0.003)
    ap.add_argument("--ema-diff-min", type=float, default=0.0015)
    ap.add_argument("--body-min", type=float, default=0.12)
    ap.add_argument("--volz-min", type=float, default=0.0)

    ap.add_argument("--out", default="./predict/tp_entry/tp_training_base.parquet")
    args = ap.parse_args()

    def _lin_from(s: str):
        a, b, n = s.split(":"); return np.linspace(float(a), float(b), int(n), dtype=np.float32)

    tp_grid = _lin_from(args.tp_grid)
    sl_grid = _lin_from(args.sl_grid)
    symbols = _resolve_symbols(args.symbols, args.m1_dir)
    if not symbols:
        print("⚠️ no symbols found"); return

    print(f"🧩 Building training base for {len(symbols)} symbols (workers={args.workers})")
    for i, s in enumerate(symbols, 1):
        print(f"   [{i}/{len(symbols)}] {s}")
    sys.stdout.flush()

    tasks = [(s, args.m1_dir, args.lookback_days, args.ttl_hours, args.slippage_pct,
              args.min_4h_bars, tp_grid, sl_grid, args.objective, args.min_rr,
              args.atr_min_pct, args.ema_diff_min, args.body_min, args.volz_min) for s in symbols]

    all_rows = []
    if args.workers == 1:
        for k, t in enumerate(tasks, 1):
            sym, rows = process_symbol(t)
            print(f"[{k}/{len(tasks)}] {sym}: +{len(rows)} rows"); sys.stdout.flush()
            all_rows.extend(rows)
    else:
        with Pool(processes=int(args.workers)) as pool:
            for k, (sym, rows) in enumerate(pool.imap_unordered(process_symbol, tasks), 1):
                print(f"[{k}/{len(tasks)}] {sym}: +{len(rows)} rows"); sys.stdout.flush()
                all_rows.extend(rows)

    if not all_rows:
        print("⚠️ no rows produced"); return

    out = pd.DataFrame(all_rows).sort_values(["time_open","symbol"]).reset_index(drop=True)
    for c in ("time_open","time_close"):
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.tz_localize(None)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base, _ = os.path.splitext(args.out)

    if args.fmt == "parquet":
        path = base + ".parquet"
        out.to_parquet(path, index=False)
        print(f"✅ saved: {path}  rows={len(out)} symbols={out['symbol'].nunique()}")
    elif args.fmt == "csv":
        path = base + ".csv"
        out.to_csv(path, index=False)
        print(f"✅ saved: {path}  rows={len(out)} symbols={out['symbol'].nunique()}")
    else:
        path = base + ".xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as wr:
            out.to_excel(wr, index=False, sheet_name="base")
        print(f"✅ saved: {path}  rows={len(out)} symbols={out['symbol'].nunique()}")

if __name__ == "__main__":
    main()