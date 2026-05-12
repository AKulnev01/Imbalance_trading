# predict/tp_entry/build_tp_dataset.py

import os, sys, argparse, json, math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

# --- sys.path на корень проекта ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- import features helpers (работает и как пакет, и как скрипт) ---
try:
    # когда запускаем как пакет: python -m predict.tp_entry.build_tp_dataset
    from .features_shared import build_4h_features, resample_4h
except Exception:
    # когда запускаем напрямую: python predict/tp_entry/build_tp_dataset.py
    _CUR = os.path.abspath(os.path.dirname(__file__))
    if _CUR not in sys.path:
        sys.path.insert(0, _CUR)
    from features_shared import build_4h_features, resample_4h


# =========================
# TZ / IO helpers
# =========================
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    idx = pd.to_datetime(x.index, utc=True, errors="coerce")
    x.index = idx
    return x.sort_index()


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
        raise RuntimeError(f"{symbol}: parquet must have 'ts' or 'timestamp'")

    df = df.assign(ts=ts).set_index("ts")
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[cols].dropna()
    return ensure_utc_index(df)


# =========================
# Core price helpers
# =========================
def apply_entry_slip(px: float, side: str, slip_pct: float) -> float:
    return px * (1.0 + slip_pct) if side == "BUY" else px * (1.0 - slip_pct)


def side_of_bar(open_px: float, close_px: float) -> str:
    return "BUY" if close_px >= open_px else "SELL"


def resolve_tp_sl(
    df_m1: pd.DataFrame,
    side: str,
    entry_ts: pd.Timestamp,
    tp: float,
    sl: float,
    ttl_h: int,
    tie_break: str = "sl",  # "sl"|"tp"|"skip" — если в одной минуте задеты и TP, и SL
) -> str:
    """
    Возвращает один из: "tp", "sl", "timeout", "none", "skip_both".
    """
    t_end = entry_ts + pd.Timedelta(hours=int(ttl_h))
    m = df_m1[(df_m1.index >= entry_ts) & (df_m1.index <= t_end)]
    if m.empty:
        return "none"

    for _, r in m.iterrows():
        hi, lo = float(r["high"]), float(r["low"])
        hit_tp = (hi >= tp) if side == "BUY" else (lo <= tp)
        hit_sl = (lo <= sl) if side == "BUY" else (hi >= sl)

        if hit_tp and hit_sl:
            if tie_break == "tp":
                return "tp"
            elif tie_break == "sl":
                return "sl"
            else:
                return "skip_both"

        if hit_sl:
            return "sl"
        if hit_tp:
            return "tp"

    return "timeout"


# =========================
# Dynamic TP/SL (ATR + тренд + контекст)
# =========================
def _nz(x, alt=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return alt
        return float(x)
    except Exception:
        return alt


def _tanh(x: float) -> float:
    return math.tanh(x)


def dynamic_tp_sl_pct(
    fr: pd.Series,
    side: str,
    symbol: str = "",
    base_tp_atr: float = 1.6,     # базовый TP в ATR
    base_sl_atr: float = 0.8,     # базовый SL в ATR
    caps: Tuple[float, float] = (0.02, 0.35),   # капы по % (2%..35%)
    coeff: Optional[dict] = None
) -> Tuple[float, float]:
    """
    Возвращает (tp_pct, sl_pct) в долях (0.12 = 12%), адаптированные под текущую свечу.
    Использует: atr14, close, ema_diff_pct, vol_z, body_ratio, wick-и, при наличии hour_of_day, weekday.
    """
    C = {
        "k_tp_trend": 0.50,       # влияние тренда на TP
        "k_sl_trend": -0.20,      # тренд уменьшает SL
        "k_tp_vol":   0.25,       # объем повышает TP
        "k_sl_vol":  -0.10,       # объем немного сжимает SL
        "k_tp_body":  0.25,       # мощное тело → TP↑
        "k_sl_body": -0.10,       # мощное тело → SL↓
        "k_wick_pen": 0.35,       # штраф за «опасную» тень
        "k_night":   -0.10,       # ночь → TP↓
        "k_weekend": -0.08,       # выходные → TP↓
    }
    if coeff:
        C.update(coeff)

    atr14 = _nz(fr.get("atr14"), 0.0)
    close = _nz(fr.get("close"), 0.0)
    if close <= 0 or atr14 <= 0:
        return (0.08, 0.04)

    # волатильностная база
    atr_pct = (atr14 / close)
    tp_pct = base_tp_atr * atr_pct
    sl_pct = base_sl_atr * atr_pct

    # тренд
    ema_diff = _nz(fr.get("ema_diff_pct"), 0.0)
    trend_sig = _tanh(5.0 * ema_diff)      # [-1..1]
    tp_pct *= (1.0 + C["k_tp_trend"] * abs(trend_sig))
    sl_pct *= (1.0 + C["k_sl_trend"] * abs(trend_sig))

    # объём / импульс
    vol_z = _nz(fr.get("vol_z"), 0.0)
    vol_sig = _tanh(0.5 * vol_z)
    tp_pct *= (1.0 + C["k_tp_vol"] * max(0.0, vol_sig))
    sl_pct *= (1.0 + C["k_sl_vol"] * max(0.0, vol_sig))

    body = _nz(fr.get("body_ratio"), 0.0)  # 0..1
    body_sig = (body - 0.5) * 2.0          # ~[-1..1]
    tp_pct *= (1.0 + C["k_tp_body"] * max(0.0, body_sig))
    sl_pct *= (1.0 + C["k_sl_body"] * max(0.0, body_sig))

    # тени: штрафуем TP против «ветра»
    up_w = _nz(fr.get("upper_wick_ratio"), 0.0)
    lo_w = _nz(fr.get("lower_wick_ratio"), 0.0)
    wick_pen = 0.0
    if side == "BUY":
        wick_pen = C["k_wick_pen"] * max(0.0, up_w - 0.3)
    else:
        wick_pen = C["k_wick_pen"] * max(0.0, lo_w - 0.3)
    if wick_pen > 0:
        tp_pct *= max(0.6, 1.0 - wick_pen)
        sl_pct *= min(1.4, 1.0 + 0.5 * wick_pen)

    # контекст по времени (если доступно)
    hour = fr.get("hour_of_day")
    if hour is not None:
        try:
            hour = int(hour)
            if hour < 7 or hour >= 22:
                tp_pct *= (1.0 - 0.10)
        except Exception:
            pass

    wday = fr.get("weekday")
    if wday is not None:
        try:
            wday = int(wday)
            if wday in (5, 6):
                tp_pct *= (1.0 - 0.08)
        except Exception:
            pass

    # клипы диапазонов
    lo_cap, hi_cap = caps
    tp_pct = float(np.clip(tp_pct, lo_cap, hi_cap))
    sl_pct = float(np.clip(sl_pct, lo_cap * 0.5, hi_cap * 0.8))

    return tp_pct, sl_pct


# =========================
# Dataset builder
# =========================
def make_labels_for_symbol(
    sym: str,
    df_m1: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    ttl_h: int,
    slip_pct: float,
    lookback_days: int,
    min_4h_bars: int,
    tie_break: str,
    # динамика
    dynamic_mode: bool = False,
    base_tp_atr: float = 1.6,
    base_sl_atr: float = 0.8,
    caps: Tuple[float, float] = (0.02, 0.35),
    dyn_coeff: Optional[dict] = None,
) -> pd.DataFrame:
    if df_m1.empty:
        return pd.DataFrame()

    if lookback_days and lookback_days > 0:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_days))
        df_m1 = df_m1[df_m1.index >= cutoff]

    df4 = resample_4h(df_m1)
    if df4.empty or len(df4) < int(min_4h_bars):
        return pd.DataFrame()

    feats = build_4h_features(df4)
    rows = []

    for t_open, r in df4.iterrows():
        try:
            side = side_of_bar(float(r["open"]), float(r["close"]))
            entry_ts = t_open + pd.Timedelta(hours=4)
            entry_ref = float(r["close"])
            entry_px = apply_entry_slip(entry_ref, side, slip_pct)

            # выбираем проценты TP/SL для ЭТОГО бара
            if dynamic_mode:
                feat_row = feats.loc[t_open] if t_open in feats.index else None
                if feat_row is None:
                    continue
                tp_pct_bar, sl_pct_bar = dynamic_tp_sl_pct(
                    feat_row, side=side, symbol=sym,
                    base_tp_atr=base_tp_atr, base_sl_atr=base_sl_atr,
                    caps=caps, coeff=dyn_coeff
                )
            else:
                tp_pct_bar, sl_pct_bar = float(tp_pct), float(sl_pct)

            if side == "BUY":
                tp = entry_px * (1.0 + tp_pct_bar)
                sl = entry_px * (1.0 - sl_pct_bar)
            else:
                tp = entry_px * (1.0 - tp_pct_bar)
                sl = entry_px * (1.0 + sl_pct_bar)

            outcome = resolve_tp_sl(df_m1, side, entry_ts, tp, sl, ttl_h, tie_break=tie_break)
            if outcome == "skip_both":  # по желанию можно включить как негатив — сейчас пропускаем
                continue

            y = 1 if outcome == "tp" else 0

            feat_row = feats.loc[t_open] if t_open in feats.index else None
            if feat_row is None:
                continue

            row = {
                "symbol": sym,
                "time_open": t_open,
                "time_close": entry_ts,
                "side": side,
                "label_tp_first": y,
                "outcome": outcome,
                "tp_pct_used": float(tp_pct_bar),
                "sl_pct_used": float(sl_pct_bar),
                "tp_price": float(tp),
                "sl_price": float(sl),
                "entry_price": float(entry_px),
                "entry_ref_close": float(entry_ref),
            }
            # добавить все фичи
            for c in feats.columns:
                v = feat_row[c]
                row[c] = float(v) if pd.notna(v) else np.nan

            rows.append(row)
        except Exception:
            # пропустим редкие больные строки
            continue

    return pd.DataFrame(rows)


def _symbols_from_m1_dir(m1_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(m1_dir):
        return out
    for name in os.listdir(m1_dir):
        if name.endswith("_m1.parquet"):
            out.append(name.replace("_m1.parquet", "").upper())
    return sorted(list(dict.fromkeys(out)))


def _resolve_symbols(user_list: str, m1_dir: str) -> List[str]:
    # 1) если задано --symbols
    if user_list and user_list.strip():
        return [s.strip().upper() for s in user_list.split(",") if s.strip()]

    # 2) пробуем config.TRADE_UNIVERSE
    try:
        from config import TRADE_UNIVERSE, filter_universe
        syms = filter_universe(TRADE_UNIVERSE or [])
        syms = [s.upper() for s in syms]
        if syms:
            return syms
    except Exception:
        pass

    # 3) fallback: взять из m1_dir по маске *_m1.parquet
    syms = _symbols_from_m1_dir(m1_dir)
    return syms


def _drop_tz_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = pd.to_datetime(x[c], utc=True, errors="coerce").dt.tz_localize(None)
    return x


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Build TP-first dataset per 4h bar using m1 TP/SL resolution.")
    ap.add_argument("--m1-dir", default="./data/m1")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--tp-pct", type=float, default=0.135, help="фиксированный TP %, если --dynamic=0")
    ap.add_argument("--sl-pct", type=float, default=0.04,  help="фиксированный SL %, если --dynamic=0")
    ap.add_argument("--slippage-pct", type=float, default=0.004)
    ap.add_argument("--ttl-hours", type=int, default=80)
    ap.add_argument("--lookback-days", type=int, default=720)
    ap.add_argument("--min-4h-bars", type=int, default=60, help="Минимум готовых 4h баров на символ")
    ap.add_argument("--tie-break", choices=["sl", "tp", "skip"], default="sl",
                    help="Если в одной минуте задеты TP и SL: что считать результатом (консервативно sl)")
    ap.add_argument("--out", default="./predict/tp_entry/tp_dataset.xlsx")

    # динамические TP/SL
    ap.add_argument("--dynamic", type=int, default=1, help="1=динамические TP/SL, 0=фиксированные")
    ap.add_argument("--base-tp-atr", type=float, default=1.6, help="база TP в ATR для динамики")
    ap.add_argument("--base-sl-atr", type=float, default=0.8, help="база SL в ATR для динамики")
    ap.add_argument("--dyn-caps", type=str, default="0.02,0.35", help="мин-макс доли для TP (например 0.02,0.35)")
    ap.add_argument("--dyn-coeff", type=str, default="",
                    help="JSON с коэффициентами динамики (переопределит дефолты). Пример: "
                         "'{\"k_tp_trend\":0.6,\"k_wick_pen\":0.4}'")

    args = ap.parse_args()

    # разбор динамических параметров
    dynamic_mode = (int(args.dynamic) == 1)
    try:
        caps_vals = [float(x) for x in (args.dyn_caps or "").split(",")]
        caps = (caps_vals + [0.02, 0.35])[:2]  # подстрахуем длину
        caps = (float(caps[0]), float(caps[1]))
    except Exception:
        caps = (0.02, 0.35)

    dyn_coeff = None
    if args.dyn_coeff:
        try:
            dyn_coeff = json.loads(args.dyn_coeff)
        except Exception:
            dyn_coeff = None

    symbols = _resolve_symbols(args.symbols, args.m1_dir)
    if not symbols:
        print("no symbols; provide --symbols or put *_m1.parquet in --m1-dir or set config.TRADE_UNIVERSE")
        return

    all_rows = []
    for s in symbols:
        m1 = load_m1(s, args.m1_dir)
        if m1.empty:
            continue
        df = make_labels_for_symbol(
            s, m1,
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            ttl_h=int(args.ttl_hours),
            slip_pct=float(args.slippage_pct),
            lookback_days=int(args.lookback_days),
            min_4h_bars=int(args.min_4h_bars),
            tie_break=str(args.tie_break),
            dynamic_mode=dynamic_mode,
            base_tp_atr=float(args.base_tp_atr),
            base_sl_atr=float(args.base_sl_atr),
            caps=caps,
            dyn_coeff=dyn_coeff,
        )
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("no rows")
        return

    out = pd.concat(all_rows, ignore_index=True).sort_values(["time_open", "symbol"])
    for c in ["time_open", "time_close"]:
        out[c] = pd.to_datetime(out[c], utc=True).dt.tz_localize(None)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base, _ = os.path.splitext(args.out)
    out_csv = base + ".csv"

    # CSV всегда
    out.to_csv(out_csv, index=False)

    # Excel (если есть openpyxl)
    try:
        with pd.ExcelWriter(args.out, engine="openpyxl") as wr:
            _drop_tz_cols(out).to_excel(wr, index=False, sheet_name="dataset")
            meta = pd.DataFrame({
                "param": ["rows", "positives", "negatives", "symbols", "dynamic"],
                "value": [
                    len(out),
                    int(out["label_tp_first"].sum()),
                    int((1 - out["label_tp_first"]).sum()),
                    len(out["symbol"].unique()),
                    int(dynamic_mode),
                ],
            })
            meta.to_excel(wr, index=False, sheet_name="meta")
    except Exception:
        pass

    print(f"saved: {args.out} (+ {out_csv}) rows={len(out)} pos={int(out['label_tp_first'].sum())}")


if __name__ == "__main__":
    main()