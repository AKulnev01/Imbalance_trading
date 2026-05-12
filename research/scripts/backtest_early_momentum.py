# scripts/backtest_early_momentum.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==== конфиг / окружение ====
try:
    from config import TRADE_UNIVERSE
except Exception:
    TRADE_UNIVERSE = []

OUT_DIR = os.path.expanduser(os.getenv("EARLY_BT_OUT_DIR", "~/Documents/отчеты"))
M1_DIR  = os.path.expanduser(os.getenv("M1_DIR", "./data/m1"))

TAKE_PCT = float(os.getenv("MOMENTUM_TP_PCT", "0.135"))   # 13.5%
STOP_PCT = float(os.getenv("MOMENTUM_SL_PCT", "0.04"))    # 4.0%
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.004"))  # 0.4%
TTL_HOURS = int(os.getenv("MOMENTUM_TTL_HOURS", "80"))

MIN_BARS_FOR_DETECT = 60  # чтобы были EMA/ATR

@dataclass
class Signal:
    symbol: str
    side: str             # BUY / SELL
    strength: float       # скор
    imb_time: pd.Timestamp  # ВРЕМЯ ЗАКРЫТИЯ БАРА t-1 (вход на этой свече)
    # 5×4 контекст
    o_back: List[float]
    h_back: List[float]
    l_back: List[float]
    c_back: List[float]
    entry_px_ref: float
    entry_px_adj: float

# ===== базовые индикаторы =====
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ===== загрузка минуток и ресемпл в 4h =====
def load_m1(symbol: str) -> pd.DataFrame:
    path = os.path.join(M1_DIR, f"{symbol}_m1.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.rename(columns={"ts":"timestamp"})
    return df[["timestamp","open","high","low","close","volume"]]

def resample_4h(df_m1: pd.DataFrame) -> pd.DataFrame:
    if df_m1 is None or df_m1.empty:
        return pd.DataFrame()
    x = df_m1.set_index("timestamp").sort_index()
    o = x["open"].resample("4h").first()
    h = x["high"].resample("4h").max()
    l = x["low"].resample("4h").min()
    c = x["close"].resample("4h").last()
    v = x["volume"].resample("4h").sum()
    df = pd.concat([o, h, l, c, v], axis=1)
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna().copy()
    return df

# ===== ранний детектор (на t-1) =====
def early_detect_on_bar(df4h: pd.DataFrame, k: pd.Timestamp) -> Tuple[Optional[str], float]:
    """
    Возвращает (side, score) на баре k (это и есть t-1).
    side=None если сигналов нет.
    """
    O, H, L, C, V = df4h["open"], df4h["high"], df4h["low"], df4h["close"], df4h["volume"]

    if k not in df4h.index:
        return (None, 0.0)
    idx = df4h.index.get_loc(k)
    if idx < 15:  # нужно прошлое для индикаторов
        return (None, 0.0)

    # величины для k
    Rk = float(H.iloc[idx] - L.iloc[idx])
    Bk = float(abs(C.iloc[idx] - O.iloc[idx]))
    eps = 1e-12
    body_pct = Bk / max(Rk, eps)

    # подготовим окна
    H_win = H.iloc[max(0, idx-5):idx].values  # t-2..t-5 включительно, размер 4 (ниже возьмём max по ним + ещё один бар)
    L_win = L.iloc[max(0, idx-5):idx].values
    # чтобы было 4–5 баров «истории», добавим ещё один: idx-1 уже входит в back-набор — возьмём idx-2..idx-5:
    if len(H_win) < 4:
        return (None, 0.0)

    # индикаторы
    atr14 = atr(H, L, C, 14)
    atr14_k = float(atr14.iloc[idx]) if not math.isnan(atr14.iloc[idx]) else None

    ema20 = ema(C, 20)
    ema50 = ema(C, 50)
    ema20_k, ema50_k = float(ema20.iloc[idx]), float(ema50.iloc[idx])
    ema20_prev = float(ema20.iloc[idx-1])

    V_sma20 = V.rolling(20).mean()
    V_std20 = V.rolling(20).std()
    vol_z = (V.iloc[idx] - (V_sma20.iloc[idx] or 0.0)) / (V_std20.iloc[idx] or 1e-12)

    # BUY условия
    buy_close_near_high = (H.iloc[idx] - C.iloc[idx]) / max(Rk, eps) <= 0.20
    buy_breakout = C.iloc[idx] > np.max(H.iloc[idx-5:idx])  # строго выше хайов последних 5 баров
    buy_range_ok = (atr14_k is not None) and (Rk >= 1.2 * atr14_k)
    buy_vol_ok = vol_z >= 1.0
    buy_wick_ok = (H.iloc[idx] - C.iloc[idx]) / max(Bk, eps) <= 0.5
    buy_trend_ok = (ema20_k > ema50_k) and ((ema20_k - ema20_prev) > 0)

    score_buy = (
        2.0 * (body_pct >= 0.60) +
        1.5 * (buy_close_near_high) +
        1.5 * (buy_breakout) +
        1.0 * (buy_range_ok) +
        1.0 * (buy_vol_ok) +
        0.5 * (buy_trend_ok)
    )

    # SELL зеркально
    sell_close_near_low = (C.iloc[idx] - L.iloc[idx]) / max(Rk, eps) <= 0.20
    sell_breakout = C.iloc[idx] < np.min(L.iloc[idx-5:idx])
    sell_range_ok = (atr14_k is not None) and (Rk >= 1.2 * atr14_k)
    sell_vol_ok = vol_z >= 1.0
    sell_wick_ok = (C.iloc[idx] - L.iloc[idx]) / max(Bk, eps) <= 0.5
    sell_trend_ok = (ema20_k < ema50_k) and ((ema20_k - ema20_prev) < 0)

    score_sell = (
        2.0 * (body_pct >= 0.60) +
        1.5 * (sell_close_near_low) +
        1.5 * (sell_breakout) +
        1.0 * (sell_range_ok) +
        1.0 * (sell_vol_ok) +
        0.5 * (sell_trend_ok)
    )

    # анти-оверэкстенд страховка
    if atr14_k is not None and Rk > 2.5 * atr14_k:
        score_buy *= 0.5
        score_sell *= 0.5

    # выбираем сторону
    side = None
    score = 0.0
    if score_buy >= 4.0 and score_buy >= score_sell:
        side, score = "BUY", float(score_buy)
    elif score_sell >= 4.0 and score_sell > score_buy:
        side, score = "SELL", float(score_sell)

    return (side, score)

# ===== контекст 5 баров =====
def last5_context(df4h: pd.DataFrame, k: pd.Timestamp) -> Tuple[List[float],List[float],List[float],List[float]]:
    """
    back1..back5: back1 = бар k (t-1), back2 = k-1, ..., back5 = k-4
    """
    idx = df4h.index.get_loc(k)
    i0 = max(0, idx-4)
    win = df4h.iloc[i0:idx+1].copy()
    # гарантируем 5 значений: если меньше, дополним в начале NaN
    for col in ("open","high","low","close"):
        if len(win) < 5:
            pad = [np.nan] * (5 - len(win))
            win[col] = list(pad) + list(win[col].values)
    o = win["open"].values[-5:][::-1]  # k, k-1, ..., k-4
    h = win["high"].values[-5:][::-1]
    l = win["low"].values[-5:][::-1]
    c = win["close"].values[-5:][::-1]
    return list(o), list(h), list(l), list(c)

# ===== выход на минутках =====
def load_m1_window(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = load_m1(symbol)
    if df.empty:
        return df
    x = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    return x.sort_values("timestamp")

def apply_slippage(px: float, pct: float, action: str) -> float:
    if action == "BUY":
        return float(px) * (1.0 + pct)
    else:
        return float(px) * (1.0 - pct)

def simulate_exit_m1(symbol: str, side: str,
                     entry_at: pd.Timestamp,
                     tp_px: float, sl_px: float,
                     ttl_hours: int) -> Tuple[pd.Timestamp, float, str]:
    t_end = entry_at + pd.Timedelta(hours=ttl_hours)
    m1 = load_m1_window(symbol, entry_at, t_end)
    if m1.empty:
        # fallback: нет минуток — закроемся по close последней доступной 4h в окне
        return (t_end, np.nan, "no_m1_timeout")

    for _, r in m1.iterrows():
        hi = float(r["high"]); lo = float(r["low"])
        ts = pd.to_datetime(r["timestamp"], utc=True)
        if side == "BUY":
            hit_tp = hi >= tp_px
            hit_sl = lo <= sl_px
        else:
            hit_tp = lo <= tp_px
            hit_sl = hi >= sl_px
        if hit_tp and hit_sl:
            # считаем, что сначала SL
            return (ts, sl_px, "sl")
        if hit_sl:
            return (ts, sl_px, "sl")
        if hit_tp:
            return (ts, tp_px, "tp")

    # не дошли — по последней минутке
    last = m1.iloc[-1]
    return (pd.to_datetime(last["timestamp"], utc=True), float(last["close"]), "timeout_last_close")

# ===== основной прогон по символу =====
def backtest_symbol(symbol: str,
                    lookback_days: int = 360) -> Tuple[List[Signal], pd.DataFrame]:
    m1 = load_m1(symbol)
    if m1.empty:
        return ([], pd.DataFrame())
    if lookback_days > 0:
        cutoff = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=int(lookback_days))
        m1 = m1[m1["timestamp"] >= cutoff]

    df4h = resample_4h(m1)
    if df4h.empty or len(df4h) < MIN_BARS_FOR_DETECT:
        return ([], pd.DataFrame())

    signals: List[Signal] = []
    trades = []

    for k in df4h.index[MIN_BARS_FOR_DETECT:]:
        # t-1 = текущий индекс k; вход на close(k)
        side, score = early_detect_on_bar(df4h, k)
        if side is None:
            continue

        entry_ref = float(df4h.loc[k, "close"])
        # проскальзывание: вход в худшую сторону
        entry_adj = apply_slippage(entry_ref, SLIPPAGE_PCT, "BUY" if side=="BUY" else "SELL")

        # контекст 5 баров
        o_back, h_back, l_back, c_back = last5_context(df4h, k)

        sig = Signal(
            symbol=symbol,
            side=side,
            strength=float(score),
            imb_time=pd.to_datetime(k, utc=True) + pd.Timedelta(hours=4),  # время ЗАКРЫТИЯ бара k
            o_back=o_back, h_back=h_back, l_back=l_back, c_back=c_back,
            entry_px_ref=entry_ref,
            entry_px_adj=entry_adj
        )
        signals.append(sig)

        # TP/SL от цены с проскальзыванием
        if side == "BUY":
            sl = entry_adj * (1.0 - STOP_PCT)
            tp = entry_adj * (1.0 + TAKE_PCT)
        else:
            sl = entry_adj * (1.0 + STOP_PCT)
            tp = entry_adj * (1.0 - TAKE_PCT)

        # симуляция на минутках
        entry_time = sig.imb_time  # вход — ровно close(k)
        close_time, trigger_px, reason = simulate_exit_m1(symbol, side, entry_time, tp, sl, TTL_HOURS)

        # выход с худшим слиппеджем
        exit_action = "SELL" if side == "BUY" else "BUY"
        exit_px_adj = apply_slippage(trigger_px, SLIPPAGE_PCT, exit_action) if not math.isnan(trigger_px) else np.nan

        # PnL (в %), комиссии не считаем здесь (нужно — добавь FEE_TAKER_PCT)
        if not math.isnan(exit_px_adj):
            if side == "BUY":
                pnl_pct = (exit_px_adj - entry_adj) / entry_adj * 100.0
            else:
                pnl_pct = (entry_adj - exit_px_adj) / entry_adj * 100.0
        else:
            pnl_pct = np.nan

        usd_alloc = 100.0
        pnl_usd = usd_alloc * (pnl_pct / 100.0) if not math.isnan(pnl_pct) else np.nan

        trades.append({
            "symbol": symbol,
            "type": side,
            "strength": float(score),
            "imb_time": entry_time,                  # момент входа
            "entry_ref": entry_ref,
            "entry_adj": entry_adj,
            "tp": tp, "sl": sl,
            "close_time": close_time,
            "close_price_adj": float(exit_px_adj) if not math.isnan(exit_px_adj) else np.nan,
            "exit_reason": reason,
            "pnl_pct": float(pnl_pct) if not math.isnan(pnl_pct) else np.nan,
            "pnl_usd": float(pnl_usd) if not math.isnan(pnl_usd) else np.nan,
            "usd_alloc": usd_alloc,
        })

    trades_df = pd.DataFrame(trades)
    return (signals, trades_df)

# ===== сборка отчёта =====
def build_and_save(universe: List[str], out_path: str, lookback_days: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_signals: List[Signal] = []
    all_trades = []

    for sym in universe:
        print(f"[scan] {sym} …", flush=True)
        sigs, tr = backtest_symbol(sym, lookback_days=lookback_days)
        all_signals.extend(sigs)
        if not tr.empty:
            all_trades.append(tr)

    # sheet: signals
    sig_rows = []
    for s in all_signals:
        row = {
            "symbol": s.symbol, "type": s.side, "strength": s.strength,
            "imb_time": s.imb_time,
            # 20 колонок контекста (5×4)
            "o_back1": s.o_back[0], "o_back2": s.o_back[1], "o_back3": s.o_back[2], "o_back4": s.o_back[3], "o_back5": s.o_back[4],
            "h_back1": s.h_back[0], "h_back2": s.h_back[1], "h_back3": s.h_back[2], "h_back4": s.h_back[3], "h_back5": s.h_back[4],
            "l_back1": s.l_back[0], "l_back2": s.l_back[1], "l_back3": s.l_back[2], "l_back4": s.l_back[3], "l_back5": s.l_back[4],
            "c_back1": s.c_back[0], "c_back2": s.c_back[1], "c_back3": s.c_back[2], "c_back4": s.c_back[3], "c_back5": s.c_back[4],
            "entry_px_ref": s.entry_px_ref, "entry_px_adj": s.entry_px_adj,
        }
        sig_rows.append(row)
    df_signals = pd.DataFrame(sig_rows)

    # sheet: trades
    df_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # сводка
    if not df_trades.empty:
        df_exec = df_trades.copy()
        trades = len(df_exec)
        wins = int((df_exec["exit_reason"] == "tp").sum())
        winrate = round(100.0 * wins / max(trades, 1), 2)
        pnl_pct_sum = round(df_exec["pnl_pct"].sum(), 3)
        pnl_usd_sum = round(df_exec["pnl_usd"].sum(), 2)
    else:
        trades, wins, winrate, pnl_pct_sum, pnl_usd_sum = 0, 0, 0.0, 0.0, 0.0

    meta = pd.DataFrame({
        "param": ["generated_utc","symbols","rows_signals","rows_trades","tp_pct","sl_pct","slippage_pct","ttl_hours","winrate_pct","pnl_pct_sum","pnl_usd_sum"],
        "value": [
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            len(universe),
            len(df_signals),
            len(df_trades),
            TAKE_PCT, STOP_PCT, SLIPPAGE_PCT, TTL_HOURS,
            winrate, pnl_pct_sum, pnl_usd_sum
        ]
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as wr:
        df_signals.to_excel(wr, index=False, sheet_name="signals")
        df_trades.to_excel(wr, index=False, sheet_name="trades")
        meta.to_excel(wr, index=False, sheet_name="meta")

    print(f"✅ Saved: {out_path}  (signals={len(df_signals)}, trades={len(df_trades)})", flush=True)

# ===== CLI =====
def parse_symbols_arg(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

def main():
    ap = argparse.ArgumentParser(description="Backtest early-entry momentum (enter at t-1 close, TP/SL from slipped entry, exits on m1).")
    ap.add_argument("--symbols", type=str, default="", help="Символы через запятую. Если пусто — из config.TRADE_UNIVERSE.")
    ap.add_argument("--lookback-days", type=int, default=360)
    ap.add_argument("--out", type=str, default="early_momentum_eval.xlsx")
    args = ap.parse_args()

    universe = parse_symbols_arg(args.symbols) if args.symbols else (TRADE_UNIVERSE or [])
    if not universe:
        print("⚠️ Пустой список символов. Укажи --symbols или заполни TRADE_UNIVERSE в config.py")
        return

    out_path = os.path.join(OUT_DIR, args.out)
    build_and_save(universe, out_path, lookback_days=int(args.lookback_days))

if __name__ == "__main__":
    main()