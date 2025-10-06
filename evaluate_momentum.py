# evaluate_momentum.py — AUTOTRADE-MIRROR (4h close entry, entry-TP/SL, minute tie-break; fees/slippage via ENV/CLI)
import os, math
import pandas as pd
from datetime import datetime, timezone

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, POSITION_FRACTION, DEFAULT_TTL_DAYS,
    to_utc_safe, fetch_ltf_window, load_price_cache, load_signals,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, finalize_write, calc_sl_tp,
)

# =========================
# ПАРАМЕТРЫ ЧЕРЕЗ ENV
# =========================
TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))        # напр. 0.03
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))        # напр. 0.01
TP_SL_MODE = "entry"                                            # принудительно "entry"

# Комиссия (только taker) — доля, напр. 0.0006 = 6 bps
FEE_TAKER = float(get_cfg("FEE_TAKER", cast=float, default=0.0))

# Слиппедж: можно одной цифрой или раздельно
# Если заданы ENTRY_/EXIT_/STOP_ — они в приоритете, иначе берём общий SLIPPAGE_PCT
SLIPPAGE_PCT_DEFAULT = float(get_cfg("SLIPPAGE_PCT", cast=float, default=0.003))  # 0.3% дефолт
ENTRY_SLIPPAGE_PCT = float(get_cfg("ENTRY_SLIPPAGE_PCT", cast=float, default=SLIPPAGE_PCT_DEFAULT))
EXIT_SLIPPAGE_PCT  = float(get_cfg("EXIT_SLIPPAGE_PCT",  cast=float, default=SLIPPAGE_PCT_DEFAULT))
STOP_SLIPPAGE_PCT  = float(get_cfg("STOP_SLIPPAGE_PCT",  cast=float, default=EXIT_SLIPPAGE_PCT))

# Минутная дорисовка
DISABLE_MINUTE_FALLBACK = bool(get_cfg("MOMENTUM_DISABLE_MINUTE_FALLBACK", cast=bool, default=False))
MINUTE_EXIT_FOR_SINGLE  = bool(get_cfg("MOMENTUM_MINUTE_EXIT_FOR_SINGLE_HIT", cast=bool, default=True))

def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY", "LONG"):  return "BUY"
    if s in ("SELL", "SHORT"):return "SELL"
    return s

def _apply_slippage(price: float, pct: float, action: str) -> float:
    """Худший fill: BUY — дороже, SELL — дешевле."""
    sp = float(pct or 0.0)
    if not isinstance(price,(int,float)) or math.isnan(price): return float('nan')
    return float(price) * (1.0 + sp) if action == "BUY" else float(price) * (1.0 - sp)

def _pnl_pct_with_fees(entry_px_adj: float, exit_px_adj: float, side: str, fee_taker_pct: float) -> float:
    if any(not isinstance(x,(int,float)) or math.isnan(x) for x in (entry_px_adj, exit_px_adj)):
        return float('nan')
    move = ((exit_px_adj - entry_px_adj) / entry_px_adj * 100.0) if side=="BUY" \
           else ((entry_px_adj - exit_px_adj) / entry_px_adj * 100.0)
    fees = float(fee_taker_pct or 0.0) * 2.0 * 100.0
    return float(move) - float(fees)

def _entry_from_4h_close(df_4h: pd.DataFrame, t0):
    """Вход ровно в t0 по цене CLOSE 4h-свечи (бар со стартом t0-4h)."""
    t0 = to_utc_safe(t0)
    if pd.isna(t0) or df_4h is None or df_4h.empty: return (pd.NaT, float("nan"))
    pos = df_4h.index.searchsorted(t0)
    ix = pos - 1
    if ix < 0: return (pd.NaT, float("nan"))
    try:
        entry_px_ref = float(df_4h.iloc[ix]["close"])
    except Exception:
        entry_px_ref = float("nan")
    return (t0, entry_px_ref)

def _resolve_exit_minute(symbol: str, side: str, entry_at, stop_eval: float, tp_eval: float, t_end):
    """Минутная дорисовка порядка TP/SL. Если оба — считаем, что SL сработал первым."""
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end): return (None, pd.NaT, float('nan'), "bad_ts")
    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf is None or ltf.empty:      return (None, pd.NaT, float('nan'), "uncertain_no_ltf")
    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side=="BUY":
            hit_tp = hi >= float(tp_eval); hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval); hit_sl = hi >= float(stop_eval)
        if hit_tp and hit_sl: return (False, ts, float(stop_eval), "sl")
        if hit_tp:            return (True,  ts, float(tp_eval),   "tp")
        if hit_sl:            return (False, ts, float(stop_eval), "sl")
    last_ts = ltf.index[-1]; last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")

def _resolve_exit_hybrid(symbol, side, entry_at, stop_eval, tp_eval, t_end, df_4h):
    """
    Быстрый 4h проход; если бар содержит оба уровня — спускаемся на 1m;
    если один — по умолчанию тоже на 1m для точной минуты (если не отключено).
    """
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end) or df_4h is None or df_4h.empty:
        return (None, pd.NaT, float("nan"), "bad_ts", "4h")

    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)]
    for ts, c in bars.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        bar_start, bar_end = ts, ts + pd.Timedelta(hours=4)
        if side=="BUY":
            hit_tp = hi >= float(tp_eval); hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval); hit_sl = hi >= float(stop_eval)

        if hit_tp and hit_sl:
            if DISABLE_MINUTE_FALLBACK:
                return (None, bar_end, float('nan'), "uncertain_both_hit_4h", "4h")
            return _resolve_exit_minute(symbol, side, max(t0,bar_start), float(stop_eval), float(tp_eval), min(t_end,bar_end)) + ("1m",)

        if hit_tp:
            if MINUTE_EXIT_FOR_SINGLE and not DISABLE_MINUTE_FALLBACK:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0,bar_start), float(stop_eval), float(tp_eval), min(t_end,bar_end)
                )
                if reason in ("tp","sl","timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (True, bar_end, float(tp_eval), "tp", "4h")

        if hit_sl:
            if MINUTE_EXIT_FOR_SINGLE and not DISABLE_MINUTE_FALLBACK:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0,bar_start), float(stop_eval), float(tp_eval), min(t_end,bar_end)
                )
                if reason in ("tp","sl","timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (False, bar_end, float(stop_eval), "sl", "4h")

    if not bars.empty:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4)
        last_close = float(bars.iloc[-1]["close"])
    else:
        last_ts = t0; last_close = float("nan")
    return (False, last_ts, last_close, "timeout_last_close", "4h")

def evaluate_momentum(signals_path: str, result_path: str,
                      lookback_days: int = 360, interval: str = "4h",
                      max_days: int = None, only_filled: bool = False, dedup: bool = False,
                      initial_capital: float = None, capital_aware: bool = True):

    # модельная дата = mtime входного файла (как «срез истории»)
    try:
        as_of = datetime.fromtimestamp(os.path.getmtime(signals_path), tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    # грузим сигналы
    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    if df_sig.empty:
        print("⚠️ Сигналов нет после фильтров."); return

    symbols = df_sig["symbol"].dropna().unique().tolist()
    price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)

    results = []
    for _, row in df_sig.iterrows():
        symbol = row["symbol"]
        side   = _normalize_side(row.get("type"))
        t0     = to_utc_safe(row.get("imb_time"))
        df4h   = price_cache.get(symbol)

        if df4h is None or df4h.empty or side not in ("BUY","SELL") or pd.isna(t0):
            continue

        # вход по CLOSE 4h
        entry_at, entry_px_ref = _entry_from_4h_close(df4h, t0)
        if pd.isna(entry_at) or not isinstance(entry_px_ref,(int,float)) or math.isnan(entry_px_ref):
            continue

        entry_action = "BUY" if side=="BUY" else "SELL"
        entry_px_adj = _apply_slippage(entry_px_ref, ENTRY_SLIPPAGE_PCT, entry_action)

        # TP/SL от entry
        rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)
        sl_eval, tp_eval = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)

        # исход (минутная дорисовка)
        ttl_days   = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
        window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)

        win, close_time, trigger_price, exit_reason, _ = _resolve_exit_hybrid(
            symbol=symbol, side=side, entry_at=entry_at,
            stop_eval=float(sl_eval), tp_eval=float(tp_eval),
            t_end=window_end, df_4h=df4h
        )
        if win is None:
            # помечаем пропуск (например, нет 1m данных). Но оставим в таблице — для контроля качества.
            results.append({
                "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
                "imb_time": t0, "t_start": entry_at,
                "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
                "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA,
                "close_time": close_time, "close_price": pd.NA, "exit_reason": str(exit_reason),
                "exit_hours": pd.NA, "usd_alloc": pd.NA,
                "variant":"MOMENTUM", "as_of": as_of, "skipped": True
            })
            continue

        # выходная цена с худшим слиппеджем (для стопов — отдельный процент)
        exit_action = "SELL" if side=="BUY" else "BUY"
        exit_slip = STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT
        exit_px_base = float(trigger_price) if isinstance(trigger_price,(int,float)) else float('nan')
        exit_px_adj  = _apply_slippage(exit_px_base, exit_slip, exit_action)
        pnl_pct_net  = _pnl_pct_with_fees(entry_px_adj, exit_px_adj, side, FEE_TAKER)

        results.append({
            "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
            "imb_time": t0, "t_start": entry_at,
            "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
            "win": True if exit_reason=="tp" else False, "pnl_pct": float(pnl_pct_net), "pnl_usd": pd.NA,
            "close_time": close_time, "close_price": float(exit_px_adj), "exit_reason": str(exit_reason),
            "exit_hours": (pd.to_datetime(close_time, utc=True) - pd.to_datetime(entry_at, utc=True)).total_seconds()/3600.0 if pd.notna(close_time) else pd.NA,
            "usd_alloc": pd.NA,
            "variant":"MOMENTUM", "as_of": as_of, "skipped": False
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ Ничего не оценили."); return

    # запрет перекрытий по символу (нужны t_start/close_time)
    df = enforce_one_at_a_time_per_symbol(df)

    # === капитал-сим ===
    sim = df.copy()
    exec_mask = (sim.get("skipped") == False)
    sim.loc[exec_mask, "exit_reason"] = "price"  # для симулятора

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap > 0 and capital_aware:
        sim_out, eq_sheet = simulate_capital_notional(
            sim, init_cap, POSITION_FRACTION, stop_pct=STOP_PCT, take_pct=TAKE_PCT
        )
        # переносим рассчитанные поля
        for col in ("usd_alloc", "pnl_usd"):
            if col in sim_out.columns:
                df[col] = sim_out[col]
    else:
        eq_sheet = pd.DataFrame()

    # сервисные сводки
    df_exec = df[df.get("skipped")==False].copy()
    by_variant = (df_exec.groupby("variant")
                  .agg(trades=("win","size"),
                       wins=("win","sum"),
                       winrate_pct=("win", lambda s: round(100.0 * float(s.sum()) / max(int(s.size),1), 2)),
                       pnl_pct=("pnl_pct","sum"),
                       pnl_usd=("pnl_usd","sum"))
                  .reset_index()) if not df_exec.empty else pd.DataFrame()
    by_exit_reason = safe_group_exit_reason(df)

    # красиво для xlsx — без таймзоны
    for c in ["imb_time","t_start","close_time","as_of"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_convert(None)

    finalize_write(result_path, df, eq_sheet, by_variant, by_exit_reason)
    print(f"✅ MOMENTUM eval saved → {result_path}")

if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1","true","yes","y","on")
    p = argparse.ArgumentParser(description="Evaluate MOMENTUM (autotrade mirror): 4h-close entry, entry TP/SL; fees/slippage via ENV; minute tie-resolution.")
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")

    # CLI-оверрайды fee/slippage
    p.add_argument("--fee-taker-pct", type=float, default=None)
    p.add_argument("--entry-slippage-pct", type=float, default=None)
    p.add_argument("--exit-slippage-pct",  type=float, default=None)
    p.add_argument("--stop-slippage-pct",  type=float, default=None)

    args = p.parse_args()

    # Подхватываем CLI в ENV, чтобы get_cfg увидел
    if args.fee_taker_pct        is not None: os.environ["FEE_TAKER"]          = str(args.fee_taker_pct)
    if args.entry_slippage_pct   is not None: os.environ["ENTRY_SLIPPAGE_PCT"] = str(args.entry_slippage_pct)
    if args.exit_slippage_pct    is not None: os.environ["EXIT_SLIPPAGE_PCT"]  = str(args.exit_slippage_pct)
    if args.stop_slippage_pct    is not None: os.environ["STOP_SLIPPAGE_PCT"]  = str(args.stop_slippage_pct)

    # Время жизни позиции, капитал и т.д.
    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_momentum_eval.xlsx"

    evaluate_momentum(
        signals_path=sig_path, result_path=res_path,
        lookback_days=int(args.lookback_days), interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled), dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )