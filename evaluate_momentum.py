# evaluate_momentum.py
import os
import math
import pandas as pd
from datetime import datetime, timezone, timedelta

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, POSITION_FRACTION, DEFAULT_TTL_DAYS,
    to_utc_safe, fetch_ltf_window, load_price_cache, load_signals,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, finalize_write, calc_sl_tp,
)

# --- Новые/доп. флаги, чтобы быть как в лайве ---
ENABLE_BUY  = bool(get_cfg("ENABLE_BUY",  cast=bool,  default=True))
ENABLE_SELL = bool(get_cfg("ENABLE_SELL", cast=bool,  default=True))
MAX_CONCURRENT = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=0))  # 0 = без лимита

# --- Параметры RR (ценовые проценты) ---
TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))

# --- TP/SL режим: 'entry' (от входа) | 'anchored' (симметрия вокруг 4h close) ---
TP_SL_MODE_RAW = str(get_cfg("MOMENTUM_TP_SL_MODE", cast=str, default="entry")).strip().lower()
FORCE_ENTRY = bool(get_cfg("MOMENTUM_FORCE_ENTRY", cast=bool, default=False))  # если true — всегда entry
TP_SL_MODE = "entry" if FORCE_ENTRY else TP_SL_MODE_RAW

# --- 1m-логика выхода ---
DISABLE_MINUTE_FALLBACK = bool(get_cfg("MOMENTUM_DISABLE_MINUTE_FALLBACK", cast=bool, default=False))
MINUTE_EXIT_FOR_SINGLE = bool(get_cfg("MOMENTUM_MINUTE_EXIT_FOR_SINGLE_HIT", cast=bool, default=True))

# --- Лайвоподобные параметры: комиссия/слиппедж ---
FEE_TAKER = float(get_cfg("FEE_TAKER", cast=float, default=0.0))  # напр. 0.0006 = 6 bps
ENTRY_SLIPPAGE_PCT = float(get_cfg("ENTRY_SLIPPAGE_PCT", cast=float, default=0.0))  # напр. 0.0015 = 0.15%
EXIT_SLIPPAGE_PCT  = float(get_cfg("EXIT_SLIPPAGE_PCT",  cast=float, default=0.0))  # обычный выход (TP)
# Если STOP_SLIPPAGE_PCT не задан, используем EXIT_SLIPPAGE_PCT
STOP_SLIPPAGE_PCT  = float(get_cfg("STOP_SLIPPAGE_PCT",  cast=float, default=EXIT_SLIPPAGE_PCT))


def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY", "LONG"): return "BUY"
    if s in ("SELL", "SHORT"): return "SELL"
    return s


def _apply_slippage(price: float, pct: float, action: str) -> float:
    """Корректируем цену под проскальзывание для фактического fill-а (только для PnL)."""
    sp = float(pct or 0.0)
    if not isinstance(price, (int, float)) or math.isnan(price):
        return float('nan')
    # Для BUY хуже цена = дороже (умножаем), для SELL хуже = дешевле (уменьшаем)
    return float(price) * (1.0 + sp) if action == "BUY" else float(price) * (1.0 - sp)


def _pnl_pct_with_fees(entry_px_adj: float, exit_px_adj: float, side: str, fee_taker_pct: float) -> float:
    """PNL% c учётом проскальзывания уже в ценах и двойной такер-фии."""
    if any(not isinstance(x, (int, float)) or math.isnan(x) for x in (entry_px_adj, exit_px_adj)):
        return float('nan')
    move_pct = ((exit_px_adj - entry_px_adj) / entry_px_adj * 100.0) if side == "BUY" \
               else ((entry_px_adj - exit_px_adj) / entry_px_adj * 100.0)
    fees_pct = float(fee_taker_pct or 0.0) * 2.0 * 100.0
    return float(move_pct) - float(fees_pct)


def _entry_from_4h_close(df_4h: pd.DataFrame, t0) -> tuple:
    """
    Вход «в момент детекта»: price = CLOSE 4h-свечи, которая закрылась в t0.
    Индекс 4h — старт бара (как в Bybit v5). Закрытие в t0 — close у бара со стартом (t0-4h).
    Возвращаем (entry_at=t0, entry_px_ref).
    """
    t0 = to_utc_safe(t0)
    if pd.isna(t0) or df_4h is None or df_4h.empty:
        return (pd.NaT, float("nan"))
    pos = df_4h.index.searchsorted(t0)
    ix = pos - 1
    if ix < 0:
        return (pd.NaT, float("nan"))
    try:
        entry_px_ref = float(df_4h.iloc[ix]["close"])
    except Exception:
        entry_px_ref = float("nan")
    return (t0, entry_px_ref)


def _resolve_exit_minute(symbol: str, side: str, entry_at: pd.Timestamp,
                         stop_eval: float, tp_eval: float, t_end: pd.Timestamp):
    """
    Минутный разбор порядка касаний. Если нет 1m — считаем исход неопределённым.
    Возвращает: (win_bool_or_None, close_time, trigger_price, exit_reason)
    """
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end):
        return (None, pd.NaT, float('nan'), "bad_ts")

    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (None, pd.NaT, float('nan'), "uncertain_no_ltf")

    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = (hi >= float(tp_eval)); hit_sl = (lo <= float(stop_eval))
        else:
            hit_tp = (lo <= float(tp_eval)); hit_sl = (hi >= float(stop_eval))

        if hit_tp and hit_sl:
            # консервативно: SL первым
            return (False, ts, float(stop_eval), "sl")
        if hit_tp:
            return (True, ts, float(tp_eval), "tp")
        if hit_sl:
            return (False, ts, float(stop_eval), "sl")

    last_ts = ltf.index[-1]
    last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")


def _resolve_exit_hybrid(symbol: str, side: str, entry_at: pd.Timestamp,
                         stop_eval: float, tp_eval: float, t_end: pd.Timestamp,
                         df_4h: pd.DataFrame):
    """
    Быстрый проход по 4h:
      - если в баре и high≥TP, и low≤SL → спускаемся на 1m только в рамках ЭТОГО бара (если не отключено флагом);
      - если в баре сработал только один уровень → по желанию спускаемся на 1m в рамках бара, чтобы взять точную минуту;
      - если ни разу не сработало до t_end → timeout_last_close по последнему 4h close.
    Возвращает: (win_bool_or_None, close_time, trigger_price, exit_reason, lt_resolution)
    """
    t0 = to_utc_safe(entry_at); t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end) or df_4h is None or df_4h.empty:
        return (None, pd.NaT, float("nan"), "bad_ts", "4h")

    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)]
    for ts, c in bars.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = (hi >= float(tp_eval)); hit_sl = (lo <= float(stop_eval))
        else:
            hit_tp = (lo <= float(tp_eval)); hit_sl = (hi >= float(stop_eval))

        bar_start = ts
        bar_end   = ts + pd.Timedelta(hours=4)

        if hit_tp and hit_sl:
            if DISABLE_MINUTE_FALLBACK:
                return (None, ts + pd.Timedelta(hours=4), float('nan'), "uncertain_both_hit_4h", "4h")
            return _resolve_exit_minute(symbol, side, max(t0, bar_start),
                                        float(stop_eval), float(tp_eval), min(t_end, bar_end)) + ("1m",)

        if hit_tp:
            if MINUTE_EXIT_FOR_SINGLE:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
                )
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (True, ts + pd.Timedelta(hours=4), float(tp_eval), "tp", "4h")

        if hit_sl:
            if MINUTE_EXIT_FOR_SINGLE:
                win_m, close_t, trig_px, reason = _resolve_exit_minute(
                    symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
                )
                if reason in ("tp", "sl", "timeout_last_close"):
                    return (win_m, close_t, trig_px, reason, "1m")
            return (False, ts + pd.Timedelta(hours=4), float(stop_eval), "sl", "4h")

    if not bars.empty:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4)
        last_close = float(bars.iloc[-1]["close"])
    else:
        last_ts = t0
        last_close = float("nan")
    return (False, last_ts, last_close, "timeout_last_close", "4h")


def _dedup_strongest_per_symbol_time(df_sig: pd.DataFrame) -> pd.DataFrame:
    if df_sig is None or df_sig.empty or "imb_time" not in df_sig.columns:
        return df_sig
    df = df_sig.copy()
    if "strength" not in df.columns:
        return df.sort_values(["symbol", "imb_time"]).drop_duplicates(["symbol", "imb_time"], keep="first")
    df = df.sort_values(["symbol", "imb_time", "strength"], ascending=[True, True, False])
    return df.drop_duplicates(["symbol", "imb_time"], keep="first")


def _enforce_global_concurrency(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k is None or k <= 0 or df is None or df.empty:
        return df

    work = df.copy()
    work["_t0"] = pd.to_datetime(work.get("t_start"), utc=True, errors="coerce")
    work["_t1"] = pd.to_datetime(work.get("close_time"), utc=True, errors="coerce")
    ttl_days = float(DEFAULT_TTL_DAYS or 3)
    miss_end = work["_t1"].isna()
    work.loc[miss_end, "_t1"] = work.loc[miss_end, "_t0"] + pd.Timedelta(days=ttl_days)

    strength = pd.to_numeric(work.get("strength"), errors="coerce")
    work["_strength"] = strength.fillna(-1e18)

    kept_idx = []
    active_ends = []
    for t0, grp in work.sort_values("_t0").groupby("_t0", sort=True):
        active_ends = [t for t in active_ends if (pd.isna(t) or t > t0)]
        slots = k - len(active_ends)
        if slots <= 0:
            continue
        sel = grp.sort_values("_strength", ascending=False).head(slots)
        kept_idx.extend(sel.index.tolist())
        active_ends.extend(sel["_t1"].tolist())

    kept = work.loc[sorted(set(kept_idx))].copy()
    kept.drop(columns=[c for c in ["_t0", "_t1", "_strength"] if c in kept.columns], inplace=True)
    return kept.sort_values(["t_start", "symbol"]).reset_index(drop=True)


def evaluate_momentum(signals_path: str,
                      result_path: str,
                      lookback_days: int = 360,
                      interval: str = "4h",
                      max_days: int = None,
                      only_filled: bool = False,
                      dedup: bool = False,
                      initial_capital: float = None,
                      capital_aware: bool = True):

    try:
        as_of = datetime.fromtimestamp(os.path.getmtime(signals_path), tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    if df_sig.empty:
        print("⚠️ Сигналов нет после фильтров."); return

    if "type" in df_sig.columns:
        t = df_sig["type"].astype(str).str.upper()
        mask = pd.Series([True] * len(df_sig), index=df_sig.index)
        if not ENABLE_BUY:
            mask &= (t != "BUY")
        if not ENABLE_SELL:
            mask &= (t != "SELL")
        df_sig = df_sig[mask]
    if df_sig.empty:
        print("⚠️ После фильтров ENABLE_BUY/SELL сигналов нет."); return

    df_sig = _dedup_strongest_per_symbol_time(df_sig)
    if df_sig.empty:
        print("⚠️ После дедупа по strongest сигналов нет."); return

    symbols = df_sig["symbol"].dropna().unique().tolist()
    price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)

    results = []
    for _, row in df_sig.iterrows():
        symbol = row["symbol"]
        side   = _normalize_side(row.get("type"))
        t0     = to_utc_safe(row.get("imb_time"))
        df4h   = price_cache.get(symbol)

        if df4h is None or df4h.empty or side not in ("BUY", "SELL") or pd.isna(t0):
            continue

        # 1) Вход в t0 по 4h-close, плюс лайв-слиппедж на entry
        entry_at, entry_px_ref = _entry_from_4h_close(df4h, t0)
        if pd.isna(entry_at) or not isinstance(entry_px_ref, (int, float)) or math.isnan(entry_px_ref):
            out = row.to_dict(); out.update({
                "variant": "MOMENTUM", "as_of": as_of,
                "stop_eval": pd.NA, "tp_eval": pd.NA, "win": pd.NA,
                "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
                "close_time": pd.NaT, "close_price": pd.NA,
                "exit_reason": "no_4h_close", "is_open_mark": False,
                "t_start": pd.NaT, "size_weight": 1.0,
                "skipped": True, "lt_resolution": "none", "entry_note": "no_4h_close",
                "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
            }); results.append(out); continue

        entry_action = "BUY" if side == "BUY" else "SELL"
        entry_px_adj = _apply_slippage(entry_px_ref, ENTRY_SLIPPAGE_PCT, entry_action)

        # 2) TP/SL по выбранной схеме
        rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)
        mode = TP_SL_MODE
        if mode == "anchored":
            anchor = float(entry_px_ref)
            k = float(STOP_PCT)
            width = anchor * k * (1.0 + rr)
            if side == "BUY":
                sl_eval = anchor - width / 2.0
                tp_eval = anchor + width / 2.0
            else:
                sl_eval = anchor + width / 2.0
                tp_eval = anchor - width / 2.0
        else:
            sl_eval, tp_eval = calc_sl_tp(float(entry_px_adj), side, float(STOP_PCT), rr)

        # 3) Что сработало раньше (с точной минутой при необходимости)
        ttl_days   = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
        window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)
        win, close_time, trigger_price, exit_reason, tf_used = _resolve_exit_hybrid(
            symbol=symbol, side=side, entry_at=entry_at,
            stop_eval=float(sl_eval), tp_eval=float(tp_eval),
            t_end=window_end, df_4h=df4h
        )

        # 4) Неинформативный исход
        if win is None or (isinstance(exit_reason, str) and exit_reason.startswith("uncertain")):
            out = row.to_dict(); out.update({
                "variant": "MOMENTUM", "as_of": as_of,
                "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
                "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA, "move_pct": pd.NA,
                "close_time": close_time, "close_price": pd.NA,
                "exit_reason": exit_reason, "is_open_mark": False,
                "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
                "skipped": True,
                "lt_resolution": tf_used or "none",
                "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
                "exit_trigger": pd.NA,
                "exit_reason_raw": exit_reason,
                "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
                "tpsl_mode": mode,
            }); results.append(out); continue

        # 5) NET PnL: применяем разный слиппедж на выход
        exit_action = "SELL" if side == "BUY" else "BUY"
        exit_trigger = float(trigger_price) if isinstance(trigger_price, (int, float)) else float('nan')
        exit_slip = STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT
        exit_px_adj  = _apply_slippage(exit_trigger, exit_slip, exit_action)
        pnl_pct_net  = _pnl_pct_with_fees(entry_px_adj, exit_px_adj, side, FEE_TAKER)

        out = row.to_dict()
        out.update({
            "variant": "MOMENTUM", "as_of": as_of,
            "stop_eval": float(sl_eval), "tp_eval": float(tp_eval),
            "win": True if exit_reason == "tp" else False,
            "pnl_pct": pnl_pct_net, "pnl_usd": pd.NA, "move_pct": pnl_pct_net,
            "close_time": close_time, "close_price": exit_px_adj,
            "exit_reason": exit_reason,
            "is_open_mark": False,
            "t_start": entry_at, "size_weight": float(row.get("size_weight", 1.0)) if pd.notna(row.get("size_weight", 1.0)) else 1.0,
            "skipped": False,
            "lt_resolution": tf_used or "4h",
            "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
            "exit_px_adj": float(exit_px_adj),
            "exit_trigger": float(exit_trigger),
            "exit_reason_raw": exit_reason,
            "strength": float(row.get("strength", pd.NA)) if pd.notna(row.get("strength", pd.NA)) else pd.NA,
            "tpsl_mode": mode,
            "type": side,
            "fee_taker_pct": float(FEE_TAKER),
            "entry_slip_pct": float(ENTRY_SLIPPAGE_PCT),
            "exit_slip_pct": float(exit_slip),
        })
        results.append(out)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("⚠️ Ничего не оценили."); return

    # запрет перекрытий по символам
    df_res = enforce_one_at_a_time_per_symbol(df_res)

    # тайминги/exit_days
    for c in ["t_start", "close_time", "imb_time"]:
        if c in df_res.columns:
            df_res[c] = df_res[c].map(to_utc_safe)
    df_res["exit_time"] = df_res["close_time"]
    t_start_utc = pd.to_datetime(df_res["t_start"], utc=True, errors="coerce")
    t_exit_utc  = pd.to_datetime(df_res["exit_time"], utc=True, errors="coerce")
    df_res["exit_days"] = ((t_exit_utc - t_start_utc).dt.total_seconds() / 86400.0).round(3)

    # глобальный лимит одновременных позиций
    before_cnt = len(df_res)
    df_res = _enforce_global_concurrency(df_res, MAX_CONCURRENT)
    after_cnt = len(df_res)
    if MAX_CONCURRENT and MAX_CONCURRENT > 0:
        print(f"🔒 Applied global concurrency cap = {MAX_CONCURRENT}: kept {after_cnt} of {before_cnt}")

    # капитал-сим: уже net pnl_pct
    df_res_adj = df_res.copy()
    mask_exec = (df_res_adj.get("skipped") == False)
    df_res_adj.loc[mask_exec, "exit_reason"] = "price"  # для симулятора

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap <= 0 or not capital_aware:
        df_out = df_res_adj.copy(); df_out["skipped"] = df_res_adj.get("skipped", False); eq_sheet = pd.DataFrame()
    else:
        df_out, eq_sheet = simulate_capital_notional(
            df_res_adj, init_cap, POSITION_FRACTION, stop_pct=STOP_PCT, take_pct=TAKE_PCT
        )

    # сводки
    df_exec = df_out[df_out.get("skipped") == False].copy()
    by_variant = (df_exec.groupby("variant")
                        .agg(trades=("win", "size"), wins=("win", "sum"),
                             winrate_pct=("win", lambda s: round(100.0 * float(s.sum()) / max(int(s.size), 1), 2)),
                             pnl_pct=("pnl_pct", "sum"), pnl_usd=("pnl_usd", "sum"))
                        .reset_index()) if not df_exec.empty else pd.DataFrame()
    by_exit_reason = safe_group_exit_reason(df_out)

    # качество данных
    reason = df_out.get("exit_reason_raw", df_out.get("exit_reason", pd.Series([], dtype=str))).astype(str).fillna("")
    lt_res = df_out.get("lt_resolution", pd.Series([], dtype=str)).astype(str).fillna("")

    n_total = int(len(df_out))
    n_exec  = int((df_out.get("skipped") == False).sum()) if "skipped" in df_out.columns else n_total
    n_skip  = int((df_out.get("skipped") == True).sum()) if "skipped" in df_out.columns else 0

    uncertain_mask = reason.str.startswith("uncertain")
    n_uncertain = int(uncertain_mask.sum())
    uncertain_pct = round(100.0 * n_uncertain / max(n_total, 1), 2)

    exec_mask = (df_out.get("skipped") == False) if "skipped" in df_out.columns else pd.Series([True]*n_total)
    ltf_1m_exec = int((lt_res[exec_mask] == "1m").sum())
    ltf_1m_exec_pct = round(100.0 * ltf_1m_exec / max(n_exec, 1), 2)

    quality_summary = pd.DataFrame([
        {"metric": "n_total", "value": n_total},
        {"metric": "n_executed", "value": n_exec},
        {"metric": "n_skipped", "value": n_skip},
        {"metric": "n_uncertain", "value": n_uncertain},
        {"metric": "uncertain_pct", "value": uncertain_pct},
        {"metric": "ltf_fallback_1m_exec", "value": ltf_1m_exec},
        {"metric": "ltf_fallback_1m_exec_pct", "value": ltf_1m_exec_pct},
        {"metric": "max_concurrent_used", "value": MAX_CONCURRENT},
        {"metric": "kept_after_concurrency", "value": after_cnt},
    ])

    lt_resolution_all  = lt_res.value_counts(dropna=False).rename("trades").reset_index().rename(columns={"index": "lt_resolution"})
    lt_resolution_exec = lt_res[exec_mask].value_counts(dropna=False).rename("trades").reset_index().rename(columns={"index": "lt_resolution"})
    uncertain_breakdown = (reason[uncertain_mask]
                            .value_counts(dropna=False)
                            .rename("trades").reset_index()
                            .rename(columns={"index": "exit_reason"}))

    extra = {
        "quality_summary": quality_summary,
        "lt_resolution_all": lt_resolution_all,
        "lt_resolution_exec": lt_resolution_exec,
        "uncertain_breakdown": uncertain_breakdown,
    }

    finalize_write(result_path, df_out, eq_sheet, by_variant, by_exit_reason, extra_sheets=extra)
    print(f"✅ MOMENTUM eval saved → {result_path}")


if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1","true","yes","y","on")
    p = argparse.ArgumentParser(description="Evaluate MOMENTUM: live-like entry/exit slippage and taker fees; TP/SL: entry|anchored; minute-precise exits; honors ENABLE_BUY/SELL; strongest per (symbol,t0); global MAX_CONCURRENT.")
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--intrabar", default=None)  # например "1m"
    p.add_argument("--intrabar-lookback-days", type=int, default=None)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")
    # Новые CLI флаги для live-like параметров
    p.add_argument("--fee-taker-pct", type=float, default=None)
    p.add_argument("--entry-slippage-pct", type=float, default=None)
    p.add_argument("--exit-slippage-pct", type=float, default=None)
    p.add_argument("--stop-slippage-pct", type=float, default=None)
    args = p.parse_args()

    # Обновим ENV из CLI, чтобы get_cfg подхватил значения
    if args.fee_taker_pct        is not None: os.environ["FEE_TAKER"]          = str(args.fee_taker_pct)
    if args.entry_slippage_pct   is not None: os.environ["ENTRY_SLIPPAGE_PCT"] = str(args.entry_slippage_pct)
    if args.exit_slippage_pct    is not None: os.environ["EXIT_SLIPPAGE_PCT"]  = str(args.exit_slippage_pct)
    if args.stop_slippage_pct    is not None: os.environ["STOP_SLIPPAGE_PCT"]  = str(args.stop_slippage_pct)

    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_momentum_eval.xlsx"

    if args.intrabar is not None:
        os.environ["INTRABAR_INTERVALS"] = args.intrabar
    if args.intrabar_lookback_days is not None:
        os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(args.intrabar_lookback_days))

    evaluate_momentum(
        signals_path=sig_path, result_path=res_path,
        lookback_days=int(args.lookback_days), interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled), dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )