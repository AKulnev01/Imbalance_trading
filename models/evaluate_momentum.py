import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, DEFAULT_TTL_DAYS,
    to_utc_safe, fetch_ltf_window, load_price_cache, load_signals,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, finalize_write, compute_entry_features,
)

# ======== ФЛАГИ ПОВЕДЕНИЯ (используем уже существующие переменные) ========
EVAL_PICK_STRONGEST = bool(get_cfg("EVAL_PICK_STRONGEST", cast=bool, default=True))
MAX_CONCURRENT_POSITIONS = int(get_cfg("MAX_CONCURRENT_POSITIONS", cast=int, default=0))

# ======== ЖЁСТКИЕ ОГРАНИЧЕНИЯ ДАТА-ИСТОЧНИКА ========
os.environ["USE_LOCAL_MINUTES"] = "1"
os.environ["USE_LOCAL_4H"]      = "0"
os.environ.setdefault("LTF_ROOT", "./data/m1")
# никакого «минутного фоллбэка отключено» — всегда используем 1m
os.environ["DISABLE_MINUTE_FALLBACK"] = "0"
os.environ["MINUTE_EXIT_FOR_SINGLE"]  = "1"

# ======== Индикаторы (жёстко OFF) ========
USE_FIB_4H = False
USE_DIV_4H = False
USE_OBOS_4H = False

# ======== Параметры TP/SL/комиссии/слиппедж ========
TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))      # например 0.03
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))       # например 0.01
FEE_TAKER = float(get_cfg("FEE_TAKER", cast=float, default=0.0))

SLIPPAGE_PCT_DEFAULT = float(get_cfg("SLIPPAGE_PCT", cast=float, default=0.003))
ENTRY_SLIPPAGE_PCT = float(get_cfg("ENTRY_SLIPPAGE_PCT", cast=float, default=SLIPPAGE_PCT_DEFAULT))
EXIT_SLIPPAGE_PCT  = float(get_cfg("EXIT_SLIPPAGE_PCT",  cast=float, default=SLIPPAGE_PCT_DEFAULT))
STOP_SLIPPAGE_PCT  = float(get_cfg("STOP_SLIPPAGE_PCT",  cast=float, default=EXIT_SLIPPAGE_PCT))

# ======== Лимит жизни сделки (часы) ========
EVAL_MAX_HOURS = int(get_cfg("EVAL_MAX_HOURS", cast=int, default=80))

# ======== Хелперы ========
def _normalize_side(v: str) -> str:
    s = str(v).strip().upper()
    if s in ("BUY", "LONG"):  return "BUY"
    if s in ("SELL", "SHORT"):return "SELL"
    return s

def _choose_one_per_same_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Если на один и тот же t_start пришли несколько сигналов (разные символы),
    оставляем ровно один — с максимальной strength.
    """
    if df.empty or "t_start" not in df.columns:
        return df
    x = df.copy()
    x["t_start"] = pd.to_datetime(x["t_start"], utc=True, errors="coerce")
    x["strength"] = pd.to_numeric(x.get("strength"), errors="coerce")
    x = x.sort_values(["t_start", "strength"], ascending=[True, False], kind="mergesort")
    x = x[~x["t_start"].duplicated(keep="first")].reset_index(drop=True)
    return x

def _enforce_single_position_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Глобально один слот капитала: следующая сделка открывается только
    после закрытия предыдущей (между символами).
    """
    if df.empty:
        return df
    x = df.copy()
    x["t_start"] = pd.to_datetime(x["t_start"], utc=True, errors="coerce")
    x["close_time"] = pd.to_datetime(x["close_time"], utc=True, errors="coerce")
    x = x.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").reset_index(drop=True)

    keep_idx = []
    last_close = pd.Timestamp.min.tz_localize("UTC")
    for i, r in x.iterrows():
        ts = r["t_start"]
        ct = r.get("close_time")
        if pd.isna(ts):
            continue
        if ts >= last_close:
            keep_idx.append(i)
            last_close = ct if pd.notna(ct) else ts
    return x.loc[keep_idx].reset_index(drop=True)

def _apply_slippage(price: float, pct: float, action: str) -> float:
    sp = float(pct or 0.0)
    if not isinstance(price, (int, float)) or math.isnan(price):
        return float('nan')
    # худший fill: BUY дороже, SELL дешевле
    return float(price) * (1.0 + sp) if action == "BUY" else float(price) * (1.0 - sp)

def _pnl_pct_with_fees(entry_px_adj: float, exit_px_adj: float, side: str, fee_taker_pct: float) -> float:
    if any(not isinstance(x, (int, float)) or math.isnan(x) for x in (entry_px_adj, exit_px_adj)):
        return float('nan')
    move = ((exit_px_adj - entry_px_adj) / entry_px_adj * 100.0) if side == "BUY" \
           else ((entry_px_adj - exit_px_adj) / entry_px_adj * 100.0)
    fees = float(fee_taker_pct or 0.0) * 2.0 * 100.0
    return float(move) - float(fees)

def _entry_from_4h_close(df_4h: pd.DataFrame, t0):
    """Вход в момент t0 по цене CLOSE 4h-свечи (бар со стартом t0-4h)."""
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

def _tp_sl_from_entry(entry_px_adj: float, side: str, take_pct: float, stop_pct: float):
    """TP/SL только от entry; никаких фибо."""
    e = float(entry_px_adj)
    if side == "BUY":
        tp = e * (1.0 + float(take_pct))
        sl = e * (1.0 - float(stop_pct))
    else:
        tp = e * (1.0 - float(take_pct))
        sl = e * (1.0 + float(stop_pct))
    return float(sl), float(tp)

def _resolve_exit_minute(symbol: str, side: str, entry_at, stop_eval: float, tp_eval: float, t_end):
    """Минутная дорисовка порядка TP/SL. Если оба — считаем, что SL первым."""
    t0 = to_utc_safe(entry_at)
    t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end):
        return (None, pd.NaT, float('nan'), "bad_ts")

    ltf = fetch_ltf_window(symbol, t0, t_end, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (None, pd.NaT, float('nan'), "uncertain_no_ltf")

    for ts, c in ltf.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if side == "BUY":
            hit_tp = hi >= float(tp_eval)
            hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval)
            hit_sl = hi >= float(stop_eval)

        if hit_tp and hit_sl:
            return (False, ts, float(stop_eval), "sl")
        if hit_tp:
            return (True, ts, float(tp_eval), "tp")
        if hit_sl:
            return (False, ts, float(stop_eval), "sl")

    last_ts = ltf.index[-1]
    last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_last_close")

def _resolve_exit_hybrid(symbol, side, entry_at, stop_eval, tp_eval, t_end, df_4h):
    """
    Тонкость на 4h: если бар содержит оба уровня — спускаемся на 1m.
    Для одиночного хита — тоже можем уточнить минутой.
    """
    t0 = to_utc_safe(entry_at)
    t_end = to_utc_safe(t_end)
    if pd.isna(t0) or pd.isna(t_end) or df_4h is None or df_4h.empty:
        return (None, pd.NaT, float("nan"), "bad_ts", "4h")

    bars = df_4h[(df_4h.index >= t0) & (df_4h.index < t_end)]
    for ts, c in bars.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        bar_start, bar_end = ts, ts + pd.Timedelta(hours=4)
        if side == "BUY":
            hit_tp = hi >= float(tp_eval)
            hit_sl = lo <= float(stop_eval)
        else:
            hit_tp = lo <= float(tp_eval)
            hit_sl = hi >= float(stop_eval)

        if hit_tp and hit_sl:
            win_m, close_t, trig_px, reason = _resolve_exit_minute(
                symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
            )
            return (win_m, close_t, trig_px, reason, "1m")

        if hit_tp:
            win_m, close_t, trig_px, reason = _resolve_exit_minute(
                symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
            )
            if reason in ("tp", "sl", "timeout_last_close"):
                return (win_m, close_t, trig_px, reason, "1m")
            return (True, bar_end, float(tp_eval), "tp", "4h")

        if hit_sl:
            win_m, close_t, trig_px, reason = _resolve_exit_minute(
                symbol, side, max(t0, bar_start), float(stop_eval), float(tp_eval), min(t_end, bar_end)
            )
            if reason in ("tp", "sl", "timeout_last_close"):
                return (win_m, close_t, trig_px, reason, "1m")
            return (False, bar_end, float(stop_eval), "sl", "4h")

    if not bars.empty:
        last_ts = bars.index[-1] + pd.Timedelta(hours=4)
        last_close = float(bars.iloc[-1]["close"])
    else:
        last_ts = t0
        last_close = float("nan")
    return (False, last_ts, last_close, "timeout_last_close", "4h")

def _force_timeout_exit(symbol: str, side: str, entry_at, deadline):
    """
    Принудительное закрытие по дедлайну:
    берём последнюю закрытую минутку до 'deadline' и закрываемся по её close.
    """
    t0 = to_utc_safe(entry_at)
    t1 = to_utc_safe(deadline)
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return (False, t1, float('nan'), "timeout_max_hours", "bad")
    ltf = fetch_ltf_window(symbol, t0, t1, candidates=["1m"])
    if ltf is None or ltf.empty:
        return (False, t1, float('nan'), "timeout_max_hours", "no_m1")
    last_ts = ltf.index[-1]
    last_close = float(ltf.iloc[-1]["close"])
    return (False, last_ts, last_close, "timeout_max_hours", "1m")

# ---------------- основная функция ----------------
def evaluate_momentum(
    signals_path: str,
    result_path: str,
    lookback_days: int = 360,
    interval: str = "4h",
    max_days: int = None,
    only_filled: bool = False,
    dedup: bool = False,
    initial_capital: float = None,
    capital_aware: bool = True,
    *,
    price_cache: dict = None,
):
    VERBOSE = str(os.getenv("EVAL_VERBOSE", "1")).strip().lower() in ("1","true","yes","y","on")
    def vprint(*a, **k):
        if VERBOSE: print(*a, **k, flush=True)

    # срез истории = mtime файла сигналов
    try:
        as_of = datetime.fromtimestamp(os.path.getmtime(signals_path), tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    # сигналы
    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    vprint(f"[eval] signals loaded: {len(df_sig)} rows from {signals_path}")
    if df_sig.empty:
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        finalize_write(result_path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        vprint(f"✅ saved (empty) → {result_path}")
        return

    # список символов для загрузки 4h
    symbols = df_sig["symbol"].dropna().astype(str).str.upper().unique().tolist()
    vprint(f"[eval] symbols: {len(symbols)}")

    # кэш 4h (без сети: читается из локальных минуток)
    if price_cache is None:
        price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)
        vprint(f"[eval] price cache built for {len([k for k,v in price_cache.items() if v is not None])} symbols")

    results = []
    total = len(df_sig)

    for i, (_, row) in enumerate(df_sig.iterrows(), 1):
        if i % 25 == 0 or i == 1 or i == total:
            vprint(f"[eval] {i}/{total} …")
        try:
            symbol = str(row["symbol"]).upper()
            side = _normalize_side(row.get("type"))
            t0 = to_utc_safe(row.get("imb_time"))
            df4h = price_cache.get(symbol)
            if df4h is None or df4h.empty or side not in ("BUY", "SELL") or pd.isna(t0):
                continue

            # убедимся, что индекс 4h — UTC-aware
            if not isinstance(df4h.index, pd.DatetimeIndex):
                df4h = df4h.copy()
                df4h.index = pd.to_datetime(df4h.index, utc=True, errors="coerce")
                df4h = df4h[~df4h.index.isna()]
            elif df4h.index.tz is None:
                df4h = df4h.copy()
                df4h.index = df4h.index.tz_localize("UTC")
            else:
                df4h = df4h.tz_convert("UTC")

            # вход: CLOSE 4h (бар до t0)
            entry_at, entry_px_ref = _entry_from_4h_close(df4h, t0)
            if pd.isna(entry_at) or not isinstance(entry_px_ref, (int, float)) or math.isnan(entry_px_ref):
                continue

            # слиппедж на входе
            entry_action = "BUY" if side == "BUY" else "SELL"
            entry_px_adj = _apply_slippage(entry_px_ref, ENTRY_SLIPPAGE_PCT, entry_action)

            # признаки входа (MA и т.п.); фибо/див/obos отключены
            feat = compute_entry_features(
                df4h=df4h,
                entry_at=entry_at,
                side=side,
                ma_fast=int(get_cfg("MA_FAST", cast=int, default=50)),
                ma_slow=int(get_cfg("MA_SLOW", cast=int, default=200)),
                ma_use_ema=bool(get_cfg("MA_USE_EMA", cast=bool, default=False)),
                ma_slope_lookback=int(get_cfg("MA_SLOPE_LOOKBACK", cast=int, default=1)),
                use_fib=False,
                fib_lookback_bars=0,
                fib_pivot_len=0,
                fib_set="",
                fib_tp_index=0,
                div_type="off",
                rsi_period=14,
                macd_fast=12, macd_slow=26, macd_signal=9,
                div_piv_len=0,
                div_lookback=0,
                div_confirm=0,
                div_price_eps=0.0,
                div_osc_eps=0.0,
            )

            # TP/SL только от entry
            sl_eval, tp_eval = _tp_sl_from_entry(entry_px_adj, side, TAKE_PCT, STOP_PCT)

            # TTL-окно по days + «жёсткий» лимит max_hours
            ttl_days = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
            window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)
            deadline = entry_at + pd.Timedelta(hours=EVAL_MAX_HOURS)
            time_cap = min(deadline, window_end)

            # выход через гибрид (4h + обязательная 1m дорисовка) в рамках time_cap
            win, close_time, trigger_price, exit_reason, lt_res = _resolve_exit_hybrid(
                symbol=symbol, side=side, entry_at=entry_at,
                stop_eval=float(sl_eval), tp_eval=float(tp_eval),
                t_end=time_cap, df_4h=df4h,
            )

            # если дошли до cap (deadline) и ничего не сработало — принудительно закрываем по последней минутке
            reached_cap = pd.notna(close_time) and pd.to_datetime(close_time, utc=True) >= pd.to_datetime(time_cap, utc=True)
            if exit_reason in ("timeout_last_close", "uncertain_no_ltf") and reached_cap:
                win, close_time, trigger_price, exit_reason, lt_res = _force_timeout_exit(
                    symbol=symbol, side=side, entry_at=entry_at, deadline=time_cap
                )

            if win is None:
                out = row.to_dict()
                out.update({
                    "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
                    "imb_time": t0, "t_start": entry_at,
                    "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
                    "win": pd.NA, "pnl_pct": pd.NA, "pnl_usd": pd.NA,
                    "close_time": close_time, "close_price": pd.NA, "exit_reason": str(exit_reason),
                    "exit_hours": pd.NA, "usd_alloc": pd.NA,
                    "variant": "MOMENTUM", "as_of": as_of, "skipped": True,
                    "lt_resolution": lt_res,
                    # совместимость колонок
                    "fib_tp_4h": pd.NA, "fib_set": pd.NA,
                    "fib_anchor_L": pd.NA, "fib_anchor_H": pd.NA,
                    "div4h_flag": False, "div4h_type": pd.NA, "div4h_side": pd.NA, "div4h_confirm_at": pd.NaT,
                    "obos_flag": "none", "obos_type": pd.NA, "obos_value": pd.NA,
                    "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),
                })
                out.update(feat)
                results.append(out)
                continue

            # слиппедж на выходе
            exit_action = "SELL" if side == "BUY" else "BUY"
            exit_slip = STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT
            exit_px_base = float(trigger_price) if isinstance(trigger_price, (int, float)) else float('nan')
            exit_px_adj = _apply_slippage(exit_px_base, exit_slip, exit_action)

            pnl_pct_net = _pnl_pct_with_fees(entry_px_adj, exit_px_adj, side, FEE_TAKER)

            # страховка от артефактов: sl не может быть +, tp не может быть -
            if exit_reason == "sl" and pd.notna(pnl_pct_net) and pnl_pct_net > 0:
                pnl_pct_net = -abs(pnl_pct_net)
            if exit_reason == "tp" and pd.notna(pnl_pct_net) and pnl_pct_net < 0:
                pnl_pct_net = abs(pnl_pct_net)

            # Диагностика
            fees_pct = float(FEE_TAKER or 0.0) * 2.0 * 100.0
            slip_in_pct  = float(ENTRY_SLIPPAGE_PCT or 0.0) * 100.0
            slip_out_pct = float((STOP_SLIPPAGE_PCT if exit_reason == "sl" else EXIT_SLIPPAGE_PCT) or 0.0) * 100.0
            theory_move_pct = (
                (exit_px_adj - entry_px_adj) / entry_px_adj * 100.0
                if side == "BUY" else
                (entry_px_adj - exit_px_adj) / entry_px_adj * 100.0
            )

            out = row.to_dict()
            out.update({
                "symbol": symbol, "type": side, "strength": row.get("strength", pd.NA),
                "imb_time": t0, "t_start": entry_at,
                "entry": float(entry_px_adj), "stop": float(sl_eval), "tp": float(tp_eval),
                "win": True if exit_reason == "tp" else False, "pnl_pct": float(pnl_pct_net), "pnl_usd": pd.NA,
                "close_time": close_time, "close_price": float(exit_px_adj), "exit_reason": str(exit_reason),
                "exit_hours": (
                    (pd.to_datetime(close_time, utc=True) - pd.to_datetime(entry_at, utc=True)).total_seconds() / 3600.0
                    if pd.notna(close_time) else pd.NA
                ),
                "usd_alloc": pd.NA,
                "variant": "MOMENTUM", "as_of": as_of, "skipped": False,
                "lt_resolution": lt_res,
                "fib_tp_4h": pd.NA, "fib_set": pd.NA,
                "fib_anchor_L": pd.NA, "fib_anchor_H": pd.NA,
                "div4h_flag": False, "div4h_type": pd.NA, "div4h_side": pd.NA, "div4h_confirm_at": pd.NaT,
                "obos_flag": "none", "obos_type": pd.NA, "obos_value": pd.NA,
                "entry_px_ref": float(entry_px_ref), "entry_px_adj": float(entry_px_adj),

                # Диагностика
                "fees_pct": fees_pct,
                "slip_in_pct": slip_in_pct,
                "slip_out_pct": slip_out_pct,
                "theory_move_pct": theory_move_pct,
            })
            out.update(feat)
            results.append(out)

        except Exception as e:
            vprint(f"[eval][warn] symbol {row.get('symbol')} failed: {e}")

    # ---------- сборка и запись ----------
    df = pd.DataFrame(results)
    df["sl"] = df.get("stop")
    for c in ("imb_time", "t_start", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    df = df.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").reset_index(drop=True)
    vprint(f"[eval] evaluated rows: {len(df)}")
    if df.empty:
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        finalize_write(result_path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        vprint(f"✅ saved (empty) → {result_path}")
        return

    # 1) запрет перекрытий как в лайве (по символу)
    df = enforce_one_at_a_time_per_symbol(df)

    # 2) ФИЛЬТР СИЛЫ: на каждый t_start оставляем только самый сильный сигнал
    if EVAL_PICK_STRONGEST:
        df = _choose_one_per_same_start(df)

    # 3) Глобально одна позиция одновременно, если так настроено в лайве
    if MAX_CONCURRENT_POSITIONS == 1:
        df = _enforce_single_position_global(df)

    df = df.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").reset_index(drop=True)

    # симуляция капитала/аллокаций (full equity, без дробления)
    sim = df.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").copy()

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap > 0 and capital_aware:
        from evaluate_common import simulate_capital_passthrough
        sim_out, eq_sheet = simulate_capital_passthrough(
            sim, initial_equity=init_cap, position_fraction=1.0  # весь депозит
        )
        for col in ("usd_alloc", "pnl_usd", "equity_after", "pnl_usd_comp"):
            if col in sim_out.columns:
                df[col] = sim_out[col]
    else:
        eq_sheet = pd.DataFrame()

    # (ПОСЛЕ симуляции капитала)
    df_exec = df[df.get("skipped") == False].copy()
    by_variant = (
        df_exec.groupby("variant")
        .agg(
            trades=("win", "size"),
            wins=("win", "sum"),
            winrate_pct=("win", lambda s: round(100.0 * float(s.sum()) / max(int(s.size), 1), 2)),
            pnl_pct=("pnl_pct", "sum"),
            pnl_usd=("pnl_usd", "sum"),
        )
        .reset_index()
    ) if not df_exec.empty else pd.DataFrame()
    by_exit_reason = safe_group_exit_reason(df)

    # финальный порядок и строгие типы
    cols_order = [
        "symbol", "type", "strength",
        "imb_time", "t_start", "close_time", "exit_hours", "exit_reason", "lt_resolution",
        "entry", "stop", "sl", "tp", "close_price", "win", "pnl_pct", "pnl_usd", "usd_alloc",
        "variant", "as_of",
        "entry_px_ref", "entry_px_adj",
        "fib_anchor_L", "fib_anchor_H", "fib_tp_4h", "fib_set",
        "div4h_flag", "div4h_type", "div4h_side", "div4h_confirm_at",
        "obos_type", "obos_flag", "obos_value",
        "equity_after", "pnl_usd_comp",
    ]
    df = df[[c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]]

    NUMERIC_COLS = [
        "strength","entry","stop","tp","pnl_pct","pnl_usd","close_price","exit_hours",
        "usd_alloc","alloc_usd_comp","pnl_usд_comp","equity_after",
        "ma_50","ma_200","ma_slow_slope",
        "fib_anchor_L","fib_anchor_H","fib_tp_4h","obos_value",
        "entry_px_ref","entry_px_adj", "fees_pct", "slip_in_pct", "slip_out_pct", "theory_move_pct",
    ]
    BOOL_COLS = ["win","skipped","trend_bull","trend_bear","div4h_flag","is_open_mark"]
    DATETIME_COLS = ["imb_time","t_start","close_time","as_of","div4h_confirm_at"]
    STRING_COLS = ["symbol","type","exit_reason","lt_resolution","variant","fib_set","div4h_type","div4h_side","obos_flag","obos_type"]

    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_COLS:
        if c in df.columns: df[c] = df[c].map(lambda x: bool(x) if pd.notna(x) else x)
    for c in DATETIME_COLS:
        if c in df.columns: df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in STRING_COLS:
        if c in df.columns: df[c] = df[c].astype("string").fillna(pd.NA)

    df = df.sort_values(["t_start", "imb_time", "symbol"], kind="mergesort").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    try:
        finalize_write(result_path, df, eq_sheet, by_variant, by_exit_reason)
        vprint(f"✅ MOMENTUM eval saved → {result_path}")
    except Exception as e:
        vprint(f"[eval][error] finalize_write failed: {e} — writing CSV fallbacks")
        base = os.path.splitext(result_path)[0]
        df.to_csv(base + "_trades.csv", index=False)
        if not eq_sheet.empty: eq_sheet.to_csv(base + "_equity.csv", index=False)
        if not by_variant.empty: by_variant.to_csv(base + "_by_variant.csv", index=False)
        if not by_exit_reason.empty: by_exit_reason.to_csv(base + "_by_exit.csv", index=False)

if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
    p = argparse.ArgumentParser(description="Evaluate MOMENTUM (4h close entry, entry-based TP/SL only; minute tie-resolution; local data only).")
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--only-filled", action="store_true")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--fee-taker-pct", type=float, default=None)
    p.add_argument("--entry-slippage-pct", type=float, default=None)
    p.add_argument("--exit-slippage-pct", type=float, default=None)
    p.add_argument("--stop-slippage-pct", type=float, default=None)
    # NEW:
    p.add_argument("--max-hours", type=int, default=None, help="Жёсткий лимит жизни сделки в часах (default 80, env EVAL_MAX_HOURS).")
    args = p.parse_args()

    if args.fee_taker_pct        is not None: os.environ["FEE_TAKER"]          = str(args.fee_taker_pct)
    if args.entry_slippage_pct   is not None: os.environ["ENTRY_SLIPPAGE_PCT"] = str(args.entry_slippage_pct)
    if args.exit_slippage_pct    is not None: os.environ["EXIT_SLIPPAGE_PCT"]  = str(args.exit_slippage_pct)
    if args.stop_slippage_pct    is not None: os.environ["STOP_SLIPPAGE_PCT"]  = str(args.stop_slippage_pct)
    if args.max_hours            is not None: os.environ["EVAL_MAX_HOURS"]     = str(int(args.max_hours))

    # обновить глобальную переменную после CLI
    globals()["EVAL_MAX_HOURS"] = int(get_cfg("EVAL_MAX_HOURS", cast=int, default=80))

    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_momentum_eval.xlsx"

    evaluate_momentum(
        signals_path=sig_path,
        result_path=res_path,
        lookback_days=int(args.lookback_days),
        interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled),
        dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )