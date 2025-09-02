# evaluate_retest.py
import os
import math
import pandas as pd
from datetime import datetime, timezone

from evaluate_common import (
    get_cfg, INITIAL_CAPITAL, POSITION_FRACTION, DEFAULT_TTL_DAYS,
    ensure_dt_index, to_num, to_utc_safe, first_touch_after_ltf, exit_on_ltf,
    enforce_one_at_a_time_per_symbol, simulate_capital_notional,
    safe_group_exit_reason, load_signals, load_price_cache, finalize_write, calc_sl_tp
)

TAKE_PCT = float(get_cfg("MOMENTUM_TP_PCT", cast=float))  # используем те же TP/SL проценты
STOP_PCT = float(get_cfg("MOMENTUM_SL_PCT", cast=float))


def evaluate_retest(signals_path: str,
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

    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup)
    if df_sig.empty:
        print("⚠️ Сигналов нет после фильтров."); return

    symbols = df_sig['symbol'].dropna().unique().tolist()
    price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)

    results = []
    for _, row in df_sig.iterrows():
        symbol = row['symbol']; side = str(row['type']).upper().strip()
        t0 = to_utc_safe(row['imb_time']); entry_base = to_num(row['entry'])
        df = price_cache.get(symbol)
        if df is None or df.empty or pd.isna(entry_base) or side not in ('BUY','SELL'):
            continue

        rr = float(TAKE_PCT) / max(float(STOP_PCT), 1e-9)
        stop_eval, tp_eval = calc_sl_tp(float(entry_base), side, float(STOP_PCT), rr)

        ttl_days  = int(DEFAULT_TTL_DAYS if max_days is None else max_days)
        window_end = min(t0 + pd.Timedelta(days=ttl_days), as_of)

        entry_at = first_touch_after_ltf(symbol, float(entry_base), t0, window_end)
        if pd.notna(entry_at):
            win, close_time, close_price, exit_reason = exit_on_ltf(
                symbol=symbol, side=side, entry_at=entry_at,
                stop_eval=float(stop_eval), tp_eval=float(tp_eval), t_end=window_end
            )
        else:
            close_time = t0 + pd.Timedelta(days=ttl_days); close_price = float('nan')
            exit_reason = "timeout_no_fill"; win = False

        if pd.notna(entry_at) and pd.notna(close_time) and not (isinstance(close_price,float) and math.isnan(close_price)):
            if side == 'BUY':
                move_pct = (float(close_price) - float(entry_base))/float(entry_base)*100.0
            else:
                move_pct = (float(entry_base) - float(close_price))/float(entry_base)*100.0
        else:
            move_pct = float('nan')

        out = row.to_dict()
        out.update({
            "variant": "RETEST",
            "as_of": as_of,
            "stop_eval": float(stop_eval), "tp_eval": float(tp_eval),
            "win": True if exit_reason == "tp" else False,
            "risk_pct": STOP_PCT*100.0, "profit_pct": TAKE_PCT*100.0,
            "move_pct": move_pct, "pnl_pct": move_pct, "pnl_usd": pd.NA,
            "close_time": close_time,
            "close_price": float(close_price) if close_price==close_price else pd.NA,
            "exit_reason": exit_reason, "is_open_mark": False,
            "t_start": entry_at, "size_weight": 1.0,
        })
        results.append(out)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("⚠️ Ничего не оценили."); return

    df_res = enforce_one_at_a_time_per_symbol(df_res)
    for c in ['t_start','close_time','imb_time']:
        if c in df_res.columns:
            df_res[c] = df_res[c].map(to_utc_safe)
    df_res['exit_time'] = df_res['close_time']
    t_start_utc = pd.to_datetime(df_res['t_start'], utc=True, errors='coerce')
    t_exit_utc  = pd.to_datetime(df_res['exit_time'], utc=True, errors='coerce')
    df_res['exit_days'] = ((t_exit_utc - t_start_utc).dt.total_seconds()/86400.0).round(3)

    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    if init_cap <= 0 or not capital_aware:
        df_out = df_res.copy(); df_out['skipped'] = False; eq_sheet = pd.DataFrame()
    else:
        df_out, eq_sheet = simulate_capital_notional(df_res, init_cap, POSITION_FRACTION, stop_pct=STOP_PCT, take_pct=TAKE_PCT)

    df_exec = df_out[df_out['skipped']==False].copy()
    by_variant = (df_exec.groupby('variant')
                        .agg(trades=('win','size'), wins=('win','sum'),
                             winrate_pct=('win', lambda s: round(100.0*float(s.sum())/max(int(s.size),1),2)),
                             pnl_pct=('pnl_pct','sum'), pnl_usd=('pnl_usd','sum'))
                        .reset_index()) if not df_exec.empty else pd.DataFrame()
    by_exit_reason = safe_group_exit_reason(df_out)

    finalize_write(result_path, df_out, eq_sheet, by_variant, by_exit_reason)
    print(f"✅ RETEST eval saved → {result_path}")


if __name__ == "__main__":
    import argparse
    def _b(v): return str(v).strip().lower() in ("1","true","yes","y","on")
    p = argparse.ArgumentParser(description="Evaluate RETEST (first 1m touch after t0).")
    p.add_argument("signals"); p.add_argument("--out", default=None)
    p.add_argument("--interval", default="4h"); p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--ttl-days", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--capital-aware", type=_b, default=True)
    p.add_argument("--intrabar", default=None)
    p.add_argument("--intrabar-lookback-days", type=int, default=None)
    p.add_argument("--only-filled", action="store_true"); p.add_argument("--dedup", action="store_true")
    args = p.parse_args()

    sig_path = os.path.expanduser(args.signals)
    res_path = os.path.expanduser(args.out) if args.out else os.path.splitext(sig_path)[0] + "_retest_eval.xlsx"

    if args.intrabar is not None:
        os.environ["INTRABAR_INTERVALS"] = args.intrabar
    if args.intrabar_lookback_days is not None:
        os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(args.intrabar_lookback_days))

    evaluate_retest(
        signals_path=sig_path, result_path=res_path,
        lookback_days=int(args.lookback_days), interval=str(args.interval),
        max_days=(int(args.ttl_days) if args.ttl_days is not None else None),
        only_filled=bool(args.only_filled), dedup=bool(args.dedup),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )