# optimize_params.py
import os, math, itertools, tempfile
import pandas as pd
from datetime import timezone, datetime
from typing import List, Tuple
from utils.fetch_data import get_bybit_klines
from utils.detect_fvg import detect_fvg_imbalances
from utils.strategy import select_entry_price
from config import TRADE_UNIVERSE, DEFAULT_TTL_DAYS, DEFAULT_MIN_STRENGTH, INITIAL_CAPITAL
from evaluate_signals import evaluate_signals  # импортируем твою функцию

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _detect_signals_for_symbol(df4h: pd.DataFrame,
                               symbol: str,
                               min_strength: float,
                               vol_mult: float,
                               tol_pct: float,
                               max_days_to_fill: int) -> List[dict]:
    imbs = detect_fvg_imbalances(
        df4h,
        volume_multiplier=float(vol_mult),
        tolerance_pct=float(tol_pct),
        min_strength_pct=float(min_strength),
        max_days_to_fill=int(max_days_to_fill),
    ) or []
    rows = []
    for imb in imbs:
        side = str(imb.get("type","")).upper()
        if side not in ("BUY","SELL"): continue
        entry = select_entry_price(df4h, symbol, imb)
        if entry is None or entry <= 0: continue
        t0 = pd.to_datetime(imb["time"], utc=True)
        rows.append({
            "symbol":   symbol,
            "imb_time": t0,
            "type":     side,
            "entry":    float(entry),
            "stop":     pd.NA,   # фактические SL/TP будет считать evaluate_signals из STOP_PCT/TAKE_PCT
            "tp":       pd.NA,
            "strength": float(imb.get("strength", 0.0)),
        })
    return rows

def _prefetch_ohlc(symbols: List[str], interval: str, days: int) -> dict:
    cache = {}
    for s in symbols:
        df = get_bybit_klines(symbol=s, interval=interval, lookback_days=days)
        df = _ensure_dt_index(df)
        cache[s] = df
    return cache

def run_sweep(symbols: List[str],
              interval: str = "4h",
              lookback_days: int = 360,
              grid_strength: List[float] = (2.0, 2.5, 3.0),
              grid_vol: List[float] = (1.1, 1.3, 1.5),
              grid_tol: List[float] = (0.0, 0.0005),
              rr_list: List[int]   = (3,),
              initial_capital: float = None,
              out_path: str = None,
              intrabar: str = None,
              intrabar_lookback_days: int = None):
    symbols = list(dict.fromkeys(symbols))
    df_cache = _prefetch_ohlc(symbols, interval, lookback_days)

    results = []
    tmpdir = tempfile.gettempdir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    if out_path is None:
        reports_dir = os.path.expanduser("~/Documents/отчеты")
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, f"param_sweep_{len(symbols)}sym_{lookback_days}d_{interval}_{ts}.xlsx")

    # используем один и тот же initial_capital, если не задан — берём из config
    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)

    # sweep
    for (ms, vm, tol, rr) in itertools.product(grid_strength, grid_vol, grid_tol, rr_list):
        # подготовим STOP/TAKE в ENV для evaluate_signals (цена-ориентированные %)
        os.environ["STOP_PCT"] = "0.01"                 # фиксируем риск 1% по цене
        os.environ["TAKE_PCT"] = f"{0.01*float(rr):.6f}"  # RR = take/stop

        # детект по всем символам с конкретной комбинацией параметров
        rows = []
        for s in symbols:
            df4h = df_cache.get(s)
            if df4h is None or df4h.empty: continue
            rows.extend(_detect_signals_for_symbol(
                df4h, s, min_strength=ms, vol_mult=vm, tol_pct=tol, max_days_to_fill=DEFAULT_TTL_DAYS
            ))

        if not rows:
            results.append({
                "min_strength": ms, "vol_mult": vm, "tol_pct": tol, "rr": rr,
                "signals": 0, "trades": 0, "wins": 0, "winrate_pct": 0.0,
                "pnl_usd": 0.0, "total_return_pct": 0.0
            })
            continue

        # сохраняем сигналы во временный Excel в формате, который ждёт evaluate_signals()
        tmp_signals = os.path.join(tmpdir, f"signals_tmp_{ms}_{vm}_{tol}_{rr}.xlsx")
        df_sig = pd.DataFrame(rows)
        # Excel-friendly (без tz)
        for c in ("imb_time",):
            if c in df_sig.columns:
                df_sig[c] = pd.to_datetime(df_sig[c], utc=True, errors="coerce").dt.tz_localize(None)
        df_sig.to_excel(tmp_signals, index=False)

        # прокидываем LTF-окна (если надо)
        if intrabar:
            os.environ["INTRABAR_INTERVALS"] = intrabar
        if intrabar_lookback_days is not None:
            os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(intrabar_lookback_days))

        # запустим оценку (внутри она сама подтянет цены для исходов)
        tmp_out = os.path.join(tmpdir, f"signals_eval_{ms}_{vm}_{tol}_{rr}.xlsx")
        evaluate_signals(
            signals_path=tmp_signals,
            result_path=tmp_out,
            lookback_days=lookback_days,
            interval=interval,
            max_days=DEFAULT_TTL_DAYS,
            include_open=False,
            compounding=True,
            initial_capital=init_cap,
            capital_aware=True,
        )

        # вытащим сводные метрики из результата
        try:
            by_var = pd.read_excel(tmp_out, sheet_name="by_variant")
        except Exception:
            by_var = pd.DataFrame()

        try:
            eq_sum = pd.read_excel(tmp_out, sheet_name="equity_summary")
            tr = float(eq_sum.loc[eq_sum["metric"]=="total_return_pct","value"].iloc[0]) if not eq_sum.empty else 0.0
        except Exception:
            tr = 0.0

        if not by_var.empty:
            # там одна «variant» (RETEST), берём её
            rec = by_var.iloc[0].to_dict()
            results.append({
                "min_strength": ms, "vol_mult": vm, "tol_pct": tol, "rr": rr,
                "signals": len(df_sig),
                "trades": int(rec.get("trades", 0)),
                "wins": int(rec.get("wins", 0)),
                "winrate_pct": float(rec.get("winrate_pct", 0.0)),
                "pnl_usd": float(rec.get("pnl_usd", 0.0)),
                "pnl_pct_sum": float(rec.get("pnl_pct", 0.0)),
                "total_return_pct": tr,
            })
        else:
            results.append({
                "min_strength": ms, "vol_mult": vm, "tol_pct": tol, "rr": rr,
                "signals": len(df_sig),
                "trades": 0, "wins": 0, "winrate_pct": 0.0, "pnl_usd": 0.0,
                "pnl_pct_sum": 0.0, "total_return_pct": tr,
            })

    # собираем финальный excel
    df_res = pd.DataFrame(results).sort_values(
        ["pnl_usd","winrate_pct","trades"], ascending=[False, False, False]
    ).reset_index(drop=True)

    topN = df_res.head(20).copy()
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(out_path) as wr:
        df_res.to_excel(wr, sheet_name="summary", index=False)
        topN.to_excel(wr, sheet_name="top20", index=False)
    print(f"✅ Sweep saved: {out_path}")
    return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Grid-search параметров детектора FVG с авто-оценкой PnL")
    p.add_argument("--interval", default="4h")
    p.add_argument("--lookback-days", type=int, default=360)
    p.add_argument("--symbols", default=None, help="через запятую; по умолчанию — из config.TRADE_UNIVERSE")
    p.add_argument("--grid-strength", default="2.0,2.5,3.0")
    p.add_argument("--grid-vol", default="1.1,1.3,1.5")
    p.add_argument("--grid-tol", default="0,0.0005")
    p.add_argument("--rr", default="3", help="список RR через запятую, например 3,4,5")
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--intrabar", default=None, help='например "1m,5m"')
    p.add_argument("--intrabar-lookback-days", type=int, default=None)
    args = p.parse_args()

    symbols = [s.strip() for s in (args.symbols.split(",") if args.symbols else TRADE_UNIVERSE) if s.strip()]
    g_strength = [float(x) for x in args.grid_strength.split(",") if x.strip()]
    g_vol      = [float(x) for x in args.grid_vol.split(",") if x.strip()]
    g_tol      = [float(x) for x in args.grid_tol.split(",") if x.strip()]
    rr_list    = [int(x) for x in args.rr.split(",") if x.strip()]

    run_sweep(
        symbols=symbols,
        interval=args.interval,
        lookback_days=int(args.lookback_days),
        grid_strength=g_strength,
        grid_vol=g_vol,
        grid_tol=g_tol,
        rr_list=rr_list,
        initial_capital=args.initial_capital,
        out_path=args.out,
        intrabar=args.intrabar,
        intrabar_lookback_days=args.intrabar_lookback_days,
    )