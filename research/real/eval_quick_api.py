import os, sys, pathlib, asyncio
import pandas as pd
from typing import Dict
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real.utils.market_data_api import fetch_kline
from evaluate_momentum import evaluate_momentum
from evaluate_common import load_signals

def _save_m1_temp(m1_cache: Dict[str, pd.DataFrame], root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    for sym, df in m1_cache.items():
        if df is None or df.empty:
            continue
        df.reset_index().to_parquet(root / f"{sym}.parquet", index=False)

def _make_4h_cache(h4_cache: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym, df in h4_cache.items():
        out[sym] = df.copy() if df is not None and not df.empty else pd.DataFrame()
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Quick eval via Bybit API (m1+4h)")
    p.add_argument("signals")
    p.add_argument("--out", default=None)
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--use-mainnet", action="store_true")
    p.add_argument("--category", default="linear")
    args = p.parse_args()

    sig_path = os.path.expanduser(args.signals)
    df_sig = load_signals(sig_path, only_filled=False, dedup=False, require_entry=False)
    df_sig = df_sig.sort_values("imb_time").tail(args.last)
    symbols = sorted(df_sig["symbol"].dropna().astype(str).str.upper().unique().tolist())

    lookback_minutes = args.lookback_days * 24 * 60
    m1 = asyncio.run(fetch_kline(symbols, interval="1", category=args.category, lookback_minutes=lookback_minutes, use_mainnet=args.use_mainnet))
    h4 = asyncio.run(fetch_kline(symbols, interval="240", category=args.category, lookback_minutes=lookback_minutes, use_mainnet=args.use_mainnet))

    tmp_root = pathlib.Path("./data/m1_api_tmp")
    os.environ["USE_LOCAL_MINUTES"] = "1"
    os.environ["LTF_ROOT"] = str(tmp_root)
    _save_m1_temp(m1, tmp_root)

    price_cache = _make_4h_cache(h4)
    out_path = args.out or os.path.splitext(sig_path)[0] + "_quick_api_eval.xlsx"

    evaluate_momentum(
        signals_path=sig_path,
        result_path=out_path,
        lookback_days=args.lookback_days,
        interval="4h",
        only_filled=False,
        dedup=False,
        capital_aware=True,
        price_cache=price_cache,
    )

    print(f"✅ Saved → {out_path}")