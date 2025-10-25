import os
import sys
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# гарантируем, что корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.strategy import scan_universe, filter_universe_to_local  # ← локальные минутки
from utils.symbols import fetch_top_symbols
from config import TRADE_UNIVERSE


def _drop_tz_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ['imb_time', 'filled_at', 'as_of', 'time', 'close_time', 'exit_time', 'timestamp']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
            except Exception:
                pass
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)
    return df


def main():
    """
    Usage:
      python scripts/generate_signals_variant.py <filename.xlsx> [lookback_days] [interval] [mode]
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_signals_variant.py <filename.xlsx> [lookback_days] [interval] [mode]")
        sys.exit(1)

    filename = sys.argv[1]
    out_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    lookback = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    interval = sys.argv[3] if len(sys.argv) > 3 else "4h"
    mode     = sys.argv[4].lower() if len(sys.argv) > 4 else "all"

    universe = list(dict.fromkeys(TRADE_UNIVERSE)) or fetch_top_symbols()[:100]
    # оставляем только те символы, у которых есть локальные минутки (./data/ohlcv/1m/*.parquet)
    universe = filter_universe_to_local(universe)

    print(f"🔄 Generating signals (variant) → {out_path}")
    print(f"   universe={len(universe)}, lookback={lookback}d, TF={interval}, mode={mode}")

    df = scan_universe(universe=universe, lookback_days=lookback, mode=mode, interval=interval)
    if df is None or df.empty:
        print("⚠️ Сигналов нет, файл не создан.")
        sys.exit(0)

    df = _drop_tz_cols(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as wr:
        df.to_excel(wr, index=False, sheet_name="data")
        meta = pd.DataFrame({
            "param": ["lookback_days","interval","mode","symbols","generated_utc","rows"],
            "value": [lookback, interval, mode, len(universe),
                      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), len(df)]
        })
        meta.to_excel(wr, index=False, sheet_name="meta")

    print(f"✅ Saved: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()