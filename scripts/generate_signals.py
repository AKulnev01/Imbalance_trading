import os
import sys
import pandas as pd
from datetime import datetime, timezone

# гарантируем, что корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.strategy import scan_universe, filter_universe_to_local
from utils.symbols import fetch_top_symbols
from config import TRADE_UNIVERSE, DEFAULT_MIN_STRENGTH


def _drop_tz_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in df.select_dtypes(include=["datetimetz"]).columns:
        df[c] = df[c].dt.tz_localize(None)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_signals.py <filename.xlsx> [lookback_days] [interval] [mode]")
        sys.exit(1)

    filename = sys.argv[1]
    out_dir = os.path.expanduser("~/Documents/отчеты")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    lookback = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.getenv("LOOKBACK_DAYS", "360"))
    interval = sys.argv[3] if len(sys.argv) > 3 else "4h"
    mode = sys.argv[4].lower() if len(sys.argv) > 4 else "all"

    universe = list(dict.fromkeys(TRADE_UNIVERSE)) or fetch_top_symbols()[:100]

    print(f"🔄 Generating signals → {out_path}")
    print(f"   universe={len(universe)}, lookback={lookback}d, TF={interval}, mode={mode}, min_strength={DEFAULT_MIN_STRENGTH}")

    universe = filter_universe_to_local(universe)
    print(f"   после фильтра по минуткам: {len(universe)} символов")

    if not universe:
        print("⚠️ Нет символов с локальными минутками. Проверь OHLCV_ROOT=./data/ohlcv/1m")
        sys.exit(0)

    df = scan_universe(universe=universe, lookback_days=lookback, mode=mode, interval=interval)
    if df is None or df.empty:
        print("⚠️ Сигналов нет, файл не создан.")
        sys.exit(0)

    df = df.rename(columns={"side": "type"})
    if "type" not in df.columns:
        print("⚠️ Нет колонки type/side в данных — проверь scan_universe().")
        sys.exit(1)

    df["type"] = df["type"].astype(str).str.upper()
    df["imb_time"] = pd.to_datetime(df["imb_time"], utc=True, errors="coerce")

    df = _drop_tz_cols(df)

    with pd.ExcelWriter(out_path, engine="openpyxl") as wr:
        df.to_excel(wr, index=False, sheet_name="data")
        meta = pd.DataFrame({
            "param": ["lookback_days", "interval", "mode", "symbols", "generated_utc", "rows"],
            "value": [
                lookback,
                interval,
                mode,
                len(universe),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                len(df),
            ],
        })
        meta.to_excel(wr, index=False, sheet_name="meta")

    print(f"✅ Saved: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()