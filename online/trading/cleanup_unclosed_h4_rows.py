from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from online.trading.db import db_cursor


def latest_closed_h4_open_utc() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    current_h4_open = now.floor("4h")
    return current_h4_open - pd.Timedelta(hours=4)


def cleanup_unclosed_h4_rows() -> None:
    latest_closed = latest_closed_h4_open_utc()

    targets: List[Tuple[str, str]] = [
        ("public.online_gate4_features", "entry_ts"),
        ("public.online_gate2_predictions", "entry_ts"),
        ("public.online_gate4_predictions_no_raw_refs", "signal_ts"),
        ("public.online_gate5_1_scores", "signal_ts"),
        ("public.online_gate5_2_ranker", "signal_ts"),
        ("public.online_gate5_3_decisions", "signal_ts"),
        ("public.trading_signals", "signal_ts"),
    ]

    print("=" * 120, flush=True)
    print("CLEANUP_UNCLOSED_H4_ROWS", flush=True)
    print("latest_closed_allowed:", latest_closed, flush=True)

    with db_cursor(commit=True) as (_, cur):
        for table_name, ts_col in targets:
            sql = "DELETE FROM {} WHERE {} > %s".format(table_name, ts_col)
            cur.execute(sql, [latest_closed.to_pydatetime()])
            print("{} deleted: {}".format(table_name, cur.rowcount), flush=True)


def main() -> None:
    cleanup_unclosed_h4_rows()


if __name__ == "__main__":
    main()
