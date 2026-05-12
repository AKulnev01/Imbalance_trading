from __future__ import annotations

import pandas as pd

from online.trading.db import db_cursor, read_sql


def refresh_latency_audit() -> int:
    sql = """
        INSERT INTO public.trading_audit_latency (
            signal_key,
            symbol,
            h4_open_ts,
            h4_close_ts,
            orchestrator_started_at,
            selector_finished_at,
            order_sent_at,
            order_ack_at,
            first_fill_at,
            seconds_close_to_orchestrator,
            seconds_close_to_order_sent,
            seconds_order_sent_to_ack,
            seconds_order_sent_to_first_fill
        )
        SELECT
            s.signal_key,
            s.symbol,
            s.signal_ts AS h4_open_ts,
            s.signal_ts + INTERVAL '4 hours' AS h4_close_ts,
            s.created_at AS orchestrator_started_at,
            s.updated_at AS selector_finished_at,
            MIN(o.sent_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET') AS order_sent_at,
            MIN(o.acknowledged_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET') AS order_ack_at,
            MIN(f.executed_at) AS first_fill_at,

            EXTRACT(EPOCH FROM (s.created_at - (s.signal_ts + INTERVAL '4 hours'))) AS seconds_close_to_orchestrator,
            EXTRACT(EPOCH FROM (MIN(o.sent_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET') - (s.signal_ts + INTERVAL '4 hours'))) AS seconds_close_to_order_sent,
            EXTRACT(EPOCH FROM (
                MIN(o.acknowledged_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET')
                - MIN(o.sent_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET')
            )) AS seconds_order_sent_to_ack,
            EXTRACT(EPOCH FROM (
                MIN(f.executed_at)
                - MIN(o.sent_at) FILTER (WHERE o.order_role = 'ENTRY_MARKET')
            )) AS seconds_order_sent_to_first_fill

        FROM public.trading_signals s
        LEFT JOIN public.trading_orders o
            ON o.signal_key = s.signal_key
        LEFT JOIN public.trading_positions p
            ON p.signal_key = s.signal_key
        LEFT JOIN public.trading_fills f
            ON f.trade_id = p.trade_id
        WHERE s.selected = TRUE
        GROUP BY
            s.signal_key,
            s.symbol,
            s.signal_ts,
            s.created_at,
            s.updated_at
        ON CONFLICT DO NOTHING
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)
        return int(cur.rowcount)


def print_live_summary() -> None:
    sql = """
        SELECT
            status,
            COUNT(*) AS n,
            SUM(COALESCE(pnl_usd, 0.0)) AS pnl_usd
        FROM public.trading_positions
        GROUP BY status
        ORDER BY status
    """
    df = read_sql(sql)
    if df.empty:
        print("NO_TRADING_POSITIONS")
    else:
        print(df.to_string(index=False))


def main() -> None:
    inserted = refresh_latency_audit()
    print("LATENCY_AUDIT_INSERTED:", inserted)
    print_live_summary()


if __name__ == "__main__":
    main()
