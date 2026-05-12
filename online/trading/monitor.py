from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from online.trading.bybit_client import BybitClient
from online.trading.db import db_cursor, json_default, read_sql
from online.trading.locks import acquire_lock, release_lock


LOCK_NAME = "trading_monitor"
DRY_RUN = os.environ.get("IMB_TRADING_DRY_RUN", "1").strip() != "0"


ACTIVE_STATUSES = [
    "ENTRY_ORDER_SENT",
    "ENTRY_PARTIALLY_FILLED",
    "ENTRY_FILLED",
    "TP_SL_PLACED",
    "POSITION_OPEN",
]


def ensure_trade_events_table() -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS public.trading_trade_events (
            event_id BIGSERIAL PRIMARY KEY,
            trade_id BIGINT NULL,
            signal_key TEXT NULL,
            symbol TEXT NULL,
            event_type TEXT NOT NULL,
            event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_trading_trade_events_trade_id
        ON public.trading_trade_events (trade_id);

        CREATE INDEX IF NOT EXISTS idx_trading_trade_events_symbol_ts
        ON public.trading_trade_events (symbol, event_ts DESC);
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)


def load_active_positions() -> pd.DataFrame:
    placeholders = ", ".join(["%s"] * len(ACTIVE_STATUSES))
    sql = """
        SELECT *
        FROM public.trading_positions
        WHERE status IN ({})
        ORDER BY created_at ASC
    """.format(placeholders)
    return read_sql(sql, ACTIVE_STATUSES)


def insert_event(trade_id: int, signal_key: str, symbol: str, event_type: str, details: Dict[str, object]) -> None:
    sql = """
        INSERT INTO public.trading_trade_events (
            trade_id,
            signal_key,
            symbol,
            event_type,
            event_ts,
            details
        )
        VALUES (%s, %s, %s, %s, NOW(), %s::jsonb)
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                int(trade_id),
                signal_key,
                symbol,
                event_type,
                json.dumps(details, ensure_ascii=False, default=json_default),
            ),
        )


def mark_manual_or_external_close_if_position_zero(client: BybitClient, row: Dict[str, object]) -> None:
    trade_id = int(row["trade_id"])
    signal_key = str(row["signal_key"])
    symbol = str(row["symbol"]).upper()

    resp = client.get_position(symbol)
    positions = (resp.get("result") or {}).get("list") or []

    size = 0.0
    for pos in positions:
        if str(pos.get("symbol")).upper() == symbol:
            try:
                size += abs(float(pos.get("size") or 0.0))
            except Exception:
                pass

    if size > 0:
        return

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE public.trading_positions
            SET status = 'POSITION_CLOSED_MANUAL',
                exit_reason = 'MANUAL_CLOSE',
                exit_filled_at = NOW(),
                updated_at = NOW()
            WHERE trade_id = %s
              AND status NOT LIKE 'POSITION_CLOSED%%'
            """,
            (trade_id,),
        )

    insert_event(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        event_type="MANUAL_CLOSE_DETECTED",
        details={"bybit_position_response": resp},
    )


def main() -> None:
    ensure_trade_events_table()
    owner = "monitor:{}".format(pd.Timestamp.now(tz="UTC"))

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=600):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        active = load_active_positions()

        if active.empty:
            print("NO_ACTIVE_POSITIONS")
            return

        print("ACTIVE_POSITIONS:", len(active))

        if DRY_RUN:
            print("DRY_RUN: monitor does not call Bybit")
            print(active[["trade_id", "symbol", "side", "status"]].to_string(index=False))
            return

        client = BybitClient()

        for _, r in active.iterrows():
            mark_manual_or_external_close_if_position_zero(client, r.to_dict())

        print("MONITOR_DONE")

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
