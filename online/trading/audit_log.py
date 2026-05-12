from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import psycopg2

from online.trading import config


def connect_db():
    return psycopg2.connect(config.DB_DSN)


def utc_now() -> datetime:
    return pd.Timestamp.now(tz="UTC").to_pydatetime()


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def json_dumps(payload: Optional[Dict[str, Any]]) -> str:
    if payload is None:
        payload = {}

    clean = {}
    for k, v in payload.items():
        clean[str(k)] = json_safe(v)

    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def ensure_audit_tables() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS public.trading_trades (
        trade_id BIGSERIAL PRIMARY KEY,
        signal_key TEXT NOT NULL UNIQUE,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        status TEXT NOT NULL,

        signal_ts TIMESTAMPTZ NULL,
        entry_ts_plan TIMESTAMPTZ NULL,
        entry_ts_actual TIMESTAMPTZ NULL,
        exit_ts_actual TIMESTAMPTZ NULL,

        entry_px_plan DOUBLE PRECISION NULL,
        entry_px_actual DOUBLE PRECISION NULL,
        exit_px_actual DOUBLE PRECISION NULL,

        qty DOUBLE PRECISION NULL,
        tp_px DOUBLE PRECISION NULL,
        sl_px DOUBLE PRECISION NULL,

        tp_order_id TEXT NULL,
        sl_order_id TEXT NULL,
        entry_order_id TEXT NULL,
        exit_order_id TEXT NULL,

        exit_reason TEXT NULL,

        gross_ret DOUBLE PRECISION NULL,
        net_ret DOUBLE PRECISION NULL,
        pnl_usd DOUBLE PRECISION NULL,

        backtest_entry_px DOUBLE PRECISION NULL,
        backtest_exit_px DOUBLE PRECISION NULL,
        backtest_exit_reason TEXT NULL,
        backtest_net_ret DOUBLE PRECISION NULL,
        backtest_pnl_usd DOUBLE PRECISION NULL,

        entry_slippage_abs DOUBLE PRECISION NULL,
        entry_slippage_pct DOUBLE PRECISION NULL,
        exit_slippage_abs DOUBLE PRECISION NULL,
        exit_slippage_pct DOUBLE PRECISION NULL,

        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_trading_trades_symbol_created
    ON public.trading_trades(symbol, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_trading_trades_status
    ON public.trading_trades(status);

    CREATE TABLE IF NOT EXISTS public.trading_order_events (
        event_id BIGSERIAL PRIMARY KEY,
        trade_id BIGINT NULL,
        signal_key TEXT NULL,
        symbol TEXT NULL,
        side TEXT NULL,

        event_type TEXT NOT NULL,
        order_role TEXT NULL,
        order_id TEXT NULL,
        client_order_id TEXT NULL,
        bybit_status TEXT NULL,

        request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,

        event_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_trading_order_events_trade_id
    ON public.trading_order_events(trade_id);

    CREATE INDEX IF NOT EXISTS idx_trading_order_events_signal_key
    ON public.trading_order_events(signal_key);

    CREATE INDEX IF NOT EXISTS idx_trading_order_events_event_ts
    ON public.trading_order_events(event_ts DESC);

    CREATE TABLE IF NOT EXISTS public.trading_audit_events (
        event_id BIGSERIAL PRIMARY KEY,
        trade_id BIGINT NULL,
        signal_key TEXT NULL,
        symbol TEXT NULL,
        side TEXT NULL,

        event_type TEXT NOT NULL,
        status TEXT NULL,
        message TEXT NULL,

        payload JSONB NOT NULL DEFAULT '{}'::jsonb,

        event_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_trading_audit_events_trade_id
    ON public.trading_audit_events(trade_id);

    CREATE INDEX IF NOT EXISTS idx_trading_audit_events_signal_key
    ON public.trading_audit_events(signal_key);

    CREATE INDEX IF NOT EXISTS idx_trading_audit_events_event_ts
    ON public.trading_audit_events(event_ts DESC);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def signal_get(signal: Any, key: str, default: Any = None) -> Any:
    if signal is None:
        return default

    if isinstance(signal, dict):
        return signal.get(key, default)

    try:
        value = signal[key]
        return value
    except Exception:
        pass

    try:
        return getattr(signal, key)
    except Exception:
        return default


def log_audit_event(
    event_type: str,
    trade_id: Optional[int] = None,
    signal_key: Optional[str] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    status: Optional[str] = None,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_audit_tables()

    sql = """
    INSERT INTO public.trading_audit_events (
        trade_id,
        signal_key,
        symbol,
        side,
        event_type,
        status,
        message,
        payload,
        event_ts
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    trade_id,
                    signal_key,
                    symbol,
                    side,
                    event_type,
                    status,
                    message,
                    json_dumps(payload),
                ),
            )
        conn.commit()


def log_order_event(
    event_type: str,
    order_role: str,
    trade_id: Optional[int] = None,
    signal_key: Optional[str] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    bybit_status: Optional[str] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    response_payload: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_audit_tables()

    sql = """
    INSERT INTO public.trading_order_events (
        trade_id,
        signal_key,
        symbol,
        side,
        event_type,
        order_role,
        order_id,
        client_order_id,
        bybit_status,
        request_payload,
        response_payload,
        event_ts
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    trade_id,
                    signal_key,
                    symbol,
                    side,
                    event_type,
                    order_role,
                    order_id,
                    client_order_id,
                    bybit_status,
                    json_dumps(request_payload),
                    json_dumps(response_payload),
                ),
            )
        conn.commit()
