from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2

from pybit.unified_trading import WebSocket

from online.trading import audit_log
from online.trading import config
from online.trading.bybit_client import BybitClient
from online.trading.position_lifecycle import (
    ACTIVE_POSITION_STATUSES,
    cancel_remaining_protective_orders_once,
    cleanup_zero_exchange_position_for_active_trade,
    handle_early_stop_expired,
    handle_partial_tp_filled,
    load_active_position_by_order_link_id,
    load_active_positions_with_due_early_stop,
)


def load_env_file_for_ws_listener() -> None:
    env_file = config.ROOT / ".env"

    if not env_file.exists():
        return

    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_env_file_for_ws_listener()
from online.trading.db import db_cursor, json_default


SERVICE_NAME = "trading_ws_listener"
ROOT = config.ROOT

STOP = False
WS_LOCK_CONN = None

WS_RECONNECT_SLEEP_SECONDS = float(os.environ.get("IMB_WS_RECONNECT_SLEEP_SECONDS", "5"))
WS_HEARTBEAT_SECONDS = float(os.environ.get("IMB_WS_HEARTBEAT_SECONDS", "30"))
WS_LIFECYCLE_TIMER_CHECK_SECONDS = float(os.environ.get("IMB_WS_LIFECYCLE_TIMER_CHECK_SECONDS", "5"))
WS_FORCED_RECONNECT_SECONDS = float(os.environ.get("IMB_WS_FORCED_RECONNECT_SECONDS", "900"))
WS_NO_EVENT_RECONNECT_SECONDS = float(os.environ.get("IMB_WS_NO_EVENT_RECONNECT_SECONDS", "0"))
WS_ENVIRONMENT = os.environ.get("IMB_BYBIT_WS_ENVIRONMENT", "mainnet").strip().lower()

DB_LOCK_KEY_1 = 918273645
DB_LOCK_KEY_2 = 20260519

LIFECYCLE_CLIENT: Optional[BybitClient] = None
LAST_LIFECYCLE_TIMER_CHECK_MONO = 0.0
LAST_WS_EVENT_MONO = 0.0
WS_CONNECT_MONO = 0.0


class PlannedWsReconnect(Exception):
    pass


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def get_config_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is not None and str(value).strip():
        return str(value).strip()

    value = getattr(config, name, None)
    if value is not None and str(value).strip():
        return str(value).strip()

    return default


def get_bybit_api_key() -> str:
    for name in ["BYBIT_API_KEY", "BYBIT_KEY", "BYBIT_API_PUBLIC_KEY"]:
        value = get_config_value(name)
        if value:
            return value

    raise RuntimeError("Bybit API key not found. Expected BYBIT_API_KEY / BYBIT_KEY / BYBIT_API_PUBLIC_KEY")


def get_bybit_api_secret() -> str:
    for name in ["BYBIT_API_SECRET", "BYBIT_SECRET", "BYBIT_API_PRIVATE_KEY"]:
        value = get_config_value(name)
        if value:
            return value

    raise RuntimeError("Bybit API secret not found. Expected BYBIT_API_SECRET / BYBIT_SECRET / BYBIT_API_PRIVATE_KEY")


def is_testnet() -> bool:
    if WS_ENVIRONMENT in {"testnet", "test", "paper"}:
        return True
    return False


def install_signal_handlers() -> None:
    def handler(signum: int, frame: Any) -> None:
        global STOP
        STOP = True
        print("WS_LISTENER_STOP_SIGNAL", flush=True)
        print("signal:", int(signum), flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)




def ensure_ws_lifecycle_columns() -> None:
    sql = """
        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS partial_tp_handled_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS early_stop_replaced_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS protective_orders_cleanup_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS protective_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS closed_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS position_seen_open_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_updated_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_last_event_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_note TEXT;

        ALTER TABLE public.trading_orders
            ADD COLUMN IF NOT EXISTS ws_seen_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_event_uid TEXT;
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)



def ensure_ws_tables() -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS public.trading_ws_events (
            event_id BIGSERIAL PRIMARY KEY,
            event_uid TEXT NOT NULL,
            event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            topic TEXT NULL,
            event_type TEXT NULL,
            symbol TEXT NULL,
            side TEXT NULL,
            order_id TEXT NULL,
            order_link_id TEXT NULL,
            exec_id TEXT NULL,
            order_status TEXT NULL,
            exec_type TEXT NULL,
            position_size DOUBLE PRECISION NULL,
            raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_trading_ws_events_event_uid
        ON public.trading_ws_events (event_uid);

        CREATE INDEX IF NOT EXISTS idx_trading_ws_events_ts
        ON public.trading_ws_events (event_ts DESC);

        CREATE INDEX IF NOT EXISTS idx_trading_ws_events_symbol_ts
        ON public.trading_ws_events (symbol, event_ts DESC);

        CREATE INDEX IF NOT EXISTS idx_trading_ws_events_topic_ts
        ON public.trading_ws_events (topic, event_ts DESC);

        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS early_stop_replaced_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS partial_tp_handled_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS protective_cleanup_done_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS closed_cleanup_done_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS position_seen_open_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_updated_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_last_event_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_last_error TEXT NULL;

        ALTER TABLE public.trading_orders
            ADD COLUMN IF NOT EXISTS ws_last_event_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS ws_last_event_uid TEXT NULL;
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)



def get_db_dsn() -> str:
    return os.environ.get("IMB_DB_DSN", config.DB_DSN)


def acquire_db_lock() -> bool:
    global WS_LOCK_CONN

    dsn = get_db_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (int(DB_LOCK_KEY_1), int(DB_LOCK_KEY_2)),
        )
        row = cur.fetchone()

    locked = bool(row[0])

    if not locked:
        try:
            conn.close()
        except Exception:
            pass
        return False

    WS_LOCK_CONN = conn

    print("=" * 120, flush=True)
    print("WS_DB_ADVISORY_LOCK_ACQUIRED", flush=True)
    print("db:", dsn, flush=True)
    print("lock_key_1:", DB_LOCK_KEY_1, flush=True)
    print("lock_key_2:", DB_LOCK_KEY_2, flush=True)

    return True


def release_db_lock() -> None:
    global WS_LOCK_CONN

    conn = WS_LOCK_CONN
    WS_LOCK_CONN = None

    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(%s, %s)",
                (int(DB_LOCK_KEY_1), int(DB_LOCK_KEY_2)),
            )
            unlocked = bool(cur.fetchone()[0])

        print("=" * 120, flush=True)
        print("WS_DB_ADVISORY_LOCK_RELEASED", flush=True)
        print("unlocked:", unlocked, flush=True)

    except Exception as e:
        print("WS_DB_LOCK_RELEASE_ERROR:", e, flush=True)

    try:
        conn.close()
    except Exception:
        pass


def normalize_rows(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = message.get("data")

    if data is None:
        return []

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        return [data]

    return []


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def build_event_uid(topic: str, row: Dict[str, Any], message: Dict[str, Any]) -> str:
    parts = [
        str(topic or ""),
        str(row.get("execId") or ""),
        str(row.get("orderId") or ""),
        str(row.get("orderLinkId") or ""),
        str(row.get("symbol") or ""),
        str(row.get("updatedTime") or ""),
        str(row.get("creationTime") or ""),
        str(row.get("execTime") or ""),
        str(row.get("positionIdx") or ""),
        str(row.get("size") or ""),
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=json_default),
    ]

    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def infer_event_type(topic: str, row: Dict[str, Any]) -> str:
    topic_u = str(topic or "").upper()

    if "EXECUTION" in topic_u:
        return "EXECUTION"

    if "ORDER" in topic_u:
        return "ORDER"

    if "POSITION" in topic_u:
        return "POSITION"

    return "UNKNOWN"


def extract_event_fields(topic: str, row: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
    event_type = infer_event_type(topic, row)

    return {
        "event_uid": build_event_uid(topic, row, message),
        "topic": str(topic or ""),
        "event_type": event_type,
        "symbol": str(row.get("symbol") or "").upper() or None,
        "side": str(row.get("side") or "") or None,
        "order_id": str(row.get("orderId") or "") or None,
        "order_link_id": str(row.get("orderLinkId") or "") or None,
        "exec_id": str(row.get("execId") or "") or None,
        "order_status": str(row.get("orderStatus") or "") or None,
        "exec_type": str(row.get("execType") or "") or None,
        "position_size": safe_float(row.get("size")),
        "raw_json": row,
    }


def insert_ws_event(fields: Dict[str, Any]) -> bool:
    sql = """
        INSERT INTO public.trading_ws_events (
            event_uid,
            topic,
            event_type,
            symbol,
            side,
            order_id,
            order_link_id,
            exec_id,
            order_status,
            exec_type,
            position_size,
            raw_json
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s::jsonb
        )
        ON CONFLICT (event_uid)
        DO NOTHING
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(fields["event_uid"]),
                fields.get("topic"),
                fields.get("event_type"),
                fields.get("symbol"),
                fields.get("side"),
                fields.get("order_id"),
                fields.get("order_link_id"),
                fields.get("exec_id"),
                fields.get("order_status"),
                fields.get("exec_type"),
                fields.get("position_size"),
                json.dumps(fields.get("raw_json") or {}, ensure_ascii=False, default=json_default),
            ),
        )
        inserted = int(cur.rowcount or 0)

    return inserted > 0


def is_important_ws_event(fields: Dict[str, Any]) -> bool:
    event_type = str(fields.get("event_type") or "").upper()
    order_status = str(fields.get("order_status") or "").upper()
    exec_type = str(fields.get("exec_type") or "").upper()
    order_link_id = str(fields.get("order_link_id") or "").lower()

    if event_type == "EXECUTION":
        return True

    if order_status in {"FILLED", "PARTIALLYFILLED", "TRIGGERED", "DEACTIVATED", "CANCELLED", "REJECTED"}:
        return True

    if "partialtp" in order_link_id:
        return True

    if "finaltp" in order_link_id:
        return True

    if "earlystop" in order_link_id:
        return True

    if "rest" in order_link_id:
        return True

    if exec_type:
        return True

    return False


def print_ws_event(fields: Dict[str, Any], inserted: bool) -> None:
    if not is_important_ws_event(fields):
        return

    print("=" * 120, flush=True)
    print("WS_EVENT", flush=True)
    print("inserted:", bool(inserted), flush=True)
    print("event_type:", fields.get("event_type"), flush=True)
    print("topic:", fields.get("topic"), flush=True)
    print("symbol:", fields.get("symbol"), flush=True)
    print("side:", fields.get("side"), flush=True)
    print("order_status:", fields.get("order_status"), flush=True)
    print("exec_type:", fields.get("exec_type"), flush=True)
    print("order_id:", fields.get("order_id"), flush=True)
    print("order_link_id:", fields.get("order_link_id"), flush=True)
    print("exec_id:", fields.get("exec_id"), flush=True)
    print("position_size:", fields.get("position_size"), flush=True)


def get_lifecycle_client() -> BybitClient:
    global LIFECYCLE_CLIENT

    if LIFECYCLE_CLIENT is None:
        LIFECYCLE_CLIENT = BybitClient()

    return LIFECYCLE_CLIENT


def is_partial_tp_event(fields: Dict[str, Any]) -> bool:
    event_type = str(fields.get("event_type") or "").upper()
    order_link_id = str(fields.get("order_link_id") or "").lower()
    order_status = str(fields.get("order_status") or "").upper()

    if "partialtp" not in order_link_id:
        return False

    if event_type == "EXECUTION":
        return True

    if order_status in {"FILLED", "PARTIALLYFILLED", "TRIGGERED"}:
        return True

    return False


def is_position_zero_event(fields: Dict[str, Any]) -> bool:
    event_type = str(fields.get("event_type") or "").upper()

    if event_type != "POSITION":
        return False

    size = fields.get("position_size")
    if size is None:
        return False

    try:
        return abs(float(size)) <= 0.0
    except Exception:
        return False


def handle_lifecycle_from_ws_event(
    fields: Dict[str, Any],
    inserted: bool,
) -> None:
    if not inserted:
        return

    try:
        event_type = str(
            fields.get("event_type") or ""
        ).upper()

        symbol = str(
            fields.get("symbol") or ""
        ).upper()

        raw_position_size = fields.get("position_size")
        position_size = None

        if raw_position_size is not None:
            try:
                position_size = abs(
                    float(raw_position_size)
                )
            except Exception:
                position_size = None

        if (
            event_type == "POSITION"
            and symbol
            and position_size is not None
            and position_size > 0
        ):
            with db_cursor(commit=True) as (_, cur):
                cur.execute(
                    """
                    WITH target AS (
                        SELECT trade_id
                        FROM public.trading_positions
                        WHERE symbol = %s
                          AND status = ANY(%s)
                        ORDER BY
                            updated_at DESC,
                            trade_id DESC
                        LIMIT 1
                    )
                    UPDATE public.trading_positions p
                    SET
                        position_seen_open_at = COALESCE(
                            p.position_seen_open_at,
                            NOW()
                        ),
                        protective_cleanup_done_at = NULL,
                        closed_cleanup_done_at = NULL,
                        ws_lifecycle_updated_at = NOW(),
                        updated_at = NOW()
                    FROM target
                    WHERE p.trade_id = target.trade_id
                    """,
                    (
                        symbol,
                        ACTIVE_POSITION_STATUSES,
                    ),
                )

            return

        if is_partial_tp_event(fields):
            order_link_id = str(
                fields.get("order_link_id") or ""
            )

            if order_link_id:
                handle_partial_tp_filled(
                    client=get_lifecycle_client(),
                    order_link_id=order_link_id,
                    source="ws_partial_tp_event",
                )

            return

        if not is_position_zero_event(fields):
            return

        if not symbol:
            return

        with db_cursor(commit=True) as (_, cur):
            cur.execute(
                """
                SELECT
                    trade_id,
                    status
                FROM public.trading_positions
                WHERE symbol = %s
                  AND (
                      status = ANY(%s)
                      OR status LIKE 'POSITION_CLOSED%%'
                  )
                ORDER BY
                    CASE
                        WHEN status = ANY(%s) THEN 0
                        ELSE 1
                    END,
                    updated_at DESC,
                    trade_id DESC
                LIMIT 3
                """,
                (
                    symbol,
                    ACTIVE_POSITION_STATUSES,
                    ACTIVE_POSITION_STATUSES,
                ),
            )

            rows = cur.fetchall()

        for row in rows:
            trade_id = int(row[0])
            status = str(row[1] or "")

            if status.startswith("POSITION_CLOSED"):
                cancel_remaining_protective_orders_once(
                    client=get_lifecycle_client(),
                    trade_id=trade_id,
                    source="ws_position_zero_event",
                )
                continue

            cleanup_zero_exchange_position_for_active_trade(
                client=get_lifecycle_client(),
                trade_id=trade_id,
                source="ws_position_zero_event",
            )

    except Exception as e:
        print("=" * 120, flush=True)
        print("WS_LIFECYCLE_EVENT_ERROR", flush=True)
        print(
            "event_type:",
            fields.get("event_type"),
            flush=True,
        )
        print(
            "symbol:",
            fields.get("symbol"),
            flush=True,
        )
        print(
            "order_link_id:",
            fields.get("order_link_id"),
            flush=True,
        )
        print(
            "error_type:",
            type(e).__name__,
            flush=True,
        )
        print("error:", e, flush=True)
        print(traceback.format_exc(), flush=True)



def run_lifecycle_timer_once() -> None:
    due_positions = load_active_positions_with_due_early_stop()

    for position_row in due_positions:
        try:
            handle_early_stop_expired(
                client=get_lifecycle_client(),
                position_row=position_row,
                source="ws_early_stop_timer",
            )
        except Exception as e:
            print("=" * 120, flush=True)
            print("WS_LIFECYCLE_TIMER_ERROR", flush=True)
            print("trade_id:", position_row.get("trade_id"), flush=True)
            print("symbol:", position_row.get("symbol"), flush=True)
            print("error_type:", type(e).__name__, flush=True)
            print("error:", e, flush=True)
            print(traceback.format_exc(), flush=True)


def maybe_run_lifecycle_timer() -> None:
    global LAST_LIFECYCLE_TIMER_CHECK_MONO

    now_mono = time.time()
    if now_mono - LAST_LIFECYCLE_TIMER_CHECK_MONO < float(WS_LIFECYCLE_TIMER_CHECK_SECONDS):
        return

    LAST_LIFECYCLE_TIMER_CHECK_MONO = now_mono
    run_lifecycle_timer_once()


def handle_ws_message(message: Dict[str, Any]) -> None:
    global LAST_WS_EVENT_MONO

    LAST_WS_EVENT_MONO = time.time()

    topic = str(message.get("topic") or "")
    rows = normalize_rows(message)

    if not rows:
        return

    for row in rows:
        fields = extract_event_fields(topic=topic, row=row, message=message)
        inserted = insert_ws_event(fields)
        print_ws_event(fields, inserted)
        handle_lifecycle_from_ws_event(fields, inserted)


def close_ws_safe(ws: Any) -> None:
    for method_name in ["exit", "close"]:
        try:
            method = getattr(ws, method_name, None)
            if callable(method):
                method()
                print("WS_LISTENER_SOCKET_CLOSED_BY_{}".format(method_name.upper()), flush=True)
                return
        except Exception as e:
            print("WS_LISTENER_SOCKET_CLOSE_ERROR:", repr(e), flush=True)


def should_reconnect_ws() -> Optional[str]:
    now_mono = time.time()

    if float(WS_FORCED_RECONNECT_SECONDS) > 0:
        age = now_mono - float(WS_CONNECT_MONO or now_mono)
        if age >= float(WS_FORCED_RECONNECT_SECONDS):
            print("=" * 120, flush=True)
            print("WS_LISTENER_RECONNECT_REASON", flush=True)
            print("reason: forced_reconnect_interval", flush=True)
            print("age_seconds:", round(age, 3), flush=True)
            print("forced_reconnect_seconds:", WS_FORCED_RECONNECT_SECONDS, flush=True)
            return "forced_reconnect_interval"

    if float(WS_NO_EVENT_RECONNECT_SECONDS) > 0:
        idle = now_mono - float(LAST_WS_EVENT_MONO or now_mono)
        if idle >= float(WS_NO_EVENT_RECONNECT_SECONDS):
            print("=" * 120, flush=True)
            print("WS_LISTENER_RECONNECT_REASON", flush=True)
            print("reason: no_private_events_interval", flush=True)
            print("idle_seconds:", round(idle, 3), flush=True)
            print("no_event_reconnect_seconds:", WS_NO_EVENT_RECONNECT_SECONDS, flush=True)
            return "no_private_events_interval"

    return None


def build_ws() -> WebSocket:
    return WebSocket(
        testnet=is_testnet(),
        channel_type="private",
        api_key=get_bybit_api_key(),
        api_secret=get_bybit_api_secret(),
    )


def subscribe_private_streams(ws: WebSocket) -> None:
    ws.order_stream(callback=handle_ws_message)
    ws.execution_stream(callback=handle_ws_message)
    ws.position_stream(callback=handle_ws_message)


def log_start() -> None:
    audit_log.ensure_audit_tables()
    audit_log.log_audit_event(
        event_type="WS_LISTENER_STARTED",
        status="STARTED",
        message="Bybit private websocket listener started",
        payload={
            "service": SERVICE_NAME,
            "root": str(ROOT),
            "environment": WS_ENVIRONMENT,
            "testnet": is_testnet(),
            "heartbeat_seconds": WS_HEARTBEAT_SECONDS,
        },
    )


def main() -> None:
    install_signal_handlers()
    ensure_ws_tables()
    ensure_ws_lifecycle_columns()

    if not acquire_db_lock():
        print("WS_LISTENER_LOCK_BUSY", flush=True)
        return

    try:
        log_start()

        print("=" * 120, flush=True)
        print("WS_LISTENER_STARTED", flush=True)
        print("ROOT:", ROOT, flush=True)
        print("ENVIRONMENT:", WS_ENVIRONMENT, flush=True)
        print("TESTNET:", is_testnet(), flush=True)
        print("STREAMS: order, execution, position", flush=True)

        while not STOP:
            try:
                global WS_CONNECT_MONO
                global LAST_WS_EVENT_MONO

                ws = build_ws()
                subscribe_private_streams(ws)

                WS_CONNECT_MONO = time.time()
                if not LAST_WS_EVENT_MONO:
                    LAST_WS_EVENT_MONO = WS_CONNECT_MONO

                print("=" * 120, flush=True)
                print("WS_LISTENER_CONNECTED", flush=True)
                print("connected_at_utc:", utc_now(), flush=True)
                print("forced_reconnect_seconds:", WS_FORCED_RECONNECT_SECONDS, flush=True)
                print("no_event_reconnect_seconds:", WS_NO_EVENT_RECONNECT_SECONDS, flush=True)

                try:
                    while not STOP:
                        sleep_left = float(WS_HEARTBEAT_SECONDS)

                        while sleep_left > 0 and not STOP:
                            step = min(float(WS_LIFECYCLE_TIMER_CHECK_SECONDS), sleep_left)
                            time.sleep(step)
                            maybe_run_lifecycle_timer()
                            sleep_left -= step

                            reconnect_reason = should_reconnect_ws()
                            if reconnect_reason:
                                raise PlannedWsReconnect(str(reconnect_reason))

                        if not STOP:
                            print("WS_LISTENER_HEARTBEAT:", utc_now(), flush=True)
                finally:
                    close_ws_safe(ws)

            except PlannedWsReconnect as e:
                print("=" * 120, flush=True)
                print("WS_LISTENER_PLANNED_RECONNECT", flush=True)
                print("reason:", str(e), flush=True)
                print("sleep_seconds:", WS_RECONNECT_SLEEP_SECONDS, flush=True)

                if STOP:
                    break

                time.sleep(WS_RECONNECT_SLEEP_SECONDS)

            except Exception as e:
                print("=" * 120, flush=True)
                print("WS_LISTENER_ERROR", flush=True)
                print("error_type:", type(e).__name__, flush=True)
                print("error:", e, flush=True)
                print(traceback.format_exc(), flush=True)

                if STOP:
                    break

                time.sleep(WS_RECONNECT_SLEEP_SECONDS)

    finally:
        release_db_lock()
        print("WS_LISTENER_STOPPED", flush=True)


if __name__ == "__main__":
    main()
