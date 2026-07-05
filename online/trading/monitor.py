from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from online.trading import config
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
    "TTL_CLOSE_SENT",
    "TTL_CLOSE_FAILED",
]

PROTECTIVE_ORDER_ROLES = [
    "TAKE_PROFIT",
    "PARTIAL_TP",
    "FINAL_TP",
    "STOP_LOSS",
    "EARLY_STOP",
    "REST_STOP_AFTER_PARTIAL",
]

PARTIAL_TP_ORDER_ROLES = [
    "PARTIAL_TP",
]

REST_STOP_ORDER_ROLES = [
    "REST_STOP_AFTER_PARTIAL",
]

MAIN_STOP_ORDER_ROLES = [
    "STOP_LOSS",
]

EARLY_STOP_ORDER_ROLES = [
    "EARLY_STOP",
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


def ensure_monitor_position_columns() -> None:
    sql = """
        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS partial_tp_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS final_tp_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS early_stop_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS main_sl_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS rest_stop_after_partial_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS partial_tp_qty_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS final_tp_qty_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS early_stop_expires_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS trade_management_mode TEXT;
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)


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


def build_order_link_id(signal_key: str, role: str) -> str:
    digest = hashlib.sha1(str(signal_key).encode("utf-8")).hexdigest()[:24]
    return "imb-{}-{}".format(digest, str(role).lower())


def close_side_for_position_side(side: str) -> str:
    side_u = str(side).upper().strip()

    if side_u == "LONG":
        return "Sell"

    if side_u == "SHORT":
        return "Buy"

    raise RuntimeError("bad position side for reduce-only close: {}".format(side))


def trigger_direction_for_reduce_only(side: str, trigger_kind: str) -> int:
    side_u = str(side).upper().strip()
    kind_u = str(trigger_kind).upper().strip()

    if side_u == "LONG" and kind_u == "TP":
        return 1

    if side_u == "LONG" and kind_u == "SL":
        return 2

    if side_u == "SHORT" and kind_u == "TP":
        return 2

    if side_u == "SHORT" and kind_u == "SL":
        return 1

    raise RuntimeError("bad side/trigger_kind: {} {}".format(side, trigger_kind))


def place_reduce_only_trigger_market_order(
    client: BybitClient,
    symbol: str,
    side: str,
    qty: float,
    trigger_px: float,
    trigger_kind: str,
    order_link_id: str,
) -> Dict[str, object]:
    session = client._get_session()

    resp = session.place_order(
        category=client.category,
        symbol=str(symbol).upper(),
        side=close_side_for_position_side(side),
        orderType="Market",
        triggerPrice=str(float(trigger_px)),
        triggerDirection=trigger_direction_for_reduce_only(side, trigger_kind),
        qty=str(float(qty)),
        reduceOnly=True,
        orderLinkId=str(order_link_id),
    )

    client._raise_if_bad(resp)
    return resp


def load_orders_for_trade(trade_id: int) -> pd.DataFrame:
    sql = """
        SELECT *
        FROM public.trading_orders
        WHERE trade_id = %s
        ORDER BY created_at ASC
    """
    return read_sql(sql, [int(trade_id)])


def load_fills_for_trade(trade_id: int) -> pd.DataFrame:
    sql = """
        SELECT *
        FROM public.trading_fills
        WHERE trade_id = %s
        ORDER BY executed_at ASC, fill_id ASC
    """
    return read_sql(sql, [int(trade_id)])


def has_filled_role(trade_id: int, roles: List[str]) -> bool:
    roles_u = [str(x).upper() for x in roles]
    if not roles_u:
        return False

    placeholders = ", ".join(["%s"] * len(roles_u))
    sql = """
        SELECT 1
        FROM public.trading_fills
        WHERE trade_id = %s
          AND UPPER(order_role) IN ({})
          AND COALESCE(exec_qty, 0) > 0
        LIMIT 1
    """.format(placeholders)

    df = read_sql(sql, [int(trade_id)] + roles_u)
    return not df.empty


def active_order_exists(
    trade_id: int,
    roles: List[str],
) -> bool:
    roles_u = [str(x).upper() for x in roles]

    if not roles_u:
        return False

    placeholders = ", ".join(
        ["%s"] * len(roles_u)
    )

    sql = """
        SELECT 1
        FROM public.trading_orders
        WHERE trade_id = %s
          AND UPPER(order_role) IN ({})
          AND UPPER(COALESCE(status, '')) NOT IN (
              'CANCELLED',
              'CANCELLED_NOT_FOUND',
              'FILLED',
              'TRIGGERED',
              'FAILED',
              'ERROR',
              'TP_SL_FAILED',
              'TTL_CLOSE_FAILED'
          )
        LIMIT 1
    """.format(placeholders)

    df = read_sql(
        sql,
        [int(trade_id)] + roles_u,
    )

    return not df.empty



def update_order_cancel_result(
    local_order_key: str,
    status: str,
    response_json: Dict[str, object],
) -> None:
    sql = """
        UPDATE public.trading_orders
        SET status = %s,
            response_json = %s::jsonb,
            lifecycle_note = COALESCE(
                lifecycle_note,
                NULLIF(%s, '')
            ),
            updated_at = NOW()
        WHERE local_order_key = %s
    """
    lifecycle_note = str((response_json or {}).get("cancel_reason") or "")

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(status),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
                lifecycle_note,
                str(local_order_key),
            ),
        )


def is_bybit_order_already_gone_error(error_text: str) -> bool:
    text_l = str(error_text or "").lower()

    markers = [
        "order not exists",
        "order does not exist",
        "not found",
        "too late to cancel",
        "already canceled",
        "already cancelled",
        "already filled",
        "order status is filled",
        "order status is cancelled",
        "order status is canceled",
        "110001",
        "110008",
    ]

    return any(marker in text_l for marker in markers)


def mark_position_closed_cleanup_done(trade_id: int) -> None:
    sql = """
        UPDATE public.trading_positions
        SET
            protective_cleanup_done_at = COALESCE(protective_cleanup_done_at, NOW()),
            closed_cleanup_done_at = COALESCE(closed_cleanup_done_at, NOW()),
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))


def cancel_trade_orders_by_roles(
    client: BybitClient,
    trade_id: int,
    symbol: str,
    roles: List[str],
    reason: str,
) -> Dict[str, object]:
    orders = load_orders_for_trade(trade_id)

    result: Dict[str, object] = {
        "trade_id": int(trade_id),
        "symbol": str(symbol).upper(),
        "roles": [str(x).upper() for x in roles],
        "reason": str(reason),
        "checked": 0,
        "cancelled": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }

    if orders.empty:
        return result

    roles_u = set(str(x).upper() for x in roles)

    for _, r in orders.iterrows():
        row = r.to_dict()
        role = str(row.get("order_role") or "").upper()
        status = str(row.get("status") or "").upper()

        if role not in roles_u:
            continue

        result["checked"] = int(result["checked"]) + 1

        if status in {
            "CANCELLED",
            "CANCELLED_NOT_FOUND",
            "FILLED",
            "TRIGGERED",
            "FAILED",
            "ERROR",
            "TP_SL_FAILED",
            "TTL_CLOSE_FAILED",
        }:
            result["skipped"] = int(result["skipped"]) + 1
            continue

        local_order_key = str(row.get("local_order_key") or "")
        bybit_order_id = row.get("bybit_order_id")
        bybit_order_link_id = row.get("bybit_order_link_id")

        if pd.isna(bybit_order_id):
            bybit_order_id = None

        if pd.isna(bybit_order_link_id):
            bybit_order_link_id = None

        try:
            resp = client.cancel_order(
                symbol=str(symbol).upper(),
                order_id=str(bybit_order_id) if bybit_order_id else None,
                order_link_id=str(bybit_order_link_id) if bybit_order_link_id else None,
            )

            update_order_cancel_result(
                local_order_key=local_order_key,
                status="CANCELLED",
                response_json={
                    "cancel_reason": reason,
                    "cancel_response": resp,
                },
            )

            result["cancelled"] = int(result["cancelled"]) + 1

        except Exception as e:
            err = str(e)

            if is_bybit_order_already_gone_error(err):
                update_order_cancel_result(
                    local_order_key=local_order_key,
                    status="CANCELLED_NOT_FOUND",
                    response_json={
                        "cancel_reason": reason,
                        "already_gone": True,
                        "error": err,
                    },
                )

                result["cancelled"] = int(result["cancelled"]) + 1
                continue

            update_order_cancel_result(
                local_order_key=local_order_key,
                status="CANCEL_FAILED",
                response_json={
                    "cancel_reason": reason,
                    "error": err,
                },
            )

            result["failed"] = int(result["failed"]) + 1
            errors = result.get("errors")
            if isinstance(errors, list):
                errors.append(
                    {
                        "local_order_key": local_order_key,
                        "order_role": role,
                        "error": err,
                    }
                )

    return result


def insert_protective_trigger_order(
    trade_id: int,
    signal_key: str,
    symbol: str,
    side: str,
    order_role: str,
    local_order_key: str,
    qty: float,
    trigger_px: float,
    status: str,
    request_json: Dict[str, object],
    response_json: Dict[str, object],
    bybit_order_id: Optional[str],
    bybit_order_link_id: str,
) -> None:
    sql = """
        INSERT INTO public.trading_orders (
            local_order_key,
            trade_id,
            signal_key,
            symbol,
            side,
            order_role,
            bybit_order_id,
            bybit_order_link_id,
            order_type,
            reduce_only,
            qty_plan,
            price_plan,
            trigger_price_plan,
            status,
            sent_at,
            acknowledged_at,
            request_json,
            response_json,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s,
            'MarketTrigger',
            TRUE,
            %s, NULL, %s,
            %s,
            NOW(), NOW(),
            %s::jsonb,
            %s::jsonb,
            NOW()
        )
        ON CONFLICT (local_order_key)
        DO UPDATE SET
            status = EXCLUDED.status,
            bybit_order_id = EXCLUDED.bybit_order_id,
            bybit_order_link_id = EXCLUDED.bybit_order_link_id,
            qty_plan = EXCLUDED.qty_plan,
            trigger_price_plan = EXCLUDED.trigger_price_plan,
            request_json = EXCLUDED.request_json,
            response_json = EXCLUDED.response_json,
            updated_at = NOW()
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                local_order_key,
                int(trade_id),
                signal_key,
                str(symbol).upper(),
                str(side).upper(),
                str(order_role).upper(),
                bybit_order_id,
                bybit_order_link_id,
                float(qty),
                float(trigger_px),
                str(status),
                json.dumps(request_json or {}, ensure_ascii=False, default=json_default),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
            ),
        )


def mark_position_status(trade_id: int, status: str, exit_reason: Optional[str] = None) -> None:
    sql = """
        UPDATE public.trading_positions
        SET status = %s,
            exit_reason = COALESCE(%s, exit_reason),
            updated_at = NOW()
        WHERE trade_id = %s
          AND status NOT LIKE 'POSITION_CLOSED%%'
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (str(status), exit_reason, int(trade_id)))


def maybe_replace_early_stop_with_main_sl(
    client: BybitClient,
    row: Dict[str, object],
    exchange_size: float,
) -> bool:
    if not bool(getattr(config, "EARLY_STOP_ENABLED", False)):
        return False

    if not bool(getattr(config, "MAIN_STOP_AFTER_EARLY_WINDOW_ENABLED", True)):
        return False

    trade_id = int(row["trade_id"])
    signal_key = str(row.get("signal_key") or "")
    symbol = str(row.get("symbol") or "").upper()
    side = str(row.get("side") or "").upper()

    early_stop_expires_at = pd.to_datetime(row.get("early_stop_expires_at"), utc=True, errors="coerce")
    now_ts = pd.Timestamp.now(tz="UTC")

    if pd.isna(early_stop_expires_at):
        return False

    if now_ts < early_stop_expires_at:
        return False

    if has_filled_role(trade_id, PARTIAL_TP_ORDER_ROLES):
        return False

    if not active_order_exists(trade_id, EARLY_STOP_ORDER_ROLES):
        return False

    if active_order_exists(trade_id, MAIN_STOP_ORDER_ROLES):
        return False

    main_sl_px = safe_float(row.get("main_sl_px_plan"))
    if main_sl_px is None or main_sl_px <= 0:
        insert_event(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            event_type="MAIN_SL_SKIPPED_BAD_PRICE",
            details={
                "main_sl_px_plan": row.get("main_sl_px_plan"),
                "early_stop_expires_at": str(early_stop_expires_at),
                "now_ts": str(now_ts),
            },
        )
        return False

    qty = float(exchange_size)
    if qty <= 0:
        qty = float(safe_float(row.get("qty")) or 0.0)

    if qty <= 0:
        insert_event(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            event_type="MAIN_SL_SKIPPED_BAD_QTY",
            details={
                "exchange_size": exchange_size,
                "db_qty": row.get("qty"),
                "main_sl_px_plan": main_sl_px,
            },
        )
        return False

    cancel_result = cancel_trade_orders_by_roles(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        roles=EARLY_STOP_ORDER_ROLES,
        reason="EARLY_STOP_EXPIRED_MAIN_SL_REPLACE",
    )

    main_sl_link_id = build_order_link_id(signal_key, "mainsl")

    request_payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "main_sl_px": main_sl_px,
        "order_link_id": main_sl_link_id,
        "early_stop_expires_at": str(early_stop_expires_at),
        "now_ts": str(now_ts),
        "cancel_result": cancel_result,
    }

    try:
        response_payload = place_reduce_only_trigger_market_order(
            client=client,
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_px=main_sl_px,
            trigger_kind="SL",
            order_link_id=main_sl_link_id,
        )
        bybit_order_id = str((response_payload.get("result") or {}).get("orderId") or "") or None
        status = "SENT"

    except Exception as e:
        response_payload = {"error": str(e)}
        bybit_order_id = None
        status = "FAILED"

    insert_protective_trigger_order(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        order_role="STOP_LOSS",
        local_order_key=main_sl_link_id,
        qty=qty,
        trigger_px=main_sl_px,
        status=status,
        request_json=request_payload,
        response_json=response_payload,
        bybit_order_id=bybit_order_id,
        bybit_order_link_id=main_sl_link_id,
    )

    insert_event(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        event_type="EARLY_STOP_REPLACED_WITH_MAIN_SL" if status == "SENT" else "MAIN_SL_PLACE_FAILED",
        details={
            "request": request_payload,
            "response": response_payload,
            "status": status,
        },
    )

    if status != "SENT":
        print("MAIN_SL_PLACE_FAILED")
        print("trade_id:", trade_id)
        print("symbol:", symbol)
        print("main_sl_px:", main_sl_px)
        print("response:", response_payload)
        return False

    print("EARLY_STOP_REPLACED_WITH_MAIN_SL")
    print("trade_id:", trade_id)
    print("symbol:", symbol)
    print("qty:", qty)
    print("main_sl_px:", main_sl_px)
    return True


def maybe_place_rest_stop_after_partial(
    client: BybitClient,
    row: Dict[str, object],
    exchange_size: float,
) -> bool:
    if not bool(getattr(config, "PARTIAL_TP_ENABLED", False)):
        return False

    if not bool(getattr(config, "REST_STOP_AFTER_PARTIAL_ENABLED", True)):
        return False

    trade_id = int(row["trade_id"])
    signal_key = str(row.get("signal_key") or "")
    symbol = str(row.get("symbol") or "").upper()
    side = str(row.get("side") or "").upper()

    if not has_filled_role(trade_id, PARTIAL_TP_ORDER_ROLES):
        return False

    if active_order_exists(trade_id, REST_STOP_ORDER_ROLES):
        return False

    rest_stop_px = safe_float(row.get("rest_stop_after_partial_px_plan"))
    if rest_stop_px is None or rest_stop_px <= 0:
        insert_event(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            event_type="REST_STOP_SKIPPED_BAD_PRICE",
            details={
                "rest_stop_after_partial_px_plan": row.get("rest_stop_after_partial_px_plan"),
            },
        )
        return False

    qty = float(exchange_size)
    if qty <= 0:
        return False

    cancel_result = cancel_trade_orders_by_roles(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        roles=EARLY_STOP_ORDER_ROLES + MAIN_STOP_ORDER_ROLES,
        reason="PARTIAL_TP_FILLED_REST_STOP_REPLACE",
    )

    rest_stop_link_id = build_order_link_id(signal_key, "reststop")

    request_payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "rest_stop_after_partial_px": rest_stop_px,
        "order_link_id": rest_stop_link_id,
        "cancel_result": cancel_result,
    }

    try:
        response_payload = place_reduce_only_trigger_market_order(
            client=client,
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_px=rest_stop_px,
            trigger_kind="SL",
            order_link_id=rest_stop_link_id,
        )
        bybit_order_id = str((response_payload.get("result") or {}).get("orderId") or "") or None
        status = "SENT"

    except Exception as e:
        response_payload = {"error": str(e)}
        bybit_order_id = None
        status = "FAILED"

    insert_protective_trigger_order(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        order_role="REST_STOP_AFTER_PARTIAL",
        local_order_key=rest_stop_link_id,
        qty=qty,
        trigger_px=rest_stop_px,
        status=status,
        request_json=request_payload,
        response_json=response_payload,
        bybit_order_id=bybit_order_id,
        bybit_order_link_id=rest_stop_link_id,
    )

    insert_event(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        event_type="REST_STOP_AFTER_PARTIAL_PLACED" if status == "SENT" else "REST_STOP_AFTER_PARTIAL_FAILED",
        details={
            "request": request_payload,
            "response": response_payload,
            "status": status,
        },
    )

    if status != "SENT":
        print("REST_STOP_AFTER_PARTIAL_FAILED")
        print("trade_id:", trade_id)
        print("symbol:", symbol)
        print("qty:", qty)
        print("rest_stop_after_partial_px:", rest_stop_px)
        print("response:", response_payload)
        return False

    mark_position_status(
        trade_id=trade_id,
        status="POSITION_OPEN",
        exit_reason=None,
    )

    print("REST_STOP_AFTER_PARTIAL_PLACED")
    print("trade_id:", trade_id)
    print("symbol:", symbol)
    print("qty:", qty)
    print("rest_stop_after_partial_px:", rest_stop_px)
    return True


def opposite_position_side(side: str) -> str:
    side_u = str(side).upper().strip()

    if side_u == "LONG":
        return "SHORT"

    if side_u == "SHORT":
        return "LONG"

    raise RuntimeError("bad position side for TTL close: {}".format(side))


def safe_float(value) -> Optional[float]:
    try:
        if value in [None, ""]:
            return None
        v = float(value)
    except Exception:
        return None

    if pd.isna(v):
        return None

    return float(v)


def get_exchange_position_size(client: BybitClient, symbol: str) -> Tuple[float, Dict[str, object]]:
    resp = client.get_position(symbol)
    positions = (resp.get("result") or {}).get("list") or []

    size = 0.0

    for pos in positions:
        if str(pos.get("symbol") or "").upper() != symbol:
            continue

        try:
            size += abs(float(pos.get("size") or 0.0))
        except Exception:
            pass

    return float(size), resp


def mark_manual_or_external_close_if_position_zero(
    client: BybitClient,
    row: Dict[str, object],
    exchange_size: float,
    position_response: Dict[str, object],
) -> bool:
    trade_id = int(row["trade_id"])
    signal_key = str(row["signal_key"])
    symbol = str(row["symbol"]).upper()

    if float(exchange_size) > 0:
        return False

    cancel_result = cancel_trade_orders_by_roles(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        roles=PROTECTIVE_ORDER_ROLES,
        reason="POSITION_ZERO_CANCEL_ALL_PROTECTIVE_ORDERS",
    )

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE public.trading_positions
            SET status = 'POSITION_CLOSED_EXTERNAL',
                exit_reason = COALESCE(exit_reason, 'EXCHANGE_POSITION_NOT_FOUND'),
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
        event_type="EXCHANGE_POSITION_ZERO_DETECTED",
        details={
            "bybit_position_response": position_response,
            "cancel_protective_orders_result": cancel_result,
        },
    )

    print("EXCHANGE_POSITION_ZERO_DETECTED")
    print("trade_id:", trade_id)
    print("symbol:", symbol)
    print("cancel_protective_orders_result:", json.dumps(cancel_result, ensure_ascii=False, default=json_default))

    return True


def ttl_close_order_already_sent(trade_id: int) -> bool:
    sql = """
        SELECT 1
        FROM public.trading_orders
        WHERE trade_id = %s
          AND order_role = 'TTL_CLOSE'
          AND status NOT IN ('TTL_CLOSE_FAILED', 'FAILED', 'ERROR')
        LIMIT 1
    """
    df = read_sql(sql, [int(trade_id)])
    return not df.empty


def insert_ttl_close_order(
    trade_id: int,
    signal_key: str,
    symbol: str,
    position_side: str,
    local_order_key: str,
    qty: float,
    status: str,
    request_json: Dict[str, object],
    response_json: Dict[str, object],
    bybit_order_id: Optional[str],
    bybit_order_link_id: str,
) -> None:
    sql = """
        INSERT INTO public.trading_orders (
            local_order_key,
            trade_id,
            signal_key,
            symbol,
            side,
            order_role,
            bybit_order_id,
            bybit_order_link_id,
            order_type,
            reduce_only,
            qty_plan,
            price_plan,
            trigger_price_plan,
            status,
            sent_at,
            acknowledged_at,
            request_json,
            response_json,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, 'TTL_CLOSE',
            %s, %s,
            'Market',
            TRUE,
            %s, NULL, NULL,
            %s,
            NOW(), NOW(),
            %s::jsonb,
            %s::jsonb,
            NOW()
        )
        ON CONFLICT (local_order_key)
        DO UPDATE SET
            status = EXCLUDED.status,
            bybit_order_id = EXCLUDED.bybit_order_id,
            bybit_order_link_id = EXCLUDED.bybit_order_link_id,
            qty_plan = EXCLUDED.qty_plan,
            request_json = EXCLUDED.request_json,
            response_json = EXCLUDED.response_json,
            updated_at = NOW()
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                local_order_key,
                int(trade_id),
                signal_key,
                symbol,
                position_side,
                bybit_order_id,
                bybit_order_link_id,
                float(qty),
                status,
                json.dumps(request_json or {}, ensure_ascii=False, default=json_default),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
            ),
        )


def cancel_protective_orders_before_ttl_close(
    client: BybitClient,
    trade_id: int,
    symbol: str,
) -> Dict[str, object]:
    return cancel_trade_orders_by_roles(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        roles=PROTECTIVE_ORDER_ROLES,
        reason="TTL_CLOSE_CANCEL_ALL_PROTECTIVE_ORDERS",
    )


def mark_position_ttl_close_sent(
    trade_id: int,
    bybit_order_id: Optional[str],
) -> None:
    sql = """
        UPDATE public.trading_positions
        SET status = 'TTL_CLOSE_SENT',
            exit_reason = 'TTL_CLOSE',
            exit_order_id = COALESCE(%s, exit_order_id),
            updated_at = NOW()
        WHERE trade_id = %s
          AND status NOT LIKE 'POSITION_CLOSED%%'
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (bybit_order_id, int(trade_id)))


def mark_position_ttl_close_failed(
    trade_id: int,
    error_message: str,
) -> None:
    sql = """
        UPDATE public.trading_positions
        SET status = 'TTL_CLOSE_FAILED',
            exit_reason = 'TTL_CLOSE_FAILED',
            updated_at = NOW()
        WHERE trade_id = %s
          AND status NOT LIKE 'POSITION_CLOSED%%'
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))

    insert_event(
        trade_id=int(trade_id),
        signal_key="",
        symbol="",
        event_type="TTL_CLOSE_FAILED",
        details={"error": str(error_message)},
    )


def load_closed_positions_with_active_protective_orders() -> pd.DataFrame:
    sql = """
        SELECT DISTINCT p.*
        FROM public.trading_positions p
        JOIN public.trading_orders o
            ON o.trade_id = p.trade_id
        WHERE p.status LIKE 'POSITION_CLOSED%%'
          AND UPPER(o.order_role) IN (
              'TAKE_PROFIT',
              'PARTIAL_TP',
              'FINAL_TP',
              'STOP_LOSS',
              'EARLY_STOP',
              'REST_STOP_AFTER_PARTIAL'
          )
          AND o.status NOT IN (
              'CANCELLED',
              'CANCELLED_NOT_FOUND',
              'FILLED',
              'TRIGGERED',
              'FAILED',
              'ERROR',
              'TP_SL_FAILED',
              'TTL_CLOSE_FAILED'
          )
        ORDER BY p.updated_at ASC NULLS LAST, p.created_at ASC
    """

    return read_sql(sql)


def cleanup_closed_position_protective_orders(
    client: BybitClient,
    row: Dict[str, object],
    source: str,
) -> bool:
    trade_id = int(row["trade_id"])
    signal_key = str(row.get("signal_key") or "")
    symbol = str(row.get("symbol") or "").upper()

    cancel_result = cancel_trade_orders_by_roles(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        roles=PROTECTIVE_ORDER_ROLES,
        reason=str(source),
    )

    failed_count = int(cancel_result.get("failed") or 0)

    if failed_count > 0:
        insert_event(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            event_type="CLOSED_POSITION_PROTECTIVE_CLEANUP_FAILED",
            details={
                "source": str(source),
                "position_status": str(row.get("status") or ""),
                "cancel_protective_orders_result": cancel_result,
            },
        )

        print("CLOSED_POSITION_PROTECTIVE_CLEANUP_FAILED")
        print("trade_id:", trade_id)
        print("symbol:", symbol)
        print("source:", source)
        print("cancel_protective_orders_result:", json.dumps(cancel_result, ensure_ascii=False, default=json_default))

        return False

    mark_position_closed_cleanup_done(trade_id)

    insert_event(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        event_type="CLOSED_POSITION_PROTECTIVE_CLEANUP",
        details={
            "source": str(source),
            "position_status": str(row.get("status") or ""),
            "cancel_protective_orders_result": cancel_result,
        },
    )

    print("CLOSED_POSITION_PROTECTIVE_CLEANUP")
    print("trade_id:", trade_id)
    print("symbol:", symbol)
    print("source:", source)
    print("cancel_protective_orders_result:", json.dumps(cancel_result, ensure_ascii=False, default=json_default))

    return True


def maybe_send_ttl_close(
    client: BybitClient,
    row: Dict[str, object],
    exchange_size: float,
) -> bool:
    trade_id = int(row["trade_id"])
    signal_key = str(row.get("signal_key") or "")
    symbol = str(row.get("symbol") or "").upper()
    position_side = str(row.get("side") or "").upper()

    ttl_close_ts = pd.to_datetime(row.get("ttl_close_ts"), utc=True, errors="coerce")
    now_ts = pd.Timestamp.now(tz="UTC")

    if pd.isna(ttl_close_ts):
        return False

    if now_ts < ttl_close_ts:
        return False

    if ttl_close_order_already_sent(trade_id):
        print("TTL_CLOSE_ALREADY_SENT")
        print("trade_id:", trade_id)
        print("symbol:", symbol)
        return False

    db_qty = safe_float(row.get("qty"))
    qty = float(exchange_size) if float(exchange_size) > 0 else float(db_qty or 0.0)

    if qty <= 0:
        insert_event(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            event_type="TTL_CLOSE_SKIPPED_BAD_QTY",
            details={
                "exchange_size": exchange_size,
                "db_qty": db_qty,
                "ttl_close_ts": str(ttl_close_ts),
                "now_ts": str(now_ts),
            },
        )
        return False

    close_side = opposite_position_side(position_side)
    ttl_link_id = build_order_link_id(signal_key, "ttl")

    cancel_result = cancel_protective_orders_before_ttl_close(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
    )

    request_payload = {
        "symbol": symbol,
        "position_side": position_side,
        "close_side": close_side,
        "qty": qty,
        "order_link_id": ttl_link_id,
        "reduce_only": True,
        "ttl_close_ts": str(ttl_close_ts),
        "now_ts": str(now_ts),
        "cancel_protective_orders_result": cancel_result,
    }

    if DRY_RUN:
        response_payload = {
            "dry_run": True,
            "message": "TTL close market order not sent",
            "request": request_payload,
        }
        bybit_order_id = None
    else:
        try:
            response_payload = client.place_market_order(
                symbol=symbol,
                side=close_side,
                qty=qty,
                order_link_id=ttl_link_id,
                reduce_only=True,
            )
            bybit_order_id = str((response_payload.get("result") or {}).get("orderId") or "") or None

        except Exception as e:
            error_message = str(e)

            insert_ttl_close_order(
                trade_id=trade_id,
                signal_key=signal_key,
                symbol=symbol,
                position_side=position_side,
                local_order_key=ttl_link_id,
                qty=qty,
                status="TTL_CLOSE_FAILED",
                request_json=request_payload,
                response_json={"error": error_message},
                bybit_order_id=None,
                bybit_order_link_id=ttl_link_id,
            )

            mark_position_ttl_close_failed(
                trade_id=trade_id,
                error_message=error_message,
            )

            print("TTL_CLOSE_FAILED")
            print("trade_id:", trade_id)
            print("symbol:", symbol)
            print("position_side:", position_side)
            print("close_side:", close_side)
            print("qty:", qty)
            print("error:", error_message)

            return False

    insert_ttl_close_order(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        position_side=position_side,
        local_order_key=ttl_link_id,
        qty=qty,
        status="DRY_RUN_NOT_SENT" if DRY_RUN else "SENT",
        request_json=request_payload,
        response_json=response_payload,
        bybit_order_id=bybit_order_id,
        bybit_order_link_id=ttl_link_id,
    )

    if not DRY_RUN:
        mark_position_ttl_close_sent(
            trade_id=trade_id,
            bybit_order_id=bybit_order_id,
        )

    insert_event(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        event_type="TTL_CLOSE_SENT" if not DRY_RUN else "DRY_RUN_TTL_CLOSE_PLANNED",
        details={
            "request": request_payload,
            "response": response_payload,
        },
    )

    print("TTL_CLOSE_SENT" if not DRY_RUN else "DRY_RUN_TTL_CLOSE_PLANNED")
    print("trade_id:", trade_id)
    print("symbol:", symbol)
    print("position_side:", position_side)
    print("close_side:", close_side)
    print("qty:", qty)
    print("ttl_close_ts:", ttl_close_ts)
    print("now_ts:", now_ts)

    return True


def main() -> None:
    ensure_trade_events_table()
    ensure_monitor_position_columns()
    owner = "monitor:{}".format(pd.Timestamp.now(tz="UTC"))

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=600):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        active = load_active_positions()

        manual_closed = 0
        ttl_sent = 0
        closed_cleanup = 0

        if DRY_RUN:
            if active.empty:
                print("NO_ACTIVE_POSITIONS")
                return

            print("ACTIVE_POSITIONS:", len(active))
            print("DRY_RUN: monitor does not call Bybit")
            print(active[["trade_id", "symbol", "side", "status", "ttl_close_ts"]].to_string(index=False))
            return

        client = BybitClient()

        closed_with_protective = load_closed_positions_with_active_protective_orders()

        if not closed_with_protective.empty:
            print("CLOSED_POSITIONS_WITH_ACTIVE_PROTECTIVE_ORDERS:", len(closed_with_protective))

            for _, closed_row in closed_with_protective.iterrows():
                if cleanup_closed_position_protective_orders(
                    client=client,
                    row=closed_row.to_dict(),
                    source="MONITOR_CLOSED_POSITION_CANCEL_ACTIVE_PROTECTIVE_ORDERS",
                ):
                    closed_cleanup += 1

        if active.empty:
            print("NO_ACTIVE_POSITIONS")
            print("CLOSED_POSITION_PROTECTIVE_CLEANUP_COUNT:", closed_cleanup)
            return

        print("ACTIVE_POSITIONS:", len(active))

        for _, r in active.iterrows():
            row = r.to_dict()
            symbol = str(row.get("symbol") or "").upper()

            exchange_size, position_response = get_exchange_position_size(client, symbol)

            closed = mark_manual_or_external_close_if_position_zero(
                client=client,
                row=row,
                exchange_size=exchange_size,
                position_response=position_response,
            )

            if closed:
                manual_closed += 1
                continue

            # WSListener + position_lifecycle.py теперь отвечают за:
            # 1) PARTIAL_TP -> cancel EARLY_STOP -> place REST_STOP_AFTER_PARTIAL
            # 2) early_stop_expires_at -> cancel EARLY_STOP -> place main STOP_LOSS
            # Monitor оставляем только как fallback для TTL-close и external/manual close.
            if maybe_send_ttl_close(
                client=client,
                row=row,
                exchange_size=exchange_size,
            ):
                ttl_sent += 1

        print("MONITOR_DONE")
        print("MANUAL_OR_EXTERNAL_CLOSED:", manual_closed)
        print("TTL_CLOSE_SENT_COUNT:", ttl_sent)
        print("CLOSED_POSITION_PROTECTIVE_CLEANUP_COUNT:", closed_cleanup)

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
