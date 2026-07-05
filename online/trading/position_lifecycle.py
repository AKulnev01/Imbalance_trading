from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from online.trading import config
from online.trading.bybit_client import BybitClient, format_bybit_decimal
from online.trading.db import db_cursor, json_default, read_sql


ACTIVE_POSITION_STATUSES = [
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

TERMINAL_ORDER_STATUSES = [
    "CANCELLED",
    "CANCELLED_NOT_FOUND",
    "FILLED",
    "TRIGGERED",
    "FAILED",
    "ERROR",
    "TP_SL_FAILED",
    "TTL_CLOSE_FAILED",
]


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


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


def round_qty_down(qty: float, qty_step: float) -> float:
    qty_f = float(qty)
    step_f = float(qty_step)

    if qty_f <= 0:
        return 0.0

    if step_f <= 0:
        return qty_f

    steps = math.floor(qty_f / step_f)
    return float(steps * step_f)


def close_side_for_position_side(side: str) -> str:
    side_u = str(side).upper()

    if side_u == "LONG":
        return "Sell"

    if side_u == "SHORT":
        return "Buy"

    raise RuntimeError("bad side for close order: {}".format(side))


def trigger_direction_for_reduce_only(side: str, trigger_kind: str) -> int:
    side_u = str(side).upper()
    kind_u = str(trigger_kind).upper()

    if side_u == "LONG" and kind_u == "TP":
        return 1

    if side_u == "LONG" and kind_u == "SL":
        return 2

    if side_u == "SHORT" and kind_u == "TP":
        return 2

    if side_u == "SHORT" and kind_u == "SL":
        return 1

    raise RuntimeError("bad side/trigger_kind: {} {}".format(side, trigger_kind))


def calc_directional_price(
    side: str,
    entry_price: float,
    atr14: float,
    atr_mult: float,
    direction: str,
) -> float:
    side_u = str(side).upper()
    direction_u = str(direction).upper()

    entry = float(entry_price)
    atr = float(atr14)
    mult = float(atr_mult)

    if entry <= 0:
        raise RuntimeError("bad entry_price: {}".format(entry_price))

    if atr <= 0:
        raise RuntimeError("bad atr14: {}".format(atr14))

    if side_u == "LONG" and direction_u == "PROFIT":
        return float(entry + atr * mult)

    if side_u == "LONG" and direction_u == "LOSS":
        return float(entry - atr * mult)

    if side_u == "SHORT" and direction_u == "PROFIT":
        return float(entry - atr * mult)

    if side_u == "SHORT" and direction_u == "LOSS":
        return float(entry + atr * mult)

    raise RuntimeError("bad side/direction: {} {}".format(side, direction))


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
        triggerPrice=format_bybit_decimal(trigger_px, max_decimals=10),
        triggerDirection=trigger_direction_for_reduce_only(side, trigger_kind),
        qty=format_bybit_decimal(qty),
        reduceOnly=True,
        orderLinkId=str(order_link_id),
    )

    client._raise_if_bad(resp)
    return resp


def is_bybit_order_already_gone_error(error_text: str) -> bool:
    text = str(error_text or "").lower()

    markers = [
        "110001",
        "order not exists",
        "too late to cancel",
        "order does not exist",
        "not found",
    ]

    return any(marker in text for marker in markers)


def is_bybit_duplicate_order_link_id_error(error_text: str) -> bool:
    text = str(error_text or "").lower()

    markers = [
        "110072",
        "orderlinkedid is duplicate",
        "order link id is duplicate",
        "duplicate",
    ]

    return any(marker in text for marker in markers)


def cancel_bybit_order_safe(
    client: BybitClient,
    symbol: str,
    order_id: Optional[str],
    order_link_id: Optional[str],
) -> Tuple[str, Dict[str, object]]:
    try:
        resp = client.cancel_order(
            symbol=str(symbol).upper(),
            order_id=str(order_id) if order_id else None,
            order_link_id=str(order_link_id) if order_link_id else None,
        )

        return "CANCELLED", {
            "cancel_status": "CANCELLED",
            "response": resp,
        }

    except Exception as e:
        err = str(e)

        if is_bybit_order_already_gone_error(err):
            return "CANCELLED_NOT_FOUND", {
                "cancel_status": "CANCELLED_NOT_FOUND",
                "error": err,
            }

        raise


def build_order_link_id(signal_key: str, role: str) -> str:
    import hashlib

    digest = hashlib.sha1(str(signal_key).encode("utf-8")).hexdigest()[:24]
    return "imb-{}-{}".format(digest, str(role).lower())

def is_active_lifecycle_order(order: Optional[Dict[str, Any]]) -> bool:
    if order is None:
        return False

    status = str(order.get("status") or "").upper()

    if status in set(TERMINAL_ORDER_STATUSES):
        return False

    return True


def ensure_lifecycle_columns() -> None:
    sql = """
        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS partial_tp_handled_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS early_stop_replaced_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS protective_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS closed_cleanup_done_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS position_seen_open_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ws_lifecycle_updated_at TIMESTAMPTZ;

        ALTER TABLE public.trading_orders
            ADD COLUMN IF NOT EXISTS lifecycle_note TEXT;
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)



def load_active_position_by_order_link_id(order_link_id: str) -> Optional[Dict[str, Any]]:
    sql = """
        SELECT
            p.*,
            o.local_order_key,
            o.order_role,
            o.bybit_order_id,
            o.bybit_order_link_id
        FROM public.trading_orders o
        JOIN public.trading_positions p
            ON p.trade_id = o.trade_id
        WHERE o.bybit_order_link_id = %s
          AND p.status = ANY(%s)
        ORDER BY p.updated_at DESC
        LIMIT 1
    """

    df = read_sql(sql, [str(order_link_id), ACTIVE_POSITION_STATUSES])

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def load_active_positions_with_due_early_stop(now_ts: Optional[pd.Timestamp] = None) -> List[Dict[str, Any]]:
    now_value = utc_now() if now_ts is None else pd.to_datetime(now_ts, utc=True)

    sql = """
        SELECT *
        FROM public.trading_positions
        WHERE status = ANY(%s)
          AND early_stop_expires_at IS NOT NULL
          AND early_stop_expires_at <= %s
          AND early_stop_replaced_at IS NULL
          AND trade_management_mode IS NOT NULL
        ORDER BY early_stop_expires_at ASC
    """

    df = read_sql(sql, [ACTIVE_POSITION_STATUSES, pd.Timestamp(now_value).to_pydatetime()])

    if df.empty:
        return []

    return [row.to_dict() for _, row in df.iterrows()]


def load_order_for_trade(trade_id: int, role: str) -> Optional[Dict[str, Any]]:
    sql = """
        SELECT *
        FROM public.trading_orders
        WHERE trade_id = %s
          AND UPPER(order_role) = %s
        ORDER BY created_at DESC
        LIMIT 1
    """

    df = read_sql(sql, [int(trade_id), str(role).upper()])

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def insert_lifecycle_order(
    trade_id: int,
    signal_key: str,
    symbol: str,
    side: str,
    order_role: str,
    local_order_key: str,
    order_type: str,
    qty_plan: Optional[float],
    price_plan: Optional[float],
    trigger_price_plan: Optional[float],
    status: str,
    request_json: Optional[Dict[str, object]],
    response_json: Optional[Dict[str, object]],
    bybit_order_id: Optional[str],
    bybit_order_link_id: Optional[str],
    lifecycle_note: str,
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
            lifecycle_note,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s, TRUE,
            %s, %s, %s,
            %s,
            NOW(), NOW(),
            %s::jsonb,
            %s::jsonb,
            %s,
            NOW()
        )
        ON CONFLICT (local_order_key)
        DO UPDATE SET
            status = EXCLUDED.status,
            bybit_order_id = EXCLUDED.bybit_order_id,
            bybit_order_link_id = EXCLUDED.bybit_order_link_id,
            request_json = EXCLUDED.request_json,
            response_json = EXCLUDED.response_json,
            lifecycle_note = EXCLUDED.lifecycle_note,
            updated_at = NOW()
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(local_order_key),
                int(trade_id),
                str(signal_key),
                str(symbol).upper(),
                str(side).upper(),
                str(order_role).upper(),
                bybit_order_id,
                bybit_order_link_id,
                str(order_type),
                qty_plan,
                price_plan,
                trigger_price_plan,
                str(status),
                json.dumps(request_json or {}, ensure_ascii=False, default=json_default),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
                str(lifecycle_note),
            ),
        )


def update_order_cancelled(
    local_order_key: str,
    status: str,
    response_json: Dict[str, object],
    lifecycle_note: str,
) -> None:
    sql = """
        UPDATE public.trading_orders
        SET status = %s,
            response_json = %s::jsonb,
            lifecycle_note = %s,
            updated_at = NOW()
        WHERE local_order_key = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(status),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
                str(lifecycle_note),
                str(local_order_key),
            ),
        )


def mark_partial_tp_handled(trade_id: int) -> None:
    sql = """
        UPDATE public.trading_positions
        SET partial_tp_handled_at = COALESCE(partial_tp_handled_at, NOW()),
            ws_lifecycle_updated_at = NOW(),
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))


def mark_early_stop_replaced(trade_id: int) -> None:
    sql = """
        UPDATE public.trading_positions
        SET early_stop_replaced_at = COALESCE(early_stop_replaced_at, NOW()),
            ws_lifecycle_updated_at = NOW(),
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))


def mark_protective_cleanup_done(trade_id: int) -> None:
    sql = """
        UPDATE public.trading_positions
        SET protective_cleanup_done_at = COALESCE(protective_cleanup_done_at, NOW()),
            ws_lifecycle_updated_at = NOW(),
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))


def append_order_lifecycle_note(local_order_key: str, note: str, status: Optional[str], response_json: Optional[Dict[str, Any]]) -> None:
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE public.trading_orders
            SET
                status = COALESCE(%s, status),
                lifecycle_note = COALESCE(lifecycle_note, '') ||
                    CASE
                        WHEN lifecycle_note IS NULL OR lifecycle_note = '' THEN ''
                        ELSE ';'
                    END ||
                    %s,
                response_json = COALESCE(response_json, '{}'::jsonb) || %s::jsonb,
                updated_at = NOW()
            WHERE local_order_key = %s
               OR bybit_order_link_id = %s
            """,
            (
                status,
                str(note),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
                str(local_order_key),
                str(local_order_key),
            ),
        )


def cancel_order_row_on_bybit(
    client: BybitClient,
    symbol: str,
    order_row: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    local_key = str(order_row.get("local_order_key") or "")
    order_id = order_row.get("bybit_order_id")
    order_link_id = order_row.get("bybit_order_link_id") or local_key

    if order_id is not None and pd.isna(order_id):
        order_id = None
    if order_link_id is not None and pd.isna(order_link_id):
        order_link_id = None

    session = client._get_session()

    try:
        response = session.cancel_order(
            category=client.category,
            symbol=str(symbol).upper(),
            orderId=None if order_id is None else str(order_id),
            orderLinkId=None if order_link_id is None else str(order_link_id),
        )
        status = "CANCELLED"
        payload = {
            "reason": str(reason),
            "cancel_status": status,
            "response": response,
        }
    except Exception as e:
        error_text = str(e)
        if is_bybit_order_not_exists_error(error_text):
            status = "CANCELLED_NOT_FOUND"
        else:
            status = "CANCEL_FAILED"

        payload = {
            "reason": str(reason),
            "cancel_status": status,
            "error": error_text,
        }

        if status == "CANCEL_FAILED":
            append_order_lifecycle_note(
                local_order_key=local_key,
                note=str(reason).lower(),
                status=status,
                response_json={str(reason).lower(): payload},
            )
            raise

    append_order_lifecycle_note(
        local_order_key=local_key,
        note=str(reason).lower(),
        status=status,
        response_json={str(reason).lower(): payload},
    )

    return payload


def cancel_active_main_stop_after_partial_tp(
    client: BybitClient,
    trade_id: int,
    symbol: str,
    source: str,
) -> Dict[str, Any]:
    main_stop = load_order_for_trade(int(trade_id), "STOP_LOSS")

    if not is_active_lifecycle_order(main_stop):
        return {
            "checked": 1,
            "cancelled": 0,
            "skipped": 1,
            "reason": "NO_ACTIVE_MAIN_STOP",
        }

    result = cancel_order_row_on_bybit(
        client=client,
        symbol=str(symbol).upper(),
        order_row=main_stop,
        reason="PARTIAL_TP_FILLED_CANCEL_MAIN_SL",
    )

    insert_event(
        trade_id=int(trade_id),
        signal_key=str(main_stop.get("signal_key") or ""),
        symbol=str(symbol).upper(),
        event_type="MAIN_SL_CANCELLED_AFTER_PARTIAL_TP",
        details={
            "source": str(source),
            "result": result,
        },
    )

    print("=" * 120, flush=True)
    print("WS_LIFECYCLE_MAIN_SL_CANCELLED_AFTER_PARTIAL_TP", flush=True)
    print("trade_id:", int(trade_id), flush=True)
    print("symbol:", str(symbol).upper(), flush=True)
    print("result:", result, flush=True)

    return {
        "checked": 1,
        "cancelled": 1,
        "skipped": 0,
        "result": result,
    }


def mark_position_closed_external_synced(
    trade_id: int,
    reason: str,
) -> None:
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE public.trading_positions
            SET
                status = 'POSITION_CLOSED_EXTERNAL_SYNCED',
                exit_reason = COALESCE(exit_reason, %s),
                closed_cleanup_done_at = COALESCE(
                    closed_cleanup_done_at,
                    NOW()
                ),
                ws_lifecycle_updated_at = NOW(),
                updated_at = NOW()
            WHERE trade_id = %s
            """,
            (
                str(reason),
                int(trade_id),
            ),
        )



def cleanup_zero_exchange_position_for_active_trade(
    client: BybitClient,
    trade_id: int,
    source: str = "ws_position_zero_event",
) -> Dict[str, Any]:
    ensure_lifecycle_columns()

    pos_df = read_sql(
        """
        SELECT *
        FROM public.trading_positions
        WHERE trade_id = %s
        LIMIT 1
        """,
        [int(trade_id)],
    )

    if pos_df.empty:
        return {
            "checked": 0,
            "cancelled": 0,
            "failed": 0,
            "reason": "POSITION_NOT_FOUND",
        }

    pos = pos_df.iloc[0].to_dict()

    symbol = str(pos.get("symbol") or "").upper()
    signal_key = str(pos.get("signal_key") or "")
    old_status = str(pos.get("status") or "")

    position_seen_open_at = pos.get("position_seen_open_at")
    partial_tp_handled_at = pos.get("partial_tp_handled_at")

    position_was_seen_open = (
        position_seen_open_at is not None
        and pd.notna(position_seen_open_at)
    )

    partial_tp_was_handled = (
        partial_tp_handled_at is not None
        and pd.notna(partial_tp_handled_at)
    )

    if not position_was_seen_open and not partial_tp_was_handled:
        result = {
            "checked": 0,
            "cancelled": 0,
            "failed": 0,
            "reason": "POSITION_ZERO_BEFORE_OPEN_CONFIRMATION",
        }

        insert_event(
            trade_id=int(trade_id),
            signal_key=signal_key,
            symbol=symbol,
            event_type="POSITION_ZERO_BEFORE_OPEN_CONFIRMATION",
            details={
                "source": str(source),
                "status": old_status,
            },
        )

        print("=" * 120, flush=True)
        print(
            "POSITION_ZERO_BEFORE_OPEN_CONFIRMATION",
            flush=True,
        )
        print("trade_id:", int(trade_id), flush=True)
        print("symbol:", symbol, flush=True)
        print("status:", old_status, flush=True)
        print("source:", str(source), flush=True)

        return result

    cleanup_result = cancel_remaining_protective_orders_once(
        client=client,
        trade_id=int(trade_id),
        source=str(source),
    )

    mark_position_closed_external_synced(
        trade_id=int(trade_id),
        reason="EXCHANGE_POSITION_ZERO_AUTO_CLEANUP",
    )

    insert_event(
        trade_id=int(trade_id),
        signal_key=signal_key,
        symbol=symbol,
        event_type="EXCHANGE_POSITION_ZERO_AUTO_CLEANUP",
        details={
            "source": str(source),
            "old_status": old_status,
            "new_status": "POSITION_CLOSED_EXTERNAL_SYNCED",
            "cleanup_result": cleanup_result,
            "position_was_seen_open": bool(
                position_was_seen_open
            ),
            "partial_tp_was_handled": bool(
                partial_tp_was_handled
            ),
        },
    )

    print("=" * 120, flush=True)
    print("EXCHANGE_POSITION_ZERO_AUTO_CLEANUP", flush=True)
    print("trade_id:", int(trade_id), flush=True)
    print("symbol:", symbol, flush=True)
    print("old_status:", old_status, flush=True)
    print(
        "new_status: POSITION_CLOSED_EXTERNAL_SYNCED",
        flush=True,
    )
    print("cleanup_result:", cleanup_result, flush=True)

    return cleanup_result



def is_position_closed_in_db(trade_id: int) -> bool:
    df = read_sql(
        """
        SELECT status
        FROM public.trading_positions
        WHERE trade_id = %s
        LIMIT 1
        """,
        [int(trade_id)],
    )

    if df.empty:
        return False

    status = str(df.iloc[0].get("status") or "").upper()
    return status.startswith("POSITION_CLOSED")


def handle_partial_tp_filled(
    client: BybitClient,
    order_link_id: str,
    source: str = "ws",
) -> bool:
    ensure_lifecycle_columns()

    pos = load_active_position_by_order_link_id(order_link_id)
    if pos is None:
        return False

    trade_id = int(pos["trade_id"])

    if pos.get("partial_tp_handled_at") is not None and pd.notna(pos.get("partial_tp_handled_at")):
        return False

    symbol = str(pos["symbol"]).upper()
    side = str(pos["side"]).upper()
    signal_key = str(pos["signal_key"])

    early_stop_order = load_order_for_trade(trade_id, "EARLY_STOP")
    if early_stop_order is not None:
        early_local_key = str(early_stop_order.get("local_order_key") or "")
        early_order_id = early_stop_order.get("bybit_order_id")
        early_link_id = early_stop_order.get("bybit_order_link_id")

        if pd.isna(early_order_id):
            early_order_id = None
        if pd.isna(early_link_id):
            early_link_id = None

        try:
            cancel_status, cancel_resp = cancel_bybit_order_safe(
                client=client,
                symbol=symbol,
                order_id=str(early_order_id) if early_order_id else None,
                order_link_id=str(early_link_id) if early_link_id else None,
            )
            update_order_cancelled(
                local_order_key=early_local_key,
                status=cancel_status,
                response_json={
                    "source": source,
                    "reason": "PARTIAL_TP_FILLED_CANCEL_EARLY_STOP",
                    "response": cancel_resp,
                },
                lifecycle_note="partial_tp_filled_cancel_early_stop",
            )
        except Exception as e:
            update_order_cancelled(
                local_order_key=early_local_key,
                status="CANCEL_FAILED",
                response_json={
                    "source": source,
                    "reason": "PARTIAL_TP_FILLED_CANCEL_EARLY_STOP_FAILED",
                    "error": str(e),
                },
                lifecycle_note="partial_tp_filled_cancel_early_stop_failed",
            )
            raise

    qty = safe_float(pos.get("final_tp_qty_plan"))
    if qty is None or qty <= 0:
        qty = safe_float(pos.get("qty"))

    rest_stop_px = safe_float(pos.get("rest_stop_after_partial_px_plan"))
    if rest_stop_px is None or rest_stop_px <= 0:
        entry_avg = safe_float(pos.get("entry_avg_px"))
        atr14 = safe_float(pos.get("atr14"))
        if entry_avg is None or atr14 is None:
            raise RuntimeError("cannot calculate rest stop: missing entry_avg_px/atr14 for trade_id={}".format(trade_id))

        rest_stop_atr = float(getattr(config, "REST_STOP_AFTER_PARTIAL_ATR_MULT", float(config.TP_ATR) * 0.125))
        rest_stop_px = calc_directional_price(
            side=side,
            entry_price=float(entry_avg),
            atr14=float(atr14),
            atr_mult=float(rest_stop_atr),
            direction="PROFIT",
        )

    if qty is None or qty <= 0:
        raise RuntimeError("bad rest stop qty for trade_id={}".format(trade_id))

    rest_link_id = build_order_link_id(signal_key, "reststop")

    try:
        rest_resp = place_reduce_only_trigger_market_order(
            client=client,
            symbol=symbol,
            side=side,
            qty=float(qty),
            trigger_px=float(rest_stop_px),
            trigger_kind="SL",
            order_link_id=rest_link_id,
        )
        rest_order_id = str(((rest_resp or {}).get("result") or {}).get("orderId") or "")
    except Exception as e:
        if not is_bybit_duplicate_order_link_id_error(str(e)):
            raise

        existing_rest_order = load_order_for_trade(trade_id, "REST_STOP_AFTER_PARTIAL")
        rest_order_id = str((existing_rest_order or {}).get("bybit_order_id") or "")
        rest_resp = {
            "idempotent_duplicate_order_link_id": True,
            "error": str(e),
            "existing_order": existing_rest_order or {},
        }

    insert_lifecycle_order(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        order_role="REST_STOP_AFTER_PARTIAL",
        local_order_key=rest_link_id,
        order_type="MarketTrigger",
        qty_plan=float(qty),
        price_plan=None,
        trigger_price_plan=float(rest_stop_px),
        status="SENT",
        request_json={
            "source": source,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "trigger_px": float(rest_stop_px),
            "reason": "PARTIAL_TP_FILLED_PLACE_REST_STOP",
        },
        response_json=rest_resp,
        bybit_order_id=rest_order_id,
        bybit_order_link_id=rest_link_id,
        lifecycle_note="partial_tp_filled_place_rest_stop",
    )

    cancel_active_main_stop_after_partial_tp(
        client=client,
        trade_id=trade_id,
        symbol=symbol,
        source=source,
    )

    mark_partial_tp_handled(trade_id)

    print("=" * 120, flush=True)
    print("WS_LIFECYCLE_PARTIAL_TP_HANDLED", flush=True)
    print("trade_id:", trade_id, flush=True)
    print("symbol:", symbol, flush=True)
    print("side:", side, flush=True)
    print("rest_stop_px:", rest_stop_px, flush=True)
    print("rest_stop_qty:", qty, flush=True)

    return True


def handle_early_stop_expired(
    client: BybitClient,
    position_row: Dict[str, Any],
    source: str = "ws_timer",
) -> bool:
    ensure_lifecycle_columns()

    trade_id = int(position_row["trade_id"])

    if position_row.get("early_stop_replaced_at") is not None and pd.notna(position_row.get("early_stop_replaced_at")):
        return False

    if is_position_closed_in_db(trade_id):
        return False

    symbol = str(position_row["symbol"]).upper()
    side = str(position_row["side"]).upper()
    signal_key = str(position_row["signal_key"])


    existing_main_sl_order = load_order_for_trade(trade_id, "STOP_LOSS")
    if is_active_lifecycle_order(existing_main_sl_order):
        mark_early_stop_replaced(trade_id)

        print("=" * 120, flush=True)
        print("WS_LIFECYCLE_EARLY_STOP_ALREADY_REPLACED_WITH_MAIN_SL", flush=True)
        print("trade_id:", trade_id, flush=True)
        print("symbol:", symbol, flush=True)
        print("side:", side, flush=True)
        print("existing_main_sl_status:", existing_main_sl_order.get("status"), flush=True)
        print("existing_main_sl_order_link_id:", existing_main_sl_order.get("bybit_order_link_id"), flush=True)

        return True

    early_stop_order = load_order_for_trade(trade_id, "EARLY_STOP")
    if early_stop_order is not None:
        early_local_key = str(early_stop_order.get("local_order_key") or "")
        early_order_id = early_stop_order.get("bybit_order_id")
        early_link_id = early_stop_order.get("bybit_order_link_id")

        if pd.isna(early_order_id):
            early_order_id = None
        if pd.isna(early_link_id):
            early_link_id = None

        try:
            cancel_status, cancel_resp = cancel_bybit_order_safe(
                client=client,
                symbol=symbol,
                order_id=str(early_order_id) if early_order_id else None,
                order_link_id=str(early_link_id) if early_link_id else None,
            )
            update_order_cancelled(
                local_order_key=early_local_key,
                status=cancel_status,
                response_json={
                    "source": source,
                    "reason": "EARLY_STOP_EXPIRED_CANCEL_EARLY_STOP",
                    "response": cancel_resp,
                },
                lifecycle_note="early_stop_expired_cancel_early_stop",
            )
        except Exception as e:
            update_order_cancelled(
                local_order_key=early_local_key,
                status="CANCEL_FAILED",
                response_json={
                    "source": source,
                    "reason": "EARLY_STOP_EXPIRED_CANCEL_EARLY_STOP_FAILED",
                    "error": str(e),
                },
                lifecycle_note="early_stop_expired_cancel_early_stop_failed",
            )
            raise

    qty = safe_float(position_row.get("qty"))
    main_sl_px = safe_float(position_row.get("main_sl_px_plan"))

    if qty is None or qty <= 0:
        raise RuntimeError("bad main stop qty for trade_id={}".format(trade_id))

    if main_sl_px is None or main_sl_px <= 0:
        entry_avg = safe_float(position_row.get("entry_avg_px"))
        atr14 = safe_float(position_row.get("atr14"))
        if entry_avg is None or atr14 is None:
            raise RuntimeError("cannot calculate main SL: missing entry_avg_px/atr14 for trade_id={}".format(trade_id))

        main_sl_px = calc_directional_price(
            side=side,
            entry_price=float(entry_avg),
            atr14=float(atr14),
            atr_mult=float(config.SL_ATR),
            direction="LOSS",
        )
    main_sl_link_id = build_order_link_id(signal_key, "mainsl")

    try:
        main_sl_resp = place_reduce_only_trigger_market_order(
            client=client,
            symbol=symbol,
            side=side,
            qty=float(qty),
            trigger_px=float(main_sl_px),
            trigger_kind="SL",
            order_link_id=main_sl_link_id,
        )
        main_sl_order_id = str(((main_sl_resp or {}).get("result") or {}).get("orderId") or "")
    except Exception as e:
        if not is_bybit_duplicate_order_link_id_error(str(e)):
            raise

        existing_main_sl_order = load_order_for_trade(trade_id, "STOP_LOSS")
        main_sl_order_id = str((existing_main_sl_order or {}).get("bybit_order_id") or "")
        main_sl_resp = {
            "idempotent_duplicate_order_link_id": True,
            "error": str(e),
            "existing_order": existing_main_sl_order or {},
        }

        print("=" * 120, flush=True)
        print("WS_LIFECYCLE_MAIN_SL_DUPLICATE_ORDER_LINK_ID_IDEMPOTENT", flush=True)
        print("trade_id:", trade_id, flush=True)
        print("symbol:", symbol, flush=True)
        print("side:", side, flush=True)
        print("main_sl_order_link_id:", main_sl_link_id, flush=True)
        print("existing_bybit_order_id:", main_sl_order_id, flush=True)

    insert_lifecycle_order(
        trade_id=trade_id,
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        order_role="STOP_LOSS",
        local_order_key=main_sl_link_id,
        order_type="MarketTrigger",
        qty_plan=float(qty),
        price_plan=None,
        trigger_price_plan=float(main_sl_px),
        status="SENT",
        request_json={
            "source": source,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "trigger_px": float(main_sl_px),
            "reason": "EARLY_STOP_EXPIRED_PLACE_MAIN_SL",
        },
        response_json=main_sl_resp,
        bybit_order_id=main_sl_order_id,
        bybit_order_link_id=main_sl_link_id,
        lifecycle_note="early_stop_expired_place_main_sl",
    )

    mark_early_stop_replaced(trade_id)

    print("=" * 120, flush=True)
    print("WS_LIFECYCLE_EARLY_STOP_REPLACED_WITH_MAIN_SL", flush=True)
    print("trade_id:", trade_id, flush=True)
    print("symbol:", symbol, flush=True)
    print("side:", side, flush=True)
    print("main_sl_px:", main_sl_px, flush=True)
    print("qty:", qty, flush=True)

    return True


def cancel_remaining_protective_orders_once(
    client: BybitClient,
    trade_id: int,
    source: str = "ws_position_closed",
) -> Dict[str, int]:
    ensure_lifecycle_columns()

    pos_df = read_sql(
        """
        SELECT trade_id
        FROM public.trading_positions
        WHERE trade_id = %s
        LIMIT 1
        """,
        [int(trade_id)],
    )

    if pos_df.empty:
        return {
            "checked": 0,
            "cancelled": 0,
            "failed": 0,
            "remaining": 0,
        }

    placeholders = ", ".join(
        ["%s"] * len(PROTECTIVE_ORDER_ROLES)
    )

    terminal_placeholders = ", ".join(
        ["%s"] * len(TERMINAL_ORDER_STATUSES)
    )

    query_params = (
        [int(trade_id)]
        + PROTECTIVE_ORDER_ROLES
        + TERMINAL_ORDER_STATUSES
    )

    orders = read_sql(
        """
        SELECT *
        FROM public.trading_orders
        WHERE trade_id = %s
          AND UPPER(order_role) IN ({roles})
          AND UPPER(COALESCE(status, '')) NOT IN ({terminal})
        ORDER BY created_at ASC
        """.format(
            roles=placeholders,
            terminal=terminal_placeholders,
        ),
        query_params,
    )

    result = {
        "checked": 0,
        "cancelled": 0,
        "failed": 0,
        "remaining": 0,
    }

    for _, row in orders.iterrows():
        order = row.to_dict()

        local_order_key = str(
            order.get("local_order_key") or ""
        )

        symbol = str(
            order.get("symbol") or ""
        ).upper()

        role = str(
            order.get("order_role") or ""
        ).upper()

        bybit_order_id = order.get("bybit_order_id")
        bybit_order_link_id = order.get(
            "bybit_order_link_id"
        )

        if pd.isna(bybit_order_id):
            bybit_order_id = None

        if pd.isna(bybit_order_link_id):
            bybit_order_link_id = None

        result["checked"] += 1

        try:
            cancel_status, resp = cancel_bybit_order_safe(
                client=client,
                symbol=symbol,
                order_id=(
                    str(bybit_order_id)
                    if bybit_order_id
                    else None
                ),
                order_link_id=(
                    str(bybit_order_link_id)
                    if bybit_order_link_id
                    else None
                ),
            )

            update_order_cancelled(
                local_order_key=local_order_key,
                status=cancel_status,
                response_json={
                    "source": source,
                    "reason": (
                        "POSITION_CLOSED_CANCEL_REMAINING_"
                        "PROTECTIVE_ORDER"
                    ),
                    "order_role": role,
                    "response": resp,
                },
                lifecycle_note=(
                    "position_closed_cancel_remaining_"
                    "protective_order"
                ),
            )

            result["cancelled"] += 1

        except Exception as e:
            update_order_cancelled(
                local_order_key=local_order_key,
                status="CANCEL_FAILED",
                response_json={
                    "source": source,
                    "reason": (
                        "POSITION_CLOSED_CANCEL_REMAINING_"
                        "PROTECTIVE_ORDER_FAILED"
                    ),
                    "order_role": role,
                    "error": str(e),
                },
                lifecycle_note=(
                    "position_closed_cancel_remaining_"
                    "protective_order_failed"
                ),
            )

            result["failed"] += 1

    remaining_orders = read_sql(
        """
        SELECT local_order_key
        FROM public.trading_orders
        WHERE trade_id = %s
          AND UPPER(order_role) IN ({roles})
          AND UPPER(COALESCE(status, '')) NOT IN ({terminal})
        ORDER BY created_at ASC
        """.format(
            roles=placeholders,
            terminal=terminal_placeholders,
        ),
        query_params,
    )

    result["remaining"] = int(len(remaining_orders))

    if (
        result["failed"] == 0
        and result["remaining"] == 0
    ):
        mark_protective_cleanup_done(
            int(trade_id)
        )

    print("=" * 120, flush=True)
    print(
        "WS_LIFECYCLE_CLOSED_POSITION_PROTECTIVE_CLEANUP",
        flush=True,
    )
    print("trade_id:", int(trade_id), flush=True)
    print("checked:", int(result["checked"]), flush=True)
    print("cancelled:", int(result["cancelled"]), flush=True)
    print("failed:", int(result["failed"]), flush=True)
    print("remaining:", int(result["remaining"]), flush=True)

    return result

