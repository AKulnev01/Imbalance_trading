
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from online.trading import audit_log
from online.trading.bybit_client import BybitClient
from online.trading.db import db_cursor, json_default, read_sql
from online.trading.locks import acquire_lock, release_lock


LOCK_NAME = "cancel_stale_protective_orders"

DRY_RUN = os.environ.get("IMB_TRADING_DRY_RUN", "1").strip() != "0"


def is_bybit_order_not_exists_error(error_text: str) -> bool:
    text = str(error_text or "").lower()

    if "110001" in text:
        return True

    if "order not exists" in text:
        return True

    if "too late to cancel" in text:
        return True

    return False

PROTECTIVE_ORDER_ROLES = [
    "TAKE_PROFIT",
    "STOP_LOSS",
]

TERMINAL_ORDER_STATUSES = [
    "CANCELLED",
    "CANCELED",
    "CANCELLED_NOT_FOUND",
    "FILLED",
    "REJECTED",
    "FAILED",
    "TP_SL_FAILED",
    "DRY_RUN_NOT_SENT",
    "DRY_RUN_STALE_CANCEL_SKIPPED",
    "STALE_CANCELLED",
    "STALE_CANCELLED_NOT_FOUND",
    "STALE_CANCEL_FAILED",
]


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_active_protective_orders() -> pd.DataFrame:
    sql = """
        SELECT
            o.local_order_key,
            o.trade_id,
            o.signal_key,
            o.symbol,
            o.side,
            o.order_role,
            o.bybit_order_id,
            o.bybit_order_link_id,
            o.status AS order_status,
            o.qty_plan,
            o.trigger_price_plan,
            o.created_at AS order_created_at,
            o.updated_at AS order_updated_at,

            p.status AS position_status,
            p.qty AS position_qty,
            p.entry_avg_px,
            p.exit_reason,
            p.exit_filled_at
        FROM public.trading_orders o
        LEFT JOIN public.trading_positions p
            ON p.trade_id = o.trade_id
        WHERE o.order_role = ANY(%s)
          AND COALESCE(o.status, '') <> ALL(%s)
          AND (
              NULLIF(o.bybit_order_id, '') IS NOT NULL
              OR NULLIF(o.bybit_order_link_id, '') IS NOT NULL
          )
        ORDER BY
            o.symbol ASC,
            o.trade_id ASC,
            o.order_role ASC
    """

    return read_sql(sql, [PROTECTIVE_ORDER_ROLES, TERMINAL_ORDER_STATUSES])


def get_exchange_open_symbols(client: BybitClient) -> Set[str]:
    positions = client.get_open_positions()
    out: Set[str] = set()

    for p in positions:
        symbol = normalize_symbol(p.get("symbol"))
        size = safe_float(p.get("size"))

        if symbol and size is not None and abs(size) > 0:
            out.add(symbol)

    return out


def mark_order_cancelled(
    local_order_key: str,
    status: str,
    response: Dict[str, Any],
    error_message: Optional[str] = None,
) -> None:
    sql = """
        UPDATE public.trading_orders
        SET
            status = %s,
            response_json = %s::jsonb,
            error_message = %s,
            updated_at = NOW()
        WHERE local_order_key = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(status),
                json.dumps(response or {}, ensure_ascii=False, default=json_default),
                error_message,
                str(local_order_key),
            ),
        )


def cancel_one_order(client: BybitClient, row: Dict[str, Any]) -> Dict[str, Any]:
    symbol = normalize_symbol(row.get("symbol"))
    local_order_key = str(row.get("local_order_key") or "")
    order_role = str(row.get("order_role") or "")
    bybit_order_id = row.get("bybit_order_id")
    bybit_order_link_id = row.get("bybit_order_link_id")

    if pd.isna(bybit_order_id):
        bybit_order_id = None
    if pd.isna(bybit_order_link_id):
        bybit_order_link_id = None

    bybit_order_id_s = str(bybit_order_id or "").strip()
    bybit_order_link_id_s = str(bybit_order_link_id or "").strip()

    if not symbol:
        raise RuntimeError("empty symbol for local_order_key={}".format(local_order_key))

    if not bybit_order_id_s and not bybit_order_link_id_s:
        raise RuntimeError("empty bybit order ids for local_order_key={}".format(local_order_key))

    if DRY_RUN:
        response = {
            "dry_run": True,
            "message": "stale protective order would be cancelled",
            "symbol": symbol,
            "local_order_key": local_order_key,
            "order_role": order_role,
            "bybit_order_id": bybit_order_id_s,
            "bybit_order_link_id": bybit_order_link_id_s,
        }
        mark_order_cancelled(
            local_order_key=local_order_key,
            status="DRY_RUN_STALE_CANCEL_SKIPPED",
            response=response,
            error_message=None,
        )
        return response

    response = client.cancel_order(
        symbol=symbol,
        order_id=bybit_order_id_s if bybit_order_id_s else None,
        order_link_id=bybit_order_link_id_s if bybit_order_link_id_s else None,
    )

    mark_order_cancelled(
        local_order_key=local_order_key,
        status="STALE_CANCELLED",
        response=response,
        error_message=None,
    )

    return response


def run_once() -> Dict[str, Any]:
    orders = load_active_protective_orders()

    if orders.empty:
        result = {
            "status": "NO_ACTIVE_PROTECTIVE_ORDERS",
            "checked_orders": 0,
            "checked_symbols": 0,
            "cancelled": 0,
            "failed": 0,
            "dry_run": DRY_RUN,
        }
        print(json.dumps(result, ensure_ascii=False, default=json_default))
        return result

    symbols = sorted(set(normalize_symbol(x) for x in orders["symbol"].dropna().tolist()))
    symbols = [x for x in symbols if x]

    client = BybitClient()
    open_symbols = get_exchange_open_symbols(client)

    stale = orders[~orders["symbol"].astype(str).str.upper().isin(open_symbols)].copy()

    cancelled = 0
    cancelled_not_found = 0
    failed = 0
    errors: List[Dict[str, Any]] = []

    for _, item in stale.iterrows():
        row = item.to_dict()
        local_order_key = str(row.get("local_order_key") or "")

        try:
            cancel_one_order(client, row)
            cancelled += 1

            audit_log.log_audit_event(
                event_type="STALE_PROTECTIVE_ORDER_CANCELLED",
                status="OK",
                trade_id=int(row["trade_id"]) if pd.notna(row.get("trade_id")) else None,
                signal_key=str(row.get("signal_key") or ""),
                symbol=normalize_symbol(row.get("symbol")),
                side=str(row.get("side") or "").upper(),
                message="Protective TP/SL order was cancelled because exchange position is absent",
                payload={
                    "local_order_key": local_order_key,
                    "order_role": str(row.get("order_role") or ""),
                    "order_status": str(row.get("order_status") or ""),
                    "position_status": str(row.get("position_status") or ""),
                    "dry_run": DRY_RUN,
                },
            )

        except Exception as e:
            error_message = str(e)

            if is_bybit_order_not_exists_error(error_message):
                cancelled_not_found += 1

                try:
                    mark_order_cancelled(
                        local_order_key=local_order_key,
                        status="STALE_CANCELLED_NOT_FOUND",
                        response={
                            "cleanup_status": "ORDER_ALREADY_GONE_ON_BYBIT",
                            "error": error_message,
                        },
                        error_message=None,
                    )
                except Exception:
                    pass

                audit_log.log_audit_event(
                    event_type="STALE_PROTECTIVE_ORDER_CANCELLED_NOT_FOUND",
                    status="OK",
                    trade_id=int(row["trade_id"]) if pd.notna(row.get("trade_id")) else None,
                    signal_key=str(row.get("signal_key") or ""),
                    symbol=normalize_symbol(row.get("symbol")),
                    side=str(row.get("side") or "").upper(),
                    message="Protective TP/SL order is already absent on Bybit",
                    payload={
                        "local_order_key": local_order_key,
                        "order_role": str(row.get("order_role") or ""),
                        "order_status": str(row.get("order_status") or ""),
                        "position_status": str(row.get("position_status") or ""),
                        "dry_run": DRY_RUN,
                        "error": error_message,
                    },
                )
                continue

            failed += 1

            errors.append(
                {
                    "local_order_key": local_order_key,
                    "symbol": normalize_symbol(row.get("symbol")),
                    "order_role": str(row.get("order_role") or ""),
                    "error": error_message,
                }
            )

            try:
                mark_order_cancelled(
                    local_order_key=local_order_key,
                    status="STALE_CANCEL_FAILED",
                    response={"error": error_message},
                    error_message=error_message,
                )
            except Exception:
                pass

            audit_log.log_audit_event(
                event_type="STALE_PROTECTIVE_ORDER_CANCEL_FAILED",
                status="ERROR",
                trade_id=int(row["trade_id"]) if pd.notna(row.get("trade_id")) else None,
                signal_key=str(row.get("signal_key") or ""),
                symbol=normalize_symbol(row.get("symbol")),
                side=str(row.get("side") or "").upper(),
                message=error_message,
                payload={
                    "local_order_key": local_order_key,
                    "order_role": str(row.get("order_role") or ""),
                    "order_status": str(row.get("order_status") or ""),
                    "position_status": str(row.get("position_status") or ""),
                    "dry_run": DRY_RUN,
                },
            )

    result = {
        "status": "DONE",
        "checked_orders": int(len(orders)),
        "checked_symbols": int(len(symbols)),
        "exchange_open_symbols": sorted(open_symbols),
        "stale_orders": int(len(stale)),
        "cancelled": int(cancelled),
        "cancelled_not_found": int(cancelled_not_found),
        "failed": int(failed),
        "dry_run": DRY_RUN,
        "errors": errors,
    }

    print(json.dumps(result, ensure_ascii=False, default=json_default))
    return result


def main() -> None:
    audit_log.ensure_audit_tables()

    owner = "cancel_stale_protective_orders:{}".format(pd.Timestamp.now(tz="UTC"))

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=300):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        run_once()
    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()