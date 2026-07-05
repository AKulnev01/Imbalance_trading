from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from online.trading.bybit_client import BybitClient
from online.trading.db import db_cursor, json_default, read_sql
from online.trading.locks import acquire_lock, release_lock


LOCK_NAME = "trading_reconcile"
DRY_RUN = os.environ.get("IMB_TRADING_DRY_RUN", "1").strip() != "0"


ENTRY_ROLES = ["ENTRY_MARKET"]
EXIT_ROLES = [
    "TAKE_PROFIT",
    "PARTIAL_TP",
    "FINAL_TP",
    "STOP_LOSS",
    "EARLY_STOP",
    "REST_STOP_AFTER_PARTIAL",
    "TTL_CLOSE",
    "EMERGENCY_CLOSE",
    "MANUAL_CLOSE",
]
PARTIAL_EXIT_ROLES = ["PARTIAL_TP"]
FINAL_EXIT_ROLES = [
    "TAKE_PROFIT",
    "FINAL_TP",
    "STOP_LOSS",
    "EARLY_STOP",
    "REST_STOP_AFTER_PARTIAL",
    "TTL_CLOSE",
    "EMERGENCY_CLOSE",
    "MANUAL_CLOSE",
]

PROTECTIVE_ORDER_ROLES = [
    "TAKE_PROFIT",
    "PARTIAL_TP",
    "FINAL_TP",
    "STOP_LOSS",
    "EARLY_STOP",
    "REST_STOP_AFTER_PARTIAL",
]

ORDER_TERMINAL_STATUSES = {
    "CANCELLED",
    "CANCELLED_NOT_FOUND",
    "CANCEL_FAILED",
    "FILLED",
    "TRIGGERED",
    "FAILED",
    "ERROR",
    "TP_SL_FAILED",
    "TTL_CLOSE_FAILED",
}


def is_bybit_order_not_exists_error(error_text: str) -> bool:
    text = str(error_text or "").lower()

    if "110001" in text:
        return True

    if "order not exists" in text:
        return True

    if "too late to cancel" in text:
        return True

    return False


def ensure_reconcile_tables() -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS public.trading_exchange_position_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            snapshot_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT NOT NULL,
            side TEXT NULL,
            size DOUBLE PRECISION NULL,
            avg_price DOUBLE PRECISION NULL,
            mark_price DOUBLE PRECISION NULL,
            unrealised_pnl DOUBLE PRECISION NULL,
            raw_json JSONB NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_trading_exchange_position_snapshots_symbol_ts
        ON public.trading_exchange_position_snapshots (symbol, snapshot_ts DESC);

        CREATE TABLE IF NOT EXISTS public.trading_fills (
            fill_id BIGSERIAL PRIMARY KEY
        );

        ALTER TABLE public.trading_fills
            ADD COLUMN IF NOT EXISTS trade_id BIGINT NULL,
            ADD COLUMN IF NOT EXISTS signal_key TEXT NULL,
            ADD COLUMN IF NOT EXISTS local_order_key TEXT NULL,
            ADD COLUMN IF NOT EXISTS bybit_order_id TEXT NULL,
            ADD COLUMN IF NOT EXISTS bybit_order_link_id TEXT NULL,
            ADD COLUMN IF NOT EXISTS bybit_exec_id TEXT NULL,
            ADD COLUMN IF NOT EXISTS symbol TEXT NULL,
            ADD COLUMN IF NOT EXISTS side TEXT NULL,
            ADD COLUMN IF NOT EXISTS order_role TEXT NULL,
            ADD COLUMN IF NOT EXISTS exec_price DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS exec_qty DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS exec_value DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS exec_fee DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS fee_currency TEXT NULL,
            ADD COLUMN IF NOT EXISTS executed_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

        UPDATE public.trading_fills
        SET raw_json = '{}'::jsonb
        WHERE raw_json IS NULL;

        UPDATE public.trading_fills
        SET created_at = NOW()
        WHERE created_at IS NULL;

        UPDATE public.trading_fills
        SET updated_at = NOW()
        WHERE updated_at IS NULL;

        CREATE UNIQUE INDEX IF NOT EXISTS ux_trading_fills_bybit_exec_id
        ON public.trading_fills (bybit_exec_id)
        WHERE bybit_exec_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_trading_fills_trade_id
        ON public.trading_fills (trade_id);

        CREATE INDEX IF NOT EXISTS idx_trading_fills_symbol_executed_at
        ON public.trading_fills (symbol, executed_at DESC);

        ALTER TABLE public.trading_positions
            ADD COLUMN IF NOT EXISTS h4_close DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS atr14 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS entry_px_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS entry_slippage_abs DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS entry_slippage_pct DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS exit_avg_px DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS exit_slippage_abs DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS exit_slippage_pct DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS entry_filled_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS exit_filled_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS exit_reason TEXT,
            ADD COLUMN IF NOT EXISTS pnl_usd DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS fee_usd DOUBLE PRECISION,
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

def safe_float(value) -> Optional[float]:
    try:
        if value in [None, ""]:
            return None
        v = float(value)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v


def safe_ts_from_ms(value) -> Optional[pd.Timestamp]:
    try:
        if value in [None, ""]:
            return None
        return pd.to_datetime(int(value), unit="ms", utc=True)
    except Exception:
        return None


def write_exchange_positions_snapshot(positions: List[Dict[str, object]]) -> int:
    if not positions:
        return 0

    sql = """
        INSERT INTO public.trading_exchange_position_snapshots (
            symbol,
            side,
            size,
            avg_price,
            mark_price,
            unrealised_pnl,
            raw_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
    """

    n = 0
    with db_cursor(commit=True) as (_, cur):
        for p in positions:
            cur.execute(
                sql,
                (
                    str(p.get("symbol") or "").upper(),
                    str(p.get("side") or ""),
                    safe_float(p.get("size")),
                    safe_float(p.get("avgPrice")),
                    safe_float(p.get("markPrice")),
                    safe_float(p.get("unrealisedPnl")),
                    json.dumps(p, ensure_ascii=False, default=json_default),
                ),
            )
            n += 1

    return n


def load_recent_orders() -> pd.DataFrame:
    sql = """
        SELECT
            o.*,
            p.entry_px_plan,
            p.entry_avg_px,
            p.exit_avg_px,
            COALESCE(p.h4_close, s.h4_close) AS h4_close,
            COALESCE(p.atr14, s.atr14) AS atr14,
            p.tp_px_plan,
            p.sl_px_plan,
            p.ttl_close_ts,
            p.exit_reason
        FROM public.trading_orders o
        LEFT JOIN public.trading_positions p
            ON p.trade_id = o.trade_id
        LEFT JOIN public.trading_signals s
            ON s.signal_key = o.signal_key
        WHERE o.created_at >= NOW() - INTERVAL '14 days'
        ORDER BY o.created_at DESC
    """
    return read_sql(sql)


def update_order_snapshot(local_order_key: str, response: Dict[str, object]) -> None:
    sql = """
        UPDATE public.trading_orders
        SET response_json = %s::jsonb,
            updated_at = NOW()
        WHERE local_order_key = %s
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                json.dumps(response, ensure_ascii=False, default=json_default),
                local_order_key,
            ),
        )

def update_order_status_from_execution(
    local_order_key: str,
    exec_rows: List[Dict[str, object]],
) -> None:
    filled_qty = 0.0

    for r in exec_rows:
        qty = safe_float(r.get("execQty"))
        if qty is not None and qty > 0:
            filled_qty += float(qty)

    if filled_qty <= 0:
        return

    sql = """
        UPDATE public.trading_orders
        SET status = 'FILLED',
            updated_at = NOW()
        WHERE local_order_key = %s
          AND status NOT IN (
              'CANCELLED',
              'CANCEL_FAILED',
              'FILLED',
              'FAILED',
              'ERROR',
              'TP_SL_FAILED',
              'TTL_CLOSE_FAILED'
          )
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (str(local_order_key),))

def extract_execution_rows(resp: Dict[str, object]) -> List[Dict[str, object]]:
    result = resp.get("result") or {}
    rows = result.get("list") or []
    if not isinstance(rows, list):
        return []
    return rows


def insert_fills_for_order(order_row: Dict[str, object], exec_rows: List[Dict[str, object]]) -> int:
    if not exec_rows:
        return 0

    sql = """
        INSERT INTO public.trading_fills (
            trade_id,
            signal_key,
            local_order_key,
            bybit_order_id,
            bybit_order_link_id,
            bybit_exec_id,
            symbol,
            side,
            order_role,
            exec_price,
            exec_qty,
            exec_value,
            exec_fee,
            fee_currency,
            executed_at,
            raw_json,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s::jsonb, NOW()
        )
        ON CONFLICT (bybit_exec_id)
        DO UPDATE SET
            trade_id = EXCLUDED.trade_id,
            signal_key = EXCLUDED.signal_key,
            local_order_key = EXCLUDED.local_order_key,
            bybit_order_id = EXCLUDED.bybit_order_id,
            bybit_order_link_id = EXCLUDED.bybit_order_link_id,
            symbol = EXCLUDED.symbol,
            side = EXCLUDED.side,
            order_role = EXCLUDED.order_role,
            exec_price = EXCLUDED.exec_price,
            exec_qty = EXCLUDED.exec_qty,
            exec_value = EXCLUDED.exec_value,
            exec_fee = EXCLUDED.exec_fee,
            fee_currency = EXCLUDED.fee_currency,
            executed_at = EXCLUDED.executed_at,
            raw_json = EXCLUDED.raw_json,
            updated_at = NOW()
    """

    n = 0
    with db_cursor(commit=True) as (_, cur):
        for r in exec_rows:
            exec_id = str(r.get("execId") or "")
            if not exec_id:
                continue

            exec_price = safe_float(r.get("execPrice"))
            exec_qty = safe_float(r.get("execQty"))

            if exec_price is None or exec_qty is None or exec_qty <= 0:
                continue

            exec_value = safe_float(r.get("execValue"))
            exec_fee = safe_float(r.get("execFee"))
            executed_at = safe_ts_from_ms(r.get("execTime"))

            cur.execute(
                sql,
                (
                    int(order_row["trade_id"]) if pd.notna(order_row.get("trade_id")) else None,
                    str(order_row.get("signal_key") or ""),
                    str(order_row.get("local_order_key") or ""),
                    str(order_row.get("bybit_order_id") or "") if pd.notna(order_row.get("bybit_order_id")) else None,
                    str(order_row.get("bybit_order_link_id") or "") if pd.notna(order_row.get("bybit_order_link_id")) else None,
                    exec_id,
                    str(order_row.get("symbol") or "").upper(),
                    str(order_row.get("side") or ""),
                    str(order_row.get("order_role") or ""),
                    float(exec_price),
                    float(exec_qty),
                    exec_value,
                    exec_fee,
                    str(r.get("feeCurrency") or ""),
                    None if executed_at is None else executed_at.to_pydatetime(),
                    json.dumps(r, ensure_ascii=False, default=json_default),
                ),
            )
            n += 1

    return n


def weighted_avg_price(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None

    qty = pd.to_numeric(df["exec_qty"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(df["exec_price"], errors="coerce")

    total_qty = float(qty.sum())
    if total_qty <= 0:
        return None

    return float((px * qty).sum() / total_qty)


def sum_qty(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["exec_qty"], errors="coerce").fillna(0.0).sum())


def sum_fee(df: pd.DataFrame) -> float:
    if df.empty or "exec_fee" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["exec_fee"], errors="coerce").fillna(0.0).sum())


def first_ts(df: pd.DataFrame):
    if df.empty:
        return None
    ts = pd.to_datetime(df["executed_at"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.min().to_pydatetime()


def last_ts(df: pd.DataFrame):
    if df.empty:
        return None
    ts = pd.to_datetime(df["executed_at"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.max().to_pydatetime()


def load_fills_for_trade(trade_id: int) -> pd.DataFrame:
    sql = """
        SELECT *
        FROM public.trading_fills
        WHERE trade_id = %s
        ORDER BY executed_at ASC, fill_id ASC
    """
    return read_sql(sql, [int(trade_id)])
def update_order_statuses_from_fills(trade_id: int) -> None:
    sql = """
        UPDATE public.trading_orders o
        SET status = 'FILLED',
            updated_at = NOW()
        WHERE o.trade_id = %s
          AND o.status NOT IN (
              'CANCELLED',
              'CANCEL_FAILED',
              'FILLED',
              'FAILED',
              'ERROR',
              'TP_SL_FAILED',
              'TTL_CLOSE_FAILED'
          )
          AND EXISTS (
              SELECT 1
              FROM public.trading_fills f
              WHERE f.trade_id = o.trade_id
                AND f.local_order_key = o.local_order_key
                AND COALESCE(f.exec_qty, 0) > 0
          )
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))


def calc_pnl_usd(side: str, entry_avg: Optional[float], exit_avg: Optional[float], qty: float, fee_usd: float) -> Optional[float]:
    if entry_avg is None or exit_avg is None or qty <= 0:
        return None

    side_u = str(side).upper()
    if side_u == "LONG":
        gross = (exit_avg - entry_avg) * qty
    elif side_u == "SHORT":
        gross = (entry_avg - exit_avg) * qty
    else:
        return None

    return float(gross - fee_usd)



def calc_realized_pnl_usd(side: str, entry_avg: Optional[float], exit_df: pd.DataFrame, fee_usd: float) -> Optional[float]:
    if entry_avg is None or exit_df.empty:
        return None

    qty = pd.to_numeric(exit_df["exec_qty"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(exit_df["exec_price"], errors="coerce")

    valid = (qty > 0) & px.notna() & (px > 0)
    if not bool(valid.any()):
        return None

    qty = qty[valid]
    px = px[valid]

    side_u = str(side).upper()
    if side_u == "LONG":
        gross = ((px - float(entry_avg)) * qty).sum()
    elif side_u == "SHORT":
        gross = ((float(entry_avg) - px) * qty).sum()
    else:
        return None

    return float(gross - float(fee_usd))

def detect_exit_reason(exit_df: pd.DataFrame) -> Optional[str]:
    if exit_df.empty:
        return None

    roles = set(str(x).upper() for x in exit_df["order_role"].dropna().tolist())

    if "REST_STOP_AFTER_PARTIAL" in roles:
        return "REST_STOP_AFTER_PARTIAL"
    if "EARLY_STOP" in roles:
        return "EARLY_STOP"
    if "STOP_LOSS" in roles:
        return "STOP_LOSS"
    if "TTL_CLOSE" in roles:
        return "TTL_CLOSE"
    if "EMERGENCY_CLOSE" in roles:
        return "EMERGENCY_CLOSE"
    if "MANUAL_CLOSE" in roles:
        return "MANUAL_CLOSE"
    if "FINAL_TP" in roles:
        return "FINAL_TP"
    if "TAKE_PROFIT" in roles:
        return "TAKE_PROFIT"
    if "PARTIAL_TP" in roles:
        return "PARTIAL_TP"

    return "UNKNOWN_EXIT"



def update_position_from_fills(trade_id: int) -> None:
    fills = load_fills_for_trade(trade_id)
    if fills.empty:
        return

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
        return

    pos = pos_df.iloc[0].to_dict()
    side = str(pos.get("side") or "").upper()

    fills["order_role"] = fills["order_role"].astype(str).str.upper()

    entry_df = fills[fills["order_role"].isin(ENTRY_ROLES)].copy()
    exit_df = fills[fills["order_role"].isin(EXIT_ROLES)].copy()
    partial_exit_df = fills[fills["order_role"].isin(PARTIAL_EXIT_ROLES)].copy()
    final_exit_df = fills[fills["order_role"].isin(FINAL_EXIT_ROLES)].copy()

    entry_avg = weighted_avg_price(entry_df)
    exit_avg = weighted_avg_price(exit_df)

    entry_qty = sum_qty(entry_df)
    exit_qty = sum_qty(exit_df)
    partial_exit_qty = sum_qty(partial_exit_df)
    final_exit_qty = sum_qty(final_exit_df)

    fee_usd = sum_fee(fills)

    h4_close = safe_float(pos.get("h4_close"))
    entry_slippage_abs = None
    entry_slippage_pct = None

    if entry_avg is not None and h4_close is not None and h4_close > 0:
        if side == "LONG":
            entry_slippage_abs = entry_avg - h4_close
        elif side == "SHORT":
            entry_slippage_abs = h4_close - entry_avg

        if entry_slippage_abs is not None:
            entry_slippage_pct = entry_slippage_abs / h4_close

    exit_reason = detect_exit_reason(exit_df)
    pnl_usd = calc_realized_pnl_usd(side, entry_avg, exit_df, fee_usd)

    if entry_qty > 0 and exit_qty <= 0:
        status = "POSITION_OPEN"
    elif entry_qty > 0 and exit_qty >= entry_qty * 0.999:
        status = "POSITION_CLOSED_" + str(exit_reason or "UNKNOWN_EXIT")
    elif entry_qty > 0 and partial_exit_qty > 0 and final_exit_qty <= 0:
        status = "POSITION_OPEN"
        exit_reason = "PARTIAL_TP_FILLED"
    else:
        status = str(pos.get("status") or "UNKNOWN")

    sql = """
        UPDATE public.trading_positions
        SET
            qty = CASE WHEN %s > 0 THEN %s ELSE qty END,
            entry_avg_px = COALESCE(%s, entry_avg_px),
            entry_filled_at = COALESCE(%s, entry_filled_at),
            entry_slippage_abs = COALESCE(%s, entry_slippage_abs),
            entry_slippage_pct = COALESCE(%s, entry_slippage_pct),

            exit_avg_px = COALESCE(%s, exit_avg_px),
            exit_filled_at = CASE WHEN %s IS NOT NULL THEN %s ELSE exit_filled_at END,
            exit_reason = CASE WHEN %s IS NOT NULL THEN %s ELSE exit_reason END,

            fee_usd = %s,
            pnl_usd = COALESCE(%s, pnl_usd),
            status = CASE
                WHEN status LIKE 'POSITION_CLOSED%%' THEN status
                ELSE %s
            END,
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                float(entry_qty),
                float(entry_qty),
                entry_avg,
                first_ts(entry_df),
                entry_slippage_abs,
                entry_slippage_pct,
                exit_avg,
                last_ts(exit_df),
                last_ts(exit_df),
                exit_reason,
                exit_reason,
                float(fee_usd),
                pnl_usd,
                status,
                int(trade_id),
            ),
        )



def update_cancelled_order_after_closed_position(
    local_order_key: str,
    status: str,
    response_json: Dict[str, object],
) -> None:
    sql = """
        UPDATE public.trading_orders
        SET status = %s,
            response_json = %s::jsonb,
            updated_at = NOW()
        WHERE local_order_key = %s
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(status),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
                str(local_order_key),
            ),
        )


def load_unfilled_protective_orders_for_closed_trade(trade_id: int) -> pd.DataFrame:
    roles = [str(x).upper() for x in PROTECTIVE_ORDER_ROLES]
    terminal_statuses = [str(x).upper() for x in ORDER_TERMINAL_STATUSES]

    role_placeholders = ", ".join(["%s"] * len(roles))
    status_placeholders = ", ".join(["%s"] * len(terminal_statuses))

    sql = """
        SELECT
            o.local_order_key,
            o.trade_id,
            o.signal_key,
            o.symbol,
            o.side,
            o.order_role,
            o.status,
            o.bybit_order_id,
            o.bybit_order_link_id,
            p.status AS position_status,
            p.exit_reason
        FROM public.trading_orders o
        JOIN public.trading_positions p
            ON p.trade_id = o.trade_id
        WHERE o.trade_id = %s
          AND p.status LIKE 'POSITION_CLOSED%%'
          AND UPPER(o.order_role) IN ({role_placeholders})
          AND UPPER(COALESCE(o.status, '')) NOT IN ({status_placeholders})
          AND NOT EXISTS (
              SELECT 1
              FROM public.trading_fills f
              WHERE f.trade_id = o.trade_id
                AND COALESCE(f.local_order_key, '') = COALESCE(o.local_order_key, '')
                AND UPPER(COALESCE(f.order_role, '')) = UPPER(COALESCE(o.order_role, ''))
                AND COALESCE(f.exec_qty, 0) > 0
          )
        ORDER BY o.created_at ASC
    """.format(
        role_placeholders=role_placeholders,
        status_placeholders=status_placeholders,
    )

    return read_sql(sql, [int(trade_id)] + roles + terminal_statuses)


def cancel_unfilled_protective_orders_after_position_closed(
    client: BybitClient,
    trade_id: int,
) -> Dict[str, object]:
    orders = load_unfilled_protective_orders_for_closed_trade(trade_id)

    result: Dict[str, object] = {
        "trade_id": int(trade_id),
        "checked": 0,
        "cancelled": 0,
        "cancelled_not_found": 0,
        "failed": 0,
        "errors": [],
    }

    if orders.empty:
        return result

    for _, r in orders.iterrows():
        row = r.to_dict()

        local_order_key = str(row.get("local_order_key") or "")
        symbol = str(row.get("symbol") or "").upper()
        role = str(row.get("order_role") or "").upper()

        bybit_order_id = row.get("bybit_order_id")
        bybit_order_link_id = row.get("bybit_order_link_id")

        if pd.isna(bybit_order_id):
            bybit_order_id = None

        if pd.isna(bybit_order_link_id):
            bybit_order_link_id = None

        result["checked"] = int(result["checked"]) + 1

        try:
            resp = client.cancel_order(
                symbol=symbol,
                order_id=str(bybit_order_id) if bybit_order_id else None,
                order_link_id=str(bybit_order_link_id) if bybit_order_link_id else None,
            )

            update_cancelled_order_after_closed_position(
                local_order_key=local_order_key,
                status="CANCELLED",
                response_json={
                    "cancel_reason": "POSITION_CLOSED_CANCEL_UNFILLED_PROTECTIVE_ORDER",
                    "trade_id": int(trade_id),
                    "order_role": role,
                    "cancel_response": resp,
                },
            )

            result["cancelled"] = int(result["cancelled"]) + 1

            print("CLOSED_POSITION_PROTECTIVE_ORDER_CANCELLED")
            print("trade_id:", int(trade_id))
            print("symbol:", symbol)
            print("order_role:", role)
            print("local_order_key:", local_order_key)

        except Exception as e:
            err = str(e)

            if is_bybit_order_not_exists_error(err):
                update_cancelled_order_after_closed_position(
                    local_order_key=local_order_key,
                    status="CANCELLED_NOT_FOUND",
                    response_json={
                        "cancel_reason": "POSITION_CLOSED_PROTECTIVE_ORDER_ALREADY_GONE",
                        "trade_id": int(trade_id),
                        "order_role": role,
                        "error": err,
                    },
                )

                result["cancelled_not_found"] = int(result["cancelled_not_found"]) + 1
                continue

            update_cancelled_order_after_closed_position(
                local_order_key=local_order_key,
                status="CANCEL_FAILED",
                response_json={
                    "cancel_reason": "POSITION_CLOSED_CANCEL_UNFILLED_PROTECTIVE_ORDER",
                    "trade_id": int(trade_id),
                    "order_role": role,
                    "error": err,
                },
            )

            result["failed"] = int(result["failed"]) + 1

            errors = result.get("errors")
            if isinstance(errors, list):
                errors.append(
                    {
                        "local_order_key": local_order_key,
                        "symbol": symbol,
                        "order_role": role,
                        "error": err,
                    }
                )

            print("CLOSED_POSITION_PROTECTIVE_ORDER_CANCEL_FAILED")
            print("trade_id:", int(trade_id))
            print("symbol:", symbol)
            print("order_role:", role)
            print("local_order_key:", local_order_key)
            print("error:", err)

    return result

def is_retryable_bybit_error(error_text: str) -> bool:
    text = str(error_text or "").lower()

    retry_markers = [
        "read timed out",
        "readtimeout",
        "timeout",
        "retryable error occurred",
        "recv_window",
        "10002",
        "temporarily unavailable",
        "connection aborted",
        "connection reset",
        "max retries exceeded",
    ]

    return any(marker in text for marker in retry_markers)


def print_reconcile_order_warning(
    stage: str,
    order_row: Dict[str, object],
    error: Exception,
) -> None:
    err = str(error)

    print("RECONCILE_ORDER_WARNING")
    print("stage:", str(stage))
    print("symbol:", str(order_row.get("symbol") or "").upper())
    print("order_role:", str(order_row.get("order_role") or "").upper())
    print("local_order_key:", str(order_row.get("local_order_key") or ""))
    print("bybit_order_id:", str(order_row.get("bybit_order_id") or ""))
    print("bybit_order_link_id:", str(order_row.get("bybit_order_link_id") or ""))
    print("retryable:", bool(is_retryable_bybit_error(err)))
    print("error:", err[:500])


def reconcile_orders(client: BybitClient, orders: pd.DataFrame) -> Tuple[int, int, int]:
    order_snapshots = 0
    fills_written = 0
    errors_count = 0
    touched_trade_ids = set()

    for _, row in orders.iterrows():
        order_row = row.to_dict()
        symbol = str(order_row.get("symbol") or "").upper()

        if pd.notna(order_row.get("trade_id")):
            touched_trade_ids.add(int(order_row["trade_id"]))

        bybit_order_id = order_row.get("bybit_order_id")
        bybit_order_link_id = order_row.get("bybit_order_link_id")

        if pd.isna(bybit_order_id):
            bybit_order_id = None
        if pd.isna(bybit_order_link_id):
            bybit_order_link_id = None

        try:
            order_resp = client.get_order_history(
                symbol=symbol,
                order_id=str(bybit_order_id) if bybit_order_id else None,
                order_link_id=str(bybit_order_link_id) if bybit_order_link_id else None,
            )

            update_order_snapshot(str(order_row["local_order_key"]), order_resp)
            order_snapshots += 1

        except Exception as e:
            errors_count += 1
            print_reconcile_order_warning(
                stage="get_order_history",
                order_row=order_row,
                error=e,
            )

        try:
            exec_resp = client.get_executions(
                symbol=symbol,
                order_id=str(bybit_order_id) if bybit_order_id else None,
                order_link_id=str(bybit_order_link_id) if bybit_order_link_id else None,
            )

            exec_rows = extract_execution_rows(exec_resp)
            fills_written += insert_fills_for_order(order_row, exec_rows)
            update_order_status_from_execution(
                local_order_key=str(order_row["local_order_key"]),
                exec_rows=exec_rows,
            )

        except Exception as e:
            errors_count += 1
            print_reconcile_order_warning(
                stage="get_executions",
                order_row=order_row,
                error=e,
            )

    for trade_id in sorted(touched_trade_ids):
        update_order_statuses_from_fills(trade_id)
        update_position_from_fills(trade_id)

    return order_snapshots, fills_written, errors_count


def main() -> None:
    ensure_reconcile_tables()
    owner = "reconcile:{}".format(pd.Timestamp.now(tz="UTC"))

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=600):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        orders = load_recent_orders()

        if orders.empty:
            print("NO_RECENT_ORDERS")
            return

        if DRY_RUN:
            print("DRY_RUN: reconcile does not call Bybit")
            print(orders[["local_order_key", "symbol", "order_role", "status"]].head(30).to_string(index=False))
            return

        client = BybitClient()

        exchange_positions = client.get_open_positions()
        snapshot_n = write_exchange_positions_snapshot(exchange_positions)

        order_snapshots, fills_written, errors_count = reconcile_orders(client, orders)

        print("EXCHANGE_OPEN_POSITIONS:", len(exchange_positions))
        print("EXCHANGE_POSITION_SNAPSHOTS_WRITTEN:", snapshot_n)
        print("RECONCILED_ORDERS:", order_snapshots)
        print("FILLS_WRITTEN_OR_UPDATED:", fills_written)
        print("RECONCILE_SOFT_ERRORS:", errors_count)

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
