from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional

import pandas as pd

from online.trading import config
from online.trading import audit_log
from online.trading import notify
from online.trading.bybit_client import BybitClient
from online.trading.db import db_cursor, json_default, read_sql
from online.trading.locks import acquire_lock, release_lock
from online.trading.risk import calc_order_qty, calc_tp_sl_prices
from online.trading.state import can_open_new_position


LOCK_NAME = "trading_execution"

def env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)

    value = str(raw).strip().lower()
    if value in {"0", "false", "no", "off", "n"}:
        return False
    if value in {"1", "true", "yes", "on", "y"}:
        return True

    return bool(default)


DRY_RUN = env_bool("IMB_TRADING_DRY_RUN", True)
ACTIVE_LIVE_POSITION_STATUSES = [
    "CREATED",
    "ENTRY_ORDER_SENT",
    "TP_SL_PLACED",
    "POSITION_OPEN",
]
def resolve_trade_capital_usdt(available_usdt: float) -> float:
    available = float(available_usdt)

    if available <= 0:
        return 0.0

    chulan_enabled = bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0))

    if not chulan_enabled:
        return available

    base_capital = float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0)

    if base_capital <= 0:
        raise RuntimeError(
            "CHULAN_BASE_CAPITAL_USDT must be > 0 when CHULAN_ENABLED=1"
        )

    return min(available, base_capital)

def build_order_link_id(signal_key: str, role: str) -> str:
    digest = hashlib.sha1(str(signal_key).encode("utf-8")).hexdigest()[:24]
    return "imb-{}-{}".format(digest, str(role).lower())

def load_latest_selected_signal() -> Optional[Dict[str, object]]:
    sql = """
        SELECT *
        FROM public.trading_signals
        WHERE selected = TRUE
          AND rejected = FALSE
        ORDER BY signal_ts DESC
        LIMIT 1
    """
    df = read_sql(sql)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def signal_already_has_position(signal_key: str) -> bool:
    sql = """
        SELECT 1
        FROM public.trading_positions
        WHERE signal_key = %s
          AND status NOT IN (
              'ENTRY_FAILED',
              'ENTRY_REJECTED',
              'CANCELLED',
              'FAILED'
          )
        LIMIT 1
    """
    df = read_sql(sql, [signal_key])
    return not df.empty


def reject_if_signal_is_stale(signal: Dict[str, object]) -> bool:
    entry_ts_plan = pd.to_datetime(signal.get("entry_ts_plan"), utc=True, errors="coerce")
    now_ts = pd.Timestamp.now(tz="UTC")

    if pd.isna(entry_ts_plan):
        raise RuntimeError("bad entry_ts_plan in selected signal: {}".format(signal.get("entry_ts_plan")))

    max_age_seconds = int(getattr(config, "MAX_SIGNAL_AGE_SECONDS", 900))
    deadline_ts = entry_ts_plan + pd.Timedelta(seconds=max_age_seconds)

    if now_ts <= deadline_ts:
        return False

    signal_key = str(signal.get("signal_key") or "")
    symbol = str(signal.get("symbol") or "").upper()
    side = str(signal.get("side") or "").upper()

    message = (
        "Selected signal is stale; execution blocked. "
        "entry_ts_plan={}, now_utc={}, max_age_seconds={}"
    ).format(entry_ts_plan, now_ts, max_age_seconds)

    audit_log.log_audit_event(
        event_type="STALE_SIGNAL_BLOCKED",
        status="SKIP_EXECUTION",
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        message=message,
        payload={
            "entry_ts_plan": str(entry_ts_plan),
            "now_utc": str(now_ts),
            "max_age_seconds": max_age_seconds,
            "deadline_ts": str(deadline_ts),
        },
    )

    notify.notify_event(
        event_type="STALE_SIGNAL_BLOCKED",
        status="SKIP_EXECUTION",
        symbol=symbol,
        side=side,
        signal_key=signal_key,
        payload={
            "entry_ts_plan": str(entry_ts_plan),
            "now_utc": str(now_ts),
            "max_age_seconds": max_age_seconds,
            "deadline_ts": str(deadline_ts),
        },
        force=True,
    )

    print("STALE_SIGNAL_BLOCKED")
    print("signal_key:", signal_key)
    print("symbol:", symbol)
    print("side:", side)
    print("entry_ts_plan:", entry_ts_plan)
    print("now_utc:", now_ts)
    print("max_age_seconds:", max_age_seconds)

    return True


def insert_position_stub(signal: Dict[str, object], status: str) -> int:
    sql = """
        INSERT INTO public.trading_positions (
            signal_key,
            symbol,
            side,
            status,
            h4_close,
            atr14,
            tp_px_plan,
            sl_px_plan,
            ttl_close_ts,
            created_at,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, NOW(), NOW())
        ON CONFLICT (signal_key)
        DO UPDATE SET
            status = EXCLUDED.status,
            updated_at = NOW()
        RETURNING trade_id
    """

    signal_ts = pd.to_datetime(signal["signal_ts"], utc=True)
    ttl_close_ts = signal_ts + pd.Timedelta(hours=config.TTL_HOURS + 4) + pd.Timedelta(seconds=config.ENTRY_DELAY_SECONDS)

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(signal["signal_key"]),
                str(signal["symbol"]).upper(),
                str(signal["side"]).upper(),
                status,
                float(signal.get("h4_close") or 0.0),
                float(signal.get("atr14") or 0.0),
                ttl_close_ts.to_pydatetime(),
            ),
        )
        row = cur.fetchone()

    return int(row[0])


def insert_order(
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
    bybit_order_id: Optional[str] = None,
    bybit_order_link_id: Optional[str] = None,
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
            %s, %s,
            %s, %s, %s,
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
                side,
                order_role,
                bybit_order_id,
                bybit_order_link_id,
                order_type,
                order_role in ["TAKE_PROFIT", "STOP_LOSS", "TTL_CLOSE", "EMERGENCY_CLOSE", "MANUAL_CLOSE"],
                qty_plan,
                price_plan,
                trigger_price_plan,
                status,
                json.dumps(request_json or {}, ensure_ascii=False, default=json_default),
                json.dumps(response_json or {}, ensure_ascii=False, default=json_default),
            ),
        )


def compact_exchange_positions(positions):
    out = []
    for p in positions:
        item = {
            "symbol": str(p.get("symbol") or "").upper(),
            "side": p.get("side"),
            "size": p.get("size"),
            "avgPrice": p.get("avgPrice"),
            "markPrice": p.get("markPrice"),
            "unrealisedPnl": p.get("unrealisedPnl"),
            "positionValue": p.get("positionValue"),
            "updatedTime": p.get("updatedTime"),
        }
        out.append(item)
    return out


def load_active_live_db_positions() -> pd.DataFrame:
    sql = """
        SELECT
            trade_id,
            signal_key,
            symbol,
            side,
            status,
            qty,
            entry_px_plan,
            entry_avg_px,
            created_at,
            updated_at
        FROM public.trading_positions
        WHERE status = ANY(%s)
        ORDER BY updated_at DESC
    """

    return read_sql(sql, [ACTIVE_LIVE_POSITION_STATUSES])


def mark_db_position_as_exchange_empty(row: pd.Series) -> None:
    trade_id = int(row["trade_id"])
    old_status = str(row.get("status") or "").upper()

    if old_status in {"CREATED", "ENTRY_ORDER_SENT"}:
        new_status = "ENTRY_FAILED"
        exit_reason = "NO_EXCHANGE_POSITION_AFTER_ENTRY_ATTEMPT"
    else:
        new_status = "POSITION_CLOSED_EXTERNAL"
        exit_reason = "EXCHANGE_POSITION_NOT_FOUND"

    sql = """
        UPDATE public.trading_positions
        SET
            status = %s,
            exit_reason = COALESCE(exit_reason, %s),
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                new_status,
                exit_reason,
                trade_id,
            ),
        )

    audit_log.log_audit_event(
        event_type="DB_POSITION_SYNCED_WITH_EXCHANGE_EMPTY",
        trade_id=trade_id,
        signal_key=str(row.get("signal_key") or ""),
        symbol=str(row.get("symbol") or "").upper(),
        side=str(row.get("side") or "").upper(),
        status="SYNCED",
        message="Bybit has no open position; local DB active position was marked inactive",
        payload={
            "trade_id": trade_id,
            "signal_key": str(row.get("signal_key") or ""),
            "symbol": str(row.get("symbol") or "").upper(),
            "side": str(row.get("side") or "").upper(),
            "old_status": old_status,
            "new_status": new_status,
            "exit_reason": exit_reason,
            "qty": None if pd.isna(row.get("qty")) else float(row.get("qty")),
            "entry_px_plan": None if pd.isna(row.get("entry_px_plan")) else float(row.get("entry_px_plan")),
            "entry_avg_px": None if pd.isna(row.get("entry_avg_px")) else float(row.get("entry_avg_px")),
        },
    )

    print("DB_POSITION_SYNCED_WITH_EXCHANGE_EMPTY")
    print("trade_id:", trade_id)
    print("signal_key:", str(row.get("signal_key") or ""))
    print("symbol:", str(row.get("symbol") or "").upper())
    print("side:", str(row.get("side") or "").upper())
    print("old_status:", old_status)
    print("new_status:", new_status)


def sync_db_positions_with_exchange_before_entry(client: BybitClient) -> bool:
    positions = client.get_open_positions()

    if positions:
        compact = compact_exchange_positions(positions)

        audit_log.log_audit_event(
            event_type="EXCHANGE_POSITION_EXISTS_BEFORE_ENTRY",
            status="SKIP_EXECUTION",
            message="Bybit has open position; new entry is blocked",
            payload={
                "open_positions": compact,
            },
        )

        notify.notify_event(
            event_type="EXCHANGE_POSITION_EXISTS_BEFORE_ENTRY",
            status="SKIP_EXECUTION",
            symbol=str(compact[0].get("symbol") or "") if compact else None,
            side=str(compact[0].get("side") or "") if compact else None,
            payload={
                "open_positions": compact,
            },
            force=True,
        )

        print("EXCHANGE_POSITION_EXISTS: skip execution")
        print(compact)
        return True

    db_active = load_active_live_db_positions()

    if db_active.empty:
        return False

    print("EXCHANGE_EMPTY_BUT_DB_HAS_ACTIVE_POSITIONS")
    print("db_active_count:", len(db_active))

    for _, row in db_active.iterrows():
        mark_db_position_as_exchange_empty(row)

    return False


def ensure_execution_position_columns() -> None:
    sql = """
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
            ADD COLUMN IF NOT EXISTS fee_usd DOUBLE PRECISION;
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)



def update_position_after_entry_plan(
    trade_id: int,
    qty: float,
    entry_px_plan: float,
    entry_avg_px: Optional[float],
    tp_px: float,
    sl_px: float,
) -> None:
    if entry_avg_px is None:
        entry_avg_px_value = float(entry_px_plan) if DRY_RUN else None
    else:
        entry_avg_px_value = float(entry_avg_px)

    entry_slippage_abs = None
    entry_slippage_pct = None

    if entry_avg_px_value is not None and float(entry_px_plan) > 0:
        entry_slippage_abs = float(entry_avg_px_value) - float(entry_px_plan)
        entry_slippage_pct = float(entry_slippage_abs) / float(entry_px_plan)

    sql = """
        UPDATE public.trading_positions
        SET
            qty = %s,
            entry_px_plan = %s,
            entry_avg_px = %s,
            entry_slippage_abs = %s,
            entry_slippage_pct = %s,
            tp_px_plan = %s,
            sl_px_plan = %s,
            status = %s,
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                float(qty),
                float(entry_px_plan),
                entry_avg_px_value,
                entry_slippage_abs,
                entry_slippage_pct,
                float(tp_px),
                float(sl_px),
                "ENTRY_ORDER_SENT" if not DRY_RUN else "DRY_RUN_ENTRY_PLANNED",
                int(trade_id),
            ),
        )

def mark_position_entry_failed(
    trade_id: int,
    error_message: str,
) -> None:
    sql = """
        UPDATE public.trading_positions
        SET
            status = 'ENTRY_FAILED',
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))

    audit_log.log_audit_event(
        event_type="ENTRY_ORDER_FAILED",
        trade_id=int(trade_id),
        status="ERROR",
        message=str(error_message),
        payload={
            "trade_id": int(trade_id),
            "error": str(error_message),
        },
    )

def mark_position_tp_sl_failed(
    trade_id: int,
    error_message: str,
) -> None:
    sql = """
        UPDATE public.trading_positions
        SET
            status = 'TP_SL_FAILED',
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (int(trade_id),))

    audit_log.log_audit_event(
        event_type="TP_SL_ORDER_FAILED",
        trade_id=int(trade_id),
        status="ERROR",
        message=str(error_message),
        payload={
            "trade_id": int(trade_id),
            "error": str(error_message),
        },
    )
def wait_entry_avg_price_by_link_id(
    client: BybitClient,
    symbol: str,
    order_link_id: str,
    fallback_price: float,
) -> float:
    timeout_seconds = float(getattr(config, "ENTRY_FILL_WAIT_TIMEOUT_SECONDS", 15.0))
    sleep_seconds = float(getattr(config, "ENTRY_FILL_WAIT_SLEEP_SECONDS", 0.5))

    deadline = time.time() + timeout_seconds

    while time.time() <= deadline:
        avg_px = client.get_avg_fill_price_by_link_id(
            symbol=symbol,
            order_link_id=order_link_id,
        )

        if avg_px is not None and float(avg_px) > 0:
            return float(avg_px)

        time.sleep(sleep_seconds)

    if float(fallback_price) <= 0:
        raise RuntimeError(
            "entry avg fill price was not found and fallback_price is bad: {}".format(
                fallback_price
            )
        )
    print("ENTRY_AVG_PRICE_FALLBACK_USED")
    print("symbol:", symbol)
    print("order_link_id:", order_link_id)
    print("fallback_price:", fallback_price)
    return float(fallback_price)

def main() -> None:
    ensure_execution_position_columns()
    audit_log.ensure_audit_tables()
    audit_log.log_audit_event(event_type="EXECUTION_MAIN_START", status="STARTED", message="execution.py started")
    owner = "execution:{}".format(pd.Timestamp.now(tz="UTC"))

    if not acquire_lock(LOCK_NAME, owner=owner, ttl_seconds=600):
        print("LOCK_BUSY:", LOCK_NAME)
        return

    try:
        print("DRY_RUN:", DRY_RUN)

        client = BybitClient()

        if DRY_RUN:
            if not can_open_new_position():
                print("ACTIVE_POSITION_EXISTS_DB_DRY_RUN: skip execution")
                return
        else:
            if sync_db_positions_with_exchange_before_entry(client):
                return

        signal = load_latest_selected_signal()
        if signal is None:
            print("NO_SELECTED_SIGNAL")
            return

        signal_key = str(signal["signal_key"])
        symbol = str(signal["symbol"]).upper()
        side = str(signal["side"]).upper()

        if reject_if_signal_is_stale(signal):
            return

        if signal_already_has_position(signal_key):
            print("SIGNAL_ALREADY_HAS_POSITION:", signal_key)
            return

        if DRY_RUN:
            available_usdt = float(os.environ.get("IMB_DRY_RUN_AVAILABLE_USDT", "100"))
            entry_px_plan = client.get_ticker_last_price(symbol) if os.environ.get("IMB_DRY_RUN_USE_BYBIT_PRICE", "0") == "1" else float(signal["h4_close"])

            atr14_check = float(signal["atr14"])
            if entry_px_plan <= 0:
                raise RuntimeError("bad entry_px_plan: %s" % entry_px_plan)
            if atr14_check <= 0:
                raise RuntimeError("bad atr14: %s" % atr14_check)
            instrument = {
                "qty_step": float(os.environ.get("IMB_DRY_RUN_QTY_STEP", "0.001")),
                "min_order_qty": float(os.environ.get("IMB_DRY_RUN_MIN_QTY", "0.001")),
                "min_notional": float(os.environ.get("IMB_DRY_RUN_MIN_NOTIONAL", "5")),
                "tick_size": float(os.environ.get("IMB_DRY_RUN_TICK_SIZE", "0.0001")),
            }
        else:
            available_usdt = client.get_wallet_balance_usdt()
            entry_px_plan = client.get_ticker_last_price(symbol)
            instrument = client.get_instrument_info(symbol)

        trade_capital_usdt = resolve_trade_capital_usdt(available_usdt)

        qty_info = calc_order_qty(
            available_usdt=trade_capital_usdt,
            entry_price=entry_px_plan,
            qty_step=float(instrument["qty_step"]),
            min_order_qty=float(instrument["min_order_qty"]),
            min_notional=float(instrument["min_notional"]),
            use_balance_pct=config.USE_AVAILABLE_BALANCE_PCT,
        )

        if not bool(qty_info["ok"]):
            print("RISK_REJECT:", qty_info)
            return

        atr14 = float(signal.get("atr14") or 0.0)
        if atr14 <= 0:
            raise RuntimeError("atr14 missing in selected signal. Need selector to carry atr14 from online features.")

        qty = float(qty_info["qty"])

        tp_px = 0.0
        sl_px = 0.0

        trade_id = insert_position_stub(signal=signal, status="DRY_RUN_CREATED" if DRY_RUN else "CREATED")

        entry_link_id = build_order_link_id(signal_key, "entry")
        tp_link_id = build_order_link_id(signal_key, "tp")
        sl_link_id = build_order_link_id(signal_key, "sl")

        entry_request = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_link_id": entry_link_id,
            "reduce_only": False,
            "available_usdt": available_usdt,
            "trade_capital_usdt": trade_capital_usdt,
            "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
            "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
        }

        if DRY_RUN:
            entry_response = {
                "dry_run": True,
                "message": "market entry not sent",
                "request": entry_request,
            }
            bybit_order_id = None

        else:
            try:
                entry_response = client.place_market_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_link_id=entry_link_id,
                    reduce_only=False,
                )
                bybit_order_id = str((entry_response.get("result") or {}).get("orderId") or "")

            except Exception as e:
                error_message = str(e)

                mark_position_entry_failed(
                    trade_id=trade_id,
                    error_message=error_message,
                )

                insert_order(
                    trade_id=trade_id,
                    signal_key=signal_key,
                    symbol=symbol,
                    side=side,
                    order_role="ENTRY_MARKET",
                    local_order_key=entry_link_id,
                    order_type="Market",
                    qty_plan=qty,
                    price_plan=None,
                    trigger_price_plan=None,
                    status="ENTRY_FAILED",
                    request_json=entry_request,
                    response_json={
                        "error": error_message,
                    },
                    bybit_order_id=None,
                    bybit_order_link_id=entry_link_id,
                )

                notify.notify_event(
                    event_type="ENTRY_ORDER_FAILED",
                    status="ERROR",
                    symbol=symbol,
                    side=side,
                    signal_key=signal_key,
                    trade_id=int(trade_id),
                    payload={
                        "error": error_message,
                        "available_usdt": available_usdt,
                        "trade_capital_usdt": trade_capital_usdt,
                        "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
                        "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
                        "entry_px_plan": entry_px_plan,
                        "qty": qty,
                        "tp_px": tp_px,
                        "sl_px": sl_px,
                    },
                    force=True,
                )

                print("ENTRY_ORDER_FAILED")
                print("trade_id:", trade_id)
                print("signal_key:", signal_key)
                print("symbol:", symbol)
                print("side:", side)
                print("available_usdt:", available_usdt)
                print("trade_capital_usdt:", trade_capital_usdt)
                print("chulan_enabled:", bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)))
                print("chulan_base_capital_usdt:", float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0))
                print("qty:", qty)
                print("error:", error_message)

                return

        insert_order(
            trade_id=trade_id,
            signal_key=signal_key,
            symbol=symbol,
            side=side,
            order_role="ENTRY_MARKET",
            local_order_key=entry_link_id,
            order_type="Market",
            qty_plan=qty,
            price_plan=None,
            trigger_price_plan=None,
            status="DRY_RUN_NOT_SENT" if DRY_RUN else "SENT",
            request_json=entry_request,
            response_json=entry_response,
            bybit_order_id=bybit_order_id,
            bybit_order_link_id=entry_link_id,
        )
        if DRY_RUN:
            entry_avg_px_actual = float(entry_px_plan)
        else:
            entry_avg_px_actual = wait_entry_avg_price_by_link_id(
                client=client,
                symbol=symbol,
                order_link_id=entry_link_id,
                fallback_price=entry_px_plan,
            )

        tp_sl = calc_tp_sl_prices(
            side=side,
            entry_price=entry_avg_px_actual,
            atr14=atr14,
            tp_atr=config.TP_ATR,
            sl_atr=config.SL_ATR,
        )

        tp_px = float(tp_sl["tp_px"])
        sl_px = float(tp_sl["sl_px"])

        update_position_after_entry_plan(
            trade_id=trade_id,
            qty=qty,
            entry_px_plan=entry_px_plan,
            entry_avg_px=entry_avg_px_actual,
            tp_px=tp_px,
            sl_px=sl_px,
        )

        if DRY_RUN:
            audit_log.log_audit_event(
                event_type="DRY_RUN_ENTRY_PLANNED",
                trade_id=int(trade_id),
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                status="DRY_RUN_ENTRY_PLANNED",
                message="Dry-run entry was planned and written to local trading tables",
                payload={
                    "available_usdt": available_usdt,
                    "trade_capital_usdt": trade_capital_usdt,
                    "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
                    "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
                    "entry_px_actual": entry_avg_px_actual,
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                    "dry_run": True,
                },
            )

            audit_log.log_order_event(
                event_type="DRY_RUN_ORDER_NOT_SENT",
                order_role="ENTRY_MARKET",
                trade_id=int(trade_id),
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                bybit_status="DRY_RUN_NOT_SENT",
                request_payload={
                    "order_type": "Market",
                    "available_usdt": available_usdt,
                    "trade_capital_usdt": trade_capital_usdt,
                    "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
                    "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
                    "entry_px_plan": entry_px_plan,
                    "entry_avg_px": entry_avg_px_actual,
                    "entry_slippage_pct": (
                        (float(entry_avg_px_actual) - float(entry_px_plan)) / float(entry_px_plan)
                        if float(entry_px_plan) > 0 else None
                    ),
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                },
                response_payload={
                    "dry_run": True,
                    "sent_to_bybit": False,
                },
            )

            print("DRY_RUN_ENTRY_PLANNED")
            print("trade_id:", trade_id)
            print("signal_key:", signal_key)
            print("symbol:", symbol)
            print("side:", side)
            print("available_usdt:", available_usdt)
            print("trade_capital_usdt:", trade_capital_usdt)
            print("entry_avg_px_actual:", entry_avg_px_actual)
            print("chulan_enabled:", bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)))
            print("chulan_base_capital_usdt:", float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0))
            print("entry_px_plan:", entry_px_plan)
            print("entry_px_plan:", entry_px_plan)
            print("entry_avg_px_actual:", entry_avg_px_actual)
            print(
                "entry_slippage_pct:",
                (
                    (float(entry_avg_px_actual) - float(entry_px_plan)) / float(entry_px_plan)
                    if float(entry_px_plan) > 0 else None
                ),
            )
            print("qty:", qty)
            print("tp_px:", tp_px)
            print("sl_px:", sl_px)

            notify.notify_event(
                event_type="DRY_RUN_ENTRY_PLANNED",
                status="OK",
                symbol=symbol,
                side=side,
                signal_key=signal_key,
                trade_id=int(trade_id),
                payload={
                    "available_usdt": available_usdt,
                    "trade_capital_usdt": trade_capital_usdt,
                    "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
                    "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
                    "entry_px_plan": entry_px_plan,
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                    "gate2": signal.get("gate2_p"),
                    "gate4": signal.get("gate4_p"),
                    "gate5_1": signal.get("gate5_1_proba"),
                    "gate5_3": signal.get("gate5_3_proba"),
                },
                force=True,
            )
            return

        try:
            tp_sl_response = client.place_tp_sl_orders(
                symbol=symbol,
                side=side,
                qty=qty,
                tp_px=tp_px,
                sl_px=sl_px,
                tp_order_link_id=tp_link_id,
                sl_order_link_id=sl_link_id,
            )

            insert_order(
                trade_id=trade_id,
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                order_role="TAKE_PROFIT",
                local_order_key=tp_link_id,
                order_type="MarketTrigger",
                qty_plan=qty,
                price_plan=None,
                trigger_price_plan=tp_px,
                status="SENT",
                request_json={"symbol": symbol, "side": side, "qty": qty, "tp_px": tp_px},
                response_json=tp_sl_response.get("take_profit"),
                bybit_order_id=str(((tp_sl_response.get("take_profit") or {}).get("result") or {}).get("orderId") or ""),
                bybit_order_link_id=tp_link_id,
            )

            insert_order(
                trade_id=trade_id,
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                order_role="STOP_LOSS",
                local_order_key=sl_link_id,
                order_type="MarketTrigger",
                qty_plan=qty,
                price_plan=None,
                trigger_price_plan=sl_px,
                status="SENT",
                request_json={"symbol": symbol, "side": side, "qty": qty, "sl_px": sl_px},
                response_json=tp_sl_response.get("stop_loss"),
                bybit_order_id=str(((tp_sl_response.get("stop_loss") or {}).get("result") or {}).get("orderId") or ""),
                bybit_order_link_id=sl_link_id,
            )

        except Exception as e:
            error_message = str(e)

            mark_position_tp_sl_failed(
                trade_id=trade_id,
                error_message=error_message,
            )

            insert_order(
                trade_id=trade_id,
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                order_role="TAKE_PROFIT",
                local_order_key=tp_link_id,
                order_type="MarketTrigger",
                qty_plan=qty,
                price_plan=None,
                trigger_price_plan=tp_px,
                status="TP_SL_FAILED",
                request_json={"symbol": symbol, "side": side, "qty": qty, "tp_px": tp_px},
                response_json={"error": error_message},
                bybit_order_id=None,
                bybit_order_link_id=tp_link_id,
            )

            insert_order(
                trade_id=trade_id,
                signal_key=signal_key,
                symbol=symbol,
                side=side,
                order_role="STOP_LOSS",
                local_order_key=sl_link_id,
                order_type="MarketTrigger",
                qty_plan=qty,
                price_plan=None,
                trigger_price_plan=sl_px,
                status="TP_SL_FAILED",
                request_json={"symbol": symbol, "side": side, "qty": qty, "sl_px": sl_px},
                response_json={"error": error_message},
                bybit_order_id=None,
                bybit_order_link_id=sl_link_id,
            )

            notify.notify_event(
                event_type="TP_SL_ORDER_FAILED",
                status="ERROR",
                symbol=symbol,
                side=side,
                signal_key=signal_key,
                trade_id=int(trade_id),
                payload={
                    "error": error_message,
                    "available_usdt": available_usdt,
                    "trade_capital_usdt": trade_capital_usdt,
                    "entry_px_plan": entry_px_plan,
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                    "entry_order_link_id": entry_link_id,
                    "tp_order_link_id": tp_link_id,
                    "sl_order_link_id": sl_link_id,
                },
                force=True,
            )

            print("TP_SL_ORDER_FAILED")
            print("trade_id:", trade_id)
            print("signal_key:", signal_key)
            print("symbol:", symbol)
            print("side:", side)
            print("qty:", qty)
            print("tp_px:", tp_px)
            print("sl_px:", sl_px)
            print("error:", error_message)

            return

        with db_cursor(commit=True) as (_, cur):
            cur.execute(
                """
                UPDATE public.trading_positions
                SET status = 'TP_SL_PLACED',
                    updated_at = NOW()
                WHERE trade_id = %s
                """,
                (trade_id,),
            )

        print("LIVE_ORDER_SENT")
        print("trade_id:", trade_id)
        print("symbol:", symbol)
        print("side:", side)
        print("available_usdt:", available_usdt)
        print("trade_capital_usdt:", trade_capital_usdt)
        print("chulan_enabled:", bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)))
        print("chulan_base_capital_usdt:", float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0))
        print("qty:", qty)
        print("tp_px:", tp_px)
        print("sl_px:", sl_px)

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
