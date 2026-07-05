from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from online.trading import config
from online.trading import audit_log
from online.trading import notify
from online.trading.bybit_client import BybitClient, format_bybit_decimal
from online.trading.db import db_cursor, json_default, read_sql
from online.trading.locks import acquire_lock, release_lock
from online.trading.risk import calc_order_qty, calc_tp_sl_prices, calc_risk_capped_position_notional
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

def resolve_trading_leverage() -> float:
    leverage = float(getattr(config, "TRADING_LEVERAGE", 1.0) or 1.0)

    if leverage <= 0.0:
        raise RuntimeError("TRADING_LEVERAGE must be > 0")

    return float(leverage)



def resolve_position_notional_multiplier() -> float:
    multiplier = float(getattr(config, "POSITION_NOTIONAL_MULTIPLIER", 1.0) or 1.0)

    if multiplier <= 0.0:
        raise RuntimeError("POSITION_NOTIONAL_MULTIPLIER must be > 0")

    return float(multiplier)


def build_position_sizing_plan(
    trade_capital_usdt: float,
    trading_leverage: float,
    position_notional_multiplier: float,
) -> Dict[str, float]:
    capital = float(trade_capital_usdt)
    leverage = float(trading_leverage)
    multiplier = float(position_notional_multiplier)

    if capital <= 0.0:
        return {
            "position_notional_usdt": 0.0,
            "estimated_initial_margin_usdt": 0.0,
        }

    if leverage <= 0.0:
        raise RuntimeError("TRADING_LEVERAGE must be > 0")

    if multiplier <= 0.0:
        raise RuntimeError("POSITION_NOTIONAL_MULTIPLIER must be > 0")

    position_notional_usdt = capital * multiplier
    estimated_initial_margin_usdt = position_notional_usdt / leverage

    if estimated_initial_margin_usdt > capital + 1e-9:
        raise RuntimeError(
            "POSITION_MARGIN_LIMIT_EXCEEDED: "
            "trade_capital_usdt={:.8f}, position_notional_usdt={:.8f}, "
            "trading_leverage={:.8f}, estimated_initial_margin_usdt={:.8f}. "
            "Need POSITION_NOTIONAL_MULTIPLIER <= TRADING_LEVERAGE."
            .format(
                capital,
                position_notional_usdt,
                leverage,
                estimated_initial_margin_usdt,
            )
        )

    return {
        "position_notional_usdt": float(position_notional_usdt),
        "estimated_initial_margin_usdt": float(estimated_initial_margin_usdt),
    }


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


def assert_trading_positions_schema_ready_before_entry() -> None:
    required_columns = {
        "qty",
        "entry_px_plan",
        "entry_avg_px",
        "entry_slippage_abs",
        "entry_slippage_pct",
        "tp_px_plan",
        "sl_px_plan",
        "trading_leverage",
        "position_notional_multiplier",
        "position_notional_usdt_plan",
        "estimated_initial_margin_usdt",
        "position_risk_cap_enabled",
        "position_risk_cap_applied",
        "position_risk_fraction",
        "raw_position_notional_usdt_plan",
        "risk_capped_position_notional_usdt_plan",
        "max_full_sl_capital_risk",
        "full_main_sl_capital_risk_abs",
        "status",
        "updated_at",
    }

    df = read_sql("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'trading_positions'
    """)

    existing_columns = set(df["column_name"].astype(str).tolist())
    missing_columns = sorted(required_columns - existing_columns)

    if missing_columns:
        raise RuntimeError(
            "TRADING_POSITIONS_SCHEMA_NOT_READY_BEFORE_ENTRY: missing columns: {}".format(
                ",".join(missing_columns)
            )
        )


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

def get_conditional_side_rule_for_execution(symbol: str, side: str) -> Optional[Dict[str, object]]:
    if not bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True)):
        return None

    rules = getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST", {}) or {}
    symbol_rules = rules.get(str(symbol or "").upper())

    if not isinstance(symbol_rules, dict):
        return None

    side_rule = symbol_rules.get(str(side or "").upper())

    if not isinstance(side_rule, dict):
        return None

    return side_rule


def is_signal_allowed_by_prod_side_rules(signal: Dict[str, object]) -> Tuple[bool, str]:
    symbol = str(signal.get("symbol") or "").upper()
    side = str(signal.get("side") or "").upper()

    side_rules = getattr(config, "SIDE_AWARE_WHITELIST", {}) or {}

    if side_rules and config.is_allowed_by_side_rules(symbol, side, side_rules):
        return True, "CURRENT_WHITELIST"

    conditional_rule = get_conditional_side_rule_for_execution(symbol, side)

    if conditional_rule is None:
        return False, "NO_WHITELIST"

    admission_source = str(signal.get("admission_source") or "").upper()
    if admission_source == "CONDITIONAL_WHITELIST":
        return True, "CONDITIONAL_WHITELIST"

    min_margin = float(conditional_rule.get("min_gate2_side_margin", 0.0) or 0.0)
    margin = signal.get("gate2_side_margin")

    if margin is None or pd.isna(margin):
        return False, "MISSING_GATE2_SIDE_MARGIN"

    if float(margin) < min_margin:
        return False, "BELOW_CONDITIONAL_GATE2_MARGIN"

    return True, "CONDITIONAL_WHITELIST"


def reject_if_signal_not_in_side_whitelist(signal: Dict[str, object]) -> bool:
    side_rules = getattr(config, "SIDE_AWARE_WHITELIST", {}) or {}
    conditional_rules = (
        getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST", {}) or {}
        if bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True))
        else {}
    )

    if not side_rules and not conditional_rules:
        return False

    signal_key = str(signal.get("signal_key") or "")
    symbol = str(signal.get("symbol") or "").upper()
    side = str(signal.get("side") or "").upper()

    allowed, reason = is_signal_allowed_by_prod_side_rules(signal)

    if allowed:
        print("SIDE_ADMISSION_OK")
        print("signal_key:", signal_key)
        print("symbol:", symbol)
        print("side:", side)
        print("reason:", reason)
        return False

    message = "Selected signal is not allowed by side whitelist rules; execution blocked."

    audit_log.log_audit_event(
        event_type="SIDE_AWARE_WHITELIST_BLOCKED",
        status="SKIP_EXECUTION",
        signal_key=signal_key,
        symbol=symbol,
        side=side,
        message=message,
        payload={
            "symbol": symbol,
            "side": side,
            "pair_model_name": signal.get("pair_model_name"),
            "grid_name": signal.get("grid_name"),
            "gate2_proba": signal.get("gate2_proba"),
            "gate2_side_margin": signal.get("gate2_side_margin"),
            "gate4_confidence": signal.get("gate4_confidence"),
            "gate5_1_proba": signal.get("gate5_1_proba"),
            "gate5_3_proba": signal.get("gate5_3_proba"),
            "admission_source": signal.get("admission_source"),
            "side_aware_whitelist": side_rules,
            "conditional_side_aware_whitelist_enabled": bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True)),
            "conditional_side_aware_whitelist": conditional_rules,
            "reason": reason,
        },
    )

    notify.notify_event(
        event_type="SIDE_AWARE_WHITELIST_BLOCKED",
        status="SKIP_EXECUTION",
        symbol=symbol,
        side=side,
        signal_key=signal_key,
        payload={
            "symbol": symbol,
            "side": side,
            "reason": reason,
        },
        force=True,
    )

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            UPDATE public.trading_signals
            SET
                selected = FALSE,
                rejected = TRUE,
                reject_reason = COALESCE(reject_reason, %s),
                skipped_reason = %s,
                dynamic_symbol_allowed = FALSE,
                dynamic_symbol_reason = %s,
                updated_at = NOW()
            WHERE signal_key = %s
            """,
            (
                reason,
                reason,
                reason,
                signal_key,
            ),
        )

    print("SIDE_AWARE_WHITELIST_BLOCKED")
    print("signal_key:", signal_key)
    print("symbol:", symbol)
    print("side:", side)
    print("reason:", reason)

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
                order_role in [
                    "TAKE_PROFIT",
                    "PARTIAL_TP",
                    "FINAL_TP",
                    "STOP_LOSS",
                    "EARLY_STOP",
                    "REST_STOP_AFTER_PARTIAL",
                    "TTL_CLOSE",
                    "EMERGENCY_CLOSE",
                    "MANUAL_CLOSE",
                ],
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
            ADD COLUMN IF NOT EXISTS trading_leverage DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS position_notional_multiplier DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS position_notional_usdt_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS estimated_initial_margin_usdt DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS position_risk_cap_enabled BOOLEAN,
            ADD COLUMN IF NOT EXISTS position_risk_cap_applied BOOLEAN,
            ADD COLUMN IF NOT EXISTS position_risk_fraction DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS raw_position_notional_usdt_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS risk_capped_position_notional_usdt_plan DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS max_full_sl_capital_risk DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS full_main_sl_gross_ret DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS full_main_sl_net_ret DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS full_main_sl_capital_risk_abs DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS raw_position_capital_risk_abs DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS capped_position_capital_risk_abs DOUBLE PRECISION,
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



def update_position_after_entry_plan(
    trade_id: int,
    qty: float,
    entry_px_plan: float,
    entry_avg_px: Optional[float],
    tp_px: float,
    sl_px: float,
    trading_leverage: Optional[float] = None,
    position_notional_multiplier: Optional[float] = None,
    position_notional_usdt_plan: Optional[float] = None,
    estimated_initial_margin_usdt: Optional[float] = None,
    position_risk_cap_enabled: Optional[bool] = None,
    position_risk_cap_applied: Optional[bool] = None,
    position_risk_fraction: Optional[float] = None,
    raw_position_notional_usdt_plan: Optional[float] = None,
    risk_capped_position_notional_usdt_plan: Optional[float] = None,
    max_full_sl_capital_risk: Optional[float] = None,
    full_main_sl_gross_ret: Optional[float] = None,
    full_main_sl_net_ret: Optional[float] = None,
    full_main_sl_capital_risk_abs: Optional[float] = None,
    raw_position_capital_risk_abs: Optional[float] = None,
    capped_position_capital_risk_abs: Optional[float] = None,
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
            trading_leverage = %s,
            position_notional_multiplier = %s,
            position_notional_usdt_plan = %s,
            estimated_initial_margin_usdt = %s,
            position_risk_cap_enabled = %s,
            position_risk_cap_applied = %s,
            position_risk_fraction = %s,
            raw_position_notional_usdt_plan = %s,
            risk_capped_position_notional_usdt_plan = %s,
            max_full_sl_capital_risk = %s,
            full_main_sl_gross_ret = %s,
            full_main_sl_net_ret = %s,
            full_main_sl_capital_risk_abs = %s,
            raw_position_capital_risk_abs = %s,
            capped_position_capital_risk_abs = %s,
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
                None if trading_leverage is None else float(trading_leverage),
                None if position_notional_multiplier is None else float(position_notional_multiplier),
                None if position_notional_usdt_plan is None else float(position_notional_usdt_plan),
                None if estimated_initial_margin_usdt is None else float(estimated_initial_margin_usdt),
                None if position_risk_cap_enabled is None else bool(position_risk_cap_enabled),
                None if position_risk_cap_applied is None else bool(position_risk_cap_applied),
                None if position_risk_fraction is None else float(position_risk_fraction),
                None if raw_position_notional_usdt_plan is None else float(raw_position_notional_usdt_plan),
                None if risk_capped_position_notional_usdt_plan is None else float(risk_capped_position_notional_usdt_plan),
                None if max_full_sl_capital_risk is None else float(max_full_sl_capital_risk),
                None if full_main_sl_gross_ret is None else float(full_main_sl_gross_ret),
                None if full_main_sl_net_ret is None else float(full_main_sl_net_ret),
                None if full_main_sl_capital_risk_abs is None else float(full_main_sl_capital_risk_abs),
                None if raw_position_capital_risk_abs is None else float(raw_position_capital_risk_abs),
                None if capped_position_capital_risk_abs is None else float(capped_position_capital_risk_abs),
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
def round_qty_down(qty: float, qty_step: float) -> float:
    qty_f = float(qty)
    step_f = float(qty_step)

    if qty_f <= 0:
        return 0.0

    if step_f <= 0:
        return qty_f

    steps = math.floor(qty_f / step_f)
    return float(steps * step_f)


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


def calc_prod_management_prices(side: str, entry_price: float, atr14: float) -> Dict[str, float]:
    partial_tp_atr = float(config.TP_ATR) * float(getattr(config, "PARTIAL_TP_LEVEL_FRACTION", 0.75))
    final_tp_atr = float(config.TP_ATR)
    early_stop_atr = float(config.SL_ATR) * float(getattr(config, "EARLY_STOP_SL_FRACTION", 0.5))
    main_sl_atr = float(config.SL_ATR)
    rest_stop_atr = float(getattr(config, "REST_STOP_AFTER_PARTIAL_ATR_MULT", float(config.TP_ATR) * 0.125))
    return {
        "partial_tp_px": calc_directional_price(
            side=side,
            entry_price=entry_price,
            atr14=atr14,
            atr_mult=partial_tp_atr,
            direction="PROFIT",
        ),
        "final_tp_px": calc_directional_price(
            side=side,
            entry_price=entry_price,
            atr14=atr14,
            atr_mult=final_tp_atr,
            direction="PROFIT",
        ),
        "early_stop_px": calc_directional_price(
            side=side,
            entry_price=entry_price,
            atr14=atr14,
            atr_mult=early_stop_atr,
            direction="LOSS",
        ),
        "main_sl_px": calc_directional_price(
            side=side,
            entry_price=entry_price,
            atr14=atr14,
            atr_mult=main_sl_atr,
            direction="LOSS",
        ),
        "rest_stop_after_partial_px": calc_directional_price(
            side=side,
            entry_price=entry_price,
            atr14=atr14,
            atr_mult=rest_stop_atr,
            direction="PROFIT",
        ),
    }


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


def close_side_for_position_side(side: str) -> str:
    side_u = str(side).upper()

    if side_u == "LONG":
        return "Sell"

    if side_u == "SHORT":
        return "Buy"

    raise RuntimeError("bad side for close order: {}".format(side))


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


def update_position_trade_management_plan(
    trade_id: int,
    trade_management_mode: str,
    partial_tp_px: Optional[float],
    final_tp_px: Optional[float],
    early_stop_px: Optional[float],
    main_sl_px: Optional[float],
    rest_stop_after_partial_px: Optional[float],
    partial_tp_qty: Optional[float],
    final_tp_qty: Optional[float],
    early_stop_expires_at: Optional[pd.Timestamp],
) -> None:
    sql = """
        UPDATE public.trading_positions
        SET
            partial_tp_px_plan = %s,
            final_tp_px_plan = %s,
            early_stop_px_plan = %s,
            main_sl_px_plan = %s,
            rest_stop_after_partial_px_plan = %s,
            partial_tp_qty_plan = %s,
            final_tp_qty_plan = %s,
            early_stop_expires_at = %s,
            trade_management_mode = %s,
            updated_at = NOW()
        WHERE trade_id = %s
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                partial_tp_px,
                final_tp_px,
                early_stop_px,
                main_sl_px,
                rest_stop_after_partial_px,
                partial_tp_qty,
                final_tp_qty,
                None if early_stop_expires_at is None else pd.to_datetime(early_stop_expires_at, utc=True).to_pydatetime(),
                str(trade_management_mode),
                int(trade_id),
            ),
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

        assert_trading_positions_schema_ready_before_entry()

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

        if reject_if_signal_not_in_side_whitelist(signal):
            return

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
        trading_leverage = resolve_trading_leverage()
        position_notional_multiplier = resolve_position_notional_multiplier()

        atr14 = float(signal.get("atr14") or 0.0)
        if atr14 <= 0:
            raise RuntimeError("atr14 missing in selected signal. Need selector to carry atr14 from online features.")

        planned_tp_sl = calc_tp_sl_prices(
            side=side,
            entry_price=entry_px_plan,
            atr14=atr14,
            tp_atr=config.TP_ATR,
            sl_atr=config.SL_ATR,
        )

        tp_px = float(planned_tp_sl["tp_px"])
        sl_px = float(planned_tp_sl["sl_px"])

        sizing_plan = build_position_sizing_plan(
            trade_capital_usdt=trade_capital_usdt,
            trading_leverage=trading_leverage,
            position_notional_multiplier=position_notional_multiplier,
        )

        raw_position_notional_usdt = float(sizing_plan["position_notional_usdt"])

        risk_sizing_plan = calc_risk_capped_position_notional(
            trade_capital_usdt=trade_capital_usdt,
            base_position_notional_usdt=raw_position_notional_usdt,
            trading_leverage=trading_leverage,
            side=side,
            entry_price=entry_px_plan,
            main_sl_price=sl_px,
            max_full_sl_capital_risk=float(getattr(config, "MAX_FULL_SL_CAPITAL_RISK", 0.0) or 0.0),
            enabled=bool(getattr(config, "POSITION_RISK_CAP_ENABLED", True)),
            include_round_trip_cost=bool(getattr(config, "POSITION_RISK_CAP_INCLUDE_ROUND_TRIP_COST", True)),
            fee_side=float(getattr(config, "POSITION_RISK_CAP_FEE_SIDE", 0.0) or 0.0),
            slippage_side=float(getattr(config, "POSITION_RISK_CAP_SLIPPAGE_SIDE", 0.0) or 0.0),
        )

        position_notional_usdt = float(risk_sizing_plan["position_notional_usdt"])
        estimated_initial_margin_usdt = float(risk_sizing_plan["estimated_initial_margin_usdt"])

        qty_info = calc_order_qty(
            available_usdt=position_notional_usdt,
            entry_price=entry_px_plan,
            qty_step=float(instrument["qty_step"]),
            min_order_qty=float(instrument["min_order_qty"]),
            min_notional=float(instrument["min_notional"]),
            use_balance_pct=config.USE_AVAILABLE_BALANCE_PCT,
        )

        if not bool(qty_info["ok"]):
            print("RISK_REJECT:", qty_info)
            print("RISK_SIZING_PLAN:", json.dumps(risk_sizing_plan, ensure_ascii=False, default=json_default))
            return

        qty = float(qty_info["qty"])

        trade_id = insert_position_stub(signal=signal, status="DRY_RUN_CREATED" if DRY_RUN else "CREATED")

        entry_link_id = build_order_link_id(signal_key, "entry")
        tp_link_id = build_order_link_id(signal_key, "tp")
        sl_link_id = build_order_link_id(signal_key, "sl")
        partial_tp_link_id = build_order_link_id(signal_key, "partialtp")
        final_tp_link_id = build_order_link_id(signal_key, "finaltp")
        early_stop_link_id = build_order_link_id(signal_key, "earlystop")

        entry_request = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_link_id": entry_link_id,
            "reduce_only": False,
            "available_usdt": available_usdt,
            "trade_capital_usdt": trade_capital_usdt,
            "trading_leverage": trading_leverage,
            "position_notional_multiplier": position_notional_multiplier,
            "position_notional_usdt_plan": position_notional_usdt,
            "estimated_initial_margin_usdt": estimated_initial_margin_usdt,
            "risk_sizing_plan": risk_sizing_plan,
            "position_risk_cap_enabled": bool(risk_sizing_plan.get("position_risk_cap_enabled")),
            "position_risk_cap_applied": bool(risk_sizing_plan.get("position_risk_cap_applied")),
            "position_risk_fraction": float(risk_sizing_plan.get("position_fraction") or 0.0),
            "raw_position_notional_usdt_plan": raw_position_notional_usdt,
            "risk_capped_position_notional_usdt_plan": position_notional_usdt,
            "max_full_sl_capital_risk": float(risk_sizing_plan.get("max_full_sl_capital_risk") or 0.0),
            "full_main_sl_capital_risk_abs": float(risk_sizing_plan.get("capped_position_capital_risk_abs") or 0.0),
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
                leverage_response = client.set_symbol_leverage(
                    symbol=symbol,
                    leverage=trading_leverage,
                )

                entry_request["leverage_response"] = leverage_response

                print("BYBIT_LEVERAGE_SET")
                print("symbol:", symbol)
                print("trading_leverage:", trading_leverage)
                print("already_set:", bool(leverage_response.get("already_set")))

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

        management_prices = calc_prod_management_prices(
            side=side,
            entry_price=entry_avg_px_actual,
            atr14=atr14,
        )

        partial_tp_px = float(management_prices["partial_tp_px"])
        final_tp_px = float(management_prices["final_tp_px"])
        early_stop_px = float(management_prices["early_stop_px"])
        main_sl_px = float(management_prices["main_sl_px"])
        rest_stop_after_partial_px = float(management_prices["rest_stop_after_partial_px"])

        update_position_after_entry_plan(
            trade_id=trade_id,
            qty=qty,
            entry_px_plan=entry_px_plan,
            entry_avg_px=entry_avg_px_actual,
            tp_px=tp_px,
            sl_px=sl_px,
            trading_leverage=trading_leverage,
            position_notional_multiplier=position_notional_multiplier,
            position_notional_usdt_plan=position_notional_usdt,
            estimated_initial_margin_usdt=estimated_initial_margin_usdt,
            position_risk_cap_enabled=bool(risk_sizing_plan.get("position_risk_cap_enabled")),
            position_risk_cap_applied=bool(risk_sizing_plan.get("position_risk_cap_applied")),
            position_risk_fraction=float(risk_sizing_plan.get("position_fraction") or 0.0),
            raw_position_notional_usdt_plan=float(risk_sizing_plan.get("raw_position_notional_usdt") or raw_position_notional_usdt),
            risk_capped_position_notional_usdt_plan=float(risk_sizing_plan.get("risk_capped_position_notional_usdt") or position_notional_usdt),
            max_full_sl_capital_risk=float(risk_sizing_plan.get("max_full_sl_capital_risk") or 0.0),
            full_main_sl_gross_ret=float(risk_sizing_plan.get("full_main_sl_gross_ret") or 0.0),
            full_main_sl_net_ret=float(risk_sizing_plan.get("full_main_sl_net_ret") or 0.0),
            full_main_sl_capital_risk_abs=float(risk_sizing_plan.get("capped_position_capital_risk_abs") or 0.0),
            raw_position_capital_risk_abs=float(risk_sizing_plan.get("raw_position_capital_risk_abs") or 0.0),
            capped_position_capital_risk_abs=float(risk_sizing_plan.get("capped_position_capital_risk_abs") or 0.0),
        )

        use_partial_mode = bool(getattr(config, "PARTIAL_TP_ENABLED", False))
        use_early_stop_mode = bool(getattr(config, "EARLY_STOP_ENABLED", False))

        qty_step = float(instrument["qty_step"])
        partial_tp_qty = round_qty_down(
            qty * float(getattr(config, "PARTIAL_TP_QTY_FRACTION", 0.5)),
            qty_step,
        )
        final_tp_qty = round_qty_down(float(qty) - float(partial_tp_qty), qty_step)

        if partial_tp_qty <= 0 or final_tp_qty <= 0:
            use_partial_mode = False

        early_stop_expires_at = (
            pd.Timestamp.now(tz="UTC")
            + pd.Timedelta(minutes=int(getattr(config, "EARLY_STOP_WINDOW_MINUTES", 60)))
        )

        trade_management_mode = (
            "partial75_early_stop"
            if use_partial_mode or use_early_stop_mode
            else "legacy_full_tp_sl"
        )

        update_position_trade_management_plan(
            trade_id=trade_id,
            trade_management_mode=trade_management_mode,
            partial_tp_px=partial_tp_px if use_partial_mode else None,
            final_tp_px=final_tp_px if use_partial_mode else None,
            early_stop_px=early_stop_px if use_early_stop_mode else None,
            main_sl_px=main_sl_px,
            rest_stop_after_partial_px=rest_stop_after_partial_px,
            partial_tp_qty=partial_tp_qty if use_partial_mode else None,
            final_tp_qty=final_tp_qty if use_partial_mode else None,
            early_stop_expires_at=early_stop_expires_at if use_early_stop_mode else None,
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
                    "trading_leverage": trading_leverage,
                    "position_notional_multiplier": position_notional_multiplier,
                    "position_notional_usdt_plan": position_notional_usdt,
                    "estimated_initial_margin_usdt": estimated_initial_margin_usdt,
                    "chulan_enabled": bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)),
                    "chulan_base_capital_usdt": float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0),
                    "entry_px_actual": entry_avg_px_actual,
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                    "partial_tp_enabled": use_partial_mode,
                    "early_stop_enabled": use_early_stop_mode,
                    "partial_tp_px": partial_tp_px,
                    "final_tp_px": final_tp_px,
                    "early_stop_px": early_stop_px,
                    "main_sl_px": main_sl_px,
                    "rest_stop_after_partial_px": rest_stop_after_partial_px,
                    "partial_tp_qty": partial_tp_qty,
                    "final_tp_qty": final_tp_qty,
                    "early_stop_expires_at": str(early_stop_expires_at),
                    "trade_management_mode": trade_management_mode,
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
                    "partial_tp_enabled": use_partial_mode,
                    "early_stop_enabled": use_early_stop_mode,
                    "partial_tp_px": partial_tp_px,
                    "final_tp_px": final_tp_px,
                    "early_stop_px": early_stop_px,
                    "main_sl_px": main_sl_px,
                    "rest_stop_after_partial_px": rest_stop_after_partial_px,
                    "partial_tp_qty": partial_tp_qty,
                    "final_tp_qty": final_tp_qty,
                    "trade_management_mode": trade_management_mode,
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
            print("trading_leverage:", trading_leverage)
            print("position_notional_multiplier:", position_notional_multiplier)
            print("position_notional_usdt_plan:", position_notional_usdt)
            print("estimated_initial_margin_usdt:", estimated_initial_margin_usdt)
            print("risk_sizing_plan:", json.dumps(risk_sizing_plan, ensure_ascii=False, default=json_default))
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
            print("partial_tp_enabled:", use_partial_mode)
            print("early_stop_enabled:", use_early_stop_mode)
            print("partial_tp_px:", partial_tp_px)
            print("final_tp_px:", final_tp_px)
            print("early_stop_px:", early_stop_px)
            print("main_sl_px:", main_sl_px)
            print("rest_stop_after_partial_px:", rest_stop_after_partial_px)
            print("partial_tp_qty:", partial_tp_qty)
            print("final_tp_qty:", final_tp_qty)
            print("trade_management_mode:", trade_management_mode)

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
                    "partial_tp_enabled": use_partial_mode,
                    "early_stop_enabled": use_early_stop_mode,
                    "partial_tp_px": partial_tp_px,
                    "final_tp_px": final_tp_px,
                    "early_stop_px": early_stop_px,
                    "main_sl_px": main_sl_px,
                    "rest_stop_after_partial_px": rest_stop_after_partial_px,
                    "partial_tp_qty": partial_tp_qty,
                    "final_tp_qty": final_tp_qty,
                    "trade_management_mode": trade_management_mode,
                    "gate2": signal.get("gate2_p"),
                    "gate4": signal.get("gate4_p"),
                    "gate5_1": signal.get("gate5_1_proba"),
                    "gate5_3": signal.get("gate5_3_proba"),
                },
                force=True,
            )
            return

        try:
            if use_partial_mode or use_early_stop_mode:
                protective_responses = {}

                if use_early_stop_mode:
                    early_stop_response = place_reduce_only_trigger_market_order(
                        client=client,
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        trigger_px=early_stop_px,
                        trigger_kind="SL",
                        order_link_id=early_stop_link_id,
                    )
                    protective_responses["early_stop"] = early_stop_response

                    insert_order(
                        trade_id=trade_id,
                        signal_key=signal_key,
                        symbol=symbol,
                        side=side,
                        order_role="EARLY_STOP",
                        local_order_key=early_stop_link_id,
                        order_type="MarketTrigger",
                        qty_plan=qty,
                        price_plan=None,
                        trigger_price_plan=early_stop_px,
                        status="SENT",
                        request_json={
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "early_stop_px": early_stop_px,
                            "early_stop_expires_at": str(early_stop_expires_at),
                        },
                        response_json=early_stop_response,
                        bybit_order_id=str(((early_stop_response or {}).get("result") or {}).get("orderId") or ""),
                        bybit_order_link_id=early_stop_link_id,
                    )

                else:
                    main_sl_response = place_reduce_only_trigger_market_order(
                        client=client,
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        trigger_px=main_sl_px,
                        trigger_kind="SL",
                        order_link_id=sl_link_id,
                    )
                    protective_responses["stop_loss"] = main_sl_response

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
                        trigger_price_plan=main_sl_px,
                        status="SENT",
                        request_json={"symbol": symbol, "side": side, "qty": qty, "sl_px": main_sl_px},
                        response_json=main_sl_response,
                        bybit_order_id=str(((main_sl_response or {}).get("result") or {}).get("orderId") or ""),
                        bybit_order_link_id=sl_link_id,
                    )

                if use_partial_mode:
                    partial_tp_response = place_reduce_only_trigger_market_order(
                        client=client,
                        symbol=symbol,
                        side=side,
                        qty=partial_tp_qty,
                        trigger_px=partial_tp_px,
                        trigger_kind="TP",
                        order_link_id=partial_tp_link_id,
                    )
                    protective_responses["partial_tp"] = partial_tp_response

                    insert_order(
                        trade_id=trade_id,
                        signal_key=signal_key,
                        symbol=symbol,
                        side=side,
                        order_role="PARTIAL_TP",
                        local_order_key=partial_tp_link_id,
                        order_type="MarketTrigger",
                        qty_plan=partial_tp_qty,
                        price_plan=None,
                        trigger_price_plan=partial_tp_px,
                        status="SENT",
                        request_json={
                            "symbol": symbol,
                            "side": side,
                            "qty": partial_tp_qty,
                            "partial_tp_px": partial_tp_px,
                            "partial_tp_level_fraction": float(getattr(config, "PARTIAL_TP_LEVEL_FRACTION", 0.75)),
                        },
                        response_json=partial_tp_response,
                        bybit_order_id=str(((partial_tp_response or {}).get("result") or {}).get("orderId") or ""),
                        bybit_order_link_id=partial_tp_link_id,
                    )

                    final_tp_response = place_reduce_only_trigger_market_order(
                        client=client,
                        symbol=symbol,
                        side=side,
                        qty=final_tp_qty,
                        trigger_px=final_tp_px,
                        trigger_kind="TP",
                        order_link_id=final_tp_link_id,
                    )
                    protective_responses["final_tp"] = final_tp_response

                    insert_order(
                        trade_id=trade_id,
                        signal_key=signal_key,
                        symbol=symbol,
                        side=side,
                        order_role="FINAL_TP",
                        local_order_key=final_tp_link_id,
                        order_type="MarketTrigger",
                        qty_plan=final_tp_qty,
                        price_plan=None,
                        trigger_price_plan=final_tp_px,
                        status="SENT",
                        request_json={
                            "symbol": symbol,
                            "side": side,
                            "qty": final_tp_qty,
                            "final_tp_px": final_tp_px,
                        },
                        response_json=final_tp_response,
                        bybit_order_id=str(((final_tp_response or {}).get("result") or {}).get("orderId") or ""),
                        bybit_order_link_id=final_tp_link_id,
                    )

                else:
                    final_tp_response = place_reduce_only_trigger_market_order(
                        client=client,
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        trigger_px=tp_px,
                        trigger_kind="TP",
                        order_link_id=tp_link_id,
                    )
                    protective_responses["take_profit"] = final_tp_response

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
                        response_json=final_tp_response,
                        bybit_order_id=str(((final_tp_response or {}).get("result") or {}).get("orderId") or ""),
                        bybit_order_link_id=tp_link_id,
                    )

            else:
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

            failed_orders = [
                ("TAKE_PROFIT", tp_link_id, tp_px, qty),
                ("STOP_LOSS", sl_link_id, sl_px, qty),
                ("PARTIAL_TP", partial_tp_link_id, partial_tp_px, partial_tp_qty),
                ("FINAL_TP", final_tp_link_id, final_tp_px, final_tp_qty),
                ("EARLY_STOP", early_stop_link_id, early_stop_px, qty),
            ]

            for failed_role, failed_link_id, failed_trigger_px, failed_qty in failed_orders:
                insert_order(
                    trade_id=trade_id,
                    signal_key=signal_key,
                    symbol=symbol,
                    side=side,
                    order_role=failed_role,
                    local_order_key=failed_link_id,
                    order_type="MarketTrigger",
                    qty_plan=failed_qty,
                    price_plan=None,
                    trigger_price_plan=failed_trigger_px,
                    status="TP_SL_FAILED",
                    request_json={
                        "symbol": symbol,
                        "side": side,
                        "qty": failed_qty,
                        "trigger_px": failed_trigger_px,
                        "trade_management_mode": trade_management_mode,
                    },
                    response_json={"error": error_message},
                    bybit_order_id=None,
                    bybit_order_link_id=failed_link_id,
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
                    "trading_leverage": trading_leverage,
                    "position_notional_multiplier": position_notional_multiplier,
                    "position_notional_usdt_plan": position_notional_usdt,
                    "estimated_initial_margin_usdt": estimated_initial_margin_usdt,
                    "entry_px_plan": entry_px_plan,
                    "qty": qty,
                    "tp_px": tp_px,
                    "sl_px": sl_px,
                    "partial_tp_enabled": use_partial_mode,
                    "early_stop_enabled": use_early_stop_mode,
                    "partial_tp_px": partial_tp_px,
                    "final_tp_px": final_tp_px,
                    "early_stop_px": early_stop_px,
                    "main_sl_px": main_sl_px,
                    "rest_stop_after_partial_px": rest_stop_after_partial_px,
                    "partial_tp_qty": partial_tp_qty,
                    "final_tp_qty": final_tp_qty,
                    "trade_management_mode": trade_management_mode,
                    "entry_order_link_id": entry_link_id,
                    "tp_order_link_id": tp_link_id,
                    "sl_order_link_id": sl_link_id,
                    "partial_tp_order_link_id": partial_tp_link_id,
                    "final_tp_order_link_id": final_tp_link_id,
                    "early_stop_order_link_id": early_stop_link_id,
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
            print("partial_tp_enabled:", use_partial_mode)
            print("early_stop_enabled:", use_early_stop_mode)
            print("partial_tp_px:", partial_tp_px)
            print("final_tp_px:", final_tp_px)
            print("early_stop_px:", early_stop_px)
            print("main_sl_px:", main_sl_px)
            print("rest_stop_after_partial_px:", rest_stop_after_partial_px)
            print("partial_tp_qty:", partial_tp_qty)
            print("final_tp_qty:", final_tp_qty)
            print("trade_management_mode:", trade_management_mode)
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
        print("trading_leverage:", trading_leverage)
        print("position_notional_multiplier:", position_notional_multiplier)
        print("position_notional_usdt_plan:", position_notional_usdt)
        print("estimated_initial_margin_usdt:", estimated_initial_margin_usdt)
        print("risk_sizing_plan:", json.dumps(risk_sizing_plan, ensure_ascii=False, default=json_default))
        print("chulan_enabled:", bool(int(getattr(config, "CHULAN_ENABLED", 0) or 0)))
        print("chulan_base_capital_usdt:", float(getattr(config, "CHULAN_BASE_CAPITAL_USDT", 0.0) or 0.0))
        print("qty:", qty)
        print("tp_px:", tp_px)
        print("sl_px:", sl_px)
        print("partial_tp_enabled:", use_partial_mode)
        print("early_stop_enabled:", use_early_stop_mode)
        print("partial_tp_px:", partial_tp_px)
        print("final_tp_px:", final_tp_px)
        print("early_stop_px:", early_stop_px)
        print("main_sl_px:", main_sl_px)
        print("rest_stop_after_partial_px:", rest_stop_after_partial_px)
        print("partial_tp_qty:", partial_tp_qty)
        print("final_tp_qty:", final_tp_qty)
        print("trade_management_mode:", trade_management_mode)

    finally:
        release_lock(LOCK_NAME, owner=owner)


if __name__ == "__main__":
    main()
