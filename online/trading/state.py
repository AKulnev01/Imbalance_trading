from __future__ import annotations

from typing import Dict, Optional

from online.trading.db import read_sql
from online.trading.bybit_client import BybitClient


ACTIVE_POSITION_STATUSES = [
    "ENTRY_ORDER_SENT",
    "ENTRY_PARTIALLY_FILLED",
    "ENTRY_FILLED",
    "TP_SL_PLACED",
    "POSITION_OPEN",
    "TTL_CLOSE_SENT",
    "EMERGENCY_CLOSE_SENT",
]


def get_active_position() -> Optional[Dict[str, object]]:
    placeholders = ", ".join(["%s"] * len(ACTIVE_POSITION_STATUSES))
    sql = """
        SELECT *
        FROM public.trading_positions
        WHERE status IN ({})
        ORDER BY created_at DESC
        LIMIT 1
    """.format(placeholders)

    df = read_sql(sql, ACTIVE_POSITION_STATUSES)

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def has_active_position() -> bool:
    return get_active_position() is not None


def can_open_new_position() -> bool:
    return not has_active_position()


def get_last_selected_signal_key() -> Optional[str]:
    sql = """
        SELECT signal_key
        FROM public.trading_signals
        WHERE selected = TRUE
        ORDER BY signal_ts DESC
        LIMIT 1
    """
    df = read_sql(sql)
    if df.empty:
        return None
    return str(df.iloc[0]["signal_key"])


def get_exchange_open_positions(symbol: Optional[str] = None) -> list:
    client = BybitClient()
    return client.get_open_positions(symbol=symbol)


def exchange_has_open_position(symbol: Optional[str] = None) -> bool:
    return len(get_exchange_open_positions(symbol=symbol)) > 0

