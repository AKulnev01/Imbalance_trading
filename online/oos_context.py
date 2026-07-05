from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class OnlineOOSContext:
    enabled: bool
    symbols: List[str]
    start_ts: Optional[pd.Timestamp]
    end_ts: Optional[pd.Timestamp]
    start_text: str
    end_text: str


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_oos_symbols(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    parts = text.replace(";", ",").replace(" ", ",").split(",")
    out: List[str] = []

    for part in parts:
        symbol = str(part or "").strip().upper().replace("_", "")
        if not symbol:
            continue
        if symbol not in out:
            out.append(symbol)

    return out


def parse_oos_ts(raw: str, field_name: str) -> Optional[pd.Timestamp]:
    text = str(raw or "").strip()
    if not text:
        return None

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError("bad {}: {}".format(field_name, raw))

    return pd.Timestamp(ts)


def get_online_oos_context() -> OnlineOOSContext:
    enabled = _truthy(os.environ.get("IMB_ONLINE_OOS_MODE", ""))

    symbols = normalize_oos_symbols(os.environ.get("IMB_ONLINE_OOS_SYMBOLS", ""))
    start_raw = str(os.environ.get("IMB_ONLINE_OOS_START", "") or "").strip()
    end_raw = str(os.environ.get("IMB_ONLINE_OOS_END", "") or "").strip()

    start_ts = parse_oos_ts(start_raw, "IMB_ONLINE_OOS_START")
    end_ts = parse_oos_ts(end_raw, "IMB_ONLINE_OOS_END")

    if enabled:
        if not symbols:
            raise RuntimeError("IMB_ONLINE_OOS_MODE=1 but IMB_ONLINE_OOS_SYMBOLS is empty")

        if start_ts is None:
            raise RuntimeError("IMB_ONLINE_OOS_MODE=1 but IMB_ONLINE_OOS_START is empty")

        if end_ts is None:
            raise RuntimeError("IMB_ONLINE_OOS_MODE=1 but IMB_ONLINE_OOS_END is empty")

        if start_ts >= end_ts:
            raise RuntimeError("bad OOS window: IMB_ONLINE_OOS_START must be earlier than IMB_ONLINE_OOS_END")

    return OnlineOOSContext(
        enabled=bool(enabled),
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        start_text=start_raw,
        end_text=end_raw,
    )


def append_oos_sql_filters(
    where_parts: List[str],
    params: List[object],
    table_alias: str,
    ts_column: str,
    symbol_column: str = "symbol",
) -> OnlineOOSContext:
    ctx = get_online_oos_context()

    if not ctx.enabled:
        return ctx

    alias = str(table_alias or "").strip()
    ts_col = str(ts_column or "").strip()
    symbol_col = str(symbol_column or "").strip()

    if not alias:
        raise RuntimeError("append_oos_sql_filters: table_alias is empty")

    if not ts_col:
        raise RuntimeError("append_oos_sql_filters: ts_column is empty")

    if not symbol_col:
        raise RuntimeError("append_oos_sql_filters: symbol_column is empty")

    where_parts.append("UPPER({}.{}) = ANY(%s)".format(alias, symbol_col))
    params.append(ctx.symbols)

    where_parts.append("{}.{} >= %s".format(alias, ts_col))
    params.append(ctx.start_ts.to_pydatetime())

    where_parts.append("{}.{} < %s".format(alias, ts_col))
    params.append(ctx.end_ts.to_pydatetime())

    return ctx
