from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, Optional, Tuple, Any

import pandas as pd

from online.trading import config
from online.trading.db import db_cursor, read_sql


OUTCOME_TABLE = "public.trading_symbol_outcomes"


def ensure_symbol_outcome_table() -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS public.trading_symbol_outcomes (
            outcome_id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            source_run_id TEXT NULL,
            signal_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            signal_ts TIMESTAMPTZ NULL,
            entry_ts TIMESTAMPTZ NOT NULL,
            exit_ts TIMESTAMPTZ NOT NULL,
            net_ret DOUBLE PRECISION NOT NULL,
            exit_reason TEXT NULL,
            pair_model_name TEXT NULL,
            grid_name TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (source, signal_key)
        );

        CREATE INDEX IF NOT EXISTS idx_trading_symbol_outcomes_symbol_exit_ts
        ON public.trading_symbol_outcomes (symbol, exit_ts);

        CREATE INDEX IF NOT EXISTS idx_trading_symbol_outcomes_source_exit_ts
        ON public.trading_symbol_outcomes (source, exit_ts);

        CREATE INDEX IF NOT EXISTS idx_trading_symbol_outcomes_source_symbol_exit_ts
        ON public.trading_symbol_outcomes (source, symbol, exit_ts);

        CREATE TABLE IF NOT EXISTS public.trading_dynamic_symbol_state (
            source TEXT NOT NULL DEFAULT 'prod',
            symbol TEXT NOT NULL,
            is_blocked BOOLEAN NOT NULL DEFAULT FALSE,
            blocked_until TIMESTAMPTZ NULL,
            cooldown_days INTEGER NOT NULL DEFAULT 0,
            bad_streak INTEGER NOT NULL DEFAULT 0,
            lookback_days INTEGER NOT NULL DEFAULT 30,
            hist_n INTEGER NOT NULL DEFAULT 0,
            hist_k INTEGER NOT NULL DEFAULT 0,
            hist_wilson DOUBLE PRECISION NULL,
            hist_sum_net_ret DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            probation_active BOOLEAN NOT NULL DEFAULT FALSE,
            last_reason TEXT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (source, symbol)
        );

        ALTER TABLE public.trading_dynamic_symbol_state
        ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'prod';

        ALTER TABLE public.trading_dynamic_symbol_state
        DROP CONSTRAINT IF EXISTS trading_dynamic_symbol_state_pkey;

        ALTER TABLE public.trading_dynamic_symbol_state
        ADD PRIMARY KEY (source, symbol);

        CREATE INDEX IF NOT EXISTS idx_trading_dynamic_symbol_state_source_blocked
        ON public.trading_dynamic_symbol_state (source, is_blocked, blocked_until);

        CREATE INDEX IF NOT EXISTS idx_trading_dynamic_symbol_state_symbol
        ON public.trading_dynamic_symbol_state (symbol);
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)

def record_symbol_outcome(
    source: str,
    signal_key: str,
    symbol: str,
    side: str,
    signal_ts: Optional[pd.Timestamp],
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    net_ret: float,
    exit_reason: Optional[str],
    pair_model_name: Optional[str] = None,
    grid_name: Optional[str] = None,
    source_run_id: Optional[str] = None,
) -> None:
    ensure_symbol_outcome_table()

    sql = """
        INSERT INTO public.trading_symbol_outcomes (
            source,
            source_run_id,
            signal_key,
            symbol,
            side,
            signal_ts,
            entry_ts,
            exit_ts,
            net_ret,
            exit_reason,
            pair_model_name,
            grid_name,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (source, signal_key)
        DO UPDATE SET
            source_run_id = EXCLUDED.source_run_id,
            symbol = EXCLUDED.symbol,
            side = EXCLUDED.side,
            signal_ts = EXCLUDED.signal_ts,
            entry_ts = EXCLUDED.entry_ts,
            exit_ts = EXCLUDED.exit_ts,
            net_ret = EXCLUDED.net_ret,
            exit_reason = EXCLUDED.exit_reason,
            pair_model_name = EXCLUDED.pair_model_name,
            grid_name = EXCLUDED.grid_name,
            updated_at = NOW()
    """

    signal_ts_dt = None
    if signal_ts is not None and pd.notna(signal_ts):
        signal_ts_dt = pd.Timestamp(signal_ts).to_pydatetime()

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(source),
                source_run_id,
                str(signal_key),
                str(symbol).upper(),
                str(side).upper(),
                signal_ts_dt,
                pd.Timestamp(entry_ts).to_pydatetime(),
                pd.Timestamp(exit_ts).to_pydatetime(),
                float(net_ret),
                exit_reason,
                pair_model_name,
                grid_name,
            ),
        )


def reset_backtest_outcomes(source: str = "backtest_approved") -> None:
    ensure_symbol_outcome_table()

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            DELETE FROM public.trading_symbol_outcomes
            WHERE source = %s
            """,
            (str(source),),
        )

        cur.execute(
            """
            DELETE FROM public.trading_dynamic_symbol_state
            WHERE source = %s
            """,
            (str(source),),
        )


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")

    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    margin = z * math.sqrt(
        (phat * (1.0 - phat) / float(n)) + ((z * z) / (4.0 * n * n))
    )
    return float((center - margin) / denom)


def get_symbol_recent_stats(
    symbol: str,
    now_ts: pd.Timestamp,
    lookback_days: int,
    source: str = "prod",
) -> Dict[str, object]:
    ensure_symbol_outcome_table()

    start_ts = now_ts - pd.Timedelta(days=int(lookback_days))

    sql = """
        SELECT
            symbol,
            exit_ts AS exit_filled_at,
            net_ret
        FROM public.trading_symbol_outcomes
        WHERE source = %s
          AND symbol = %s
          AND exit_ts >= %s
          AND exit_ts < %s
        ORDER BY exit_ts ASC
    """

    df = read_sql(
        sql,
        [
            str(source),
            str(symbol).upper(),
            start_ts.to_pydatetime(),
            now_ts.to_pydatetime(),
        ],
    )

    if df.empty:
        return {
            "n": 0,
            "k": 0,
            "win_rate": float("nan"),
            "wilson": float("nan"),
            "sum_net_ret": 0.0,
            "bad_streak": 0,
        }

    df["net_ret"] = pd.to_numeric(df["net_ret"], errors="coerce").fillna(0.0)
    wins = df["net_ret"] > 0

    bad_streak = 0
    for x in df["net_ret"].tolist()[::-1]:
        if float(x) <= 0:
            bad_streak += 1
        else:
            break

    n = int(len(df))
    k = int(wins.sum())

    return {
        "n": n,
        "k": k,
        "win_rate": float(k) / float(n) if n > 0 else float("nan"),
        "wilson": wilson_lower_bound(k, n),
        "sum_net_ret": float(df["net_ret"].sum()),
        "bad_streak": int(bad_streak),
    }

def get_symbol_state(symbol: str, source: str = "prod") -> Optional[Dict[str, object]]:
    ensure_symbol_outcome_table()

    sql = """
        SELECT *
        FROM public.trading_dynamic_symbol_state
        WHERE source = %s
          AND symbol = %s
        LIMIT 1
    """

    df = read_sql(sql, [str(source), str(symbol).upper()])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def upsert_symbol_state(
    symbol: str,
    is_blocked: bool,
    blocked_until: Optional[pd.Timestamp],
    cooldown_days: int,
    bad_streak: int,
    stats: Dict[str, object],
    probation_active: bool,
    reason: str,
    source: str = "prod",
) -> None:
    ensure_symbol_outcome_table()

    sql = """
        INSERT INTO public.trading_dynamic_symbol_state (
            source,
            symbol,
            is_blocked,
            blocked_until,
            cooldown_days,
            bad_streak,
            lookback_days,
            hist_n,
            hist_k,
            hist_wilson,
            hist_sum_net_ret,
            probation_active,
            last_reason,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (source, symbol)
        DO UPDATE SET
            is_blocked = EXCLUDED.is_blocked,
            blocked_until = EXCLUDED.blocked_until,
            cooldown_days = EXCLUDED.cooldown_days,
            bad_streak = EXCLUDED.bad_streak,
            lookback_days = EXCLUDED.lookback_days,
            hist_n = EXCLUDED.hist_n,
            hist_k = EXCLUDED.hist_k,
            hist_wilson = EXCLUDED.hist_wilson,
            hist_sum_net_ret = EXCLUDED.hist_sum_net_ret,
            probation_active = EXCLUDED.probation_active,
            last_reason = EXCLUDED.last_reason,
            updated_at = NOW()
    """

    blocked_until_dt = None
    if blocked_until is not None:
        blocked_until_dt = pd.Timestamp(blocked_until).to_pydatetime()

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            sql,
            (
                str(source),
                str(symbol).upper(),
                bool(is_blocked),
                blocked_until_dt,
                int(cooldown_days),
                int(bad_streak),
                int(config.LOOKBACK_DAYS),
                int(stats.get("n", 0)),
                int(stats.get("k", 0)),
                float(stats["wilson"]) if pd.notna(stats.get("wilson")) else None,
                float(stats.get("sum_net_ret", 0.0)),
                bool(probation_active),
                reason,
            ),
        )


def is_symbol_allowed(
    symbol: str,
    now_ts: Optional[pd.Timestamp] = None,
    source: str = "prod",
) -> Tuple[bool, str, Dict[str, object]]:
    if not config.DYNAMIC_SYMBOL_FILTER_ENABLED:
        return True, "dynamic_filter_disabled", {}

    ensure_symbol_outcome_table()

    symbol = str(symbol).upper()
    source = str(source)

    now = pd.Timestamp.now(tz="UTC") if now_ts is None else pd.Timestamp(now_ts).tz_convert("UTC")

    if symbol in set(config.FORCED_EXCLUDED_SYMBOLS):
        return False, "forced_excluded_symbol", {}

    state = get_symbol_state(symbol=symbol, source=source)

    if state is not None and bool(state.get("is_blocked")):
        blocked_until = pd.to_datetime(state.get("blocked_until"), utc=True, errors="coerce")
        if pd.notna(blocked_until) and now < blocked_until:
            return False, "active_cooldown_until_{}".format(blocked_until), state

        upsert_symbol_state(
            source=source,
            symbol=symbol,
            is_blocked=False,
            blocked_until=None,
            cooldown_days=int(state.get("cooldown_days") or config.BASE_COOLDOWN_DAYS),
            bad_streak=int(state.get("bad_streak") or 0),
            stats={
                "n": int(state.get("hist_n") or 0),
                "k": int(state.get("hist_k") or 0),
                "wilson": state.get("hist_wilson"),
                "sum_net_ret": float(state.get("hist_sum_net_ret") or 0.0),
            },
            probation_active=True,
            reason="probation_after_cooldown",
        )
        return True, "probation_after_cooldown", state

    stats = get_symbol_recent_stats(
        source=source,
        symbol=symbol,
        now_ts=now,
        lookback_days=config.LOOKBACK_DAYS,
    )

    n = int(stats["n"])
    wilson = stats["wilson"]
    bad_streak = int(stats["bad_streak"])

    should_block = False
    reason = "allowed"

    if n >= int(config.MIN_TRADES):
        if pd.notna(wilson) and float(wilson) < float(config.MIN_WILSON):
            should_block = True
            reason = "wilson_below_min"
        if bad_streak >= int(config.MAX_BAD_STREAK):
            should_block = True
            reason = "bad_streak"

    if should_block:
        prev_cooldown = int(state.get("cooldown_days") or 0) if state is not None else 0
        cooldown = max(
            int(config.BASE_COOLDOWN_DAYS),
            prev_cooldown * 2 if prev_cooldown > 0 else int(config.BASE_COOLDOWN_DAYS),
        )
        cooldown = min(cooldown, int(config.MAX_COOLDOWN_DAYS))
        blocked_until = now + pd.Timedelta(days=cooldown)

        upsert_symbol_state(
            source=source,
            symbol=symbol,
            is_blocked=True,
            blocked_until=blocked_until,
            cooldown_days=cooldown,
            bad_streak=bad_streak,
            stats=stats,
            probation_active=False,
            reason=reason,
        )
        return False, reason, stats

    upsert_symbol_state(
        source=source,
        symbol=symbol,
        is_blocked=False,
        blocked_until=None,
        cooldown_days=int(state.get("cooldown_days") or 0) if state is not None else 0,
        bad_streak=bad_streak,
        stats=stats,
        probation_active=False,
        reason="allowed",
    )

    return True, "allowed", stats
