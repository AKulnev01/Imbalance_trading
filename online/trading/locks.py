from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from online.trading.db import db_cursor


def acquire_lock(
    lock_name: str,
    owner: str,
    ttl_seconds: int = 900,
) -> bool:
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    expires_at = now + timedelta(seconds=int(ttl_seconds))

    sql = """
        INSERT INTO public.trading_runtime_locks (
            lock_name,
            owner,
            acquired_at,
            expires_at
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (lock_name)
        DO UPDATE SET
            owner = EXCLUDED.owner,
            acquired_at = EXCLUDED.acquired_at,
            expires_at = EXCLUDED.expires_at
        WHERE public.trading_runtime_locks.expires_at < NOW()
        RETURNING lock_name
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (lock_name, owner, now, expires_at))
        row = cur.fetchone()

    return row is not None


def release_lock(lock_name: str, owner: Optional[str] = None) -> int:
    if owner is None:
        sql = """
            DELETE FROM public.trading_runtime_locks
            WHERE lock_name = %s
        """
        params = [lock_name]
    else:
        sql = """
            DELETE FROM public.trading_runtime_locks
            WHERE lock_name = %s
              AND owner = %s
        """
        params = [lock_name, owner]

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, params)
        return int(cur.rowcount)


def refresh_lock(
    lock_name: str,
    owner: str,
    ttl_seconds: int = 900,
) -> bool:
    expires_at = pd.Timestamp.now(tz="UTC").to_pydatetime() + timedelta(seconds=int(ttl_seconds))

    sql = """
        UPDATE public.trading_runtime_locks
        SET expires_at = %s
        WHERE lock_name = %s
          AND owner = %s
        RETURNING lock_name
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, (expires_at, lock_name, owner))
        row = cur.fetchone()

    return row is not None
