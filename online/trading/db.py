from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from online.trading.config import DB_DSN


def connect_db():
    return psycopg2.connect(DB_DSN)


@contextmanager
def db_cursor(commit: bool = False):
    conn = connect_db()
    try:
        cur = conn.cursor()
        try:
            yield conn, cur
            if commit:
                conn.commit()
        finally:
            cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def split_table_name(table_name: str) -> Tuple[str, str]:
    parts = str(table_name).split(".")
    if len(parts) == 1:
        return "public", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise RuntimeError("table name must be table or schema.table: {}".format(table_name))


def table_qname(table_name: str) -> str:
    schema, table = split_table_name(table_name)
    return "{}.{}".format(qident(schema), qident(table))


def table_exists(table_name: str) -> bool:
    schema, table = split_table_name(table_name)
    sql = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """
    with db_cursor(commit=False) as (_, cur):
        cur.execute(sql, (schema, table))
        return bool(cur.fetchone()[0])


def get_table_columns(table_name: str) -> List[str]:
    schema, table = split_table_name(table_name)
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """
    with connect_db() as conn:
        df = pd.read_sql_query(sql, conn, params=(schema, table))
    return [str(x) for x in df["column_name"].tolist()]


def read_sql(sql: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
    with connect_db() as conn:
        return pd.read_sql_query(sql, conn, params=list(params or []))


def execute(sql: str, params: Optional[Iterable[Any]] = None) -> int:
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, list(params or []))
        return int(cur.rowcount)


def execute_values(
    sql: str,
    rows: List[Tuple[Any, ...]],
    page_size: int = 1000,
) -> int:
    if not rows:
        return 0

    with db_cursor(commit=True) as (_, cur):
        psycopg2.extras.execute_values(cur, sql, rows, page_size=page_size)
        return len(rows)


def to_utc_dt(value: Any) -> Optional[datetime]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def clean_db_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return to_utc_dt(value)

    if isinstance(value, datetime):
        return value

    if isinstance(value, np.generic):
        return value.item()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
