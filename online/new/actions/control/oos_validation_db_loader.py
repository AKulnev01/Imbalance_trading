from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_DSN_ENV = "IMB_DB_DSN"

DEFAULT_SCHEMA = "public"
DEFAULT_M1_TABLE = "candles_m1"
DEFAULT_H4_TABLE = "candles_h4"
DEFAULT_MARKET_CATEGORY = "linear"
DEFAULT_SOURCE = "bybit"

DB_COLUMNS = [
    "symbol",
    "market_category",
    "entry_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "source",
]


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    if not text:
        raise ValueError("empty symbol")

    return text


def parse_ts(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("invalid timestamp: {}".format(value))
    return pd.Timestamp(ts)


def qident(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("empty SQL identifier")

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(ch not in allowed for ch in text):
        raise ValueError("unsafe SQL identifier: {}".format(name))

    return '"' + text.replace('"', '""') + '"'


def fq_table(schema_name: str, table_name: str) -> str:
    return "{}.{}".format(qident(schema_name), qident(table_name))


def detect_ts_col(df: pd.DataFrame) -> str:
    for candidate in ["entry_ts", "ts", "timestamp", "open_time", "open_ts", "datetime", "time"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "timestamp column not found. Expected one of: entry_ts, ts, timestamp, open_time, open_ts, datetime, time"
    )


def normalize_candle_frame(
    df: pd.DataFrame,
    symbol: str,
    market_category: str,
    source: str,
) -> pd.DataFrame:
    ts_col = detect_ts_col(df)
    out = df.copy()

    out["entry_ts"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=["entry_ts"]).copy()

    required_price_cols = ["open", "high", "low", "close", "volume"]
    for col in required_price_cols:
        if col not in out.columns:
            raise ValueError("required parquet column not found: {}".format(col))
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).copy()

    if "turnover" in out.columns:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
    else:
        out["turnover"] = None

    out["symbol"] = normalize_symbol(symbol)
    out["market_category"] = str(market_category).strip()
    out["source"] = str(source).strip()

    out = out[DB_COLUMNS].copy()
    out = out.sort_values("entry_ts").drop_duplicates(
        subset=["symbol", "market_category", "entry_ts"],
        keep="last",
    ).reset_index(drop=True)

    return out


def load_oos_frame(
    parquet_path: Path,
    symbol: str,
    market_category: str,
    source: str,
    oos_start: pd.Timestamp,
    oos_end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(str(parquet_path))

    df = pd.read_parquet(parquet_path)
    df = normalize_candle_frame(
        df=df,
        symbol=symbol,
        market_category=market_category,
        source=source,
    )

    mask = df["entry_ts"] >= oos_start
    if oos_end is not None:
        mask = mask & (df["entry_ts"] < oos_end)

    df = df.loc[mask].copy()
    return df.reset_index(drop=True)


def frame_bounds(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "rows": 0,
            "min_entry_ts": None,
            "max_entry_ts": None,
        }

    return {
        "rows": int(len(df)),
        "min_entry_ts": pd.Timestamp(df["entry_ts"].min()).isoformat(),
        "max_entry_ts": pd.Timestamp(df["entry_ts"].max()).isoformat(),
    }


def get_psycopg2():
    try:
        import psycopg2
        import psycopg2.extras
        return psycopg2
    except Exception as exc:
        raise RuntimeError(
            "psycopg2 is required for DB operations. Error: {}".format(repr(exc))
        )


def fetch_existing_stats(
    conn,
    schema_name: str,
    table_name: str,
    symbol: str,
    market_category: str,
    oos_start: pd.Timestamp,
    oos_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    table = fq_table(schema_name, table_name)

    if oos_end is None:
        sql = """
        SELECT
            COUNT(*) AS rows,
            MIN(entry_ts) AS min_entry_ts,
            MAX(entry_ts) AS max_entry_ts
        FROM {table}
        WHERE symbol = %s
          AND market_category = %s
          AND entry_ts >= %s
        """.format(table=table)
        params = (symbol, market_category, oos_start.to_pydatetime())
    else:
        sql = """
        SELECT
            COUNT(*) AS rows,
            MIN(entry_ts) AS min_entry_ts,
            MAX(entry_ts) AS max_entry_ts
        FROM {table}
        WHERE symbol = %s
          AND market_category = %s
          AND entry_ts >= %s
          AND entry_ts < %s
        """.format(table=table)
        params = (symbol, market_category, oos_start.to_pydatetime(), oos_end.to_pydatetime())

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    rows = int(row[0]) if row and row[0] is not None else 0
    min_ts = row[1] if row and row[1] is not None else None
    max_ts = row[2] if row and row[2] is not None else None

    return {
        "existing_rows_in_period": rows,
        "existing_min_entry_ts": str(min_ts) if min_ts is not None else None,
        "existing_max_entry_ts": str(max_ts) if max_ts is not None else None,
    }


def rows_for_insert(df: pd.DataFrame) -> List[Tuple[Any, ...]]:
    rows: List[Tuple[Any, ...]] = []

    for row in df.itertuples(index=False):
        turnover = None if pd.isna(row.turnover) else float(row.turnover)

        rows.append((
            str(row.symbol),
            str(row.market_category),
            pd.Timestamp(row.entry_ts).to_pydatetime(),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
            turnover,
            str(row.source),
        ))

    return rows


def insert_sql(schema_name: str, table_name: str, on_conflict: str) -> str:
    table = fq_table(schema_name, table_name)

    base = """
    INSERT INTO {table} (
        symbol,
        market_category,
        entry_ts,
        open,
        high,
        low,
        close,
        volume,
        turnover,
        source
    )
    VALUES %s
    """.format(table=table)

    if on_conflict == "error":
        return base

    if on_conflict == "skip":
        return base + """
        ON CONFLICT (symbol, market_category, entry_ts) DO NOTHING
        """

    if on_conflict == "update":
        return base + """
        ON CONFLICT (symbol, market_category, entry_ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            turnover = EXCLUDED.turnover,
            source = EXCLUDED.source,
            updated_at = NOW()
        """

    raise ValueError("unsupported on_conflict: {}".format(on_conflict))


def write_frame_to_db(
    dsn: str,
    schema_name: str,
    table_name: str,
    df: pd.DataFrame,
    on_conflict: str,
) -> int:
    if len(df) == 0:
        return 0

    psycopg2 = get_psycopg2()
    rows = rows_for_insert(df)

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                insert_sql(schema_name=schema_name, table_name=table_name, on_conflict=on_conflict),
                rows,
                page_size=5000,
            )
        conn.commit()

    return int(len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load validation/OOS candles from parquet into existing public.candles_m1/candles_h4 tables."
    )

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--m1-parquet", required=True)
    parser.add_argument("--h4-parquet", required=True)

    parser.add_argument("--oos-start", required=True)
    parser.add_argument("--oos-end", default="")

    parser.add_argument("--market-category", default=DEFAULT_MARKET_CATEGORY)
    parser.add_argument("--source", default=DEFAULT_SOURCE)

    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--m1-table", default=DEFAULT_M1_TABLE)
    parser.add_argument("--h4-table", default=DEFAULT_H4_TABLE)

    parser.add_argument("--dsn", default="")
    parser.add_argument("--dsn-env", default=DEFAULT_DSN_ENV)

    parser.add_argument(
        "--on-conflict",
        choices=["error", "skip", "update"],
        default="error",
        help="error = fail on existing candles; skip = do nothing on conflict; update = overwrite existing candle values.",
    )

    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json-out", default="")

    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    market_category = str(args.market_category).strip()
    source = str(args.source).strip()

    if not market_category:
        raise ValueError("empty market_category")
    if not source:
        raise ValueError("empty source")

    oos_start = parse_ts(args.oos_start)
    oos_end = parse_ts(args.oos_end) if str(args.oos_end).strip() else None

    m1_df = load_oos_frame(
        parquet_path=Path(str(args.m1_parquet)),
        symbol=symbol,
        market_category=market_category,
        source=source,
        oos_start=oos_start,
        oos_end=oos_end,
    )

    h4_df = load_oos_frame(
        parquet_path=Path(str(args.h4_parquet)),
        symbol=symbol,
        market_category=market_category,
        source=source,
        oos_start=oos_start,
        oos_end=oos_end,
    )

    dsn = str(args.dsn).strip() or str(os.environ.get(str(args.dsn_env), "")).strip()

    result: Dict[str, Any] = {
        "symbol": symbol,
        "market_category": market_category,
        "source": source,
        "schema": str(args.schema),
        "m1_table": str(args.m1_table),
        "h4_table": str(args.h4_table),
        "m1_parquet": str(args.m1_parquet),
        "h4_parquet": str(args.h4_parquet),
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat() if oos_end is not None else None,
        "on_conflict": str(args.on_conflict),
        "write": bool(args.write),
        "m1": frame_bounds(m1_df),
        "h4": frame_bounds(h4_df),
    }

    if dsn:
        psycopg2 = get_psycopg2()
        with psycopg2.connect(dsn) as conn:
            result["m1"].update(fetch_existing_stats(
                conn=conn,
                schema_name=str(args.schema),
                table_name=str(args.m1_table),
                symbol=symbol,
                market_category=market_category,
                oos_start=oos_start,
                oos_end=oos_end,
            ))
            result["h4"].update(fetch_existing_stats(
                conn=conn,
                schema_name=str(args.schema),
                table_name=str(args.h4_table),
                symbol=symbol,
                market_category=market_category,
                oos_start=oos_start,
                oos_end=oos_end,
            ))
    else:
        result["db_existing_stats"] = "skipped because DSN is empty"

    if args.write:
        if not dsn:
            raise RuntimeError("DB DSN is required for --write. Pass --dsn or set {}".format(args.dsn_env))

        if str(args.on_conflict) == "error":
            existing_m1 = int(result["m1"].get("existing_rows_in_period") or 0)
            existing_h4 = int(result["h4"].get("existing_rows_in_period") or 0)

            if existing_m1 > 0 or existing_h4 > 0:
                raise RuntimeError(
                    "Existing candles found in target period. "
                    "Refusing to write with --on-conflict error. "
                    "Use --on-conflict skip for idempotent append or --on-conflict update only if you intentionally want overwrite."
                )

        m1_written = write_frame_to_db(
            dsn=dsn,
            schema_name=str(args.schema),
            table_name=str(args.m1_table),
            df=m1_df,
            on_conflict=str(args.on_conflict),
        )

        h4_written = write_frame_to_db(
            dsn=dsn,
            schema_name=str(args.schema),
            table_name=str(args.h4_table),
            df=h4_df,
            on_conflict=str(args.on_conflict),
        )

        result["m1"]["attempted_write_rows"] = int(m1_written)
        result["h4"]["attempted_write_rows"] = int(h4_written)
        result["status"] = "WRITTEN"
    else:
        result["status"] = "DRY_RUN"

    text = json.dumps(result, ensure_ascii=True, indent=2)
    print(text)

    if str(args.json_out).strip():
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
