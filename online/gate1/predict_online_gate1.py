from __future__ import annotations

from online.trading import config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import traceback

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise RuntimeError(
        "catboost is not installed or cannot be imported. "
        "Install/activate the same project venv where CatBoost models were trained."
    ) from e


ROOT = Path(__file__).resolve().parents[2]

DB_DSN = config.DB_DSN

ONLINE_FEATURES_TABLE = "public.online_gate1_features"
ONLINE_PREDICTIONS_TABLE = "public.online_gate1_predictions"

GATE1_MODELS_DIR = ROOT / "production" / "models" / "final_gate1"

REPORT_DIR = ROOT / "online" / "_reports_gate1"
REPORT_CSV = REPORT_DIR / "online_gate1_predictions_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate1_predictions_report.json"

DEFAULT_THRESHOLD = 0.50

EXCLUDED_SYMBOLS = {
    "AGTUSDT",
    "GORKUSDT",
    "DMCUSDT",
    "MILKUSDT",
    "EPTUSDT",
    "A2ZUSDT",
    "OBTUSDT",
    "AINUSDT",
}


def connect_db():
    return psycopg2.connect(DB_DSN)


def parse_table_name(table_name: str) -> Tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) != 2:
        raise ValueError("table name must be schema.table, got: %s" % table_name)
    return parts[0], parts[1]


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_table(table_name: str) -> str:
    schema, table = parse_table_name(table_name)
    return quote_ident(schema) + "." + quote_ident(table)


def utc_now_str() -> str:
    return str(pd.Timestamp.now(tz="UTC").floor("s").tz_convert(None))


def to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert(None)


def to_db_utc_datetime(ts: Any) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("json root is not dict: %s" % path)
    return obj


def find_gate1_model_path(symbol: str) -> Path:
    base = GATE1_MODELS_DIR / symbol / "gate1"

    candidates = [
        base / "gate1_impulse_abs_move_atr_16h.cbm",
        base / "gate1.cbm",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = sorted(base.glob("*.cbm")) if base.exists() else []
    if len(found):
        return found[0]

    raise FileNotFoundError("cannot find Gate1 .cbm model for %s in %s" % (symbol, base))


def find_gate1_meta_path(symbol: str) -> Path:
    base = GATE1_MODELS_DIR / symbol / "gate1"

    candidates = [
        base / "meta.json",
        base / "report.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = sorted(base.glob("*.json")) if base.exists() else []
    if len(found):
        return found[0]

    raise FileNotFoundError("cannot find Gate1 meta json for %s in %s" % (symbol, base))


def extract_meta_feature_names(meta: Dict[str, Any]) -> List[str]:
    features = meta.get("features")

    if isinstance(features, dict):
        names = features.get("feature_names")
        if isinstance(names, list):
            return [str(x) for x in names]

    for key in ["feature_names", "features", "model_features"]:
        value = meta.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]

    raise RuntimeError("cannot find feature names in meta keys=%s" % list(meta.keys()))


def extract_threshold_from_obj(obj: Any) -> Optional[float]:
    if isinstance(obj, dict):
        priority_keys = [
            "threshold",
            "thr",
            "best_threshold",
            "selected_threshold",
            "proba_threshold",
            "gate1_threshold",
            "decision_threshold",
            "threshold_selected",
            "best_thr",
            "selected_thr",
        ]

        for key in priority_keys:
            if key in obj:
                try:
                    value = float(obj[key])
                    if np.isfinite(value):
                        return value
                except Exception:
                    pass

        for value in obj.values():
            found = extract_threshold_from_obj(value)
            if found is not None:
                return found

    if isinstance(obj, list):
        for value in obj:
            found = extract_threshold_from_obj(value)
            if found is not None:
                return found

    return None


def extract_threshold(meta: Dict[str, Any]) -> float:
    thresholding = meta.get("thresholding")
    found = extract_threshold_from_obj(thresholding)
    if found is not None:
        return float(found)

    found = extract_threshold_from_obj(meta)
    if found is not None:
        return float(found)

    return float(DEFAULT_THRESHOLD)


def get_model_feature_names(model: CatBoostClassifier, meta: Dict[str, Any]) -> List[str]:
    model_names = list(getattr(model, "feature_names_", []) or [])
    model_names = [str(x) for x in model_names if str(x).strip()]

    if len(model_names) > 0:
        return model_names

    return extract_meta_feature_names(meta)


def create_predictions_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS public.online_gate1_predictions (
        symbol text NOT NULL,
        entry_ts timestamptz NOT NULL,

        gate1_proba double precision NOT NULL,
        gate1_pred integer NOT NULL,
        gate1_threshold double precision NOT NULL,
        gate1_pass boolean NOT NULL,

        model_feature_count integer NOT NULL,
        model_path text NOT NULL,
        meta_path text NOT NULL,

        source_feature_table text NOT NULL DEFAULT 'public.online_gate1_features',
        prediction_builder text NOT NULL DEFAULT 'online/gate1/predict_online_gate1.py',

        online_inserted_at timestamptz NOT NULL DEFAULT now(),
        online_updated_at timestamptz NOT NULL DEFAULT now(),

        PRIMARY KEY (symbol, entry_ts)
    );

    CREATE INDEX IF NOT EXISTS idx_online_gate1_predictions_symbol_ts_desc
    ON public.online_gate1_predictions (symbol, entry_ts DESC);

    CREATE INDEX IF NOT EXISTS idx_online_gate1_predictions_entry_ts
    ON public.online_gate1_predictions (entry_ts);

    CREATE INDEX IF NOT EXISTS idx_online_gate1_predictions_pass
    ON public.online_gate1_predictions (gate1_pass);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def get_symbols_from_online_features() -> List[str]:
    query = """
        SELECT DISTINCT symbol
        FROM public.online_gate1_features
        ORDER BY symbol
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    symbols = [str(x).upper() for x in df["symbol"].tolist()]
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    return symbols

def load_missing_feature_rows_batch(
    symbols: List[str],
    rebuild: bool,
    limit_latest: Optional[int],
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    feature_table = sql_table(ONLINE_FEATURES_TABLE)

    if rebuild:
        query = """
            SELECT f.*
            FROM {feature_table} f
            WHERE UPPER(f.symbol) = ANY(%s)
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """.format(feature_table=feature_table)
        params = [symbols]
    else:
        query = """
            SELECT f.*
            FROM {feature_table} f
            LEFT JOIN public.online_gate1_predictions p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE UPPER(f.symbol) = ANY(%s)
              AND p.symbol IS NULL
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """.format(feature_table=feature_table)
        params = [symbols]

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["symbol", "entry_ts"]).sort_values(["symbol", "entry_ts"]).reset_index(drop=True)

    if limit_latest is not None and int(limit_latest) > 0:
        df = (
            df.groupby("symbol", group_keys=False)
            .tail(int(limit_latest))
            .sort_values(["symbol", "entry_ts"])
            .reset_index(drop=True)
        )

    return df


def split_feature_rows_by_symbol(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}

    for symbol, part in df.groupby("symbol", sort=True):
        result[str(symbol).upper()] = part.sort_values("entry_ts").reset_index(drop=True)

    return result


def get_missing_feature_rows(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    feature_table = sql_table(ONLINE_FEATURES_TABLE)

    if rebuild:
        query = """
            SELECT f.*
            FROM {feature_table} f
            WHERE f.symbol = %s
            ORDER BY f.entry_ts ASC
        """.format(feature_table=feature_table)
        params = [symbol]
    else:
        query = """
            SELECT f.*
            FROM {feature_table} f
            LEFT JOIN public.online_gate1_predictions p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE f.symbol = %s
              AND p.symbol IS NULL
            ORDER BY f.entry_ts ASC
        """.format(feature_table=feature_table)
        params = [symbol]

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    if limit_latest is not None and int(limit_latest) > 0 and len(df) > int(limit_latest):
        df = df.tail(int(limit_latest)).reset_index(drop=True)

    return df


def prepare_x(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise RuntimeError("missing model features in online table: %s" % missing[:50])

    x = df[feature_names].copy()

    for c in x.columns:
        if c == "symbol":
            x[c] = x[c].astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x


def predict_proba_positive(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    cat_features = []
    if "symbol" in x.columns:
        cat_features.append("symbol")

    pool = Pool(x, cat_features=cat_features if cat_features else None)
    pred = model.predict_proba(pool)

    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr.astype(float)

    if arr.shape[1] == 1:
        return arr[:, 0].astype(float)

    return arr[:, 1].astype(float)


def upsert_predictions(
    df_src: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    model_path: Path,
    meta_path: Path,
    feature_count: int,
) -> int:
    if df_src.empty:
        return 0

    records = []

    for i, row in enumerate(df_src.itertuples(index=False)):
        p = float(proba[i])
        pred = int(p >= float(threshold))
        gate_pass = bool(pred == 1)

        records.append(
            (
                str(getattr(row, "symbol")).upper(),
                to_db_utc_datetime(getattr(row, "entry_ts")),
                p,
                pred,
                float(threshold),
                gate_pass,
                int(feature_count),
                str(model_path),
                str(meta_path),
                ONLINE_FEATURES_TABLE,
            )
        )

    sql = """
        INSERT INTO public.online_gate1_predictions (
            symbol,
            entry_ts,
            gate1_proba,
            gate1_pred,
            gate1_threshold,
            gate1_pass,
            model_feature_count,
            model_path,
            meta_path,
            source_feature_table
        )
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            gate1_proba = EXCLUDED.gate1_proba,
            gate1_pred = EXCLUDED.gate1_pred,
            gate1_threshold = EXCLUDED.gate1_threshold,
            gate1_pass = EXCLUDED.gate1_pass,
            model_feature_count = EXCLUDED.model_feature_count,
            model_path = EXCLUDED.model_path,
            meta_path = EXCLUDED.meta_path,
            source_feature_table = EXCLUDED.source_feature_table,
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=5000)
        conn.commit()

    return len(records)


def predict_symbol(
    symbol: str,
    feature_rows_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "symbol": symbol,
        "status": "init",
        "feature_rows_to_predict": 0,
        "inserted": 0,
        "gate1_pass_rows": 0,
        "gate1_fail_rows": 0,
        "proba_min": None,
        "proba_max": None,
        "proba_mean": None,
        "threshold": None,
        "model_feature_count": None,
        "model_path": "",
        "meta_path": "",
        "error": "",
    }

    try:
        model_path = find_gate1_model_path(symbol)
        meta_path = find_gate1_meta_path(symbol)

        report["model_path"] = str(model_path)
        report["meta_path"] = str(meta_path)

        df = feature_rows_by_symbol.get(symbol)
        if df is None or df.empty:
            report["status"] = "no_missing_rows"
            return report

        report["feature_rows_to_predict"] = int(len(df))

        meta = read_json(meta_path)

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        feature_names = get_model_feature_names(model, meta)
        threshold = extract_threshold(meta)

        report["threshold"] = float(threshold)
        report["model_feature_count"] = int(len(feature_names))

        x = prepare_x(df, feature_names)
        proba = predict_proba_positive(model, x)

        inserted = upsert_predictions(
            df_src=df,
            proba=proba,
            threshold=threshold,
            model_path=model_path,
            meta_path=meta_path,
            feature_count=len(feature_names),
        )

        gate_pass = proba >= float(threshold)

        report["inserted"] = int(inserted)
        report["gate1_pass_rows"] = int(gate_pass.sum())
        report["gate1_fail_rows"] = int((~gate_pass).sum())
        report["proba_min"] = float(np.nanmin(proba)) if len(proba) else None
        report["proba_max"] = float(np.nanmax(proba)) if len(proba) else None
        report["proba_mean"] = float(np.nanmean(proba)) if len(proba) else None
        report["status"] = "ok"
        return report

    except Exception as e:
        report["status"] = "error"
        report["error"] = "%s: %s" % (type(e).__name__, str(e))
        report["traceback"] = traceback.format_exc()
        return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="", help="optional single symbol, e.g. BTCUSDT")
    ap.add_argument("--rebuild", action="store_true", help="re-score all existing online_gate1_features rows")
    ap.add_argument("--limit-latest", type=int, default=0, help="optional limit of latest rows per symbol")
    args = ap.parse_args()

    limit_latest = int(args.limit_latest) if int(args.limit_latest) > 0 else None

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("ONLINE_FEATURES_TABLE:", ONLINE_FEATURES_TABLE)
    print("ONLINE_PREDICTIONS_TABLE:", ONLINE_PREDICTIONS_TABLE)
    print("GATE1_MODELS_DIR:", GATE1_MODELS_DIR)
    print("REBUILD:", bool(args.rebuild))
    print("LIMIT_LATEST:", limit_latest)
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    create_predictions_table()

    if args.symbol.strip():
        symbols = [args.symbol.strip().upper()]
    else:
        symbols = get_symbols_from_online_features()

    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: missing online_gate1_features rows")
    print()

    missing_features = load_missing_feature_rows_batch(
        symbols=symbols,
        rebuild=bool(args.rebuild),
        limit_latest=limit_latest,
    )
    feature_rows_by_symbol = split_feature_rows_by_symbol(missing_features)

    print("MISSING_FEATURE_ROWS_TOTAL:", len(missing_features))
    print("MISSING_FEATURE_SYMBOLS:", len(feature_rows_by_symbol))
    print()

    reports: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols, start=1):
        print("[%d/%d] %s" % (idx, len(symbols), symbol))

        rep = predict_symbol(
            symbol=symbol,
            feature_rows_by_symbol=feature_rows_by_symbol,
        )
        reports.append(rep)

        if rep["status"] == "ok":
            print(
                "    status=ok | rows=%s | inserted=%s | pass=%s | fail=%s | thr=%s | proba_mean=%s"
                % (
                    rep["feature_rows_to_predict"],
                    rep["inserted"],
                    rep["gate1_pass_rows"],
                    rep["gate1_fail_rows"],
                    rep["threshold"],
                    rep["proba_mean"],
                )
            )
        elif rep["status"] == "no_missing_rows":
            print("    status=no_missing_rows")
        else:
            print("    ERROR:", rep.get("error", ""))

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    status_counts = rep_df["status"].value_counts(dropna=False).to_dict() if len(rep_df) else {}

    summary = {
        "created_at_utc": utc_now_str(),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "online_features_table": ONLINE_FEATURES_TABLE,
        "online_predictions_table": ONLINE_PREDICTIONS_TABLE,
        "gate1_models_dir": str(GATE1_MODELS_DIR),
        "rebuild": bool(args.rebuild),
        "limit_latest": limit_latest,
        "symbols_count": int(len(symbols)),
        "status_counts": status_counts,
        "total_rows_predicted": int(sum(int(r.get("feature_rows_to_predict", 0)) for r in reports)),
        "total_inserted": int(sum(int(r.get("inserted", 0)) for r in reports)),
        "total_gate1_pass": int(sum(int(r.get("gate1_pass_rows", 0)) for r in reports)),
        "total_gate1_fail": int(sum(int(r.get("gate1_fail_rows", 0)) for r in reports)),
        "report_csv": str(REPORT_CSV),
        "report_json": str(REPORT_JSON),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", status_counts)
    print("TOTAL ROWS PREDICTED:", summary["total_rows_predicted"])
    print("TOTAL INSERTED:", summary["total_inserted"])
    print("TOTAL GATE1 PASS:", summary["total_gate1_pass"])
    print("TOTAL GATE1 FAIL:", summary["total_gate1_fail"])
    print("WROTE:", REPORT_CSV)
    print("WROTE:", REPORT_JSON)


if __name__ == "__main__":
    main()
