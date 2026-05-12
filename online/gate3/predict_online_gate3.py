from __future__ import annotations

from online.trading import config
from pathlib import Path
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise RuntimeError(
        "catboost is not installed or cannot be imported. "
        "Activate project venv before running this script."
    ) from e


ROOT = Path(os.environ.get("IMB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DB_DSN = config.DB_DSN

ONLINE_GATE3_FEATURES_TABLE = "public.online_gate3_features"
ONLINE_GATE3_PREDICTIONS_TABLE = "public.online_gate3_predictions"

GATE3_SCORE_ROOT = ROOT / "production" / "models" / "final_gate3_score_long_short"
POLICY_CSV = ROOT / "production" / "models" / "ks" / "gate3_symbol_policy.csv.updated"

REPORT_DIR = ROOT / "online" / "_reports_gate3"
REPORT_CSV = REPORT_DIR / "online_gate3_predictions_report.csv"
REPORT_JSON = REPORT_DIR / "online_gate3_predictions_report.json"

SOURCE_NAME = "online_gate3_score_predictions_v1"
PREDICTION_BUILDER = "online/gate3/predict_online_gate3.py"

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


def split_table_name(table_name: str) -> Tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) != 2:
        raise RuntimeError("table name must be schema.table")
    return parts[0], parts[1]


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def table_qname(table_name: str) -> str:
    schema, table = split_table_name(table_name)
    return schema + "." + qident(table)


def utc_now_floor_second() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s").tz_convert(None)


def to_db_utc_datetime(value: Any) -> Optional[datetime]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def table_exists(table_name: str) -> bool:
    schema, table = split_table_name(table_name)
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (schema, table))
            return bool(cur.fetchone()[0])


def get_table_columns(table_name: str) -> List[str]:
    schema, table = split_table_name(table_name)
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """
    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(schema, table))
    return [str(x) for x in df["column_name"].tolist()]


def load_json_safe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def get_symbols_from_features() -> List[str]:
    query = f"""
        SELECT DISTINCT symbol
        FROM {ONLINE_GATE3_FEATURES_TABLE}
        ORDER BY symbol
    """
    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return []

    symbols = [str(x).upper() for x in df["symbol"].tolist()]
    symbols = [s for s in symbols if s not in EXCLUDED_SYMBOLS]
    return sorted(set(symbols))


def get_missing_feature_rows(symbol: str, rebuild: bool, limit_latest: Optional[int]) -> pd.DataFrame:
    if rebuild or not table_exists(ONLINE_GATE3_PREDICTIONS_TABLE):
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE3_FEATURES_TABLE} f
            WHERE f.symbol = %s
            ORDER BY f.entry_ts ASC
        """
        params = [symbol]
    else:
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE3_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE3_PREDICTIONS_TABLE} p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE f.symbol = %s
              AND p.entry_ts IS NULL
            ORDER BY f.entry_ts ASC
        """
        params = [symbol]

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    if limit_latest is not None and int(limit_latest) > 0 and len(df) > int(limit_latest):
        df = df.tail(int(limit_latest)).reset_index(drop=True)

    return df



def empty_missing_feature_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "entry_ts"])


def normalize_missing_feature_rows(df: pd.DataFrame, limit_latest: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return empty_missing_feature_rows()

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["symbol", "entry_ts"])
    out = out[~out["symbol"].isin(EXCLUDED_SYMBOLS)].copy()
    out = out.sort_values(["symbol", "entry_ts"]).drop_duplicates(["symbol", "entry_ts"], keep="last")

    if limit_latest is not None and int(limit_latest) > 0:
        out = (
            out.groupby("symbol", group_keys=False)
            .tail(int(limit_latest))
            .reset_index(drop=True)
        )

    return out.reset_index(drop=True)


def load_missing_feature_rows_batch(
    symbols: List[str],
    rebuild: bool,
    limit_latest: Optional[int],
) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    symbols_clean = sorted(set(str(x).upper() for x in symbols if str(x).upper() not in EXCLUDED_SYMBOLS))

    if not symbols_clean:
        return {}

    if rebuild or not table_exists(ONLINE_GATE3_PREDICTIONS_TABLE):
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE3_FEATURES_TABLE} f
            WHERE f.symbol = ANY(%s)
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """
    else:
        query = f"""
            SELECT f.symbol, f.entry_ts
            FROM {ONLINE_GATE3_FEATURES_TABLE} f
            LEFT JOIN {ONLINE_GATE3_PREDICTIONS_TABLE} p
              ON p.symbol = f.symbol
             AND p.entry_ts = f.entry_ts
            WHERE f.symbol = ANY(%s)
              AND p.entry_ts IS NULL
            ORDER BY f.symbol ASC, f.entry_ts ASC
        """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=[symbols_clean])

    df = normalize_missing_feature_rows(df, limit_latest=limit_latest)

    result: Dict[str, pd.DataFrame] = {}
    for symbol in symbols_clean:
        part = df[df["symbol"] == symbol].copy()
        result[symbol] = part.reset_index(drop=True) if not part.empty else empty_missing_feature_rows()

    return result


def load_feature_context_batch(missing_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    bounds = []

    for symbol, missing in missing_by_symbol.items():
        if missing is None or missing.empty:
            continue

        max_ts = pd.to_datetime(missing["entry_ts"], utc=True, errors="coerce").max()
        if pd.isna(max_ts):
            continue

        bounds.append((str(symbol).upper(), to_db_utc_datetime(max_ts)))

    if not bounds:
        return {}

    query = f"""
        SELECT f.*
        FROM {ONLINE_GATE3_FEATURES_TABLE} f
        INNER JOIN (VALUES %s) AS b(symbol, max_ts)
            ON f.symbol = b.symbol
           AND f.entry_ts <= b.max_ts::timestamptz
        ORDER BY f.symbol ASC, f.entry_ts ASC
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            rows = execute_values(cur, query, bounds, fetch=True)
            cols = [desc[0] for desc in cur.description]

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        return {}

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = (
        df.dropna(subset=["symbol", "entry_ts"])
        .sort_values(["symbol", "entry_ts"])
        .drop_duplicates(["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    result: Dict[str, pd.DataFrame] = {}
    for symbol, part in df.groupby("symbol", sort=False):
        result[str(symbol).upper()] = part.copy().reset_index(drop=True)

    return result


def load_feature_context(symbol: str, max_ts: pd.Timestamp) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {ONLINE_GATE3_FEATURES_TABLE}
        WHERE symbol = %s
          AND entry_ts <= %s
        ORDER BY entry_ts ASC
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=[symbol, to_db_utc_datetime(max_ts)])

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").drop_duplicates(["symbol", "entry_ts"], keep="last").reset_index(drop=True)
    return df


def gate3_long_model_path(symbol: str) -> Path:
    return GATE3_SCORE_ROOT / symbol / "long" / "gate3_score" / "gate3_score.cbm"


def gate3_short_model_path(symbol: str) -> Path:
    return GATE3_SCORE_ROOT / symbol / "short" / "gate3_score" / "gate3_score.cbm"


def gate3_long_meta_path(symbol: str) -> Path:
    return GATE3_SCORE_ROOT / symbol / "long" / "gate3_score" / "meta.json"


def gate3_short_meta_path(symbol: str) -> Path:
    return GATE3_SCORE_ROOT / symbol / "short" / "gate3_score" / "meta.json"


def load_policy() -> pd.DataFrame:
    if not POLICY_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(POLICY_CSV)
    if "symbol" not in df.columns:
        return pd.DataFrame()

    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def policy_row_for_symbol(policy: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    if policy.empty:
        return {}
    part = policy[policy["symbol"] == symbol]
    if part.empty:
        return {}
    return part.tail(1).iloc[0].to_dict()


def numeric_from_obj(obj: Any, default: float) -> float:
    val = pd.to_numeric(obj, errors="coerce")
    if pd.isna(val) or not np.isfinite(float(val)):
        return float(default)
    return float(val)


def extract_threshold_from_meta(meta: Dict[str, Any], fallback_thr: float) -> float:
    for key in [
        "best_threshold",
        "threshold",
        "score_threshold",
        "selected_threshold",
    ]:
        if key in meta:
            val = numeric_from_obj(meta.get(key), np.nan)
            if np.isfinite(val):
                return float(val)

    return float(fallback_thr)


def extract_gate3_meta_features(meta: Dict[str, Any], fallback_thr: float) -> Dict[str, float]:
    stats = meta.get("stats", {}) if isinstance(meta.get("stats", {}), dict) else {}

    threshold = extract_threshold_from_meta(meta, fallback_thr=fallback_thr)

    return {
        "threshold": float(threshold) if np.isfinite(threshold) else float(DEFAULT_THRESHOLD),
        "precision_meta": numeric_from_obj(stats.get("precision", np.nan), np.nan),
        "wilson_meta": numeric_from_obj(stats.get("wilson_lower", np.nan), np.nan),
        "delta_wilson_meta": numeric_from_obj(stats.get("delta_wilson", np.nan), np.nan),
        "pvalue_meta": numeric_from_obj(stats.get("p_value", np.nan), np.nan),
        "kept_n_meta": numeric_from_obj(meta.get("best_threshold_kept_n", np.nan), np.nan),
        "valid_pos_rate_meta": numeric_from_obj(meta.get("best_threshold_kept_pos_rate", np.nan), np.nan),
        "thr_kept_lift_meta": numeric_from_obj(meta.get("best_threshold_kept_lift", np.nan), np.nan),
    }


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0.5).astype(int)


def select_side_pattern_cols(cols: List[str], side: str) -> List[str]:
    side = str(side).strip().lower()
    out = []

    for c in cols:
        c_low = str(c).lower()

        is_long = ("_up" in c_low) or ("buy" in c_low) or ("_lo_" in c_low)
        is_short = ("_dn" in c_low) or ("sell" in c_low) or ("_hi_" in c_low)

        if side == "long" and is_long and not is_short:
            out.append(c)

        if side == "short" and is_short and not is_long:
            out.append(c)

    return sorted(set(out))


def add_symbol_dummies(df: pd.DataFrame, symbol: str, required_features: List[str]) -> pd.DataFrame:
    sym_cols = list(dict.fromkeys([c for c in required_features if str(c).startswith("sym_")]))
    if not sym_cols:
        return df

    values = {}

    for c in sym_cols:
        if c in df.columns:
            continue

        if c == "sym_is_other":
            continue

        name = c[4:]
        values[c] = np.full(len(df), int(symbol == name), dtype=np.int8)

    if "sym_is_other" in sym_cols and "sym_is_other" not in df.columns:
        known_symbols = {c[4:] for c in sym_cols if c != "sym_is_other"}
        values["sym_is_other"] = np.full(len(df), int(symbol not in known_symbols), dtype=np.int8)

    if not values:
        return df

    block = pd.DataFrame(values, index=df.index)
    return pd.concat([df, block], axis=1).copy()


def rolling_wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window, min_periods=window).apply(_wma, raw=True)


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )

    return pd.Series(tr, index=df.index).rolling(14, min_periods=14).mean()


def add_common_missing_features(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    need = set(required_features)
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce") if "close" in out.columns else pd.Series(np.nan, index=out.index)
    open_ = pd.to_numeric(out["open"], errors="coerce") if "open" in out.columns else pd.Series(np.nan, index=out.index)
    high = pd.to_numeric(out["high"], errors="coerce") if "high" in out.columns else pd.Series(np.nan, index=out.index)
    low = pd.to_numeric(out["low"], errors="coerce") if "low" in out.columns else pd.Series(np.nan, index=out.index)
    volume = pd.to_numeric(out["volume"], errors="coerce") if "volume" in out.columns else pd.Series(np.nan, index=out.index)

    if "ret" in need and "ret" not in out.columns:
        out["ret"] = close.pct_change().replace([np.inf, -np.inf], np.nan)

    if "side_num" in need and "side_num" not in out.columns:
        out["side_num"] = 0.0

    if "dir_prev" in need and "dir_prev" not in out.columns:
        prev_body = (close.shift(1) - open_.shift(1)).fillna(0.0)
        out["dir_prev"] = np.sign(prev_body).astype(float)

    if "sma5" in need and "sma5" not in out.columns:
        out["sma5"] = close.rolling(5, min_periods=5).mean()

    if "sma10" in need and "sma10" not in out.columns:
        out["sma10"] = close.rolling(10, min_periods=10).mean()

    if "wma10" in need and "wma10" not in out.columns:
        out["wma10"] = rolling_wma(close, 10)

    if "vol_delta" in need and "vol_delta" not in out.columns:
        out["vol_delta"] = volume.diff()

    if "vol_med20" in need and "vol_med20" not in out.columns:
        out["vol_med20"] = volume.rolling(20, min_periods=20).median()

    if "vol_med48" in need and "vol_med48" not in out.columns:
        out["vol_med48"] = volume.rolling(48, min_periods=48).median()

    if "vol_z" in need and "vol_z" not in out.columns:
        vol_mean20 = volume.rolling(20, min_periods=20).mean()
        vol_std20 = volume.rolling(20, min_periods=20).std()
        out["vol_z"] = (volume - vol_mean20) / vol_std20.replace(0.0, np.nan)

    if "obv" in need and "obv" not in out.columns:
        direction = np.sign(close.diff().fillna(0.0))
        out["obv"] = (direction * volume.fillna(0.0)).cumsum()

    if "vwap20" in need and "vwap20" not in out.columns:
        typical = (high + low + close) / 3.0
        num = (typical * volume).rolling(20, min_periods=20).sum()
        den = volume.rolling(20, min_periods=20).sum()
        out["vwap20"] = num / den.replace(0.0, np.nan)

    if "atr14" not in out.columns and {"high", "low", "close"}.issubset(out.columns):
        out["atr14"] = compute_atr14(out)

    if "atr_slope" in need and "atr_slope" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["atr_slope"] = atr14.diff()

    if "body_vs_wick" in need and "body_vs_wick" not in out.columns:
        body = (close - open_).abs()
        upper = high - np.maximum(open_, close)
        lower = np.minimum(open_, close) - low
        wick = upper + lower
        out["body_vs_wick"] = body / wick.replace(0.0, np.nan)

    if "zero_vol_share48" in need and "zero_vol_share48" not in out.columns:
        out["zero_vol_share48"] = (volume.fillna(0.0) <= 0.0).astype(float).rolling(48, min_periods=1).mean()

    if "ctx_ret1" in need and "ctx_ret1" not in out.columns:
        out["ctx_ret1"] = close.pct_change(1).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret2" in need and "ctx_ret2" not in out.columns:
        out["ctx_ret2"] = close.pct_change(2).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret4" in need and "ctx_ret4" not in out.columns:
        out["ctx_ret4"] = close.pct_change(4).replace([np.inf, -np.inf], np.nan)

    if "ctx_ret8" in need and "ctx_ret8" not in out.columns:
        out["ctx_ret8"] = close.pct_change(8).replace([np.inf, -np.inf], np.nan)

    if "ctx_atrp14" in need and "ctx_atrp14" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["ctx_atrp14"] = atr14 / close.replace(0.0, np.nan)

    if "ctx_range_atr" in need and "ctx_range_atr" not in out.columns:
        atr14 = pd.to_numeric(out["atr14"], errors="coerce")
        out["ctx_range_atr"] = (high - low) / atr14.replace(0.0, np.nan)

    return out


def add_lag_features_if_needed(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    lag_feats = [c for c in required_features if str(c).endswith("_lag1") or str(c).endswith("_lag2")]
    if not lag_feats:
        return df

    block = pd.DataFrame(index=df.index)

    for feat in lag_feats:
        if feat.endswith("_lag1"):
            base = feat[:-5]
            if base in df.columns:
                block[feat] = pd.to_numeric(df[base], errors="coerce").shift(1)

        if feat.endswith("_lag2"):
            base = feat[:-5]
            if base in df.columns:
                block[feat] = pd.to_numeric(df[base], errors="coerce").shift(2)

    if block.empty:
        return df

    return pd.concat([df, block], axis=1).copy()


def add_active_set_features(df: pd.DataFrame, active_cols: List[str], prefix: str) -> pd.DataFrame:
    block = pd.DataFrame(index=df.index)

    if prefix == "g3_long":
        primary_col = "active_pa_atr_squeeze_break_up"
        secondary_col = "active_pa_bos_up_24"
    elif prefix == "g3_short":
        primary_col = "active_pa_atr_squeeze_break_dn"
        secondary_col = "active_pa_bos_dn_24"
    else:
        primary_col = ""
        secondary_col = ""

    if not active_cols:
        block[f"{prefix}_any_active"] = 0
        block[f"{prefix}_active_count"] = 0
        block[f"{prefix}_active_primary"] = 0
        block[f"{prefix}_active_secondary"] = 0
        block[f"{prefix}_active_overlap_primary_secondary"] = 0
        block[f"{prefix}_max_active_age"] = 0
        return pd.concat([df, block], axis=1).copy()

    act = df[active_cols].copy()
    for c in active_cols:
        act[c] = safe_bool_series(act[c])

    act_sum = act.sum(axis=1)

    block[f"{prefix}_any_active"] = (act_sum > 0).astype(int)
    block[f"{prefix}_active_count"] = act_sum.astype(int)

    if primary_col and primary_col in act.columns:
        block[f"{prefix}_active_primary"] = act[primary_col].astype(int)
    else:
        block[f"{prefix}_active_primary"] = 0

    if secondary_col and secondary_col in act.columns:
        block[f"{prefix}_active_secondary"] = act[secondary_col].astype(int)
    else:
        block[f"{prefix}_active_secondary"] = 0

    block[f"{prefix}_active_overlap_primary_secondary"] = (
        (block[f"{prefix}_active_primary"] == 1) &
        (block[f"{prefix}_active_secondary"] == 1)
    ).astype(int)

    max_age = np.zeros(len(df), dtype=int)

    for c in active_cols:
        x = act[c].to_numpy(dtype=int)
        age = np.zeros(len(x), dtype=int)
        run = 0

        for i in range(len(x)):
            if x[i] == 1:
                run += 1
            else:
                run = 0
            age[i] = run

        max_age = np.maximum(max_age, age)

    block[f"{prefix}_max_active_age"] = max_age

    return pd.concat([df, block], axis=1).copy()


def prepare_model_input(
    base_df: pd.DataFrame,
    symbol: str,
    required_features: List[str],
    active_cols: Optional[List[str]],
    active_prefix: Optional[str],
) -> pd.DataFrame:
    df = base_df.copy()
    required_features = list(dict.fromkeys(required_features))

    df = add_common_missing_features(df=df, required_features=required_features)
    df = add_symbol_dummies(df=df, symbol=symbol, required_features=required_features)

    if active_cols and active_prefix:
        need_active = False

        for feat in required_features:
            if str(feat).startswith(f"{active_prefix}_"):
                need_active = True
                break
            if "__age" in str(feat) or "__fresh" in str(feat) or "__mid" in str(feat) or "__late" in str(feat):
                need_active = True
                break

        if need_active:
            df = add_active_set_features(df=df, active_cols=active_cols, prefix=active_prefix)

    df = add_lag_features_if_needed(df=df, required_features=required_features)

    missing_cols = [c for c in required_features if c not in df.columns]
    if missing_cols:
        missing_block = pd.DataFrame({c: np.nan for c in missing_cols}, index=df.index)
        df = pd.concat([df, missing_block], axis=1).copy()

    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def load_model_features(model_path: Path) -> Tuple[Optional[CatBoostClassifier], List[str], str]:
    if not model_path.exists():
        return None, [], "missing_model"

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    feats = list(dict.fromkeys(model.feature_names_ or []))
    if not feats:
        return model, [], "empty_model_features"

    return model, feats, "ok"


def predict_model_proba(
    model: CatBoostClassifier,
    prepared_df: pd.DataFrame,
    feature_names: List[str],
) -> np.ndarray:
    x = prepared_df[feature_names].replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, ~x.columns.duplicated(keep="first")].copy()

    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce")

    med = x.median(numeric_only=True)
    x = x.fillna(med)
    x = x.fillna(0.0)

    return model.predict_proba(x)[:, 1]


def create_predictions_table() -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {ONLINE_GATE3_PREDICTIONS_TABLE} (
            symbol text NOT NULL,
            entry_ts timestamptz NOT NULL,

            g3_long_score_proba double precision,
            g3_short_score_proba double precision,
            g3_long_score_pass boolean NOT NULL DEFAULT false,
            g3_short_score_pass boolean NOT NULL DEFAULT false,

            gate3_proba_long double precision,
            gate3_proba_short double precision,
            gate3_pass_long boolean NOT NULL DEFAULT false,
            gate3_pass_short boolean NOT NULL DEFAULT false,
            gate3_any_pass boolean NOT NULL DEFAULT false,
            gate3_best_side text NOT NULL DEFAULT '',
            gate3_best_proba double precision,
            gate3_margin_long double precision,
            gate3_margin_short double precision,

            gate3_threshold_long double precision,
            gate3_threshold_short double precision,

            g3_score_spread double precision,
            g3_score_abs_spread double precision,
            g3_score_max double precision,

            gate3_precision_meta_long double precision,
            gate3_wilson_meta_long double precision,
            gate3_delta_wilson_meta_long double precision,
            gate3_pvalue_meta_long double precision,
            gate3_kept_n_meta_long double precision,
            gate3_valid_pos_rate_meta_long double precision,
            gate3_thr_kept_lift_meta_long double precision,

            gate3_precision_meta_short double precision,
            gate3_wilson_meta_short double precision,
            gate3_delta_wilson_meta_short double precision,
            gate3_pvalue_meta_short double precision,
            gate3_kept_n_meta_short double precision,
            gate3_valid_pos_rate_meta_short double precision,
            gate3_thr_kept_lift_meta_short double precision,

            gate3_precision_meta double precision,
            gate3_wilson_meta double precision,
            gate3_delta_wilson_meta double precision,
            gate3_pvalue_meta double precision,
            gate3_kept_n_meta double precision,
            gate3_valid_pos_rate_meta double precision,
            gate3_thr_kept_lift_meta double precision,

            has_gate3_long_bundle boolean NOT NULL DEFAULT false,
            has_gate3_short_bundle boolean NOT NULL DEFAULT false,
            has_any_gate3_bundle boolean NOT NULL DEFAULT false,
            has_full_gate3_bundle boolean NOT NULL DEFAULT false,

            long_model_path text NOT NULL DEFAULT '',
            short_model_path text NOT NULL DEFAULT '',
            long_meta_path text NOT NULL DEFAULT '',
            short_meta_path text NOT NULL DEFAULT '',
            long_feature_count integer NOT NULL DEFAULT 0,
            short_feature_count integer NOT NULL DEFAULT 0,

            online_source text NOT NULL DEFAULT '{SOURCE_NAME}',
            online_prediction_builder text NOT NULL DEFAULT '{PREDICTION_BUILDER}',
            online_inserted_at timestamptz NOT NULL DEFAULT now(),
            online_updated_at timestamptz NOT NULL DEFAULT now(),

            PRIMARY KEY (symbol, entry_ts)
        );

        CREATE INDEX IF NOT EXISTS idx_online_gate3_predictions_symbol_ts_desc
        ON public.online_gate3_predictions (symbol, entry_ts DESC);

        CREATE INDEX IF NOT EXISTS idx_online_gate3_predictions_entry_ts
        ON public.online_gate3_predictions (entry_ts);
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def py_value(v: Any) -> Any:
    if isinstance(v, (pd.Timestamp, datetime, np.datetime64)):
        return to_db_utc_datetime(v)

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, np.bool_):
        return bool(v)

    if isinstance(v, bool):
        return bool(v)

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, int):
        return int(v)

    if isinstance(v, np.floating):
        return float(v)

    if isinstance(v, float):
        return float(v)

    return v


def upsert_predictions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    create_predictions_table()

    table_cols = get_table_columns(ONLINE_GATE3_PREDICTIONS_TABLE)
    cols = [c for c in df.columns if c in table_cols]

    records = []
    for row in df[cols].itertuples(index=False, name=None):
        records.append(tuple(py_value(v) for v in row))

    quoted_cols = ", ".join(qident(c) for c in cols)

    update_cols = [
        c for c in cols
        if c not in {"symbol", "entry_ts", "online_inserted_at", "online_updated_at"}
    ]

    update_sql = ", ".join(
        f"{qident(c)} = EXCLUDED.{qident(c)}"
        for c in update_cols
    )

    sql = f"""
        INSERT INTO {ONLINE_GATE3_PREDICTIONS_TABLE} ({quoted_cols})
        VALUES %s
        ON CONFLICT (symbol, entry_ts)
        DO UPDATE SET
            {update_sql},
            online_updated_at = now()
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=5000)
        conn.commit()

    return len(records)


def empty_gate3_meta_features() -> Dict[str, float]:
    return {
        "threshold": np.nan,
        "precision_meta": np.nan,
        "wilson_meta": np.nan,
        "delta_wilson_meta": np.nan,
        "pvalue_meta": np.nan,
        "kept_n_meta": np.nan,
        "valid_pos_rate_meta": np.nan,
        "thr_kept_lift_meta": np.nan,
    }


def build_predictions_for_symbol(
    symbol: str,
    missing_df: pd.DataFrame,
    policy_row: Dict[str, Any],
    feature_context_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report = {
        "symbol": symbol,
        "status": "init",
        "feature_rows_to_predict": int(len(missing_df)),
        "context_rows": 0,
        "rows_predicted": 0,
        "inserted": 0,
        "has_gate3_long_bundle": 0,
        "has_gate3_short_bundle": 0,
        "long_feature_count": 0,
        "short_feature_count": 0,
        "err": "",
    }

    if missing_df.empty:
        report["status"] = "no_missing"
        return pd.DataFrame(), report

    max_ts = pd.Timestamp(missing_df["entry_ts"].max())

    if feature_context_df is None:
        features = load_feature_context(symbol=symbol, max_ts=max_ts)
    else:
        features = feature_context_df.copy()

    report["context_rows"] = int(len(features))

    if features.empty:
        report["status"] = "missing_online_gate3_features"
        return pd.DataFrame(), report

    target_ts = set(pd.Timestamp(x) for x in missing_df["entry_ts"].tolist())
    target_mask = features["entry_ts"].isin(target_ts)

    if int(target_mask.sum()) == 0:
        report["status"] = "target_rows_not_in_context"
        return pd.DataFrame(), report

    long_model_path = gate3_long_model_path(symbol)
    short_model_path = gate3_short_model_path(symbol)
    long_meta_path = gate3_long_meta_path(symbol)
    short_meta_path = gate3_short_meta_path(symbol)

    long_model, long_feats, long_model_status = load_model_features(long_model_path)
    short_model, short_feats, short_model_status = load_model_features(short_model_path)

    long_use_score = int(numeric_from_obj(policy_row.get("gate3_use_score_model_long", 1), 1))
    short_use_score = int(numeric_from_obj(policy_row.get("gate3_use_score_model_short", 1), 1))

    long_thr_policy = numeric_from_obj(policy_row.get("gate3_score_threshold_long", DEFAULT_THRESHOLD), DEFAULT_THRESHOLD)
    short_thr_policy = numeric_from_obj(policy_row.get("gate3_score_threshold_short", DEFAULT_THRESHOLD), DEFAULT_THRESHOLD)

    has_long_bundle = int(long_model is not None and long_model_status == "ok" and long_use_score == 1)
    has_short_bundle = int(short_model is not None and short_model_status == "ok" and short_use_score == 1)
    has_any_bundle = int(has_long_bundle or has_short_bundle)
    has_full_bundle = int(has_long_bundle and has_short_bundle)

    report["has_gate3_long_bundle"] = int(has_long_bundle)
    report["has_gate3_short_bundle"] = int(has_short_bundle)
    report["long_feature_count"] = int(len(long_feats)) if has_long_bundle else 0
    report["short_feature_count"] = int(len(short_feats)) if has_short_bundle else 0

    if has_long_bundle:
        long_meta = extract_gate3_meta_features(
            meta=load_json_safe(long_meta_path),
            fallback_thr=long_thr_policy,
        )
        long_thr = float(long_meta["threshold"])
    else:
        long_meta = empty_gate3_meta_features()
        long_thr = np.nan

    if has_short_bundle:
        short_meta = extract_gate3_meta_features(
            meta=load_json_safe(short_meta_path),
            fallback_thr=short_thr_policy,
        )
        short_thr = float(short_meta["threshold"])
    else:
        short_meta = empty_gate3_meta_features()
        short_thr = np.nan

    all_active_cols = sorted([c for c in features.columns if str(c).startswith("active_pa_")])
    long_active_cols = select_side_pattern_cols(all_active_cols, "long")
    short_active_cols = select_side_pattern_cols(all_active_cols, "short")

    work = features.copy()

    for c in work.columns:
        if c in {"symbol", "entry_ts"}:
            continue
        if pd.api.types.is_numeric_dtype(work[c]):
            continue
        work[c] = pd.to_numeric(work[c], errors="coerce")

    long_proba = np.full(len(work), np.nan, dtype=float)
    short_proba = np.full(len(work), np.nan, dtype=float)

    if has_long_bundle:
        long_input = prepare_model_input(
            base_df=work,
            symbol=symbol,
            required_features=long_feats,
            active_cols=long_active_cols,
            active_prefix="g3_long",
        )
        missing_long = [c for c in long_feats if c not in long_input.columns]
        if missing_long:
            report["status"] = "g3_long_missing_features"
            report["err"] = ",".join(missing_long[:30])
            return pd.DataFrame(), report

        long_proba = predict_model_proba(long_model, long_input, long_feats)

    if has_short_bundle:
        short_input = prepare_model_input(
            base_df=work,
            symbol=symbol,
            required_features=short_feats,
            active_cols=short_active_cols,
            active_prefix="g3_short",
        )
        missing_short = [c for c in short_feats if c not in short_input.columns]
        if missing_short:
            report["status"] = "g3_short_missing_features"
            report["err"] = ",".join(missing_short[:30])
            return pd.DataFrame(), report

        short_proba = predict_model_proba(short_model, short_input, short_feats)

    out = pd.DataFrame({
        "symbol": work["symbol"].astype(str).str.upper(),
        "entry_ts": work["entry_ts"],
        "g3_long_score_proba": long_proba,
        "g3_short_score_proba": short_proba,
    })

    out = out[out["entry_ts"].isin(target_ts)].copy()
    out = out.sort_values(["entry_ts", "symbol"]).reset_index(drop=True)

    out["gate3_threshold_long"] = long_thr
    out["gate3_threshold_short"] = short_thr

    if has_long_bundle:
        out["g3_long_score_pass"] = pd.to_numeric(out["g3_long_score_proba"], errors="coerce").ge(long_thr)
    else:
        out["g3_long_score_pass"] = False

    if has_short_bundle:
        out["g3_short_score_pass"] = pd.to_numeric(out["g3_short_score_proba"], errors="coerce").ge(short_thr)
    else:
        out["g3_short_score_pass"] = False

    out["gate3_proba_long"] = out["g3_long_score_proba"]
    out["gate3_proba_short"] = out["g3_short_score_proba"]
    out["gate3_pass_long"] = out["g3_long_score_pass"].astype(bool)
    out["gate3_pass_short"] = out["g3_short_score_pass"].astype(bool)
    out["gate3_any_pass"] = out["gate3_pass_long"] | out["gate3_pass_short"]

    long_filled_for_best = pd.to_numeric(out["gate3_proba_long"], errors="coerce").fillna(-np.inf)
    short_filled_for_best = pd.to_numeric(out["gate3_proba_short"], errors="coerce").fillna(-np.inf)

    out["gate3_best_side"] = np.where(
        long_filled_for_best > short_filled_for_best,
        "LONG",
        np.where(short_filled_for_best > long_filled_for_best, "SHORT", ""),
    )

    out["gate3_best_proba"] = np.maximum(long_filled_for_best, short_filled_for_best)
    out.loc[~np.isfinite(out["gate3_best_proba"]), "gate3_best_proba"] = np.nan

    out["gate3_margin_long"] = pd.to_numeric(out["gate3_proba_long"], errors="coerce") - out["gate3_threshold_long"]
    out["gate3_margin_short"] = pd.to_numeric(out["gate3_proba_short"], errors="coerce") - out["gate3_threshold_short"]

    long_score_filled = pd.to_numeric(out["g3_long_score_proba"], errors="coerce").fillna(0.0)
    short_score_filled = pd.to_numeric(out["g3_short_score_proba"], errors="coerce").fillna(0.0)

    if has_any_bundle:
        out["g3_score_spread"] = long_score_filled - short_score_filled
        out["g3_score_abs_spread"] = out["g3_score_spread"].abs()
        out["g3_score_max"] = np.maximum(long_score_filled, short_score_filled)
    else:
        out["g3_score_spread"] = np.nan
        out["g3_score_abs_spread"] = np.nan
        out["g3_score_max"] = np.nan

    out["gate3_precision_meta_long"] = long_meta["precision_meta"]
    out["gate3_wilson_meta_long"] = long_meta["wilson_meta"]
    out["gate3_delta_wilson_meta_long"] = long_meta["delta_wilson_meta"]
    out["gate3_pvalue_meta_long"] = long_meta["pvalue_meta"]
    out["gate3_kept_n_meta_long"] = long_meta["kept_n_meta"]
    out["gate3_valid_pos_rate_meta_long"] = long_meta["valid_pos_rate_meta"]
    out["gate3_thr_kept_lift_meta_long"] = long_meta["thr_kept_lift_meta"]

    out["gate3_precision_meta_short"] = short_meta["precision_meta"]
    out["gate3_wilson_meta_short"] = short_meta["wilson_meta"]
    out["gate3_delta_wilson_meta_short"] = short_meta["delta_wilson_meta"]
    out["gate3_pvalue_meta_short"] = short_meta["pvalue_meta"]
    out["gate3_kept_n_meta_short"] = short_meta["kept_n_meta"]
    out["gate3_valid_pos_rate_meta_short"] = short_meta["valid_pos_rate_meta"]
    out["gate3_thr_kept_lift_meta_short"] = short_meta["thr_kept_lift_meta"]

    out["gate3_precision_meta"] = out[["gate3_precision_meta_long", "gate3_precision_meta_short"]].max(axis=1, skipna=True)
    out["gate3_wilson_meta"] = out[["gate3_wilson_meta_long", "gate3_wilson_meta_short"]].max(axis=1, skipna=True)
    out["gate3_delta_wilson_meta"] = out[["gate3_delta_wilson_meta_long", "gate3_delta_wilson_meta_short"]].max(axis=1, skipna=True)
    out["gate3_pvalue_meta"] = out[["gate3_pvalue_meta_long", "gate3_pvalue_meta_short"]].min(axis=1, skipna=True)
    out["gate3_kept_n_meta"] = out[["gate3_kept_n_meta_long", "gate3_kept_n_meta_short"]].max(axis=1, skipna=True)
    out["gate3_valid_pos_rate_meta"] = out[["gate3_valid_pos_rate_meta_long", "gate3_valid_pos_rate_meta_short"]].max(axis=1, skipna=True)
    out["gate3_thr_kept_lift_meta"] = out[["gate3_thr_kept_lift_meta_long", "gate3_thr_kept_lift_meta_short"]].max(axis=1, skipna=True)

    out["has_gate3_long_bundle"] = bool(has_long_bundle)
    out["has_gate3_short_bundle"] = bool(has_short_bundle)
    out["has_any_gate3_bundle"] = bool(has_any_bundle)
    out["has_full_gate3_bundle"] = bool(has_full_bundle)

    out["long_model_path"] = str(long_model_path) if has_long_bundle else ""
    out["short_model_path"] = str(short_model_path) if has_short_bundle else ""
    out["long_meta_path"] = str(long_meta_path) if has_long_bundle and long_meta_path.exists() else ""
    out["short_meta_path"] = str(short_meta_path) if has_short_bundle and short_meta_path.exists() else ""
    out["long_feature_count"] = int(len(long_feats)) if has_long_bundle else 0
    out["short_feature_count"] = int(len(short_feats)) if has_short_bundle else 0

    out["online_source"] = SOURCE_NAME
    out["online_prediction_builder"] = PREDICTION_BUILDER

    report["rows_predicted"] = int(len(out))

    if has_any_bundle:
        report["status"] = "ok"
    else:
        report["status"] = "ok_no_gate3_score_bundle"
        report["err"] = f"long={long_model_status}, short={short_model_status}"

    return out, report

def parse_args() -> Tuple[Optional[str], bool, Optional[int]]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-latest", type=int, default=0)
    args = ap.parse_args()

    symbol = str(args.symbol).strip().upper() if args.symbol else None
    rebuild = bool(args.rebuild)
    limit_latest = int(args.limit_latest) if int(args.limit_latest or 0) > 0 else None

    return symbol, rebuild, limit_latest


def main() -> None:
    symbol_arg, rebuild, limit_latest = parse_args()

    print("ROOT:", ROOT)
    print("DB_DSN:", DB_DSN)
    print("ONLINE_GATE3_FEATURES_TABLE:", ONLINE_GATE3_FEATURES_TABLE)
    print("ONLINE_GATE3_PREDICTIONS_TABLE:", ONLINE_GATE3_PREDICTIONS_TABLE)
    print("GATE3_SCORE_ROOT:", GATE3_SCORE_ROOT)
    print("POLICY_CSV:", POLICY_CSV)
    print("REBUILD:", rebuild)
    print("LIMIT_LATEST:", limit_latest)
    print()

    if not table_exists(ONLINE_GATE3_FEATURES_TABLE):
        raise RuntimeError(f"missing table: {ONLINE_GATE3_FEATURES_TABLE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    create_predictions_table()

    policy = load_policy()

    if symbol_arg:
        symbols = [symbol_arg]
    else:
        symbols = get_symbols_from_features()

    print("SYMBOLS:", len(symbols))
    print("DB_BATCH_LOAD: missing gate3 prediction targets + online_gate3_features context")
    print()

    missing_by_symbol = load_missing_feature_rows_batch(
        symbols=symbols,
        rebuild=rebuild,
        limit_latest=limit_latest,
    )
    context_by_symbol = load_feature_context_batch(missing_by_symbol)

    missing_total = int(sum(len(x) for x in missing_by_symbol.values()))
    missing_symbols = int(sum(1 for x in missing_by_symbol.values() if x is not None and not x.empty))
    context_total = int(sum(len(x) for x in context_by_symbol.values()))

    print("MISSING_GATE3_PREDICTION_ROWS_TOTAL:", missing_total)
    print("MISSING_GATE3_PREDICTION_SYMBOLS:", missing_symbols)
    print("GATE3_FEATURE_CONTEXT_ROWS_BATCH:", context_total)
    print()

    reports = []
    total_predicted = 0
    total_inserted = 0
    total_long_pass = 0
    total_short_pass = 0

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")

        try:
            missing = missing_by_symbol.get(symbol, empty_missing_feature_rows())

            policy_row = policy_row_for_symbol(policy, symbol)

            pred_df, rep = build_predictions_for_symbol(
                symbol=symbol,
                missing_df=missing,
                policy_row=policy_row,
                feature_context_df=context_by_symbol.get(symbol),
            )

            inserted = upsert_predictions(pred_df) if not pred_df.empty else 0
            rep["inserted"] = int(inserted)

            if not pred_df.empty:
                total_long_pass += int(pred_df["gate3_pass_long"].astype(bool).sum())
                total_short_pass += int(pred_df["gate3_pass_short"].astype(bool).sum())

            total_predicted += int(rep.get("rows_predicted", 0))
            total_inserted += int(inserted)

            reports.append(rep)

            print(
                f"    status={rep['status']} | "
                f"to_predict={rep.get('feature_rows_to_predict', 0)} | "
                f"context={rep.get('context_rows', 0)} | "
                f"predicted={rep.get('rows_predicted', 0)} | "
                f"inserted={inserted} | "
                f"long_bundle={rep.get('has_gate3_long_bundle', 0)} | "
                f"short_bundle={rep.get('has_gate3_short_bundle', 0)}"
            )

            if rep.get("err"):
                print(f"    err={rep['err']}")

        except Exception as e:
            rep = {
                "symbol": symbol,
                "status": "error",
                "feature_rows_to_predict": 0,
                "context_rows": 0,
                "rows_predicted": 0,
                "inserted": 0,
                "has_gate3_long_bundle": 0,
                "has_gate3_short_bundle": 0,
                "long_feature_count": 0,
                "short_feature_count": 0,
                "err": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            reports.append(rep)
            print(f"    ERROR: {rep['err']}")

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    status_counts = rep_df["status"].value_counts(dropna=False).sort_index().to_dict() if len(rep_df) else {}

    summary = {
        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
        "root": str(ROOT),
        "db_dsn": DB_DSN,
        "online_gate3_features_table": ONLINE_GATE3_FEATURES_TABLE,
        "online_gate3_predictions_table": ONLINE_GATE3_PREDICTIONS_TABLE,
        "gate3_score_root": str(GATE3_SCORE_ROOT),
        "policy_csv": str(POLICY_CSV),
        "symbols_count": int(len(symbols)),
        "rebuild": bool(rebuild),
        "limit_latest": limit_latest,
        "status_counts": status_counts,
        "total_rows_predicted": int(total_predicted),
        "total_inserted": int(total_inserted),
        "total_gate3_long_pass": int(total_long_pass),
        "total_gate3_short_pass": int(total_short_pass),
        "report_csv": str(REPORT_CSV),
    }

    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("DONE")
    print("STATUS COUNTS:", status_counts)
    print("TOTAL ROWS PREDICTED:", total_predicted)
    print("TOTAL INSERTED:", total_inserted)
    print("TOTAL GATE3 LONG PASS:", total_long_pass)
    print("TOTAL GATE3 SHORT PASS:", total_short_pass)
    print("WROTE:", REPORT_CSV)
    print("WROTE:", REPORT_JSON)


if __name__ == "__main__":
    main()
