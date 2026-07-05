from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from online.trading import audit_log
from online.trading import config
from online.trading.db import clean_db_value, db_cursor, get_table_columns, read_sql, split_table_name, table_exists
from online.trading.dynamic_blacklist import is_symbol_allowed


def latest_closed_h4_open_utc() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    current_h4_open = now.floor("4h")
    return current_h4_open - pd.Timedelta(hours=4)

def prod_threshold_tag() -> str:
    return (
        "g2_%03d_g4_%03d_g51_%03d_g53_%03d"
        % (
            int(round(float(config.GATE2_THR) * 1000)),
            int(round(float(config.GATE4_THR) * 1000)),
            int(round(float(config.GATE5_1_THR) * 1000)),
            int(round(float(config.GATE5_3_THR) * 1000)),
        )
    )


def get_dynamic_blacklist_source() -> str:
    configured = str(getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "") or "").strip()
    if configured:
        return configured

    return "backtest_approved__" + prod_threshold_tag()


def table_ref(table_name: str) -> str:
    schema, table = split_table_name(table_name)
    return '{}."{}"'.format(schema, table)


def pick_first_existing(cols: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    existing = set(cols)
    for c in candidates:
        if c in existing:
            return c
    if required:
        raise RuntimeError(
            "Не найдена ни одна колонка из списка: {}. Доступные колонки: {}".format(
                candidates,
                cols,
            )
        )
    return None


def ensure_trading_signal_extra_columns() -> None:
    sql = """
        ALTER TABLE public.trading_signals
            ADD COLUMN IF NOT EXISTS h4_close DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS atr14 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS signal_strength DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS gate2_side_margin DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS admission_source TEXT,
            ADD COLUMN IF NOT EXISTS dynamic_symbol_allowed BOOLEAN,
            ADD COLUMN IF NOT EXISTS dynamic_symbol_reason TEXT,
            ADD COLUMN IF NOT EXISTS skipped_reason TEXT,
            ADD COLUMN IF NOT EXISTS candidate_rank INTEGER,
            ADD COLUMN IF NOT EXISTS selector_version TEXT;
    """
    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql)



def get_gate2_join_sql() -> Tuple[str, str]:
    gate2_table = config.ONLINE_GATE2_TABLE

    if not table_exists(gate2_table):
        raise RuntimeError("missing Gate2 table: {}".format(gate2_table))

    cols = get_table_columns(gate2_table)

    if "signal_key" in cols:
        join_sql = """
        LEFT JOIN {gate2_table} g2
            ON g2.signal_key = g51.signal_key
        """.format(gate2_table=table_ref(gate2_table))
    else:
        ts_col = pick_first_existing(
            cols,
            [
                "signal_ts",
                "entry_ts",
                "ts",
                "bar_ts",
                "candle_ts",
                "timestamp",
                "open_time",
            ],
            required=True,
        )

        join_sql = """
        LEFT JOIN {gate2_table} g2
            ON g2.symbol = g51.symbol
           AND g2.{ts_col} = g51.signal_ts
        """.format(
            gate2_table=table_ref(gate2_table),
            ts_col=ts_col,
        )

    if "up_reach_high_proba" in cols and "dn_reach_high_proba" in cols:
        select_sql = """
        g2.up_reach_high_proba AS gate2_up_reach_high_proba,
        g2.dn_reach_high_proba AS gate2_dn_reach_high_proba,
        CASE
            WHEN UPPER(g51.side) = 'LONG' THEN g2.up_reach_high_proba
            WHEN UPPER(g51.side) = 'SHORT' THEN g2.dn_reach_high_proba
            ELSE NULL
        END AS gate2_for_side_proba,
        CASE
            WHEN UPPER(g51.side) = 'LONG' THEN g2.up_reach_high_proba - g2.dn_reach_high_proba
            WHEN UPPER(g51.side) = 'SHORT' THEN g2.dn_reach_high_proba - g2.up_reach_high_proba
            ELSE NULL
        END AS gate2_side_margin
        """
        return join_sql, select_sql

    proba_col = pick_first_existing(
        cols,
        [
            "gate2_for_side_proba",
            "gate2_for_gate4_side_proba",
            "gate2_side_proba",
            "gate2_selected_side_proba",
            "gate2_proba",
            "gate2_best_proba",
        ],
        required=True,
    )

    select_sql = """
    NULL::DOUBLE PRECISION AS gate2_up_reach_high_proba,
    NULL::DOUBLE PRECISION AS gate2_dn_reach_high_proba,
    g2.{proba_col} AS gate2_for_side_proba,
    NULL::DOUBLE PRECISION AS gate2_side_margin
    """.format(proba_col=proba_col)

    return join_sql, select_sql

def load_latest_joined_candidates() -> pd.DataFrame:
    for table_name in [
        config.ONLINE_GATE5_1_TABLE,
        config.ONLINE_GATE5_3_TABLE,
        config.ONLINE_GATE4_TABLE,
        config.ONLINE_GATE2_TABLE,
        config.ONLINE_GATE4_FEATURES_TABLE,
    ]:
        if not table_exists(table_name):
            raise RuntimeError("missing table: {}".format(table_name))

    gate2_join_sql, gate2_select_sql = get_gate2_join_sql()

    sql = """
        WITH g51 AS (
            SELECT DISTINCT ON (signal_key, grid_name)
                signal_key,
                symbol,
                signal_ts,
                side,
                prod_pair_name,
                grid_name,
                gate4_confidence,
                pred_side_confidence,
                pred_side_ratio,
                gate5_1_proba,
                updated_at
            FROM {gate5_1_table}
            WHERE prod_pair_name = %s
              AND grid_name = %s
            ORDER BY signal_key, grid_name, updated_at DESC
        ),
        g53 AS (
            SELECT DISTINCT ON (signal_key)
                signal_key,
                chosen_grid_name,
                pred_proba AS gate5_3_proba,
                updated_at
            FROM {gate5_3_table}
            WHERE chosen_grid_name = %s
            ORDER BY signal_key, updated_at DESC
        )
        SELECT
            g51.signal_key,
            g51.symbol,
            g51.signal_ts,
            g51.side,
            g51.prod_pair_name,
            g51.grid_name,

            f.close AS h4_close,
            f.atr14 AS atr14,

            g51.gate4_confidence,
            g51.pred_side_confidence,
            g51.pred_side_ratio,
            g51.gate5_1_proba,

            g53.gate5_3_proba,

            g4.proba_long,
            g4.proba_short,
            g4.gate4_pred_side,
            g4.gate4_pred_side_gap,
            g4.gate4_pred_side_ratio,

            {gate2_select_sql}

        FROM g51
        INNER JOIN g53
            ON g53.signal_key = g51.signal_key

        LEFT JOIN {gate4_table} g4
            ON g4.signal_key = g51.signal_key

        LEFT JOIN {gate4_features_table} f
            ON f.symbol = g51.symbol
           AND f.entry_ts = g51.signal_ts

        {gate2_join_sql}

        ORDER BY g51.signal_ts DESC, g51.symbol ASC
    """.format(
        gate5_1_table=table_ref(config.ONLINE_GATE5_1_TABLE),
        gate5_3_table=table_ref(config.ONLINE_GATE5_3_TABLE),
        gate4_table=table_ref(config.ONLINE_GATE4_TABLE),
        gate4_features_table=table_ref(config.ONLINE_GATE4_FEATURES_TABLE),
        gate2_join_sql=gate2_join_sql,
        gate2_select_sql=gate2_select_sql,
    )

    df = read_sql(
        sql,
        [
            config.PAIR_MODEL_NAME,
            config.GRID_NAME,
            config.GRID_NAME,
        ],
    )

    if df.empty:
        return df

    latest_closed = latest_closed_h4_open_utc()
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce")
    df = df[df["signal_ts"] <= latest_closed].copy()

    return df.reset_index(drop=True)


def build_entry_ts_plan(signal_ts: pd.Series) -> pd.Series:
    return pd.to_datetime(signal_ts, utc=True, errors="coerce") + pd.Timedelta(
        seconds=int(config.H4_SECONDS) + int(config.ENTRY_DELAY_SECONDS)
    )


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["signal_key"] = out["signal_key"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["side"] = out["side"].astype(str).str.upper()
    out["signal_ts"] = pd.to_datetime(out["signal_ts"], utc=True, errors="coerce")

    for c in [
        "h4_close",
        "atr14",
        "gate2_up_reach_high_proba",
        "gate2_dn_reach_high_proba",
        "gate2_for_side_proba",
        "gate2_side_margin",
        "gate4_confidence",
        "gate5_1_proba",
        "gate5_3_proba",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["entry_ts_plan"] = build_entry_ts_plan(out["signal_ts"])

    out["signal_strength"] = (
        out["gate2_for_side_proba"].fillna(0.0)
        + out["gate4_confidence"].fillna(0.0)
        + out["gate5_1_proba"].fillna(0.0)
        + out["gate5_3_proba"].fillna(0.0)
    )

    out = out.dropna(
        subset=[
            "signal_key",
            "symbol",
            "signal_ts",
            "entry_ts_plan",
            "side",
            "h4_close",
            "atr14",
            "gate2_for_side_proba",
            "gate4_confidence",
            "gate5_1_proba",
            "gate5_3_proba",
        ]
    ).copy()

    out = out[out["atr14"] > 0].copy()
    out = out[out["h4_close"] > 0].copy()

    return out.reset_index(drop=True)


def keep_latest_signal_ts_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if not bool(getattr(config, "SELECT_ONLY_LATEST_SIGNAL_TS", True)):
        return df

    latest_signal_ts = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce").max()
    if pd.isna(latest_signal_ts):
        return df.iloc[0:0].copy()

    out = df[pd.to_datetime(df["signal_ts"], utc=True, errors="coerce") == latest_signal_ts].copy()
    return out.reset_index(drop=True)


def threshold_reject_reason(row: pd.Series) -> Optional[str]:
    side = str(row.get("side") or "").upper()
    if side not in ["LONG", "SHORT"]:
        return "INVALID_SIDE"

    if float(row.get("gate2_for_side_proba") or 0.0) < float(config.GATE2_THR):
        return "BELOW_GATE2"

    if float(row.get("gate4_confidence") or 0.0) < float(config.GATE4_THR):
        return "BELOW_GATE4"

    if float(row.get("gate5_1_proba") or 0.0) < float(config.GATE5_1_THR):
        return "BELOW_GATE5_1"

    if float(row.get("gate5_3_proba") or 0.0) < float(config.GATE5_3_THR):
        return "BELOW_GATE5_3"

    return None

def get_conditional_side_rule(symbol: str, side: str) -> Optional[Dict[str, object]]:
    rules = getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST", {}) or {}
    symbol_rules = rules.get(str(symbol).upper())

    if not isinstance(symbol_rules, dict):
        return None

    side_rule = symbol_rules.get(str(side).upper())

    if not isinstance(side_rule, dict):
        return None

    return side_rule

def get_side_admission_result(row: pd.Series) -> Tuple[bool, Optional[str], Optional[str]]:
    symbol = str(row.get("symbol") or "").upper()
    side = str(row.get("side") or "").upper()

    side_rules = getattr(config, "SIDE_AWARE_WHITELIST", {}) or {}

    if side_rules and config.is_allowed_by_side_rules(symbol, side, side_rules):
        return True, "CURRENT_WHITELIST", None

    if not bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True)):
        return False, None, "NO_WHITELIST"

    conditional_rule = get_conditional_side_rule(symbol, side)

    if conditional_rule is None:
        return False, None, "NO_WHITELIST"

    min_margin = float(conditional_rule.get("min_gate2_side_margin", 0.0) or 0.0)
    margin = row.get("gate2_side_margin")

    if pd.isna(margin):
        return False, None, "MISSING_GATE2_SIDE_MARGIN"

    if float(margin) < min_margin:
        return False, None, "BELOW_CONDITIONAL_GATE2_MARGIN"

    return True, "CONDITIONAL_WHITELIST", None

def apply_prod_thresholds_with_reasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["threshold_reject_reason"] = [
        threshold_reject_reason(row)
        for _, row in out.iterrows()
    ]

    if config.ALLOWED_SYMBOLS is not None:
        allowed = set(str(x).upper() for x in config.ALLOWED_SYMBOLS)
        out.loc[~out["symbol"].isin(allowed), "threshold_reject_reason"] = "NOT_IN_ALLOWED_SYMBOLS"

    if config.FORCED_EXCLUDED_SYMBOLS:
        excluded = set(str(x).upper() for x in config.FORCED_EXCLUDED_SYMBOLS)
        out.loc[out["symbol"].isin(excluded), "threshold_reject_reason"] = "FORCED_EXCLUDED_SYMBOL"

    side_rules = getattr(config, "SIDE_AWARE_WHITELIST", {}) or {}
    conditional_side_rules = (
        getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST", {}) or {}
        if bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True))
        else {}
    )

    out["admission_source"] = None

    if side_rules or conditional_side_rules:
        for idx, row in out.iterrows():
            current_reason = row.get("threshold_reject_reason")

            if current_reason is not None and not pd.isna(current_reason):
                continue

            allowed, admission_source, reject_reason = get_side_admission_result(row)

            if allowed:
                out.at[idx, "admission_source"] = admission_source
            else:
                out.at[idx, "threshold_reject_reason"] = reject_reason

    return out.reset_index(drop=True)

def passed_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["threshold_reject_reason"].isna()].copy().reset_index(drop=True)


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.sort_values(
        [
            "signal_ts",
            "signal_strength",
            "gate4_confidence",
            "gate2_for_side_proba",
            "gate5_1_proba",
            "gate5_3_proba",
            "symbol",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).copy()

    out["candidate_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def dedup_one_signal_per_h4(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = sort_candidates(df)
    out = out.drop_duplicates(["signal_ts"], keep="first")
    return out.reset_index(drop=True)


def save_signal_row(
    signal: Dict[str, object],
    selected: bool,
    rejected: bool,
    reject_reason: Optional[str],
    skipped_reason: Optional[str],
    dynamic_symbol_allowed: Optional[bool],
    dynamic_symbol_reason: Optional[str],
    candidate_rank: Optional[int],
) -> None:
    ensure_trading_signal_extra_columns()

    sql = """
        INSERT INTO public.trading_signals (
            signal_key,
            symbol,
            signal_ts,
            entry_ts_plan,
            side,

            pair_model_name,
            grid_name,
            tp_atr,
            sl_atr,
            ttl_hours,

            h4_close,
            atr14,

            gate2_proba,
            gate4_confidence,
            gate5_1_proba,
            gate5_3_proba,
            signal_strength,
            gate2_side_margin,
            admission_source,

            selected,
            rejected,
            reject_reason,
            skipped_reason,
            dynamic_symbol_allowed,
            dynamic_symbol_reason,
            candidate_rank,
            selector_version,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (signal_key)
        DO UPDATE SET
            entry_ts_plan = EXCLUDED.entry_ts_plan,
            side = EXCLUDED.side,

            pair_model_name = EXCLUDED.pair_model_name,
            grid_name = EXCLUDED.grid_name,
            tp_atr = EXCLUDED.tp_atr,
            sl_atr = EXCLUDED.sl_atr,
            ttl_hours = EXCLUDED.ttl_hours,

            h4_close = EXCLUDED.h4_close,
            atr14 = EXCLUDED.atr14,

            gate2_proba = EXCLUDED.gate2_proba,
            gate4_confidence = EXCLUDED.gate4_confidence,
            gate5_1_proba = EXCLUDED.gate5_1_proba,
            gate5_3_proba = EXCLUDED.gate5_3_proba,
            signal_strength = EXCLUDED.signal_strength,
            gate2_side_margin = EXCLUDED.gate2_side_margin,
            admission_source = EXCLUDED.admission_source,

            selected = EXCLUDED.selected,
            rejected = EXCLUDED.rejected,
            reject_reason = EXCLUDED.reject_reason,
            skipped_reason = EXCLUDED.skipped_reason,
            dynamic_symbol_allowed = EXCLUDED.dynamic_symbol_allowed,
            dynamic_symbol_reason = EXCLUDED.dynamic_symbol_reason,
            candidate_rank = EXCLUDED.candidate_rank,
            selector_version = EXCLUDED.selector_version,
            updated_at = NOW()
    """

    params = (
        str(signal["signal_key"]),
        str(signal["symbol"]).upper(),
        clean_db_value(signal["signal_ts"]),
        clean_db_value(signal["entry_ts_plan"]),
        str(signal["side"]).upper(),

        config.PAIR_MODEL_NAME,
        config.GRID_NAME,
        float(config.TP_ATR),
        float(config.SL_ATR),
        float(config.TTL_HOURS),

        float(signal.get("h4_close")) if pd.notna(signal.get("h4_close")) else None,
        float(signal.get("atr14")) if pd.notna(signal.get("atr14")) else None,

        float(signal.get("gate2_for_side_proba")) if pd.notna(signal.get("gate2_for_side_proba")) else None,
        float(signal.get("gate4_confidence")) if pd.notna(signal.get("gate4_confidence")) else None,
        float(signal.get("gate5_1_proba")) if pd.notna(signal.get("gate5_1_proba")) else None,
        float(signal.get("gate5_3_proba")) if pd.notna(signal.get("gate5_3_proba")) else None,
        float(signal.get("signal_strength")) if pd.notna(signal.get("signal_strength")) else None,
        float(signal.get("gate2_side_margin")) if pd.notna(signal.get("gate2_side_margin")) else None,
        str(signal.get("admission_source")) if signal.get("admission_source") is not None and pd.notna(signal.get("admission_source")) else None,

        bool(selected),
        bool(rejected),
        reject_reason,
        skipped_reason,
        dynamic_symbol_allowed,
        dynamic_symbol_reason,
        int(candidate_rank) if candidate_rank is not None else None,
        "selector_v2_candidate_audit",
    )

    with db_cursor(commit=True) as (_, cur):
        cur.execute(sql, params)


def save_selection_snapshot(
    df: pd.DataFrame,
    best_signal_key: Optional[str],
    entry_block_reason: Optional[str] = None,
) -> None:
    if df.empty:
        return

    for _, row in df.iterrows():
        signal = row.to_dict()
        signal_key = str(signal["signal_key"])

        threshold_reason = signal.get("threshold_reject_reason")
        if pd.isna(threshold_reason):
            threshold_reason = None

        dynamic_allowed = signal.get("dynamic_symbol_allowed")
        if pd.isna(dynamic_allowed):
            dynamic_allowed = None
        elif dynamic_allowed is not None:
            dynamic_allowed = bool(dynamic_allowed)

        dynamic_reason = signal.get("dynamic_symbol_reason")
        if pd.isna(dynamic_reason):
            dynamic_reason = None
        elif dynamic_reason is not None:
            dynamic_reason = str(dynamic_reason)

        candidate_rank = signal.get("candidate_rank")
        if pd.isna(candidate_rank):
            candidate_rank = None

        selected = best_signal_key is not None and signal_key == best_signal_key

        reject_reason = None
        skipped_reason = None

        if selected:
            rejected = False
        else:
            rejected = True
            if threshold_reason:
                reject_reason = str(threshold_reason)
            elif dynamic_allowed is False:
                reject_reason = "DYNAMIC_BLACKLIST"
                skipped_reason = dynamic_reason
            elif entry_block_reason:
                reject_reason = str(entry_block_reason)
                skipped_reason = str(entry_block_reason)
            else:
                reject_reason = "SLOT1_LOST_TO_STRONGER_SIGNAL"

        save_signal_row(
            signal=signal,
            selected=selected,
            rejected=rejected,
            reject_reason=reject_reason,
            skipped_reason=skipped_reason,
            dynamic_symbol_allowed=dynamic_allowed,
            dynamic_symbol_reason=dynamic_reason,
            candidate_rank=int(candidate_rank) if candidate_rank is not None else None,
        )


def log_no_signal(df: pd.DataFrame) -> None:
    signal_ts = None
    candidates_count = 0

    if df is not None and not df.empty:
        candidates_count = int(len(df))
        signal_ts = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce").max()

    audit_log.ensure_audit_tables()
    audit_log.log_audit_event(
        event_type="NO_SIGNAL_FOR_H4",
        status="NO_SIGNAL",
        message="No tradable signal passed selector for latest H4",
        payload={
            "signal_ts": None if signal_ts is None or pd.isna(signal_ts) else str(signal_ts),
            "candidates_count": candidates_count,
            "pair_model_name": config.PAIR_MODEL_NAME,
            "grid_name": config.GRID_NAME,
            "gate2_thr": config.GATE2_THR,
            "gate4_thr": config.GATE4_THR,
            "gate5_1_thr": config.GATE5_1_THR,
            "gate5_3_thr": config.GATE5_3_THR,
            "dynamic_blacklist_source": get_dynamic_blacklist_source(),
        },
    )



def threshold_reject_reason_verbose(row: pd.Series) -> str:
    reasons = []

    if float(row.get("gate2_for_side_proba") or 0.0) < float(config.GATE2_THR):
        reasons.append("BELOW_GATE2")

    if float(row.get("gate4_confidence") or 0.0) < float(config.GATE4_THR):
        reasons.append("BELOW_GATE4")

    if float(row.get("gate5_1_proba") or 0.0) < float(config.GATE5_1_THR):
        reasons.append("BELOW_GATE5_1")

    if float(row.get("gate5_3_proba") or 0.0) < float(config.GATE5_3_THR):
        reasons.append("BELOW_GATE5_3")

    side = str(row.get("side") or "").upper()
    if side not in ["LONG", "SHORT"]:
        reasons.append("BAD_SIDE")

    if reasons:
        return "+".join(reasons)

    side_rules = getattr(config, "SIDE_AWARE_WHITELIST", {}) or {}
    conditional_side_rules = (
        getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST", {}) or {}
        if bool(getattr(config, "CONDITIONAL_SIDE_AWARE_WHITELIST_ENABLED", True))
        else {}
    )

    if side_rules or conditional_side_rules:
        allowed, admission_source, reject_reason = get_side_admission_result(row)

        if not allowed:
            return str(reject_reason)

        return str(admission_source)

    return "PASSED_THRESHOLDS"

def build_no_signal_audit_payload() -> Dict[str, object]:
    raw = load_latest_joined_candidates()
    normalized = normalize_candidates(raw)
    latest = keep_latest_signal_ts_only(normalized)

    payload: Dict[str, object] = {
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "latest_rows": int(len(latest)),
        "pair_model_name": config.PAIR_MODEL_NAME,
        "grid_name": config.GRID_NAME,
        "gate2_thr": float(config.GATE2_THR),
        "gate4_thr": float(config.GATE4_THR),
        "gate5_1_thr": float(config.GATE5_1_THR),
        "gate5_3_thr": float(config.GATE5_3_THR),
        "dynamic_blacklist_source": get_dynamic_blacklist_source(),
    }

    if latest.empty:
        payload["latest_signal_ts"] = None
        payload["reject_summary"] = {}
        payload["top_candidates"] = []
        return payload

    latest = latest.copy()
    latest["threshold_reject_reason"] = latest.apply(threshold_reject_reason_verbose, axis=1)

    payload["latest_signal_ts"] = str(pd.to_datetime(latest["signal_ts"], utc=True, errors="coerce").max())
    payload["reject_summary"] = {
        str(k): int(v)
        for k, v in latest["threshold_reject_reason"].value_counts(dropna=False).sort_index().to_dict().items()
    }

    top_cols = [
        "signal_key",
        "symbol",
        "side",
        "signal_ts",
        "h4_close",
        "atr14",
        "gate2_up_reach_high_proba",
        "gate2_dn_reach_high_proba",
        "gate2_for_side_proba",
        "gate2_side_margin",
        "gate4_confidence",
        "gate5_1_proba",
        "gate5_3_proba",
        "signal_strength",
        "admission_source",
        "threshold_reject_reason",
    ]

    top = (
        latest.sort_values(
            ["signal_strength", "gate4_confidence", "gate2_for_side_proba", "symbol"],
            ascending=[False, False, False, True],
        )
        .head(10)
        .copy()
    )

    payload["top_candidates"] = [
        {
            c: (None if pd.isna(row.get(c)) else str(row.get(c)) if c in ["signal_ts"] else row.get(c))
            for c in top_cols
            if c in top.columns
        }
        for _, row in top.iterrows()
    ]

    return payload


def log_no_signal_for_latest_h4(source: str) -> None:
    payload = build_no_signal_audit_payload()

    latest_signal_ts = payload.get("latest_signal_ts")
    message = "No signal passed approved production thresholds for latest H4"

    audit_log.log_audit_event(
        event_type="NO_SIGNAL_FOR_LATEST_H4",
        status="NO_SIGNAL",
        message=message,
        payload={
            "source": source,
            **payload,
        },
    )


def select_best_signal(entry_block_reason: Optional[str] = None) -> Optional[Dict[str, object]]:
    raw = load_latest_joined_candidates()
    df = normalize_candidates(raw)
    df = keep_latest_signal_ts_only(df)
    df = apply_prod_thresholds_with_reasons(df)

    if df.empty:
        log_no_signal(df)
        return None

    checked_rows = []

    for _, row in sort_candidates(df).iterrows():
        threshold_reason = row.get("threshold_reject_reason")

        if pd.isna(threshold_reason):
            allowed, reason, stats = is_symbol_allowed(
                symbol=str(row["symbol"]),
                now_ts=pd.Timestamp(row["entry_ts_plan"]),
                source=str(getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "prod")),
            )
        else:
            allowed = None
            reason = None

        r = row.copy()
        r["dynamic_symbol_allowed"] = allowed
        r["dynamic_symbol_reason"] = reason
        checked_rows.append(r)

    checked = pd.DataFrame(checked_rows)
    checked = sort_candidates(checked)

    eligible = checked[
        checked["threshold_reject_reason"].isna()
        & (checked["dynamic_symbol_allowed"] == True)
    ].copy()

    eligible = dedup_one_signal_per_h4(eligible)

    if eligible.empty:
        save_selection_snapshot(checked, best_signal_key=None)
        log_no_signal(checked)
        return None

    if entry_block_reason:
        save_selection_snapshot(
            checked,
            best_signal_key=None,
            entry_block_reason=str(entry_block_reason),
        )
        log_no_signal(checked)
        return None

    best = eligible.iloc[0].to_dict()
    best_signal_key = str(best["signal_key"])

    save_selection_snapshot(checked, best_signal_key=best_signal_key)

    return best


def save_selected_signal(signal: Dict[str, object]) -> None:
    save_signal_row(
        signal=signal,
        selected=True,
        rejected=False,
        reject_reason=None,
        skipped_reason=None,
        dynamic_symbol_allowed=bool(signal.get("dynamic_symbol_allowed")) if signal.get("dynamic_symbol_allowed") is not None else None,
        dynamic_symbol_reason=str(signal.get("dynamic_symbol_reason")) if signal.get("dynamic_symbol_reason") is not None else None,
        candidate_rank=int(signal.get("candidate_rank")) if signal.get("candidate_rank") is not None and pd.notna(signal.get("candidate_rank")) else None,
    )


def main() -> None:
    print("PAIR_MODEL_NAME:", config.PAIR_MODEL_NAME)
    print("GRID_NAME:", config.GRID_NAME)
    print("THRS:", config.GATE2_THR, config.GATE4_THR, config.GATE5_1_THR, config.GATE5_3_THR)
    print("DYNAMIC_BLACKLIST_SOURCE:", str(getattr(config, "DYNAMIC_BLACKLIST_SOURCE", "prod")))
    print("DYNAMIC_BLACKLIST_SOURCE:", get_dynamic_blacklist_source())

    ensure_trading_signal_extra_columns()

    signal = select_best_signal()

    if signal is None:
        print("NO_SIGNAL")
        return

    save_selected_signal(signal)

    print("SELECTED_SIGNAL")
    print("signal_key:", signal["signal_key"])
    print("symbol:", signal["symbol"])
    print("side:", signal["side"])
    print("signal_ts:", signal["signal_ts"])
    print("entry_ts_plan:", signal["entry_ts_plan"])
    print("h4_close:", signal.get("h4_close"))
    print("atr14:", signal.get("atr14"))
    print("gate2:", signal.get("gate2_for_side_proba"))
    print("gate4:", signal.get("gate4_confidence"))
    print("gate5_1:", signal.get("gate5_1_proba"))
    print("gate5_3:", signal.get("gate5_3_proba"))
    print("signal_strength:", signal.get("signal_strength"))
    print("candidate_rank:", signal.get("candidate_rank"))


if __name__ == "__main__":
    main()
