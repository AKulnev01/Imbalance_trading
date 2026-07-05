from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from production.features.build_features_full import build_features_single_symbol


ROOT = Path(__file__).resolve().parents[3]

M1_DIR = ROOT / "data" / "m1_4"
DS_DIR = ROOT / "production" / "dataset" / "gate1"
TEMPLATE_DS_DIR = ROOT / "production" / "dataset" / "gate1"

REPORT_JSON = ROOT / "online" / "new" / "actions" / "_build_gate1_dataset_for_symbols_report.json"
REPORT_CSV = ROOT / "online" / "new" / "actions" / "_build_gate1_dataset_for_symbols_report.csv"

RESAMPLE_RULE = "4h"
RESAMPLE_LABEL = "right"
RESAMPLE_CLOSED = "right"

N_CONTEXT_BARS = 500

USE_BTC_ETH_REFS = True
BTC_SYMBOL = "BTCUSDT"
ETH_SYMBOL = "ETHUSDT"

TARGET_MOVE_ABS = 0.01
SILENCE_FEATURE_BUILDER_STDOUT = True


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_floor_minute() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min").tz_convert(None)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)



def backup_existing_artifact(path: Path, artifact_group: str, symbol: str, report=None) -> str:
    """
    Backup only real data/model artifacts before overwrite.
    This function is intentionally used by new online/new/actions wrappers
    when they are about to overwrite existing production/data artifacts.
    It does NOT create backups of the wrapper script itself.
    """
    p = Path(path)

    if not p.exists():
        return ""

    symbol_safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    backup_dir = (
        ROOT
        / "online"
        / "new"
        / "actions"
        / "_artifact_backups"
        / str(artifact_group)
        / symbol_safe
        / run_id
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / p.name

    suffix_i = 1
    while backup_path.exists():
        backup_path = backup_dir / "{}.dup{}{}".format(p.stem, suffix_i, p.suffix)
        suffix_i += 1

    shutil.copy2(str(p), str(backup_path))

    if report is not None:
        try:
            backup_paths = report.setdefault("backup_paths", [])
            backup_paths.append(str(backup_path))
        except Exception:
            pass

    print("    BACKUP_ARTIFACT:", p, "->", backup_path, flush=True)
    return str(backup_path)

def parse_symbols(raw_values: List[str]) -> List[str]:
    out: List[str] = []

    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            symbol = part.strip().upper()
            if symbol:
                out.append(symbol)

    seen = set()
    unique: List[str] = []

    for symbol in out:
        if symbol in seen:
            continue
        unique.append(symbol)
        seen.add(symbol)

    if not unique:
        fail("no symbols provided")

    return unique


def find_ts_col(df: pd.DataFrame) -> str:
    for col in ["ts", "entry_ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if col in df.columns:
            return col

    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"

    raise RuntimeError("timestamp column not found; cols={}".format(list(df.columns)[:40]))


def read_m1(symbol: str) -> pd.DataFrame:
    path = M1_DIR / "{}.parquet".format(symbol)

    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_parquet(path)

    ts_col = find_ts_col(df)

    if ts_col == "__index__":
        df = df.reset_index().rename(columns={"index": "ts"})
        ts_col = "ts"
    elif ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    required = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("{}: m1 missing columns {}; cols={}".format(symbol, missing, list(df.columns)[:40]))

    out = df[required].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=required)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )

    return out


def m1_to_4h(m1: pd.DataFrame) -> pd.DataFrame:
    x = m1.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"]).sort_values("ts")
    x = x.set_index("ts")

    open_s = x["open"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).first()
    high_s = x["high"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).max()
    low_s = x["low"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).min()
    close_s = x["close"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).last()
    volume_s = x["volume"].resample(
        RESAMPLE_RULE,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin="epoch",
    ).sum()

    out = pd.DataFrame(
        {
            "entry_ts": open_s.index,
            "open": open_s.values,
            "high": high_s.values,
            "low": low_s.values,
            "close": close_s.values,
            "volume": volume_s.values,
        }
    )

    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["entry_ts", "open", "high", "low", "close", "volume"])

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.sort_values("entry_ts")
        .drop_duplicates("entry_ts", keep="last")
        .reset_index(drop=True)
    )

    dt = out["entry_ts"].diff()
    bad = dt[(dt != pd.Timedelta(hours=4)) & (~dt.isna())]

    if len(bad):
        print("[WARN] h4 gaps detected count={} examples={}".format(len(bad), [str(v) for v in bad.head(10).tolist()]))

    mask = (dt.isna()) | (dt == pd.Timedelta(hours=4))
    out = out[mask].copy().reset_index(drop=True)

    return out


def add_refs_to_bars(bars: pd.DataFrame, btc4h: pd.DataFrame, eth4h: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")

    btc = btc4h.copy()
    eth = eth4h.copy()

    btc["entry_ts"] = pd.to_datetime(btc["entry_ts"], utc=True, errors="coerce")
    eth["entry_ts"] = pd.to_datetime(eth["entry_ts"], utc=True, errors="coerce")

    btc = btc.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last")
    eth = eth.sort_values("entry_ts").drop_duplicates("entry_ts", keep="last")

    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        btc[["entry_ts", "close"]].rename(columns={"close": "ref_btc_close"}),
        on="entry_ts",
        direction="backward",
    )

    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        eth[["entry_ts", "close"]].rename(columns={"close": "ref_eth_close"}),
        on="entry_ts",
        direction="backward",
    )

    x["ref_btc_close"] = x["ref_btc_close"].ffill()
    x["ref_eth_close"] = x["ref_eth_close"].ffill()

    x["ref_close"] = (
        x["ref_btc_close"].fillna(x["ref_eth_close"]) +
        x["ref_eth_close"].fillna(x["ref_btc_close"])
    ) / 2.0

    return x


def find_template_dataset(exclude_symbols: List[str]) -> Optional[Path]:
    exclude = set(s.upper() for s in exclude_symbols)

    if not TEMPLATE_DS_DIR.exists():
        return None

    files = sorted(
        p for p in TEMPLATE_DS_DIR.glob("*.parquet")
        if p.is_file() and not p.name.startswith("_") and p.stem.upper() not in exclude
    )

    if not files:
        return None

    return files[0]


def align_to_template(df_new: pd.DataFrame, df_tpl: pd.DataFrame) -> pd.DataFrame:
    tpl_cols = list(df_tpl.columns)
    out = df_new.reindex(columns=tpl_cols)

    for col in tpl_cols:
        dtype = df_tpl[col].dtype

        try:
            if pd.api.types.is_datetime64_any_dtype(dtype):
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            elif pd.api.types.is_numeric_dtype(dtype):
                out[col] = pd.to_numeric(out[col], errors="coerce")
            else:
                out[col] = out[col].astype(object)
        except Exception:
            pass

    out = out.drop(columns=["label_gate1"], errors="ignore")
    return out


def build_feature_rows(symbol: str, bars4h: pd.DataFrame) -> pd.DataFrame:
    x = bars4h.copy()
    x["symbol"] = symbol

    if SILENCE_FEATURE_BUILDER_STDOUT:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            feat = build_features_single_symbol(x)
    else:
        feat = build_features_single_symbol(x)

    feat = (
        feat
        .sort_values("entry_ts")
        .drop_duplicates(subset=["symbol", "entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    if "volat_ret12" in feat.columns:
        feat = feat[feat["volat_ret12"].notna()].copy()
    if "atr_to_price" in feat.columns:
        feat = feat[feat["atr_to_price"].notna()].copy()

    feat = feat.drop(columns=["label_gate1"], errors="ignore")
    return feat


def add_gate1_target(df_feat: pd.DataFrame) -> pd.DataFrame:
    x = df_feat.copy()
    x = x.sort_values("entry_ts").reset_index(drop=True)

    next_high = x["high"].shift(-1)
    next_low = x["low"].shift(-1)
    entry = x["close"]

    up_move = (next_high - entry) / entry
    down_move = (entry - next_low) / entry

    x["y"] = ((up_move > TARGET_MOVE_ABS) | (down_move > TARGET_MOVE_ABS)).astype(int)

    # Последняя строка не имеет будущего next-bar target.
    x = x.iloc[:-1].copy()

    return x


def validate_gate1_dataset(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "min_entry_ts": None,
            "max_entry_ts": None,
            "entry_gap_count": 0,
            "entry_gap_examples": [],
            "y_value_counts": {},
            "cols_count": 0,
        }

    x = df.copy()

    if "entry_ts" not in x.columns:
        raise RuntimeError("{}: gate1 dataset missing entry_ts".format(symbol))

    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["entry_ts"]).sort_values("entry_ts")

    dt = x["entry_ts"].diff()
    gaps = dt[(dt != pd.Timedelta(hours=4)) & (~dt.isna())]

    y_counts: Dict[str, int] = {}
    if "y" in x.columns:
        y_counts = {str(k): int(v) for k, v in x["y"].value_counts(dropna=False).sort_index().items()}

    return {
        "rows": int(len(x)),
        "min_entry_ts": None if x.empty else str(x["entry_ts"].min()),
        "max_entry_ts": None if x.empty else str(x["entry_ts"].max()),
        "entry_gap_count": int(len(gaps)),
        "entry_gap_examples": [str(v) for v in gaps.head(10).tolist()],
        "y_value_counts": y_counts,
        "cols_count": int(len(x.columns)),
    }


def build_symbol_dataset(
    symbol: str,
    template_df: Optional[pd.DataFrame],
    btc4h: Optional[pd.DataFrame],
    eth4h: Optional[pd.DataFrame],
    force_rebuild: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ds_path = DS_DIR / "{}.parquet".format(symbol)
    m1_path = M1_DIR / "{}.parquet".format(symbol)

    report: Dict[str, Any] = {
        "symbol": symbol,
        "status": "OK",
        "error": "",
        "m1_path": str(m1_path),
        "ds_path": str(ds_path),
        "force_rebuild": bool(force_rebuild),
        "old_rows": 0,
        "old_max_entry_ts": "",
    }

    old_df: Optional[pd.DataFrame] = None
    old_max: Optional[pd.Timestamp] = None

    if ds_path.exists() and not force_rebuild:
        old_df = pd.read_parquet(ds_path)

        if "entry_ts" not in old_df.columns:
            raise RuntimeError("{}: existing gate1 dataset missing entry_ts".format(symbol))

        old_df["entry_ts"] = pd.to_datetime(old_df["entry_ts"], utc=True, errors="coerce")
        old_df = old_df.dropna(subset=["entry_ts"]).copy()
        old_max = old_df["entry_ts"].max()

        report["old_rows"] = int(len(old_df))
        report["old_max_entry_ts"] = str(old_max)

    m1 = read_m1(symbol)
    bars4h = m1_to_4h(m1)

    if USE_BTC_ETH_REFS:
        if btc4h is None or eth4h is None:
            raise RuntimeError("BTC/ETH refs are required but not prepared")
        bars4h = add_refs_to_bars(bars4h, btc4h=btc4h, eth4h=eth4h)

    bars_new = bars4h.copy()

    if old_max is not None:
        bars_new = bars_new[bars_new["entry_ts"] > old_max].copy()

    if bars_new.empty and not force_rebuild:
        report["status"] = "NO_NEW_BARS"
        return old_df if old_df is not None else pd.DataFrame(), report

    if force_rebuild:
        old_df = None
        old_max = None
        bars_new = bars4h.copy()

    report["bars4h_total"] = int(len(bars4h))
    report["bars_new"] = int(len(bars_new))

    if old_df is not None and old_max is not None:
        hist_cols = ["entry_ts", "open", "high", "low", "close", "volume"]
        hist = old_df[hist_cols].copy()

        for ref_col in ["ref_btc_close", "ref_eth_close", "ref_close"]:
            if ref_col in old_df.columns:
                hist[ref_col] = old_df[ref_col]

        hist["entry_ts"] = pd.to_datetime(hist["entry_ts"], utc=True, errors="coerce")
        hist = (
            hist
            .dropna(subset=["entry_ts"])
            .sort_values("entry_ts")
            .drop_duplicates("entry_ts", keep="last")
            .tail(N_CONTEXT_BARS)
        )

        bars_ctx = pd.concat([hist, bars_new], ignore_index=True, sort=False)
        bars_ctx = (
            bars_ctx
            .sort_values("entry_ts")
            .drop_duplicates("entry_ts", keep="last")
            .reset_index(drop=True)
        )
    else:
        bars_ctx = bars_new.copy()

    report["bars_context"] = int(len(bars_ctx))

    df_feat = build_feature_rows(symbol, bars_ctx)
    df_feat = add_gate1_target(df_feat)

    min_new_ts = old_max if old_max is not None else pd.Timestamp.min.tz_localize("UTC")
    df_new = df_feat[df_feat["entry_ts"] > min_new_ts].copy()

    report["new_rows"] = int(len(df_new))

    if old_df is not None:
        df_new_aligned = align_to_template(df_new, old_df)
        out = pd.concat([old_df, df_new_aligned], ignore_index=True, sort=False)
    else:
        if template_df is not None:
            out = align_to_template(df_new, template_df)
        else:
            out = df_new.copy()

    if "label_gate1" in out.columns:
        out = out.drop(columns=["label_gate1"])

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)

    if "entry_ts" in out.columns:
        out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")

    if "side" in out.columns:
        out = out.sort_values(["entry_ts", "side"], kind="mergesort").reset_index(drop=True)
    else:
        out = out.sort_values(["entry_ts"], kind="mergesort").reset_index(drop=True)

    out = (
        out
        .dropna(subset=["entry_ts"])
        .drop_duplicates(subset=["symbol", "entry_ts"] if "symbol" in out.columns else ["entry_ts"], keep="last")
        .reset_index(drop=True)
    )

    validation = validate_gate1_dataset(out, symbol)
    report.update(validation)

    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate1 Dataset For Symbols. "
            "Uses data/m1_4 parquet, BTC/ETH refs, production.features.build_features_full, "
            "and writes production/dataset/gate1/<SYMBOL>.parquet."
        )
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--m1-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--template-dir", default="")
    args = parser.parse_args()

    started_at = time.time()

    global M1_DIR
    global DS_DIR
    global TEMPLATE_DS_DIR

    if str(args.m1_dir).strip():
        m1_dir_arg = Path(str(args.m1_dir).strip())
        M1_DIR = m1_dir_arg if m1_dir_arg.is_absolute() else ROOT / m1_dir_arg

    if str(args.out_dir).strip():
        out_dir_arg = Path(str(args.out_dir).strip())
        DS_DIR = out_dir_arg if out_dir_arg.is_absolute() else ROOT / out_dir_arg

    if str(args.template_dir).strip():
        template_dir_arg = Path(str(args.template_dir).strip())
        TEMPLATE_DS_DIR = template_dir_arg if template_dir_arg.is_absolute() else ROOT / template_dir_arg

    symbols = parse_symbols(args.symbols)

    ensure_dir(DS_DIR)
    ensure_dir(REPORT_JSON.parent)

    template_path = find_template_dataset(exclude_symbols=symbols)
    template_df = pd.read_parquet(template_path) if template_path is not None else None

    btc4h: Optional[pd.DataFrame] = None
    eth4h: Optional[pd.DataFrame] = None

    if USE_BTC_ETH_REFS:
        btc4h = m1_to_4h(read_m1(BTC_SYMBOL))
        eth4h = m1_to_4h(read_m1(ETH_SYMBOL))

    print("Build Gate1 Dataset For Symbols")
    print("ROOT:", ROOT)
    print("M1_DIR:", M1_DIR)
    print("DS_DIR:", DS_DIR)
    print("TEMPLATE_DS_DIR:", TEMPLATE_DS_DIR)
    print("REPORT_JSON:", REPORT_JSON)
    print("REPORT_CSV:", REPORT_CSV)
    print("SYMBOLS:", symbols)
    print("FORCE_REBUILD:", bool(args.force_rebuild))
    print("USE_BTC_ETH_REFS:", USE_BTC_ETH_REFS)
    print("TEMPLATE_PATH:", "" if template_path is None else template_path)
    print("TEMPLATE_COLS:", 0 if template_df is None else len(template_df.columns))
    print("TARGET_MOVE_ABS:", TARGET_MOVE_ABS)
    print("=" * 120)

    reports: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(symbols, start=1):
        item_started = time.time()
        print("[{}/{}] {}: build gate1 dataset".format(idx, len(symbols), symbol), flush=True)

        try:
            out, report = build_symbol_dataset(
                symbol=symbol,
                template_df=template_df,
                btc4h=btc4h,
                eth4h=eth4h,
                force_rebuild=bool(args.force_rebuild),
            )

            if report.get("status") == "NO_NEW_BARS":
                print("    NO_NEW_BARS | old_rows={} | old_max={}".format(
                    report.get("old_rows"),
                    report.get("old_max_entry_ts"),
                ), flush=True)
            else:
                ds_path = DS_DIR / "{}.parquet".format(symbol)
                backup_existing_artifact(ds_path, "gate1_dataset", symbol, report)
                out.to_parquet(ds_path, index=False)

                report["wrote_path"] = str(ds_path)
                report["elapsed_sec"] = round(time.time() - item_started, 3)

                print(
                    "    OK | rows={} | cols={} | new_rows={} | y={} | gaps={} | min={} | max={} | elapsed_sec={}".format(
                        report.get("rows"),
                        report.get("cols_count"),
                        report.get("new_rows"),
                        report.get("y_value_counts"),
                        report.get("entry_gap_count"),
                        report.get("min_entry_ts"),
                        report.get("max_entry_ts"),
                        report.get("elapsed_sec"),
                    ),
                    flush=True,
                )
                print("    WROTE:", ds_path, flush=True)

        except Exception as exc:
            report = {
                "symbol": symbol,
                "status": "ERR",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "elapsed_sec": round(time.time() - item_started, 3),
            }
            print("    ERR:", report["error"], flush=True)

        reports.append(report)

    rep_df = pd.DataFrame(reports)
    rep_df.to_csv(REPORT_CSV, index=False)

    summary = {
        "created_at_utc": str(utc_now_floor_minute()),
        "name": "Build Gate1 Dataset For Symbols",
        "root": str(ROOT),
        "m1_dir": str(M1_DIR),
        "ds_dir": str(DS_DIR),
        "template_ds_dir": str(TEMPLATE_DS_DIR),
        "symbols": symbols,
        "force_rebuild": bool(args.force_rebuild),
        "use_btc_eth_refs": USE_BTC_ETH_REFS,
        "btc_symbol": BTC_SYMBOL,
        "eth_symbol": ETH_SYMBOL,
        "template_path": "" if template_path is None else str(template_path),
        "template_cols": 0 if template_df is None else int(len(template_df.columns)),
        "target_move_abs": TARGET_MOVE_ABS,
        "status_counts": dict(rep_df["status"].value_counts().sort_index()) if not rep_df.empty else {},
        "elapsed_sec": round(time.time() - started_at, 3),
        "report_csv": str(REPORT_CSV),
        "reports": reports,
    }

    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 120)
    print("DONE")
    print("STATUS_COUNTS:", summary["status_counts"])
    print("ELAPSED_SEC:", summary["elapsed_sec"])
    print("WROTE_JSON:", REPORT_JSON)
    print("WROTE_CSV:", REPORT_CSV)

    if any(r.get("status") == "ERR" for r in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
