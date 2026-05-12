# predict/tp_entry/relabel_with_ttl.py
import argparse
import sys
import pandas as pd
import numpy as np


# ----------------------- utils: normalization -----------------------

def _norm_side(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ("BUY", "B", "LONG", "+1", "1", "TRUE", "YES", "Y", "ON"):
        return "BUY"
    if s in ("SELL", "S", "SHORT", "-1", "-", "FALSE", "NO", "N", "OFF"):
        return "SELL"
    # попытка распознать числовые
    try:
        v = float(s)
        return "BUY" if v >= 0 else "SELL"
    except Exception:
        return np.nan


# ----------------------- TTL CSV reading -----------------------

def read_ttl_csv(path: str) -> pd.DataFrame:
    ttl = pd.read_csv(path)
    cols = {c.lower(): c for c in ttl.columns}

    sym = next((cols[c] for c in ["symbol", "sym", "ticker"] if c in cols), None)
    tmax = next((cols[c] for c in ["best_tmax_hours", "ttl_hours", "tmax_hours", "best_ttl"] if c in cols), None)
    side = next((cols[c] for c in ["side", "direction", "dir"] if c in cols), None)

    if sym is None or tmax is None:
        raise ValueError(
            f"TTL CSV must have 'symbol' and 'best_tmax_hours/ttl_hours'. Got: {list(ttl.columns)}"
        )

    use_cols = [sym, tmax]
    if side:
        use_cols.append(side)

    out = ttl[use_cols].copy()
    out.columns = ["symbol", "best_tmax_hours"] + (["side"] if side else [])
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    if "side" in out.columns:
        out["side"] = out["side"].map(_norm_side)

    # необязательный вес, если есть в исходном CSV
    if "tp_sl_count" in ttl.columns:
        cnt = ttl[[sym, "tp_sl_count"]].copy()
        cnt.columns = ["symbol", "tp_sl_count"]
        out = out.merge(cnt, on="symbol", how="left")
    else:
        out["tp_sl_count"] = 1

    return out


# ----------------------- side inference in dataset -----------------------

def infer_side_in_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = df.copy()

    # ensure symbol
    if "symbol" not in df.columns:
        for alt in ["SYMBOL", "base_symbol", "ticker", "Ticker"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "symbol"})
                break
    if "symbol" not in df.columns:
        raise ValueError("Input dataset must contain a 'symbol' column.")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # 1) direct side
    if "side" in df.columns:
        df["side"] = df["side"].map(_norm_side)
        if df["side"].notna().any():
            return df, True

    # 2) alternatives
    alt_maps = [
        ("direction", lambda s: np.where(pd.to_numeric(s, errors="coerce").fillna(0) >= 0, "BUY", "SELL")),
        ("dir",       lambda s: np.where(pd.to_numeric(s, errors="coerce").fillna(0) >= 0, "BUY", "SELL")),
        ("is_long",   lambda s: np.where(s.astype(str).str.lower().isin(["1", "true", "yes", "y"]), "BUY", "SELL")),
        ("is_buy",    lambda s: np.where(s.astype(str).str.lower().isin(["1", "true", "yes", "y"]), "BUY", "SELL")),
        ("long_short",lambda s: s.astype(str).str.upper().map(_norm_side)),
        ("position",  lambda s: s.astype(str).str.upper().map(_norm_side)),
        ("signal_side", lambda s: s.astype(str).str.upper().map(_norm_side)),
    ]
    for col, fn in alt_maps:
        if col in df.columns:
            try:
                side = fn(df[col])
                if isinstance(side, pd.Series):
                    side = side.map(_norm_side)
                df["side"] = side
                if df["side"].notna().any():
                    return df, True
            except Exception:
                continue

    # no side
    return df, False


# ----------------------- hours derivation -----------------------

def _derive_from_timestamps(df, open_col, tp_col, sl_col):
    def to_ts(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        try:
            return pd.to_datetime(x).value / 1e9
        except Exception:
            return np.nan

    o = df[open_col].map(to_ts)
    tp_h_col = "_derived_h_to_tp"
    sl_h_col = "_derived_h_to_sl"

    if tp_col:
        ttp = df[tp_col].map(to_ts)
        df[tp_h_col] = (ttp - o) / 3600.0
        df.loc[pd.isna(ttp), tp_h_col] = np.nan
    else:
        df[tp_h_col] = np.nan

    if sl_col:
        tsl = df[sl_col].map(to_ts)
        df[sl_h_col] = (tsl - o) / 3600.0
        df.loc[pd.isna(tsl), sl_h_col] = np.nan
    else:
        df[sl_h_col] = np.nan

    return tp_h_col, sl_h_col


def derive_hours(df: pd.DataFrame, args=None) -> tuple[str, str]:
    """
    Возвращает названия колонок с часами до TP/SL. Может создать временные колонки.
    Приоритет:
      1) Явно указанные таймштампы флагами
      2) Готовые пары часов
      3) Типовые таймштампы
      4) Fallback: exit_hours + exit_reason (TP/SL)
    """
    cols = set(df.columns)

    # 0) explicit timestamp columns via args
    if args and args.open_col and (args.tp_col or args.sl_col):
        return _derive_from_timestamps(df, args.open_col, args.tp_col, args.sl_col)

    # 1) ready hour columns
    for a, b in [
        ("h_to_tp", "h_to_sl"),
        ("tp_hours", "sl_hours"),
        ("hours_to_tp", "hours_to_sl"),
        ("t_to_tp_h", "t_to_sl_h"),
    ]:
        if {a, b} <= cols:
            return a, b

    # 2) typical timestamp triplets
    ts_candidates = [
        ("open_ts", "tp_ts", "sl_ts"),
        ("t_open",  "t_tp",  "t_sl"),
        ("open_time", "tp_time", "sl_time"),
        ("open_at", "tp_at", "sl_at"),
    ]
    for o, t, s in ts_candidates:
        if o in cols and (t in cols or s in cols):
            return _derive_from_timestamps(df, o, t if t in cols else None, s if s in cols else None)

    # 3) fallback: exit_hours + exit_reason
    if args and args.exit_hours_col and args.exit_reason_col:
        if args.exit_hours_col in df.columns and args.exit_reason_col in df.columns:
            tp_h_col = "_derived_h_to_tp"
            sl_h_col = "_derived_h_to_sl"

            ex_h = pd.to_numeric(df[args.exit_hours_col], errors="coerce")
            reason = df[args.exit_reason_col].astype(str).str.upper()

            # набор меток TP
            tp_labels = set([x.strip().upper() for x in str(args.tp_labels).split(",") if x.strip()])
            is_tp = reason.isin(tp_labels)
            # SL: по ключевым словам
            is_sl = reason.str.contains("SL") | reason.str.contains("STOP")

            df[tp_h_col] = np.where(is_tp, ex_h, np.nan)
            df[sl_h_col] = np.where(is_sl, ex_h, np.nan)
            return tp_h_col, sl_h_col

    raise ValueError(
        "Не нашёл колонки с временем до TP/SL. "
        "Укажи --open-col/--tp-col/--sl-col или --exit-hours-col/--exit-reason-col, "
        "либо добавь пары h_to_tp/h_to_sl."
    )


# ----------------------- TTL merge -----------------------

def aggregate_ttl_per_symbol(ttl_map: pd.DataFrame, strategy: str = "max_count") -> pd.DataFrame:
    """
    Сведение до одного TTL на символ (когда side в датасете не найден).
    По умолчанию берём запись с максимумом tp_sl_count, если колонка есть.
    Иначе — медиану.
    """
    if strategy == "max_count" and "tp_sl_count" in ttl_map.columns:
        idx = ttl_map.groupby("symbol")["tp_sl_count"].idxmax()
        agg = ttl_map.loc[idx, ["symbol", "best_tmax_hours"]].copy()
    elif strategy == "median":
        agg = ttl_map.groupby("symbol", as_index=False)["best_tmax_hours"].median()
    elif strategy == "min":
        agg = ttl_map.groupby("symbol", as_index=False)["best_tmax_hours"].min()
    elif strategy == "max":
        agg = ttl_map.groupby("symbol", as_index=False)["best_tmax_hours"].max()
    else:
        # fallback на медиану
        agg = ttl_map.groupby("symbol", as_index=False)["best_tmax_hours"].median()

    agg = agg.rename(columns={"best_tmax_hours": "ttl_hours"})
    return agg


def attach_ttl(df: pd.DataFrame, ttl_map: pd.DataFrame, has_side: bool, strategy: str = "max_count") -> pd.DataFrame:
    df = df.copy()
    if has_side and "side" in ttl_map.columns and not ttl_map["side"].isna().all():
        m = ttl_map[["symbol", "side", "best_tmax_hours"]].rename(columns={"best_tmax_hours": "ttl_hours"})
        return df.merge(m, on=["symbol", "side"], how="left", validate="m:1")
    else:
        agg = aggregate_ttl_per_symbol(ttl_map, strategy=strategy)
        return df.merge(agg, on="symbol", how="left", validate="m:1")


# ----------------------- target computation -----------------------

def compute_y_ttl(df: pd.DataFrame, args=None) -> pd.Series:
    """
    y_ttl = 1, если TP успел случиться в пределах ttl_hours и раньше SL (или SL нет в пределах TTL).
    y_ttl = 0, если SL наступил раньше в пределах TTL или TP не наступил в TTL.
    ttl_hours NaN -> 0.
    """
    if "ttl_hours" not in df.columns:
        raise ValueError("ttl_hours is missing after merge.")

    tp_h_col, sl_h_col = derive_hours(df, args=args)

    tp_h = pd.to_numeric(df[tp_h_col], errors="coerce")
    sl_h = pd.to_numeric(df[sl_h_col], errors="coerce")
    ttl = pd.to_numeric(df["ttl_hours"], errors="coerce")

    tp_in = (~tp_h.isna()) & (tp_h <= ttl)
    sl_in = (~sl_h.isna()) & (sl_h <= ttl)

    y = (tp_in & (~sl_in | (tp_h < sl_h))).astype(int)

    # строки без ttl считаем 0
    y[ttl.isna()] = 0
    return y


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet with features")
    ap.add_argument("--ttl-csv", dest="ttl_csv", required=True, help="CSV with best TTL per symbol(/side)")
    ap.add_argument("--out", dest="outp", required=True, help="Output parquet path")
    ap.add_argument("--replace-y", dest="replace_y", action="store_true", help="Replace existing target 'y' with 'y_ttl'")
    ap.add_argument("--ttl-agg", dest="ttl_agg", default="max_count",
                    choices=["max_count", "median", "min", "max"],
                    help="Aggregation strategy when side is absent in dataset")

    # явные указания колонок времени
    ap.add_argument("--open-col", default=None, help="Имя колонки с временем открытия (ts)")
    ap.add_argument("--tp-col", default=None, help="Имя колонки с ts TP (если есть)")
    ap.add_argument("--sl-col", default=None, help="Имя колонки с ts SL (если есть)")

    # fallback: exit_hours + exit_reason
    ap.add_argument("--exit-hours-col", default=None, help="Колонка 'часы до выхода'")
    ap.add_argument("--exit-reason-col", default=None, help="Колонка причины выхода (например, TP/SL)")
    ap.add_argument("--tp-labels", default="TP,TAKE_PROFIT", help="Список меток, означающих TP (через запятую)")

    args = ap.parse_args()

    # load data
    df = pd.read_parquet(args.inp)
    ttl_map = read_ttl_csv(args.ttl_csv)

    # unify side
    df, has_side = infer_side_in_dataset(df)
    if not has_side:
        print("[relabel_with_ttl] WARN: 'side' не найден в датасете — применяю агрегированный TTL по символу.", file=sys.stderr)

    # merge ttl
    merged = attach_ttl(df, ttl_map, has_side=has_side, strategy=args.ttl_agg)
    rows_in = len(merged)

    merged = merged[~merged["ttl_hours"].isna()].copy()
    rows_out = len(merged)
    if rows_out == 0:
        raise ValueError("После join нет ни одной строки с ttl_hours. Проверь символы/направления в CSV и датасете.")

    # compute target
    merged["y_ttl"] = compute_y_ttl(merged, args=args)

    if args.replace_y:
        if "y" in merged.columns:
            merged.drop(columns=["y"], inplace=True)
        merged = merged.rename(columns={"y_ttl": "y"})

    merged.to_parquet(args.outp, index=False)

    pos = int((merged["y_ttl"] if "y_ttl" in merged.columns else merged["y"]).sum())
    print(f"[relabel_with_ttl] rows_in={rows_in} rows_out={rows_out} positives={pos} out={args.outp}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[relabel_with_ttl] ERROR: {e}", file=sys.stderr)
        sys.exit(1)