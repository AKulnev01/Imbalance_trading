# predict/tp_entry/build_labeled_features.py
import os, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from predict.tp_entry.data_utils import load_m1, make_4h
from predict.tp_entry.label_triple_barrier import label_entries

# === фичи по 4h ===
def features_from_4h(h4: pd.DataFrame) -> pd.DataFrame:
    df = h4.copy()
    # базовые проверки
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"В 4h нет колонки '{c}'")

    df["ret"] = (df["close"] / df["open"] - 1.0)
    rng = (df["high"] - df["low"]).replace(0, np.nan)

    df["body"] = df["close"] - df["open"]
    df["body_pct_rng"] = (df["body"] / rng).clip(-5, 5).fillna(0)

    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    for col in ["upper_wick", "lower_wick"]:
        df[col] = df[col].clip(0, 5).fillna(0)

    df["rng_pct"] = (rng / df["close"]).fillna(0)
    df["ret_l1"] = df["ret"].shift(1).fillna(0)
    df["ret_l2"] = df["ret"].shift(2).fillna(0)
    if "vol_regime" not in df.columns:
        df["vol_regime"] = 1
    if "atr14" not in df.columns:
        # если нет atr14 — пометим как NaN (ниже мы дропнем такие строки при attach)
        df["atr14"] = np.nan

    keep = [
        "open","high","low","close","volume","atr14",
        "ret","body","body_pct_rng","upper_wick","lower_wick",
        "ret_l1","ret_l2","rng_pct","vol_regime"
    ]
    return df[[c for c in keep if c in df.columns]]

def _find_reason_col(df: pd.DataFrame) -> str:
    for c in ["exit_reason", "reason", "exit_kind", "exit", "outcome"]:
        if c in df.columns:
            return c
    raise KeyError(f"Не нашли колонку с причиной выхода. Есть: {list(df.columns)}")

# === нормализация причин выхода (робастная)

def _load_h4_safe(sym: str, m1_dir: str) -> pd.DataFrame:
    m1b = load_m1(sym, m1_dir)
    if m1b.empty:
        return pd.DataFrame()
    h4b = make_4h(m1b).dropna(subset=["close"])
    h4b = h4b.reset_index()
    h4b.columns = ["bar_ts"] + list(h4b.columns[1:])
    h4b["bar_ts"] = pd.to_datetime(h4b["bar_ts"], errors="coerce", utc=True).dt.tz_localize(None)
    return h4b[["bar_ts","close"]]

def _attach_ref_benchmarks(df_entries: pd.DataFrame, m1_dir: str, bench_syms: list[str]) -> pd.DataFrame:
    out = df_entries.copy()
    for b in bench_syms:
        h4b = _load_h4_safe(b, m1_dir)
        if h4b.empty:
            print(f"[WARN] bench {b}: no 4h data → fill 0")
            col = f"ref_{b.replace('PERP','').replace('USDT','').lower()}_close"
            out[col] = 0.0
            continue

        merged = pd.merge_asof(
            out.sort_values("entry_ts"),
            h4b.sort_values("bar_ts"),
            left_on="entry_ts", right_on="bar_ts",
            direction="backward",
            tolerance=pd.Timedelta(hours=4),
            suffixes=("", "_bench"),
        )

        # Надёжно находим «правую» close
        candidates = ["close_bench", "close_y", "close_right", "close"]
        close_col = next((c for c in candidates if c in merged.columns), None)

        col = f"ref_{b.replace('PERP','').replace('USDT','').lower()}_close"
        if close_col is None:
            print(f"[WARN] bench {b}: close column not found after merge → fill 0")
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(merged[close_col], errors="coerce").fillna(0.0).values
    return out

# === нормализация причин выхода (робастная)
def _normalize_exit_reason(values) -> pd.Series:
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    s = s.astype("string").str.strip().str.lower()
    s = s.replace({
        "take_profit": "tp", "takeprofit": "tp", "tp_hit": "tp", "tp ": "tp", " tp": "tp",
        "stop_loss": "sl", "stoploss": "sl", "sl_hit": "sl", "sl ": "sl", " sl": "sl",
        "t/o": "timeout", "time": "timeout", "to": "timeout", "out_time": "timeout"
    })
    s = s.where(s.isin(["tp", "sl", "timeout"]), "other")
    return s

# === tz helpers ===
def _to_naive_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce", utc=True)
    return di.tz_localize(None)

def _to_naive_utc_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_localize(None)

# === normalize helpers ===
def _flatten_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    if any(isinstance(c, tuple) for c in df.columns):
        df = df.copy()
        df.columns = ["_".join(map(str, c)).strip("_") if isinstance(c, tuple) else c
                      for c in df.columns]
    return df

def _scalarize_any(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0] if len(v) > 0 else None
    return v

def _ensure_single_bar_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить ровно один столбец 'bar_ts': если несколько — оставить первый, остальные удалить.
    Если нет точного имени — взять первый с префиксом 'bar_ts'."""
    df = df.copy()
    cols = list(df.columns)

    exact_idx = [i for i, c in enumerate(cols) if c == "bar_ts"]
    if len(exact_idx) >= 2:
        keep_i = exact_idx[0]
        drop_idx = [i for i in exact_idx if i != keep_i]
        drop_names = [cols[i] for i in drop_idx]
        df = df.drop(columns=drop_names)
        return df

    if len(exact_idx) == 1:
        return df

    pref = [c for c in cols if str(c).startswith("bar_ts")]
    if len(pref) >= 1:
        keep = pref[0]
        # прибираем возможные дубли по имени
        dup_mask = (pd.Index(df.columns) == keep)
        if dup_mask.sum() > 1:
            first = True
            new_cols = []
            for c in df.columns:
                if c == keep:
                    if first:
                        new_cols.append(c)
                        first = False
                    else:
                        new_cols.append(None)
                else:
                    new_cols.append(c)
            to_drop = [c for c in new_cols if c is None]
            df = df.drop(columns=to_drop)
        if keep != "bar_ts":
            df = df.rename(columns={keep: "bar_ts"})
        return df

    raise ValueError("Не найден столбец времени 'bar_ts' после merge.")

def _ensure_bar_ts_1d_datetime(merged: pd.DataFrame) -> pd.DataFrame:
    merged = _flatten_tuple_columns(merged.copy())
    merged = _ensure_single_bar_ts(merged)

    sel = merged.loc[:, ["bar_ts"]]
    if hasattr(sel, "ndim") and sel.ndim == 2 and sel.shape[1] > 1:
        merged["bar_ts"] = sel.iloc[:, 0]
    else:
        merged["bar_ts"] = merged["bar_ts"]

    merged["bar_ts"] = merged["bar_ts"].map(_scalarize_any)
    merged["bar_ts"] = pd.to_datetime(merged["bar_ts"], errors="coerce", utc=True).dt.tz_localize(None)

    if merged["bar_ts"].isna().any():
        merged = merged.dropna(subset=["bar_ts"])

    return merged

# === робастное прикрепление фич к меткам ===
def _attach_features_robust(lab: pd.DataFrame, h4: pd.DataFrame, sym: str, side_name: str) -> pd.DataFrame:
    feats_full = features_from_4h(h4).copy()
    feats_full.index = _to_naive_utc_index(feats_full.index)

    lab2 = lab.copy()
    # индексируем по entry_ts (naive UTC)
    if "entry_ts" in lab2.columns:
        lab2.index = _to_naive_utc_series(lab2["entry_ts"])
    else:
        lab2.index = _to_naive_utc_index(lab2.index)

    # 1) exact
    exact = lab2.join(feats_full, how="inner")
    if len(exact):
        print(f"[INFO] {sym} {side_name}: exact join -> {len(exact)}")
        return exact

    # 2) nearest (5m)
    try:
        nearest_vals = feats_full.reindex(lab2.index, method="nearest", tolerance=pd.Timedelta(minutes=5))
        nearest = lab2.join(nearest_vals, how="left").dropna(subset=["close", "atr14"])
        if len(nearest):
            print(f"[INFO] {sym} {side_name}: nearest(5m) -> {len(nearest)}")
            return nearest
    except Exception:
        pass

    # 3) asof (4h)
    try:
        f = feats_full.reset_index().rename(columns={"index": "bar_open"})
        l = lab2.reset_index().rename(columns={"index": "entry_ts"})
        merged = pd.merge_asof(
            l.sort_values("entry_ts"),
            f.sort_values("bar_open"),
            left_on="entry_ts",
            right_on="bar_open",
            direction="backward",
            tolerance=pd.Timedelta(hours=4),
        ).dropna(subset=["bar_open", "close", "atr14"]).set_index("entry_ts")
        if len(merged):
            print(f"[INFO] {sym} {side_name}: asof(4h) -> {len(merged)}")
            return merged
    except Exception:
        pass

    print(f"[WARN] {sym} {side_name}: no features attached (0 rows after all strategies)")
    return lab2.iloc[0:0]

# === финальная санитарная очистка фрейма перед сохранением
def _finalize_df(df: pd.DataFrame, sym: str, side_name: str, fast_minutes: int) -> pd.DataFrame:
    df = df.copy()

    # гарантируем entry_ts в колонках и индексе
    if "entry_ts" not in df.columns:
        df["entry_ts"] = df.index
    df["entry_ts"] = _to_naive_utc_series(df["entry_ts"])
    df = df.dropna(subset=["entry_ts"])
    df = df.set_index("entry_ts", drop=False)

    # symbol / side
    df["symbol"] = sym
    df["side"] = 1 if side_name == "BUY" else -1
    df["side"] = df["side"].astype("int8")

    # y строго из нормализованной exit_reason
    df["y"] = (df["exit_reason"] == "tp").astype("int8")

    # y_fast: только когда известен ttm_min (>=0) и он <= fast_minutes
    df["y_fast"] = ((df["exit_reason"] == "tp") & (df["ttm_min"].ge(0)) & (df["ttm_min"] <= fast_minutes)).astype("int8")

    # essential numeric → коэрсинг
    for c in ["close", "atr14"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # выкидываем строки без базовых ценовых фич
    need = [c for c in ["close", "atr14"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)

    # убираем бесконечности и оставшиеся NaN в числовых колонках
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        # безопасно: только для числовых, кроме базовых где мы уже сделали dropna
        to_fill = [c for c in num_cols if c not in need]
        if to_fill:
            df[to_fill] = df[to_fill].fillna(0)

    # убираем дубликаты по entry_ts (если вдруг)
    df = df[~df.index.duplicated(keep="first")]

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=str, help="путь к файлу сигналов (parquet)")
    ap.add_argument("--symbols", type=str, help="список через запятую, если signals не задан")
    ap.add_argument("--m1-dir", type=str, default="./data/m1")
    ap.add_argument("--best-csv", type=str, default="./reports/tp_opt_rand/best_ks.csv")
    ap.add_argument("--tmax-hours", type=int, default=80)
    ap.add_argument("--fee-pct", type=float, default=0.001)
    ap.add_argument("--slip-exit-pct", type=float, default=0.004)
    ap.add_argument("--fast-minutes", type=int, default=120)
    ap.add_argument("--out", type=str, default="./reports/features")
    ap.add_argument("--add-ref-bench", action="store_true",
                    help="приклеить эталонные 4h закрытия BTC/ETH к entry_ts")
    ap.add_argument("--bench-syms", type=str, default="BTCUSDT,ETHUSDT",
                    help="список эталонных символов для реф. закрытий (через запятую)")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.best_csv):
        raise FileNotFoundError(f"не найден best_ks: {args.best_csv}")
    best = pd.read_csv(args.best_csv)
    need_cols = {"symbol", "side", "k_tp", "k_sl"}
    miss = need_cols - set(best.columns)
    if miss:
        raise ValueError(f"в {args.best_csv} нет колонок: {sorted(miss)}")
    best["symbol"] = best["symbol"].astype(str).str.upper()
    best["side"] = best["side"].astype(str).str.upper()

    # список символов
    if args.signals:
        signals = pd.read_parquet(args.signals)
        if "symbol" not in signals.columns:
            raise ValueError("В signals нет колонки symbol")
        symbols = sorted(signals["symbol"].unique())
        print(f"[INFO] Загружено {len(signals)} сигналов по {len(symbols)} символам")
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        signals = None
    else:
        raise ValueError("Нужно указать либо --signals, либо --symbols")

    def fallback_by_side(side_name: str) -> tuple[float, float]:
        sub = best[best["side"] == side_name]
        if not sub.empty:
            return float(sub["k_tp"].median()), float(sub["k_sl"].median())
        return 2.0, 1.0

    for sym in symbols:
        m1 = load_m1(sym, args.m1_dir)
        if m1.empty:
            print(f"[SKIP] {sym}: нет минуток")
            continue

        h4 = make_4h(m1).dropna(subset=["atr14"])
        if h4.empty:
            print(f"[SKIP] {sym}: нет 4h ATR")
            continue

        for side_name, side_val in (("BUY", +1), ("SELL", -1)):
            row = best[(best.symbol == sym) & (best.side == side_name)]
            if row.empty:
                k_tp, k_sl = fallback_by_side(side_name)
            else:
                r = row.iloc[0]
                k_tp, k_sl = float(r.k_tp), float(r.k_sl)

            # точки входа
            if signals is not None:
                subset = signals[(signals["symbol"] == sym) & (signals["side"] == side_name)]
                if "bar_ts" not in subset.columns:
                    print(f"[WARN] {sym} {side_name}: нет bar_ts в signals — пропуск")
                    continue
                subset = subset.drop_duplicates("bar_ts")
                entry_ts = _to_naive_utc_series(subset["bar_ts"])
                entries_raw = pd.DataFrame({"entry_ts": entry_ts, "side": side_val}).dropna(subset=["entry_ts"])
            else:
                # используем сами 4h бары
                h4_idx = _to_naive_utc_index(h4.index)
                entries_raw = pd.DataFrame({"entry_ts": h4_idx, "side": side_val})

            if entries_raw.empty:
                print(f"[WARN] {sym} {side_name}: нет точек входа — пропуск")
                continue

            # === asof: приклеиваем к ПРЕДЫДУЩЕМУ 4h-бару OHLC/ATR ===
            h4r = h4.reset_index()
            h4r.columns = ["bar_ts"] + list(h4r.columns[1:])
            h4r["bar_ts"] = _to_naive_utc_series(h4r["bar_ts"])
            base_cols = [c for c in ["close", "atr14"] if c in h4r.columns]
            h4mini = h4r[["bar_ts"] + base_cols].dropna(subset=base_cols)

            merged = pd.merge_asof(
                entries_raw.sort_values("entry_ts"),
                h4mini.sort_values("bar_ts"),
                left_on="entry_ts",
                right_on="bar_ts",
                direction="backward",
                tolerance=pd.Timedelta(hours=4),
            )
            before = len(merged)
            merged = merged.dropna(subset=["bar_ts"] + base_cols)
            after = len(merged)
            print(f"[INFO] {sym} {side_name}: asof attach {before} -> {after}")
            if merged.empty:
                print(f"[WARN] {sym} {side_name}: no aligned entries — skip")
                continue

            # НОРМАЛИЗАЦИЯ bar_ts (удаление дублей, 1-D, datetime64[ns], naive UTC)
            merged = _ensure_bar_ts_1d_datetime(merged)
            dup = merged.columns[merged.columns.duplicated(keep='first')]
            if len(dup) > 0:
                merged = merged.loc[:, ~merged.columns.duplicated(keep='first')]
                print(f"[FIX] dropped duplicate columns: {list(dup)}")

            # финальная таблица входов — индекс = entry_ts
            entries = (
                merged
                .drop(columns=["entry_ts"], errors="ignore")
                .rename(columns={"bar_ts": "entry_ts"})
            )
            entries["entry_ts"] = _to_naive_utc_series(entries["entry_ts"])
            entries = entries.set_index("entry_ts", drop=False)
            entries.index = _to_naive_utc_index(entries.index)

            # трипл-барьер разметка
            lab = label_entries(
                m1=m1,
                entries_4h=entries,
                side_col="side",
                k_tp=k_tp, k_sl=k_sl,
                tmax_hours=args.tmax_hours,
                fee_pct=args.fee_pct, slip_exit_pct=args.slip_exit_pct,
                atr_col="atr14", atr_n=14
            )

            # вернуть индекс = entry_ts (naive UTC)
            if "entry_ts" in lab.columns:
                lab["entry_ts"] = _to_naive_utc_series(lab["entry_ts"])
                lab = lab.set_index("entry_ts", drop=False)
            else:
                lab.index = _to_naive_utc_index(lab.index)

            # нормализуем колонку причин
            reason_col = _find_reason_col(lab)
            lab = lab.rename(columns={reason_col: "exit_reason"})
            lab["exit_reason"] = _normalize_exit_reason(lab["exit_reason"])

            # используем всё (включая timeout), чтобы y=0/1 считался от нормализованной причины
            use = lab.copy()

            # индекс = entry_ts (naive UTC)
            if "entry_ts" in use.columns:
                use["entry_ts"] = _to_naive_utc_series(use["entry_ts"])
                use = use.set_index("entry_ts", drop=False)
            else:
                use.index = _to_naive_utc_index(use.index)

            # базовые метки и время жизни
            if {"exit_ts", "entry_ts"} <= set(use.columns):
                tdiff = (_to_naive_utc_series(use["exit_ts"]) - _to_naive_utc_series(use["entry_ts"]))
                use["ttm_min"] = pd.to_numeric((tdiff.dt.total_seconds() / 60), errors="coerce")
            else:
                use["ttm_min"] = np.nan

            # нормализуем exit_reason (повтор безопасен) и пересчитываем y/y_fast
            use["exit_reason"] = _normalize_exit_reason(use["exit_reason"])
            use["y"] = use["exit_reason"].eq("tp").astype("int8")
            use["y_fast"] = (use["exit_reason"].eq("tp") & (use["ttm_min"] <= args.fast_minutes)).astype("Int8")

            # опционально приклеим эталонные закрытия BTC/ETH к entry_ts
            if args.add_ref_bench:
                bench_list = [s.strip().upper() for s in args.bench_syms.split(",") if s.strip()]
                use = _attach_ref_benchmarks(use, args.m1_dir, bench_list)

            # приклеим 4h-фичи
            out_df = _attach_features_robust(use, h4, sym, side_name)

            # финальные правки/санитарка
            out_df = _finalize_df(out_df, sym, side_name, args.fast_minutes)

            out_p = Path(args.out)/f"{sym}_{side_name}_feats.parquet"
            out_df.to_parquet(out_p, index=True)
            print(f"[OK] {sym} {side_name} → {out_p} ({len(out_df)} строк)")

if __name__ == "__main__":
    main()