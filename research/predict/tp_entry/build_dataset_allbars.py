import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from predict.tp_entry.data_utils import load_m1, make_4h
from predict.tp_entry.label_triple_barrier import label_entries

# ---------- helpers ----------

def _to_naive_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce", utc=True)
    return di.tz_localize(None)

def _save_feats(df: pd.DataFrame, path: Path):
    """Сохранить фичи, гарантируя наличие столбца entry_ts (naive UTC), без индекса."""
    if "entry_ts" not in df.columns:
        # если во время пайплайна он оказался в индексе
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "entry_ts"})
        elif "bar_ts" in df.columns:
            df = df.rename(columns={"bar_ts": "entry_ts"})
        else:
            raise ValueError(f"{path}: no 'entry_ts' column and index is not DatetimeIndex")

    # нормализуем время: naive UTC, выкидываем NaT
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["entry_ts"])

    # типы безопасности
    if "side" in df.columns:
        df["side"] = df["side"].astype(str)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

# ---------- базовые фичи из 4h ----------

def features_from_4h(h4: pd.DataFrame) -> pd.DataFrame:
    df = h4.copy()
    df["ret"] = (df["close"] / df["open"] - 1.0)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = df["close"] - df["open"]
    df["body_pct_rng"] = (df["body"] / rng).clip(-5, 5).fillna(0)

    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    for c in ["upper_wick", "lower_wick"]:
        df[c] = df[c].clip(0, 5).fillna(0)

    df["rng_pct"] = (rng / df["close"]).fillna(0)
    df["ret_l1"] = df["ret"].shift(1).fillna(0)
    df["ret_l2"] = df["ret"].shift(2).fillna(0)
    if "vol_regime" not in df.columns:
        df["vol_regime"] = 1

    keep = [
        "open","high","low","close","volume","atr14","ret",
        "body","body_pct_rng","upper_wick","lower_wick",
        "ret_l1","ret_l2","rng_pct","vol_regime"
    ]
    return df[[c for c in keep if c in df.columns]]

def _find_reason_col(df: pd.DataFrame) -> str:
    for c in ["exit_reason","reason","exit_kind","exit","outcome"]:
        if c in df.columns:
            return c
    raise KeyError(f"Нет колонки причины выхода среди: {list(df.columns)}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, required=True,
                    help="список символов через запятую (пример: BTCUSDT,ETHUSDT)")
    ap.add_argument("--m1-dir", type=str, default="./data/m1")
    ap.add_argument("--best-csv", type=str, default="./reports/summary/best_ks.csv")
    ap.add_argument("--tmax-hours", type=int, default=80)
    ap.add_argument("--fee-pct", type=float, default=0.001)
    ap.add_argument("--slip-exit-pct", type=float, default=0.004)
    ap.add_argument("--fast-minutes", type=int, default=120)
    ap.add_argument("--out", type=str, default="./reports/features")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # читаем k_tp/k_sl
    best = pd.read_csv(args.best_csv)
    # ожидаем столбцы: symbol, side, k_tp, k_sl

    def fallback_by_side(side_name: str) -> tuple[float, float]:
        sub = best[best["side"] == side_name]
        if not sub.empty:
            return float(sub["k_tp"].median()), float(sub["k_sl"].median())
        return 2.0, 1.0

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        m1 = load_m1(sym, args.m1_dir)
        if m1.empty:
            print(f"[SKIP] {sym}: нет минуток")
            continue

        h4 = make_4h(m1).dropna(subset=["atr14"])
        if h4.empty:
            print(f"[SKIP] {sym}: нет 4h ATR")
            continue

        # индекс 4h → naive UTC
        h4.index = _to_naive_utc_index(h4.index)
        feats_full = features_from_4h(h4)

        for side_name, side_val in (("BUY", +1), ("SELL", -1)):
            row = best[(best.symbol.str.upper() == sym) & (best.side.str.upper() == side_name)]
            if row.empty:
                k_tp, k_sl = fallback_by_side(side_name)
            else:
                r = row.iloc[0]
                k_tp, k_sl = float(r.k_tp), float(r.k_sl)

            # каждая 4h-свеча — потенциальный вход на следующей минуте
            entries = h4.assign(side=side_val)[["close", "atr14", "side"]].copy()
            entries.index.name = "entry_ts"  # индекс = время 4h бара

            # размечаем triple-barrier'ом
            lab = label_entries(
                m1=m1,
                entries_4h=entries,
                side_col="side",
                k_tp=k_tp, k_sl=k_sl,
                tmax_hours=args.tmax_hours,
                fee_pct=args.fee_pct, slip_exit_pct=args.slip_exit_pct,
                atr_col="atr14", atr_n=14
            )

            reason_col = _find_reason_col(lab)
            # нормализуем регистр и оставляем только tp/sl
            lab[reason_col] = lab[reason_col].astype(str).str.lower()
            lab = lab[lab[reason_col].isin(["tp", "sl"])].copy()
            lab = lab.rename(columns={reason_col: "exit_reason"})

            # целевая метка
            lab["y"] = (lab["exit_reason"] == "tp").astype(int)

            # time-to-market в минутах (только для TP, быстрые TP ≤ fast_minutes)
            if {"exit_ts", "entry_ts"}.issubset(lab.columns):
                entry_ts = pd.to_datetime(lab["entry_ts"], errors="coerce", utc=True).dt.tz_localize(None)
                exit_ts  = pd.to_datetime(lab["exit_ts"],  errors="coerce", utc=True).dt.tz_localize(None)
                tdiff = exit_ts - entry_ts
                ttm_min = tdiff.dt.total_seconds() / 60.0
                lab["ttm_min"] = np.where(lab["exit_reason"] == "tp", ttm_min, np.nan)
            else:
                lab["ttm_min"] = np.nan
            lab["y_fast"] = ((lab["exit_reason"] == "tp") & (lab["ttm_min"] <= args.fast_minutes)).astype("Int8")

            # выравниваем индексы для join (оба naive UTC)
            lab.index = _to_naive_utc_index(lab.index)
            feats_full.index = _to_naive_utc_index(feats_full.index)

            out_df = lab.join(feats_full, how="inner")
            out_df["symbol"] = sym
            out_df["side"] = side_name

            out_p = Path(args.out) / f"{sym}_{side_name}_feats.parquet"
            _save_feats(out_df, out_p)
            print(f"[OK] {sym} {side_name} → {out_p} ({len(out_df)} строк)")

if __name__ == "__main__":
    main()