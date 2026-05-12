# predict/tp_entry/prepare_dataset.py
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

FEAT_POOL = [
    "open","high","low","close","volume","atr14",
    "ret","body","body_pct_rng","upper_wick","lower_wick",
    "ret_l1","ret_l2","rng_pct","vol_regime"
]

ALIASES_TS_ORDER = [
    "entry_ts", "entry_bar_ts", "bar_open", "bar_ts", "time", "t", "ts"
]

def parse_sym_side_from_name(path: Path):
    m = re.match(r"([A-Z0-9]+)_(BUY|SELL)_feats\.parquet$", path.name)
    if m: return m.group(1), m.group(2)
    m = re.match(r"([A-Z0-9]+)_(BUY|SELL).*\.parquet$", path.name)
    if m: return m.group(1), m.group(2)
    return None, None

def coerce_ts_series(s: pd.Series) -> pd.Series:
    """Привести разные типы time-колонок к naive Datetime (UTC-без tz)."""
    if s is None:
        return None
    # если чисто числовая — вероятнее всего миллисекунды Unix
    if np.issubdtype(s.dtype, np.number):
        dt = pd.to_datetime(s, unit="ms", errors="coerce", utc=True)
    else:
        # строка/дататайп — обычный парсинг; уважаем возможную tz и конвертим к UTC
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    # в итоге делаем naive (tzinfo убираем)
    return dt.dt.tz_localize(None)

def ensure_entry_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантировать наличие колонки entry_ts (naive datetime)."""
    x = df.copy()

    # 1) если уже есть entry_ts — нормализуем и идём дальше
    if "entry_ts" in x.columns:
        x["entry_ts"] = coerce_ts_series(x["entry_ts"])
        return x.dropna(subset=["entry_ts"])

    # 2) поиск по алиасам
    for cand in ALIASES_TS_ORDER:
        if cand in x.columns:
            ts = coerce_ts_series(x[cand])
            if ts is not None:
                x["entry_ts"] = ts
                return x.dropna(subset=["entry_ts"])

    # 3) попробовать взять из индекса (частый кейс)
    if isinstance(x.index, pd.DatetimeIndex):
        ts = pd.to_datetime(x.index, utc=True)
        ts = ts.tz_localize(None)
        x = x.reset_index(drop=False)
        # имя колонки индекса может быть None → после reset_index это "index"
        idx_col = x.columns[0]
        # заменим её на entry_ts
        x = x.rename(columns={idx_col: "entry_ts"})
        x["entry_ts"] = coerce_ts_series(x["entry_ts"])
        return x.dropna(subset=["entry_ts"])

    # 4) жёсткий fallback: reset_index и попытаться распарсить первую колонку
    x = x.reset_index(drop=False)
    first = x.columns[0]
    ts_try = coerce_ts_series(x[first])
    if ts_try is not None and ts_try.notna().any():
        x = x.rename(columns={first: "entry_ts"})
        x["entry_ts"] = ts_try
        return x.dropna(subset=["entry_ts"])

    raise ValueError("не удалось восстановить entry_ts ни из колонок, ни из индекса")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str, default="./reports/features")
    ap.add_argument("--out", type=str, default="./reports/features/dataset_all.parquet")
    args = ap.parse_args()

    paths = sorted(Path(args.features_dir).glob("*_feats.parquet"))
    if not paths:
        raise SystemExit("нет *_feats.parquet в features-dir")

    pieces = []
    kept_files, kept_rows = 0, 0

    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[SKIP] {p.name}: читается с ошибкой: {e}")
            continue
        if df is None or len(df) == 0:
            print(f"[SKIP] {p.name}: пусто")
            continue

        # гарантируем entry_ts
        try:
            df = ensure_entry_ts(df)
        except Exception as e:
            print(f"[SKIP] {p.name}: нет entry_ts ({e})")
            continue

        # symbol/side → из колонок или из имени
        if "symbol" not in df.columns or df["symbol"].isna().all():
            s_from_name, side_from_name = parse_sym_side_from_name(p)
            if s_from_name:
                df["symbol"] = s_from_name
            else:
                print(f"[SKIP] {p.name}: не определить symbol")
                continue

        if "side" not in df.columns or df["side"].isna().all():
            s_from_name, side_from_name = parse_sym_side_from_name(p)
            if side_from_name:
                df["side"] = side_from_name
            else:
                print(f"[SKIP] {p.name}: не определить side (BUY/SELL)")
                continue

        # exit_reason / y
        if "exit_reason" in df.columns:
            df["exit_reason"] = df["exit_reason"].astype(str).str.lower()
        else:
            if "y" in df.columns:
                df["exit_reason"] = np.where(df["y"].astype(int) == 1, "tp", "sl")
            else:
                print(f"[SKIP] {p.name}: нет exit_reason и нет y")
                continue

        # оставляем только tp/sl
        df = df[df["exit_reason"].isin(["tp", "sl"])]
        if df.empty:
            print(f"[SKIP] {p.name}: после фильтра по tp/sl пусто")
            continue

        if "y" not in df.columns:
            df["y"] = (df["exit_reason"] == "tp").astype(int)
        else:
            df["y"] = df["y"].astype(int)

        # нормализуем side и side_num
        if df["side"].dtype != object:
            df["side"] = df["side"].map({1: "BUY", -1: "SELL"}).fillna(df["side"].astype(str))
        df["side"] = df["side"].str.upper()
        df["side_num"] = df["side"].map({"BUY": 1, "SELL": -1}).fillna(0).astype(int)

        # фичи — возьмём, что реально есть
        feat_cols = [c for c in FEAT_POOL if c in df.columns]
        if "vol_regime" not in feat_cols:
            df["vol_regime"] = 1
            feat_cols = feat_cols + (["vol_regime"] if "vol_regime" not in feat_cols else [])

        keep = ["entry_ts", "symbol", "side", "side_num", "y"]
        if "ref_close" in df.columns:
            keep.append("ref_close")

        pieces.append(df[keep + feat_cols])
        kept_files += 1
        kept_rows += len(df)

    if not pieces:
        raise SystemExit("после фильтра всё ещё пусто — покажи head() любого *_feats.parquet, подстрою парсер")

    out = pd.concat(pieces, ignore_index=True)
    out.sort_values(["symbol", "entry_ts"], inplace=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[OK] files={kept_files} rows={len(out)} → {args.out}")

if __name__ == "__main__":
    main()