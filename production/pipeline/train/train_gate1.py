from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ====== ROOT / import safety ======
ROOT = Path(r"C:\Projects\ImbalanceSearcher")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ====== INPUT / OUTPUT ======
# Gate1 учим на bar-level датасетах без дублей (у тебя это pa_gate1view)
SRC_DIR = ROOT / "production" / "dataset" / "gate1"
OUT_ROOT = ROOT / "production" / "models" / "final_gate1"

# ====== LABEL CONFIG ======
H_HOURS = 16          # горизонт в часах
K_ATR = 1.0           # порог в "ATR-единицах" (move/atr_to_price)

# ====== TRAIN CONFIG ======
VALID_FRAC = 0.2
RANDOM_SEED = 42

THR_GRID = [round(x, 3) for x in np.linspace(0.10, 0.90, 17)]
KEPT_MIN = 0.03
KEPT_MAX = 0.20


def fail(msg: str):
    raise SystemExit(msg)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def future_max_min_over_next_n(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """
    Для каждого t берём max(high[t+1..t+n]) и min(low[t+1..t+n]).
    Реализовано через reverse-rolling чтобы не было утечки.
    """
    if n <= 0:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

    hi = df["high"]
    lo = df["low"]

    hi_rev = hi.iloc[::-1]
    lo_rev = lo.iloc[::-1]

    fut_max_hi_rev = hi_rev.shift(1).rolling(window=n, min_periods=n).max()
    fut_min_lo_rev = lo_rev.shift(1).rolling(window=n, min_periods=n).min()

    fut_max_hi = fut_max_hi_rev.iloc[::-1]
    fut_min_lo = fut_min_lo_rev.iloc[::-1]

    return fut_max_hi, fut_min_lo


def build_gate1_label(df: pd.DataFrame, h_hours: int, k_atr: float) -> pd.DataFrame:
    """
    label=1 если после свечи было движение "куда-то" >= k_atr в нормировке на atr_to_price.
    move = max( (max_high/entry - 1)/atr_to_price, (entry/min_low - 1)/atr_to_price )
    entry = close текущей свечи (последняя известная инфа).
    """
    if "entry_ts" not in df.columns:
        fail("dataset missing entry_ts")
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            fail(f"dataset missing OHLC column: {c}")
    df = df.sort_values("entry_ts").copy()

    if h_hours % 4 != 0:
        fail(f"H_HOURS must be multiple of 4 for 4h bars. got={h_hours}")
    n = h_hours // 4

    fut_max_hi, fut_min_lo = future_max_min_over_next_n(df, n=n)

    entry = df["close"].astype("float64")
    atrp = (df["atr14"] / df["close"]).astype("float64")

    atrp = atrp.replace([np.inf, -np.inf], np.nan)
    atrp = atrp.where(atrp > 0.0)

    move_up = (fut_max_hi.astype("float64") / entry - 1.0) / atrp
    move_dn = (entry / fut_min_lo.astype("float64") - 1.0) / atrp
    move = pd.concat([move_up, move_dn], axis=1).max(axis=1)

    vol = df["volat_ret12"].astype("float64")
    vol_med = vol.rolling(100, min_periods=20).median()

    vol_ok = vol > vol_med

    label = ((move >= float(k_atr)) & vol_ok).astype("int8")

    usable = move.notna()

    df["g1_move_up_atr"] = move_up
    df["g1_move_dn_atr"] = move_dn
    df["g1_move_atr"] = move
    df["g1_label"] = label

    df = df[usable].copy()
    if len(df) < 50:
        raise RuntimeError("too few rows after labeling")
    return df


def pick_features(df: pd.DataFrame) -> list[str]:
    """
    Берём только числовые фичи. Исключаем любые target/trade/ks/p_ колонки и любые даты.
    """
    drop_exact = {
        "symbol", "symbol_id", "entry_ts", "side",
        "pnl_net", "y", "y_fast", "ks_ret_adj", "exit_px", "tp_px", "sl_px", "ttm_min", "exit_reason",
    }

    drop_name_substr = (
        "ts", "time", "date", "datetime",
        "open_time", "close_time", "start", "end"
    )

    cols: list[str] = []
    for c in df.columns:
        if c in drop_exact or c == "g1_label":
            continue

        cl = c.lower()
        if any(s in cl for s in drop_name_substr) and c not in ("day_of_week", "hour_of_day"):
            continue

        if c.startswith("ks_"):
            continue
        if c.startswith("p_"):
            continue

        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
        if not pd.api.types.is_numeric_dtype(s):
            continue

        if s.nunique() <= 1 and c not in ("atr_to_price",):
            continue

        cols.append(c)

    if not cols:
        print("WARNING: fallback to raw numeric features")
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("y",)]
    return cols


def time_split(df: pd.DataFrame, valid_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 200:
        cut = max(10, int(n * (1 - valid_frac)))
    else:
        cut = int(n * (1 - valid_frac))
    cut = min(max(cut, 1), n - 1)
    tr = df.iloc[:cut].copy()
    va = df.iloc[cut:].copy()
    return tr, va


def eval_thresholds(y_true, proba) -> dict:
    rows = []
    n = len(y_true)

    for thr in THR_GRID:
        kept = proba >= thr
        kept_n = int(kept.sum())
        kept_share = kept_n / n if n else 0.0

        if kept_n > 0:
            mean_y = float(y_true[kept].mean())
        else:
            mean_y = 0.0

        rows.append({
            "thr": float(thr),
            "kept_n": kept_n,
            "kept_share": float(kept_share),
            "precision": float(y_true[kept].mean()) if kept_n else 0.0,
            "lift": float(y_true[kept].mean() / (y_true.mean() + 1e-9)) if kept_n else 0.0,
        })

    best = None
    for r in rows:
        if KEPT_MIN <= r["kept_share"] <= KEPT_MAX and r["kept_n"] >= 20:
            if best is None or r["lift"] > best["lift"]:
                best = r

    if best is None:
        target = (KEPT_MIN + KEPT_MAX) / 2
        best = min(rows, key=lambda r: abs(r["kept_share"] - target))

    return {"best_thr": float(best["thr"]), "grid": rows}


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = y_true.astype(int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(proba)
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_sorted) + 1)
    s_pos = ranks[y_sorted == 1].sum()
    auc = (s_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def train_one_symbol(sym_path: Path) -> dict:
    sym = sym_path.stem
    df = pd.read_parquet(sym_path)

    if "entry_ts" not in df.columns:
        raise RuntimeError("missing entry_ts")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).copy()

    # В gate1view обычно уже 1 строка на бар; но если есть side — берём BOTH
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).copy()
    df = df.sort_values("entry_ts", kind="mergesort")
    df = df.drop_duplicates(subset=["entry_ts"], keep="last").reset_index(drop=True)

    if "side" in df.columns:
        df["side"] = df["side"].astype(str)

    # НЕ ПЕРЕСЧИТЫВАЕМ ТАРГЕТ
    if "y" not in df.columns:
        raise RuntimeError("dataset missing target y")

    feats = pick_features(df)
    print(sym, "pos_rate_total:", df["y"].mean(), "n=", len(df))

    tr, va = time_split(df, valid_frac=VALID_FRAC)
    print(sym, "pos_rate_train:", tr["y"].mean(), "pos_rate_valid:", va["y"].mean())

    X_tr = tr[feats]
    y_tr = tr["y"].astype(int).to_numpy()
    X_va = va[feats]
    y_va = va["y"].astype(int).to_numpy()

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1200,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_SEED,
        verbose=200,
        od_type="Iter",
        od_wait=100,
        auto_class_weights="Balanced",
    )
    # ===== sample weights =====
    w_tr = tr["atr_to_price"].fillna(0).values

    med_tr = np.nanmedian(w_tr)
    if med_tr == 0 or np.isnan(med_tr):
        med_tr = 1.0

    w_tr = np.clip(w_tr / med_tr, 0.5, 2.0)

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]

    auc_tr = safe_auc(y_tr, p_tr)
    auc_va = safe_auc(y_va, p_va)

    thr_info = eval_thresholds(y_true=y_va, proba=p_va)

    out_dir = OUT_ROOT / sym / "gate1"
    ensure_dir(out_dir)

    model_path = out_dir / "gate1_impulse_abs_move_atr_16h.cbm"
    meta_path = out_dir / "meta.json"
    model.save_model(str(model_path))

    best_thr = thr_info["best_thr"]
    kept = p_va >= best_thr
    kept_n = int(kept.sum())
    kept_share = float(kept_n / len(p_va)) if len(p_va) else 0.0
    precision = float(y_va[kept].mean()) if kept_n else 0.0
    mean_move = 0.0

    meta = {
        "symbol": sym,
        "gate": "gate1",
        "label": {
            "type": "next_bar_range",
            "threshold": 0.01,
            "formula": "max((high_next-close)/close, (close-low_next)/close) >= 1%"
        },
        "train": {
            "valid_frac": VALID_FRAC,
            "rows_total": int(len(df)),
            "rows_train": int(len(tr)),
            "rows_valid": int(len(va)),
            "pos_rate_total": float(df["y"].mean()),
            "pos_rate_valid": float(va["y"].mean()),
            "auc_train": auc_tr,
            "auc_valid": auc_va,
        },
        "features": {
            "n_features": int(len(feats)),
            "feature_names": feats,
        },
        "thresholding": {
            "best_thr": float(best_thr),
            "kept_min": KEPT_MIN,
            "kept_max": KEPT_MAX,
            "kept_n_valid": kept_n,
            "precision_valid": precision,
            "thr_grid": THR_GRID,
            "grid_stats": thr_info["grid"],
        }
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "symbol": sym,
        "rows_total": int(len(df)),
        "rows_valid": int(len(va)),
        "pos_rate_total": float(df["y"].mean()),
        "auc_valid": float(auc_va),
        "best_thr": float(best_thr),
        "kept_share_valid": kept_share,
        "precision_valid": float(precision),
        "mean_move_atr_valid": float(mean_move),
        "model_path": str(model_path),
        "status": "OK",
    }


def main():
    if not SRC_DIR.exists():
        fail(f"SRC_DIR not found: {SRC_DIR}")

    ensure_dir(OUT_ROOT)

    files = sorted([p for p in SRC_DIR.glob("*.parquet") if not p.name.startswith("_")])
    if not files:
        fail(f"no parquet files in {SRC_DIR}")

    rows = []
    ok = 0
    err = 0

    for fp in files:
        sym = fp.stem
        print(">>> PROCESSING:", sym)
        try:
            r = train_one_symbol(fp)
            rows.append(r)
            ok += 1
            print(f"[OK] {sym} rows={r['rows_total']} auc_valid={r['auc_valid']:.4f} best_thr={r['best_thr']:.3f} kept={r['kept_share_valid']:.3f} prec={r['precision_valid']:.3f} mean_move={r['mean_move_atr_valid']:.3f}")
        except Exception as e:
            rows.append({"symbol": sym, "status": "ERR", "error": f"{type(e).__name__}: {e}"})
            err += 1
            print(f"[ERR] {sym}: {type(e).__name__}: {e}")

    manifest = pd.DataFrame(rows)
    manifest_path = OUT_ROOT / "_MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False)

    report = {
        "src_dir": str(SRC_DIR),
        "out_root": str(OUT_ROOT),
        "h_hours": H_HOURS,
        "k_atr": K_ATR,
        "valid_frac": VALID_FRAC,
        "files_total": int(len(files)),
        "ok": ok,
        "err": err,
        "manifest": str(manifest_path),
    }
    (OUT_ROOT / "_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE")
    print("WROTE", manifest_path)
    print("OK", ok, "ERR", err)


if __name__ == "__main__":
    main()