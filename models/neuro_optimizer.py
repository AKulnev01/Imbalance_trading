# models/neuro_optimizer.py
# Обучение регрессора "параметры -> KPI" + генерация кандидатов.
# Поддерживает MLP, HistGradientBoosting и auto-режим.

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# --- пространство параметров (для генерации) ---
DEFAULT_PARAM_SPACE = {
    "USE_FIB_4H":  [0, 1],
    "USE_DIV_4H":  [0, 1],
    "USE_OBOS_4H": [0, 1],
    "FIB_4H_LOOKBACK_BARS": [60, 90, 120, 180],
    "FIB_4H_PIVOT_LEN":     [2, 3, 4, 5],
    "FIB_SET":              ["0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618"],
    "FIB_TP_INDEX":         [2, 3, 4],
    "FIB_TOUCH_MODE":       ["wick", "close"],
    "FIB_SL_MODE":          ["current", "beyond_prev_fib"],
    "DIV4H_TYPE":          ["off", "rsi", "macd"],
    "RSI_PERIOD":          [10, 14, 21],
    "MACD_FAST":           [8, 12],
    "MACD_SLOW":           [24, 26, 30],
    "MACD_SIGNAL":         [9],
    "DIV4H_PIVOT_LEN":     [2, 3, 4],
    "DIV4H_LOOKBACK_BARS": [60, 90, 120, 180],
    "DIV4H_CONFIRM_BARS":  [1, 2, 3],
    "DIV4H_POLICY":        ["tighten_tp", "skip_entry"],
    "DIV_TIGHTEN_STEP":    [0, 1, 2],
    "OBOS_TYPE":     ["off", "rsi", "stoch", "wpr", "cci"],
    "OB_RSI_OB":     [70.0, 75.0], "OB_RSI_OS": [25.0, 30.0],
    "STO_K":         [14], "STO_D": [3], "STO_SMA": [3],
    "OB_STOCH_OB":   [80.0], "OB_STOCH_OS": [20.0],
    "OB_WPR_OB":     [-20.0], "OB_WPR_OS": [-80.0],
    "OB_CCI_OB":     [100.0, 150.0], "OB_CCI_OS": [-100.0, -150.0],
    "OBOS_POLICY":   ["tp_bias", "filter_entry"],
    "MOMENTUM_TP_PCT": [0.02, 0.03, 0.04],
    "MOMENTUM_SL_PCT": [0.01, 0.015, 0.02],
    "FEE_TAKER":       [0.000, 0.0007],
    "ENTRY_SLIPPAGE_PCT": [0.001, 0.002, 0.003],
    "EXIT_SLIPPAGE_PCT":  [0.001, 0.002, 0.003],
    "STOP_SLIPPAGE_PCT":  [0.001, 0.002, 0.003],
}

# ---------- утилиты ----------
def _ensure_score(df: pd.DataFrame) -> pd.DataFrame:
    if "kpi" in df.columns:
        df["kpi_score"] = pd.to_numeric(df["kpi"], errors="coerce")
        return df
    if "kpi_score" in df.columns:
        df["kpi_score"] = pd.to_numeric(df["kpi_score"], errors="coerce")
        return df
    wr = pd.to_numeric(df.get("kpi_winrate", df.get("kpi_winrate_pct", 0.0)), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(df.get("kpi_pnl_pct", 0.0), errors="coerce").fillna(0.0)
    dd  = pd.to_numeric(df.get("kpi_dd_pct", 0.0), errors="coerce").fillna(0.0)
    df["kpi_score"] = 0.6*(wr/100.0) + 0.4*(pnl/100.0) - 0.5*(dd/100.0)
    return df

def _split_cols(df: pd.DataFrame):
    ignore = {
        "kpi","kpi_score","kpi_winrate","kpi_winrate_pct","kpi_pnl_pct","kpi_dd_pct","kpi_sharpe",
        "rows","as_of","created_at","seed","uuid"
    }
    feats = [c for c in df.columns if c not in ignore]
    cat = [c for c in feats if df[c].dtype == "object"]
    num = [c for c in feats if c not in cat]
    return num, cat

def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <=1.1

def _num_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])

def _cat_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", _ohe())])

def _preproc(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", _num_pipe(), num_cols),
        ("cat", _cat_pipe(), cat_cols),
    ], remainder="drop")

def _fmt(x):
    import numpy as np
    return f"{float(x):.4f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "n/a"

def _mlp_pipeline(num_cols, cat_cols):
    pre = _preproc(num_cols, cat_cols)
    model = MLPRegressor(hidden_layer_sizes=(256,128), activation="relu", max_iter=1200, random_state=42)
    return Pipeline([("prep", pre), ("model", model)])

def _hgb_pipeline(num_cols, cat_cols):
    pre = _preproc(num_cols, cat_cols)
    model = HistGradientBoostingRegressor(learning_rate=0.06, max_depth=6, max_iter=500, random_state=42)
    return Pipeline([("prep", pre), ("model", model)])

def _sample_from_space(space: dict, n: int) -> list:
    import random
    keys = list(space.keys())
    out = []
    for _ in range(n):
        d = {k: random.choice(space[k]) for k in keys}
        out.append(d)
    return out

def _save_json(obj, path):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _fmt(x):
    return f"{float(x):.4f}" if isinstance(x, (int,float,np.floating)) and np.isfinite(x) else "n/a"

# ---------- обучение + предложения ----------
def train_and_suggest(in_path: str,
                      out_model: str,
                      suggest_n: int = 0,
                      out_suggest: str = None,
                      param_space_path: str = None,
                      model_mode: str = "auto"):
    df = pd.read_parquet(in_path)
    df = _ensure_score(df).dropna(subset=["kpi_score"]).reset_index(drop=True)

    if df.empty:
        print("❌ Пустой датасет KPI — нечего учить.")
        return

    std = float(np.nanstd(df["kpi_score"].values))
    if std == 0.0:
        print("⚠️ KPI имеет нулевую дисперсию — модели почти нечему учиться.")

    num_cols, cat_cols = _split_cols(df)
    X = df[num_cols + cat_cols].copy()
    y = pd.to_numeric(df["kpi_score"], errors="coerce")

    # чистим y и X по маске валидных таргетов
    mask = y.notna() & np.isfinite(y.values)
    X, y = X.loc[mask], y.loc[mask]

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    report = {}
    chosen = None

    if model_mode in ("mlp","auto"):
        mlp = _mlp_pipeline(num_cols, cat_cols)
        mlp.fit(Xtr, ytr)
        r2_mlp = r2_score(yva, mlp.predict(Xva))
        report["r2_mlp"] = r2_mlp
        chosen = ("mlp", mlp, r2_mlp)

    if model_mode in ("hgb","auto"):
        hgb = _hgb_pipeline(num_cols, cat_cols)
        hgb.fit(Xtr, ytr)
        r2_hgb = r2_score(yva, hgb.predict(Xva))
        report["r2_hgb"] = r2_hgb
        if chosen is None or (isinstance(r2_hgb, (int,float,np.floating)) and r2_hgb > chosen[2]):
            chosen = ("hgb", hgb, r2_hgb)

    if chosen is None:
        print("❌ Не удалось выбрать модель.")
        return

    name, pipe, r2 = chosen
    Path(Path(out_model).parent).mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "num_cols": num_cols, "cat_cols": cat_cols, "r2_val": r2, "report": report, "model": name}, out_model)

    # печать метрик без падений
    extra = ""
    if model_mode == "auto":
        extra = f" (mlp={_fmt(report.get('r2_mlp'))}, hgb={_fmt(report.get('r2_hgb'))})"
    extra = ""
    if model_mode == "auto":
        extra = f" (mlp={_fmt(report.get('r2_mlp'))}, hgb={_fmt(report.get('r2_hgb'))})"
    print(f"💾 saved model → {out_model}; best={name}, R2_val={_fmt(r2)}{extra}")

    # — предложения
    if suggest_n and suggest_n > 0:
        space = DEFAULT_PARAM_SPACE
        if param_space_path and Path(param_space_path).exists():
            with open(param_space_path, "r", encoding="utf-8") as f:
                space = json.load(f)

        # лёгкий оверсэмпл
        cands = _sample_from_space(space, max(suggest_n * 6, suggest_n))
        cand_df = pd.DataFrame(cands).reindex(columns=X.columns)
        # имьютер категории/числа уже внутри пайплайна
        preds = pipe.predict(cand_df)
        out_df = cand_df.copy()
        out_df["kpi_pred"] = preds
        top = out_df.sort_values("kpi_pred", ascending=False).head(suggest_n)
        out_suggest = out_suggest or "./models/data/suggested_params.json"
        _save_json(top.to_dict("records"), out_suggest)
        print(f"🧠 suggested {len(top)} candidates → {out_suggest}")

def main():
    ap = argparse.ArgumentParser(description="Train regressor on params→KPI and optionally suggest candidates.")
    ap.add_argument("--in", dest="in_path", default="./models/data/params_kpi.parquet")
    ap.add_argument("--out-model", default="./models/neuro_opt.joblib")
    ap.add_argument("--suggest", type=int, default=0)
    ap.add_argument("--out-suggest", default="./models/data/suggested_params.json")
    ap.add_argument("--space", dest="param_space_path", default=None)
    ap.add_argument("--model", choices=["mlp","hgb","auto"], default="auto")
    args = ap.parse_args()

    train_and_suggest(
        in_path=args.in_path,
        out_model=args.out_model,
        suggest_n=int(args.suggest),
        out_suggest=args.out_suggest,
        param_space_path=args.param_space_path,
        model_mode=args.model,
    )

if __name__ == "__main__":
    main()