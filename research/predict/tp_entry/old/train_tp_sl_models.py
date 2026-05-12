# predict/tp_entry/train_tp_sl_models.py
import os, json, argparse, warnings
import numpy as np, pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_dataset(path):
    if path.lower().endswith(".csv"): return pd.read_csv(path)
    return pd.read_parquet(path)

BASE_DROP = {
    "symbol","time_open","time_close","side",
    "best_rr","best_tp_rate","best_sl_rate","best_exp_pnl","objective_used"
}
TARGETS = {"tp":"best_tp_pct", "sl":"best_sl_pct"}
TIE_TARGET = "tie_tp_first"  # 1 если TP раньше SL, 0 иначе

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # базовые числовые фичи
    keep = [c for c in x.columns
            if c not in BASE_DROP
            and c not in list(TARGETS.values()) + ["mfe_pct","mae_pct"]
            and pd.api.types.is_numeric_dtype(x[c])]
    feats = x[keep].copy()
    # добавим MFE/MAE как фичи (они поведенческие для данного бара)
    for c in ("mfe_pct","mae_pct"):
        if c in x.columns:
            feats[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    # скейл необязателен для LGBM, но оставим числовой тип
    for c in feats.columns:
        if not pd.api.types.is_numeric_dtype(feats[c]):
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
    feats = feats.replace([np.inf,-np.inf], 0.0).fillna(0.0)
    return feats

def build_tie_label(df: pd.DataFrame) -> pd.Series:
    # приблизительный прокси tie-лейбла: TP раньше SL => 1, иначе 0
    # если в датасете нет явного «кто раньше», используем правило по best_tp_rate / best_sl_rate
    if "best_tp_rate" in df.columns and "best_sl_rate" in df.columns:
        return (df["best_tp_rate"] > df["best_sl_rate"]).astype(int)
    # фолбэк: MFE по модулю ближе к best_tp_pct, чем MAE к best_sl_pct (оч грубо)
    y = (df["mfe_pct"].abs() >= df["mae_pct"].abs()).astype(int)
    y = y.fillna(0).astype(int)
    return y

def compute_symbol_te(train_df: pd.DataFrame):
    # агрегаты по символу на ТРЕЙНЕ (без утечек)
    g = train_df.groupby("symbol")
    te = pd.DataFrame({
        "sym_tp_mean": g["best_tp_pct"].mean(),
        "sym_sl_mean": g["best_sl_pct"].mean(),
        "sym_winrate": (g["best_tp_rate"].mean() if "best_tp_rate" in train_df.columns else g["mfe_pct"].apply(lambda s: np.mean(s > 0))),
        "sym_atr14_mean": g["atr14"].mean() if "atr14" in train_df.columns else 0.0,
        "sym_volz_mean": g["vol_z"].mean() if "vol_z" in train_df.columns else 0.0,
    })
    te = te.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    te_map = {sym: row._asdict() if hasattr(row, "_asdict") else row.to_dict()
              for sym, row in te.iterrows()}
    # глобальные средние для фолбэка
    global_vals = {
        "sym_tp_mean": float(train_df["best_tp_pct"].mean()),
        "sym_sl_mean": float(train_df["best_sl_pct"].mean()),
        "sym_winrate": float(train_df["best_tp_rate"].mean() if "best_tp_rate" in train_df.columns else (train_df["mfe_pct"]>0).mean()),
        "sym_atr14_mean": float(train_df["atr14"].mean() if "atr14" in train_df.columns else 0.0),
        "sym_volz_mean": float(train_df["vol_z"].mean() if "vol_z" in train_df.columns else 0.0),
    }
    return te_map, global_vals

def attach_te(df: pd.DataFrame, te_map, global_vals):
    out = df.copy()
    def _get(sym, key):
        d = te_map.get(sym)
        if d is None: return global_vals[key]
        v = d.get(key, None)
        return global_vals[key] if v is None or (isinstance(v, float) and not np.isfinite(v)) else v

    out["sym_tp_mean"]   = out["symbol"].map(lambda s: _get(s,"sym_tp_mean")).astype(float)
    out["sym_sl_mean"]   = out["symbol"].map(lambda s: _get(s,"sym_sl_mean")).astype(float)
    out["sym_winrate"]   = out["symbol"].map(lambda s: _get(s,"sym_winrate")).astype(float)
    out["sym_atr14_mean"]= out["symbol"].map(lambda s: _get(s,"sym_atr14_mean")).astype(float)
    out["sym_volz_mean"] = out["symbol"].map(lambda s: _get(s,"sym_volz_mean")).astype(float)
    return out

def make_lgbm_reg():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.03,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, random_state=42, n_jobs=-1
    )

def make_lgbm_cls():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=900, learning_rate=0.03,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, random_state=42, n_jobs=-1
    )

def main():
    ap = argparse.ArgumentParser(description="Train TP/SL regressors + tie classifier with symbol target encoding.")
    ap.add_argument("--data", default="./predict/tp_entry/tp_training_base_fullrecalc.parquet")
    ap.add_argument("--outdir", default="./predict/tp_entry/models_tp_sl")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rr", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(os.path.expanduser(args.outdir), exist_ok=True)
    df = load_dataset(args.data).replace([np.inf,-np.inf], np.nan)

    # базовые таргеты
    y_tp = pd.to_numeric(df[TARGETS["tp"]], errors="coerce")
    y_sl = pd.to_numeric(df[TARGETS["sl"]], errors="coerce")
    y_tie = build_tie_label(df)

    # train/test split ПО СТРОКАМ, потом TE только на train
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    # считаем TE только на train
    te_map, te_global = compute_symbol_te(df_train)
    # прикручиваем TE к обеим частям
    df_train_te = attach_te(df_train, te_map, te_global)
    df_test_te  = attach_te(df_test,  te_map, te_global)

    # набор фичей = базовые + TE
    Xtr = build_features(df_train_te)
    Xte = build_features(df_test_te)
    # добавим столбцы TE в X (если их вдруг нет в базовом списке)
    for c in ["sym_tp_mean","sym_sl_mean","sym_winrate","sym_atr14_mean","sym_volz_mean"]:
        Xtr[c] = pd.to_numeric(df_train_te[c], errors="coerce").fillna(0.0)
        Xte[c] = pd.to_numeric(df_test_te[c], errors="coerce").fillna(0.0)

    # окончательный список фичей:
    feature_names = list(Xtr.columns)

    # модели
    reg_tp = make_lgbm_reg()
    reg_sl = make_lgbm_reg()
    cls_tie = make_lgbm_cls()

    reg_tp.fit(Xtr, y_tp.iloc[train_idx])
    reg_sl.fit(Xtr, y_sl.iloc[train_idx])
    cls_tie.fit(Xtr, y_tie.iloc[train_idx])

    # метрики
    tp_pred = reg_tp.predict(Xte); sl_pred = reg_sl.predict(Xte)
    y_tp_te = y_tp.iloc[test_idx]; y_sl_te = y_sl.iloc[test_idx]
    print(f"TP:  MAE={mean_absolute_error(y_tp_te, tp_pred):.4f}  R2={r2_score(y_tp_te, tp_pred):.3f}")
    print(f"SL:  MAE={mean_absolute_error(y_sl_te, sl_pred):.4f}  R2={r2_score(y_sl_te, sl_pred):.3f}")

    try:
        p_tie = cls_tie.predict_proba(Xte)[:,1]
    except Exception:
        z = cls_tie.decision_function(Xte); p_tie = (z - z.min())/(z.max()-z.min()+1e-12)
    auc = roc_auc_score(y_tie.iloc[test_idx], p_tie)
    ap = average_precision_score(y_tie.iloc[test_idx], p_tie)
    print(f"TIE: AUC={auc:.3f}  AP={ap:.3f}")

    # сохраняем бандлы и TE-карту
    bundle_tp = {"model": reg_tp, "features": feature_names, "min_rr": float(args.min_rr)}
    bundle_sl = {"model": reg_sl, "features": feature_names, "min_rr": float(args.min_rr)}
    bundle_tie= {"model": cls_tie, "features": feature_names}

    dump(bundle_tp, os.path.join(args.outdir, "tp_reg.pkl"))
    dump(bundle_sl, os.path.join(args.outdir, "sl_reg.pkl"))
    dump(bundle_tie, os.path.join(args.outdir, "tie_cls.pkl"))

    with open(os.path.join(args.outdir, "symbol_te.json"), "w") as f:
        json.dump({"map": te_map, "global": te_global}, f, ensure_ascii=False, indent=2)

    print("Saved →", args.outdir)
    print("Features:", len(feature_names))
    print("TE keys:", list(te_global.keys()))
if __name__=="__main__":
    main()