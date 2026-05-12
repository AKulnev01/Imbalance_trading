from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_states_175")

FOCUS_MODELS  = Path("models/ks_v11_state_per_symbol_cat_focus_only")
WEIGHT_MODELS = Path("models/ks_v11_state_per_symbol_cat_weighted_gpu")

OUT_CSV = Path("reports/features/ks_v11_state_models_backtest_compare.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

results = []

for p in sorted(DATA_DIR.glob("*.parquet")):
    symbol = p.stem

    focus_path  = FOCUS_MODELS  / f"{symbol}.cbm"
    weight_path = WEIGHT_MODELS / f"{symbol}.cbm"

    # нужны обе модели
    if not focus_path.exists() or not weight_path.exists():
        print(f"=== {symbol} ===")
        print(f"  SKIP: no focus/weight model")
        continue

    print(f"=== {symbol} ===")

    df = pd.read_parquet(p)

    required = {"entry_ts", "symbol", "side", "pnl_net"}
    if not required.issubset(df.columns):
        print("  SKIP: missing", required - set(df.columns))
        continue

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values("entry_ts").reset_index(drop=True)

    # ID сделки = (entry_ts, side)
    df["_trade_id"] = df.groupby(["entry_ts", "side"]).ngroup()
    n_trades = int(df["_trade_id"].nunique())

    # фичи как в тренере
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "pnl_net",
        "label_focus",
        "state_focus",
        "is_focus",
        "sample_weight",
        "pnl_max",
        "pnl_min",
        "pnl_spread",
    }
    feature_cols = [c for c in num_cols if c not in exclude]

    if not feature_cols:
        print("  SKIP: no numeric features after exclude")
        continue

    X = df[feature_cols]

    # ---------- STATIC: глобальный ks по сетке ----------
    ks_cols = ["ks_tp_scale", "ks_sl_scale", "ks_ttl_hours"]
    if not set(ks_cols).issubset(df.columns):
        print("  SKIP: no ks columns", set(ks_cols) - set(df.columns))
        continue

    ks_grp = df.groupby(ks_cols)["pnl_net"].mean().reset_index()
    best_idx = ks_grp["pnl_net"].idxmax()

    best_tp, best_sl, best_ttl = (
        ks_grp.loc[best_idx, "ks_tp_scale"],
        ks_grp.loc[best_idx, "ks_sl_scale"],
        ks_grp.loc[best_idx, "ks_ttl_hours"],
    )

    df_static = df[
        (df["ks_tp_scale"] == best_tp)
        & (df["ks_sl_scale"] == best_sl)
        & (df["ks_ttl_hours"] == best_ttl)
    ].copy()

    static_trades = int(df_static["_trade_id"].nunique())
    if static_trades != n_trades:
        print(
            f"  WARN: static trades {static_trades} != total trades {n_trades}, "
            f"используем то, что есть"
        )

    pnl_static = float(df_static.drop_duplicates("_trade_id")["pnl_net"].sum())

    # ---------- focus_only модель ----------
    model_focus = CatBoostRegressor()
    model_focus.load_model(str(focus_path))

    df["pred_focus"] = model_focus.predict(X)
    idx_focus = df.groupby("_trade_id")["pred_focus"].idxmax()
    pnl_focus = float(df.loc[idx_focus, "pnl_net"].sum())

    # ---------- weighted модель ----------
    model_weight = CatBoostRegressor()
    model_weight.load_model(str(weight_path))

    df["pred_weight"] = model_weight.predict(X)
    idx_weight = df.groupby("_trade_id")["pred_weight"].idxmax()
    pnl_weight = float(df.loc[idx_weight, "pnl_net"].sum())

    # ---------- дельты ----------
    edge_focus_vs_static  = pnl_focus  - pnl_static
    edge_weight_vs_static = pnl_weight - pnl_static
    edge_focus_vs_weight  = pnl_focus  - pnl_weight

    denom = abs(pnl_static) + 1e-9  # чтобы не делить на 0
    edge_focus_vs_static_rel  = edge_focus_vs_static  / denom
    edge_weight_vs_static_rel = edge_weight_vs_static / denom

    results.append(
        dict(
            symbol=symbol,
            n_trades=n_trades,
            pnl_static=pnl_static,
            pnl_focus=pnl_focus,
            pnl_weight=pnl_weight,
            edge_focus_vs_static=edge_focus_vs_static,
            edge_weight_vs_static=edge_weight_vs_static,
            edge_focus_vs_weight=edge_focus_vs_weight,
            edge_focus_vs_static_rel=edge_focus_vs_static_rel,
            edge_weight_vs_static_rel=edge_weight_vs_static_rel,
        )
    )

    print(
        f"  trades={n_trades} | "
        f"static={pnl_static:.2f} | "
        f"focus={pnl_focus:.2f} | "
        f"weight={pnl_weight:.2f}"
    )

if results:
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print("\nSaved backtest compare ->", OUT_CSV, "rows:", len(out_df))

    print("\nTop 10 по edge_weight_vs_static (абсолют):")
    print(
        out_df.sort_values("edge_weight_vs_static", ascending=False)
        .head(10)[["symbol", "pnl_static", "pnl_weight", "edge_weight_vs_static"]]
    )

    print("\nTop 10 по edge_weight_vs_static_rel (отн.):")
    print(
        out_df.sort_values("edge_weight_vs_static_rel", ascending=False)
        .head(10)[["symbol", "pnl_static", "pnl_weight", "edge_weight_vs_static_rel"]]
    )

    print("\nСводка по победам:")
    print(
        "weight > static:",
        int((out_df["edge_weight_vs_static"] > 0).sum()),
        "символов",
    )
    print(
        "focus  > static:",
        int((out_df["edge_focus_vs_static"] > 0).sum()),
        "символов",
    )
    print(
        "weight > focus :",
        int((out_df["edge_focus_vs_weight"] < 0).sum()),
        "символов",
    )
else:
    print("No symbols backtested")