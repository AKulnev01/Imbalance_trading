from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


DATA_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_states_175_CLEAN")
OUT_MODELS = Path("models/ks_v11_state_per_symbol_cat_focus_only_CLEAN")
OUT_MODELS.mkdir(parents=True, exist_ok=True)

metrics = []

for p in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(p)

    required = {"entry_ts", "symbol", "side", "pnl_net", "is_focus"}
    missing = required - set(df.columns)
    if missing:
        print("SKIP", p.name, "missing", missing)
        continue

    symbol = str(df["symbol"].iloc[0])

    # сортируем по времени
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values("entry_ts").reset_index(drop=True)

    # берём только фокусные режимы и СНОВА ресетим индекс
    df = df[df["is_focus"] == 1].copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) < 1000:
        print("SKIP", symbol, "too few focus rows:", len(df))
        continue

    target_col = "pnl_net"
    y = df[target_col].astype(float)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        target_col,
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
        print("SKIP", symbol, "no numeric features after exclude")
        continue

    split_idx = int(len(df) * 0.8)

    # ИСПОЛЬЗУЕМ iloc, чтобы резать по позиции
    X_train = df.iloc[:split_idx][feature_cols]
    y_train = y.iloc[:split_idx]

    X_valid = df.iloc[split_idx:][feature_cols]
    y_valid = y.iloc[split_idx:]

    # sanity-check
    assert len(X_train) == len(y_train), f"{symbol}: X_train {len(X_train)} != y_train {len(y_train)}"
    assert len(X_valid) == len(y_valid), f"{symbol}: X_valid {len(X_valid)} != y_valid {len(y_valid)}"

    print(f"=== {symbol} ===")
    print("rows_focus:", len(df), "features:", len(feature_cols))

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        depth=8,
        learning_rate=0.03,
        iterations=8000,
        l2_leaf_reg=3.0,
        random_seed=42,
        od_type="Iter",
        od_wait=500,
        task_type="CPU",
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=False,
    )

    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_valid)

    rmse_train = float(np.sqrt(np.mean((y_tr_pred - y_train) ** 2)))
    rmse_valid = float(np.sqrt(np.mean((y_va_pred - y_valid) ** 2)))

    model_path = OUT_MODELS / f"{symbol}.cbm"
    model.save_model(model_path)

    print(
        "saved model ->",
        model_path,
        "| rmse_train:",
        rmse_train,
        "| rmse_valid:",
        rmse_valid,
    )

    metrics.append(
        dict(
            symbol=symbol,
            n_rows=len(df),
            n_features=len(feature_cols),
            rmse_train=rmse_train,
            rmse_valid=rmse_valid,
            model_path=str(model_path),
        )
    )

if metrics:
    metrics_df = pd.DataFrame(metrics)
    metrics_path = Path(
        "reports/features/ks_v11_state_per_symbol_cat_focus_only_metrics.csv"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics ->", metrics_path, "rows:", len(metrics_df))
else:
    print("No models trained")