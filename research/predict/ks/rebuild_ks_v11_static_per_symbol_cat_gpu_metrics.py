from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATA_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_feats_175")
MODELS_DIR = Path("models/ks_v11_static_per_symbol_cat_gpu")
OUT_METRICS = Path("reports/features/ks_v11_static_per_symbol_cat_gpu_metrics.csv")
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

metrics = []

for p in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(p)

    required = {"entry_ts", "symbol", "side", "pnl_net"}
    if not required.issubset(df.columns):
        print("SKIP", p.name, "missing", required - set(df.columns))
        continue

    symbol = str(df["symbol"].iloc[0])
    model_path = MODELS_DIR / f"{symbol}.cbm"
    if not model_path.exists():
        print("SKIP", symbol, "no model", model_path)
        continue

    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])

    if len(df) < 1000:
        print("SKIP", symbol, "too few rows:", len(df))
        continue

    target_col = "pnl_net"
    y = df[target_col].astype(float)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col]

    split_idx = int(len(df) * 0.8)
    X_train = df.loc[:split_idx - 1, feature_cols]
    y_train = y.iloc[:split_idx]

    X_valid = df.loc[split_idx:, feature_cols]
    y_valid = y.iloc[split_idx:]

    print(f"=== {symbol} ===")
    print("rows:", len(df), "features:", len(feature_cols))

    model = CatBoostRegressor()
    model.load_model(model_path)

    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_valid)

    train_rmse = float(np.sqrt(np.mean((y_tr_pred - y_train) ** 2)))
    valid_rmse = float(np.sqrt(np.mean((y_va_pred - y_valid) ** 2)))

    print(
        "rmse_train:", train_rmse,
        "rmse_valid:", valid_rmse,
    )

    metrics.append(
        dict(
            symbol=symbol,
            n_rows=len(df),
            n_features=len(feature_cols),
            train_rmse=train_rmse,
            valid_rmse=valid_rmse,
            model_path=str(model_path),
        )
    )

if metrics:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print("Saved metrics ->", OUT_METRICS, "rows:", len(metrics_df))
else:
    print("No metrics computed")