from pathlib import Path
import pandas as pd

# где лежат:
FEATS_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_feats_175")
STATES_DIR = Path("reports/features/dataset_ks_v11_by_symbol_states_175")

# куда кладём результат:
OUT_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_states_175")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY = ["symbol", "entry_ts", "side"]

for feats_path in sorted(FEATS_DIR.glob("*.parquet")):
    symbol = feats_path.stem
    states_path = STATES_DIR / feats_path.name

    if not states_path.exists():
        print(f"SKIP {symbol}: no states file {states_path}")
        continue

    df_f = pd.read_parquet(feats_path)
    df_s = pd.read_parquet(states_path)

    # sanity по ключам
    if not set(KEY).issubset(df_f.columns):
        print(f"SKIP {symbol}: missing keys in feats: {set(KEY) - set(df_f.columns)}")
        continue
    if not set(KEY).issubset(df_s.columns):
        print(f"SKIP {symbol}: missing keys in states: {set(KEY) - set(df_s.columns)}")
        continue

    # приводим типы времени, чтобы merge не промазал
    df_f["entry_ts"] = pd.to_datetime(df_f["entry_ts"])
    df_s["entry_ts"] = pd.to_datetime(df_s["entry_ts"])

    # оставляем в states только нужные колонки,
    # чтобы не плодить дублей вроде ret_x / ret_y
    state_cols_keep = KEY + [
        "regime_trend",
        "regime_vol",
        "label_focus",
        "state_focus",
        "is_focus",
        "sample_weight",
    ]
    missing_state_cols = [c for c in state_cols_keep if c not in df_s.columns]
    if missing_state_cols:
        print(f"SKIP {symbol}: missing state columns {missing_state_cols}")
        continue

    df_s_small = df_s[state_cols_keep].drop_duplicates(KEY)

    trades_f = df_f[KEY].drop_duplicates().shape[0]
    trades_s = df_s_small[KEY].drop_duplicates().shape[0]

    if trades_f != trades_s:
        print(
            f"WARN {symbol}: trades mismatch feats={trades_f}, states={trades_s} "
            f"(merge всё равно делаем, но стоит потом глянуть)"
        )

    df_merged = df_f.merge(
        df_s_small,
        on=KEY,
        how="left",
        validate="many_to_one",
    )

    n_rows = len(df_merged)
    n_focus = int(df_merged["is_focus"].sum())
    n_na = int(df_merged["is_focus"].isna().sum())

    out_path = OUT_DIR / feats_path.name
    df_merged.to_parquet(out_path, index=False)

    print(
        f"{symbol}: rows={n_rows}, trades_feats={trades_f}, "
        f"trades_states={trades_s}, focus_rows={n_focus}, "
        f"na_is_focus={n_na} -> {out_path}"
    )

print("DONE. Merged feats+states saved to", OUT_DIR)