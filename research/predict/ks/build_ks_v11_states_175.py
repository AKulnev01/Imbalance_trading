from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("reports/features/dataset_ks_v11_by_symbol_with_feats_175")
OUT_DIR = Path("reports/features/dataset_ks_v11_by_symbol_states_175")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Гиперпараметры ----
STATE_FOCUS_Q = 0.7         # верхние 30% по "силе" свечи

# тренд и вола
RET_TREND_ROLL_N = 5        # окно по сделкам (не по времени!)
RET_TREND_THRESH = 0.003    # ~0.3% за 4h, порог тренда

VOL_Q_LOW = 0.3
VOL_Q_HIGH = 0.7

# веса для обучения
WEIGHT_LABEL = 1.0
WEIGHT_STATE = 0.5


for path in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(path)

    required = {"symbol", "entry_ts", "side", "pnl_net"}
    missing = required - set(df.columns)
    if missing:
        print(f"SKIP {path.name} missing {missing}")
        continue

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    key = ["symbol", "entry_ts", "side"]

    has_ret = "ret" in df.columns
    has_vol = "volat_ret12" in df.columns

    agg = {
        "pnl_net": ["max", "min"],
    }
    if has_ret:
        agg["ret"] = "first"
    if has_vol:
        agg["volat_ret12"] = "first"

    g = df.groupby(key).agg(agg)
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()

    # базовые pnl-метрики
    g["pnl_max"] = g["pnl_net_max"]
    g["pnl_min"] = g["pnl_net_min"]
    g["pnl_spread"] = g["pnl_max"] - g["pnl_min"]

    # state-фичи
    if has_ret:
        g["ret"] = g["ret_first"]
    else:
        g["ret"] = 0.0

    if has_vol:
        g["volat_ret12"] = g["volat_ret12_first"]
    else:
        g["volat_ret12"] = 0.0

    # сортируем по времени для построения тренда
    g = g.sort_values("entry_ts").reset_index(drop=True)

    # режим тренда
    if (g["ret"] != 0).any():
        roll_ret = g["ret"].rolling(RET_TREND_ROLL_N, min_periods=1).mean()
        g["regime_trend"] = "range"
        g.loc[roll_ret > RET_TREND_THRESH, "regime_trend"] = "trend_up"
        g.loc[roll_ret < -RET_TREND_THRESH, "regime_trend"] = "trend_down"
    else:
        g["regime_trend"] = "range"

    # режим волатильности
    if (g["volat_ret12"] > 0).any():
        q_low = g["volat_ret12"].quantile(VOL_Q_LOW)
        q_high = g["volat_ret12"].quantile(VOL_Q_HIGH)

        g["regime_vol"] = "mid_vol"
        g.loc[g["volat_ret12"] <= q_low, "regime_vol"] = "low_vol"
        g.loc[g["volat_ret12"] >= q_high, "regime_vol"] = "high_vol"
    else:
        g["regime_vol"] = "mid_vol"

    # сила state: большое движение или высокая волька
    g["score_state"] = np.maximum(g["ret"].abs(), g["volat_ret12"])

    # ====== label_focus: адаптивно по квантилям pnl ======
    pnl_max_pos = g["pnl_max"].clip(lower=0)
    pnl_spread_pos = g["pnl_spread"].clip(lower=0)

    if (pnl_max_pos > 0).any():
        thr_best = pnl_max_pos[pnl_max_pos > 0].quantile(0.7)
    else:
        thr_best = 0.0

    if (pnl_spread_pos > 0).any():
        thr_spread = pnl_spread_pos[pnl_spread_pos > 0].quantile(0.7)
    else:
        thr_spread = 0.0

    g["label_focus"] = (
        (g["pnl_max"] >= thr_best) &
        (g["pnl_spread"] >= thr_spread)
    ).astype(int)

    # fallback: если всё равно 0, но есть положительные pnl_max → берём top 10%
    if g["label_focus"].sum() == 0 and (g["pnl_max"] > 0).any():
        n_top = max(1, int(len(g) * 0.1))
        top_idx = g["pnl_max"].nlargest(n_top).index
        g.loc[top_idx, "label_focus"] = 1

    # ====== state_focus: верхние 30% по score_state ======
    if (g["score_state"] > 0).any():
        thr_state = g["score_state"].quantile(STATE_FOCUS_Q)
        g["state_focus"] = (g["score_state"] >= thr_state).astype(int)
    else:
        g["state_focus"] = 0

    # общий фокус
    g["is_focus"] = ((g["label_focus"] == 1) | (g["state_focus"] == 1)).astype(int)

    # веса для обучения
    g["sample_weight"] = (
        1.0 +
        WEIGHT_LABEL * g["label_focus"] +
        WEIGHT_STATE * g["state_focus"]
    )

    out_path = OUT_DIR / path.name
    cols_out = key + [
        "pnl_max",
        "pnl_min",
        "pnl_spread",
        "ret",
        "volat_ret12",
        "regime_trend",
        "regime_vol",
        "score_state",
        "label_focus",
        "state_focus",
        "is_focus",
        "sample_weight",
    ]

    g[cols_out].to_parquet(out_path, index=False)

    print(
        f"{path.stem}: trades={len(g)}, "
        f"label_focus={g['label_focus'].sum()}, "
        f"state_focus={g['state_focus'].sum()}, "
        f"is_focus={g['is_focus'].sum()}"
    )

print("DONE. Saved per-trade state files to", OUT_DIR)