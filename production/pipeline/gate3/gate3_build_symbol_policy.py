import os
import numpy as np
import pandas as pd


#python production/pipeline/gate3_recearch/gate3_build_symbol_policy.py
IN_EDGE = "production/models/gate3_recearch/_ACTIVE_EDGE_full_timeline.csv"
OUT_POLICY = "production/models/ks/gate3_symbol_policy.csv"

STRONG_SCORE = 0.6

NAME_MAP = {
    "active_squeeze_up": "active_pa_atr_squeeze_break_up",
    "active_bos_up": "active_pa_bos_up_24",
    "active_squeeze_dn": "active_pa_atr_squeeze_break_dn",
    "active_bos_dn": "active_pa_bos_dn_24",
}

# =========================
# LOAD
# =========================
edge = pd.read_csv(IN_EDGE)



for c in ["lift_mfe", "lift_p_hit", "lift_time", "n"]:
    if c in edge.columns:
        edge[c] = pd.to_numeric(edge[c], errors="coerce")

edge["symbol"] = edge["symbol"].astype(str)
edge["pattern"] = edge["pattern"].replace(NAME_MAP)
edge["side"] = edge["side"].fillna("").astype(str).str.lower().str.strip()

ALL_PATTERNS = sorted(edge["pattern"].dropna().unique())
LONG_PATTERNS = [p for p in ALL_PATTERNS if p.endswith("_up")]
SHORT_PATTERNS = [p for p in ALL_PATTERNS if p.endswith("_dn")]

edge["lift_time"] = pd.to_numeric(edge["lift_time"], errors="coerce").fillna(0.0)

edge["score_raw"] = (
    edge["lift_mfe"].fillna(0.0)
    + edge["lift_p_hit"].fillna(0.0)
    + 0.1 * edge["lift_time"]
)

# =========================
# PIVOT (основа)
# =========================
pivot = edge.pivot_table(
    index="symbol",
    columns="pattern",
    values="score_raw",
    aggfunc="max",
).reset_index()

pivot.columns.name = None

for c in ALL_PATTERNS:
    if c not in pivot.columns:
        pivot[c] = 0.0

pivot = pivot[["symbol", *ALL_PATTERNS]].copy()

# =========================
# BASE DF
# =========================
symbols = pd.DataFrame({"symbol": sorted(edge["symbol"].unique())})
df = symbols.merge(pivot, on="symbol", how="left")
df[ALL_PATTERNS] = df[ALL_PATTERNS].fillna(0.0)
# =========================
# N PER PATTERN
# =========================
n_map = (
    edge.groupby(["symbol", "pattern"])["n"]
    .max()
    .unstack()
    .fillna(0.0)
)

for p in ALL_PATTERNS:
    if p not in n_map.columns:
        n_map[p] = 0.0

n_map = n_map[ALL_PATTERNS].reset_index()

df = df.merge(n_map, on="symbol", how="left", suffixes=("", "_n"))

# =========================
# WEIGHTS
# =========================
for p in ALL_PATTERNS:
    df[f"{p}_weight"] = (
            pd.to_numeric(df[p], errors="coerce").fillna(0.0)
            * np.log1p(pd.to_numeric(df[f"{p}_n"], errors="coerce").fillna(0.0))
    )

# =========================
# SCORES
# =========================
long_cols = [f"{p}_weight" for p in LONG_PATTERNS]
short_cols = [f"{p}_weight" for p in SHORT_PATTERNS]

df["gate3_score_long"] = (
    df[long_cols].sum(axis=1)
    / (df[long_cols] != 0).sum(axis=1).replace(0, np.nan)
).fillna(0.0) + df[long_cols].quantile(0.75, axis=1).fillna(0.0)

df["gate3_score_short"] = (
    df[short_cols].sum(axis=1)
    / (df[short_cols] != 0).sum(axis=1).replace(0, np.nan)
).fillna(0.0) + df[short_cols].quantile(0.75, axis=1).fillna(0.0)

df["gate3_top_pattern_long"] = df[long_cols].max(axis=1)
df["gate3_top_pattern_short"] = df[short_cols].max(axis=1)

df["gate3_side_bias"] = df["gate3_score_long"] - df["gate3_score_short"]

# =========================
# ENABLE
# =========================

df["gate3_score_long_z"] = (
    (df["gate3_score_long"] - df["gate3_score_long"].mean()) /
    (df["gate3_score_long"].std() + 1e-9)
)

df["gate3_score_short_z"] = (
    (df["gate3_score_short"] - df["gate3_score_short"].mean()) /
    (df["gate3_score_short"].std() + 1e-9)
)

df["gate3_rank_long"] = df["gate3_score_long"].rank(pct=True)
df["gate3_rank_short"] = df["gate3_score_short"].rank(pct=True)


q_long_en = df["gate3_score_long"].quantile(0.50)
q_short_en = df["gate3_score_short"].quantile(0.50)

df["gate3_enabled_long"] = (df["gate3_score_long"] >= q_long_en).astype(int)
df["gate3_enabled_short"] = (df["gate3_score_short"] >= q_short_en).astype(int)
df["gate3_enabled"] = (
    (df["gate3_enabled_long"] == 1)
    | (df["gate3_enabled_short"] == 1)
).astype(int)

# =========================
# MODE
# =========================
q_long = df["gate3_score_long"].quantile(0.75)
q_short = df["gate3_score_short"].quantile(0.75)

df["gate3_mode_long"] = np.where(
    df["gate3_score_long"] >= q_long,
    "strong",
    np.where(df["gate3_enabled_long"] == 1, "normal", "disabled"),
)

df["gate3_mode_short"] = np.where(
    df["gate3_score_short"] >= q_short,
    "strong",
    np.where(df["gate3_enabled_short"] == 1, "normal", "disabled"),
)


# =========================
# REASONS
# =========================
df["reason_long"] = np.where(
    df["gate3_enabled_long"] == 1,
    "enabled",
    "disabled"
)

df["reason_short"] = np.where(
    df["gate3_enabled_short"] == 1,
    "enabled",
    "disabled"
)

# =========================
# OUTPUT
# =========================
out = df[[
    "symbol",
    "gate3_enabled",

    "gate3_enabled_long",
    "gate3_mode_long",
    "gate3_score_long",
    "reason_long",

    "gate3_enabled_short",
    "gate3_mode_short",
    "gate3_score_short",
    "reason_short",

    "gate3_score_long_z",
    "gate3_score_short_z",
    "gate3_rank_long",
    "gate3_rank_short",

    "gate3_top_pattern_long",
    "gate3_top_pattern_short",
    "gate3_side_bias",

    *ALL_PATTERNS
]].sort_values(
    ["gate3_enabled", "gate3_score_long", "gate3_score_short"],
    ascending=[False, False, False],
).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_POLICY), exist_ok=True)
out.to_csv(OUT_POLICY, index=False)

print("WROTE", OUT_POLICY)

print("\nLONG SUMMARY")
print(out.groupby(["gate3_enabled_long", "gate3_mode_long"]).size().reset_index(name="n"))

print("\nSHORT SUMMARY")
print(out.groupby(["gate3_enabled_short", "gate3_mode_short"]).size().reset_index(name="n"))

print("\nHEAD")
print(out.head(40).to_string(index=False))

print("\nEDGE PATTERNS:")
print(sorted(edge["pattern"].unique()))

print("\nPIVOT COLS:")
print(sorted(pivot.columns))