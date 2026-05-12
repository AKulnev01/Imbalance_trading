from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRanker

# === ПУТИ ===
DATA_PATH        = Path("reports/features/bonk/bonk_v15_weighted_q.parquet")
MODEL_RANK_PATH  = Path("models/bonk_v15_rank_q_gpu.cbm")
MODEL_TRANS_PATH = Path("models/bonk_v15_transformer_rank.pt")

TP_BASE = 0.13
SL_BASE = 0.04


# ==========================
#   МОДЕЛЬ ТРАНСФОРМЕРА
#   (совпадает с train_bonk_v15_transformer_rank.py: in_proj / out_proj)
# ==========================

class KSSeqTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # линейная проекция фич → скрытое пространство
        self.in_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # голова: логит per-KS
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, n_features]
        return: [B, L] — логиты per-KS
        """
        h = self.in_proj(x)          # [B, L, d_model]
        h = self.encoder(h)          # [B, L, d_model]
        logits = self.out_proj(h).squeeze(-1)  # [B, L]
        return logits


# ==========================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def load_data():
    print("[LOAD]", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values(
        ["entry_ts", "side", "ks_tp_mult", "ks_sl_mult"]
    ).reset_index(drop=True)

    bars = (
        df[["entry_ts", "side"]]
        .drop_duplicates()
        .sort_values(["entry_ts", "side"])
        .reset_index(drop=True)
    )
    bars["group_id"] = np.arange(len(bars), dtype=np.int64)

    df = df.merge(bars, on=["entry_ts", "side"], how="left")

    if df["group_id"].isna().any():
        raise SystemExit("group_id не проставились для части строк")

    print("[INFO] total rows:", len(df))
    print("[INFO] unique bars:", bars.shape[0])
    return df, bars


def get_feature_cols(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "pnl_net",
        "ks_ttl_hours",
        "sample_weight",
        "is_focus",
        "group_id",
    }
    feature_cols = [c for c in num_cols if c not in exclude]
    print("[INFO] num features:", len(feature_cols))
    print("[INFO] first 20 features:", feature_cols[:20])
    return feature_cols


def compute_static_best_ks(df: pd.DataFrame):
    grid = (
        df.groupby(["side", "ks_tp_mult", "ks_sl_mult"], as_index=False)["pnl_net"]
        .mean()
    )
    best_rows = (
        grid.sort_values("pnl_net", ascending=False)
        .groupby("side", as_index=False)
        .first()
    )

    best_ks = {}
    print("\n[STATIC BEST_KS by side]:")
    for _, row in best_rows.iterrows():
        side = int(row["side"])
        tp_mult = float(row["ks_tp_mult"])
        sl_mult = float(row["ks_sl_mult"])
        best_ks[side] = (tp_mult, sl_mult)
        rr = (TP_BASE * tp_mult) / (SL_BASE * sl_mult)
        print(
            f"  side={side:+d}: tp_mult={tp_mult:.4f}, "
            f"sl_mult={sl_mult:.4f}, RR≈{rr:.3f}"
        )

    return best_ks


def apply_static_strategy(df: pd.DataFrame, bars: pd.DataFrame, best_ks: dict):
    rows = []
    for side, (tp, sl) in best_ks.items():
        mask = (
            (df["side"] == side)
            & (df["ks_tp_mult"] == tp)
            & (df["ks_sl_mult"] == sl)
        )
        sub = df.loc[mask, ["group_id", "pnl_net"]]
        rows.append(sub)

    static_df = pd.concat(rows, axis=0)

    if static_df["group_id"].nunique() != bars["group_id"].nunique():
        print(
            "[WARN] static: covered:",
            static_df["group_id"].nunique(),
            "bars:",
            bars["group_id"].nunique(),
        )

    static_df = static_df.groupby("group_id", as_index=False)["pnl_net"].first()
    static_df = static_df.rename(columns={"pnl_net": "pnl_static"})
    return static_df


def apply_catboost_ranker(df: pd.DataFrame, bars: pd.DataFrame, feature_cols):
    print("\n[LOAD RANKER]", MODEL_RANK_PATH)
    ranker = CatBoostRanker()
    ranker.load_model(MODEL_RANK_PATH)

    print("[PREDICT] CatBoostRanker scores...")
    df["score_rank"] = ranker.predict(df[feature_cols])

    idx_max = df.groupby("group_id")["score_rank"].idxmax()
    rank_df = (
        df.loc[idx_max, ["group_id", "pnl_net"]]
        .rename(columns={"pnl_net": "pnl_ranker"})
        .reset_index(drop=True)
    )

    if rank_df["group_id"].nunique() != bars["group_id"].nunique():
        print(
            "[WARN] ranker: covered:",
            rank_df["group_id"].nunique(),
            "bars:",
            bars["group_id"].nunique(),
        )

    return rank_df


def apply_transformer_ranker(df: pd.DataFrame, feature_cols):
    print("\n[TRANSFORMER] build group->indices map...")
    group_to_indices = {gid: g.index.values for gid, g in df.groupby("group_id")}

    group_sizes = {g: len(idxs) for g, idxs in group_to_indices.items()}
    sizes_series = pd.Series(group_sizes)
    seq_len = int(sizes_series.mode().iloc[0])
    valid_groups = [g for g, sz in group_sizes.items() if sz == seq_len]

    print(
        f"[TRANSFORMER] seq_len={seq_len}, "
        f"valid_groups={len(valid_groups)}/{len(group_to_indices)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[TRANSFORMER] device:", device)

    print("[TRANSFORMER] load model:", MODEL_TRANS_PATH)
    # ЯВНО weights_only=False, чтобы не ругался на numpy._core.multiarray._reconstruct
    try:
        state = torch.load(MODEL_TRANS_PATH, map_location=device, weights_only=False)
    except TypeError:
        # если старая версия torch без weights_only
        state = torch.load(MODEL_TRANS_PATH, map_location=device)

    if isinstance(state, nn.Module):
        model = state
    else:
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        model = KSSeqTransformer(
            n_features=len(feature_cols),
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    BATCH = 64
    rows = []

    groups_list = valid_groups

    def batch_iter(groups):
        for i in range(0, len(groups), BATCH):
            yield groups[i: i + BATCH]

    with torch.no_grad():
        for batch_groups in batch_iter(groups_list):
            batch_feats = []
            batch_group_ids = []
            batch_idx_lists = []

            for gid in batch_groups:
                idxs = group_to_indices[gid]
                feat = df.loc[idxs, feature_cols].to_numpy(dtype=np.float32)
                if feat.shape[0] != seq_len:
                    continue
                batch_feats.append(feat)
                batch_group_ids.append(gid)
                batch_idx_lists.append(idxs)

            if not batch_feats:
                continue

            x = torch.tensor(
                np.stack(batch_feats), dtype=torch.float32, device=device
            )  # [B, L, F]

            logits = model(x)  # [B, L]
            if logits.dim() == 3:
                logits = logits.squeeze(-1)

            best_idx = logits.argmax(dim=1).cpu().numpy()

            for gid, idxs, bi in zip(
                batch_group_ids, batch_idx_lists, range(len(batch_idx_lists))
            ):
                row_idx = idxs[best_idx[bi]]
                pnl = float(df.at[row_idx, "pnl_net"])
                rows.append((gid, pnl))

    trans_df = pd.DataFrame(rows, columns=["group_id", "pnl_transformer"])
    print("[TRANSFORMER] covered groups:", trans_df["group_id"].nunique())
    return trans_df


def join_bar_results(
    df: pd.DataFrame,
    static_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    trans_df: pd.DataFrame,
):
    bar_meta = (
        df.sort_values("group_id")
        .drop_duplicates("group_id")[
            [
                "group_id",
                "entry_ts",
                "side",
                "is_focus",
                "vol_rel",
                "rng_norm",
                "quality_score",
            ]
        ]
    )

    res = bar_meta.merge(static_df, on="group_id", how="left")
    res = res.merge(rank_df, on="group_id", how="left")
    res = res.merge(trans_df, on="group_id", how="left")
    return res


def summarize_subset(res: pd.DataFrame, mask: pd.Series, name: str):
    sub = res.loc[mask].copy()
    n = len(sub)
    if n == 0:
        print(f"\n=== SUBSET {name}: 0 bars ===")
        return

    print(f"\n=== SUBSET {name}: bars={n} ===")

    for col, label in [
        ("pnl_static", "STATIC"),
        ("pnl_ranker", "RANKER"),
        ("pnl_transformer", "TRANS"),
    ]:
        s = sub[col].dropna()
        if s.empty:
            print(f"{label}: no data")
            continue
        mean = float(s.mean())
        med = float(s.median())
        pos = float((s > 0).mean())
        neg = float((s < 0).mean())
        print(
            f"{label}: mean={mean:.4f}, median={med:.4f}, "
            f"pos={pos:.3%}, neg={neg:.3%}"
        )

    both_cr = sub.dropna(subset=["pnl_static", "pnl_ranker"])
    if len(both_cr):
        win = float((both_cr["pnl_ranker"] > both_cr["pnl_static"]).mean())
        tie = float((both_cr["pnl_ranker"] == both_cr["pnl_static"]).mean())
        lose = float((both_cr["pnl_ranker"] < both_cr["pnl_static"]).mean())
        print(
            f"RANKER vs STATIC: win={win:.3%}, tie={tie:.3%}, lose={lose:.3%}"
        )

    both_ct = sub.dropna(subset=["pnl_static", "pnl_transformer"])
    if len(both_ct):
        win = float((both_ct["pnl_transformer"] > both_ct["pnl_static"]).mean())
        tie = float((both_ct["pnl_transformer"] == both_ct["pnl_static"]).mean())
        lose = float((both_ct["pnl_transformer"] < both_ct["pnl_static"]).mean())
        print(
            f"TRANS   vs STATIC: win={win:.3%}, tie={tie:.3%}, lose={lose:.3%}"
        )

    both_rt = sub.dropna(subset=["pnl_ranker", "pnl_transformer"])
    if len(both_rt):
        win = float((both_rt["pnl_transformer"] > both_rt["pnl_ranker"]).mean())
        tie = float((both_rt["pnl_transformer"] == both_rt["pnl_ranker"]).mean())
        lose = float((both_rt["pnl_transformer"] < both_rt["pnl_ranker"]).mean())
        print(
            f"TRANS   vs RANKER: win={win:.3%}, tie={tie:.3%}, lose={lose:.3%}"
        )


def main():
    df, bars = load_data()
    feature_cols = get_feature_cols(df)

    best_ks = compute_static_best_ks(df)
    static_df = apply_static_strategy(df, bars, best_ks)
    rank_df = apply_catboost_ranker(df, bars, feature_cols)
    trans_df = apply_transformer_ranker(df, feature_cols)

    res = join_bar_results(df, static_df, rank_df, trans_df)

    mask_all = (
        res["pnl_static"].notna()
        & res["pnl_ranker"].notna()
        & res["pnl_transformer"].notna()
    )
    mask_focus = mask_all & (res["is_focus"] == 1)
    mask_entry = mask_all & (
        (res["quality_score"] > 0.4)
        & (res["vol_rel"] > 1.0)
        & (res["rng_norm"] > 1.0)
    )

    summarize_subset(res, mask_all, "ALL BARS")
    summarize_subset(res, mask_focus, "FOCUS BARS")
    summarize_subset(res, mask_entry, "ENTRY-ZONE")

    out_path = Path(
        "reports/features/bonk/bonk_v15_eval_ks_cat_vs_static_vs_trans.parquet"
    )
    res.to_parquet(out_path)
    print("\n[SAVED per-bar results] ->", out_path)


if __name__ == "__main__":
    main()