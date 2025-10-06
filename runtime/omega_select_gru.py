import os, json, numpy as np
from typing import Dict, Any, List
from models.nn_gru import load_model, predict
from ml.encode_theta import encode_theta, encode_fx
from ml.space_utils import load_space, sample_thetas

# ВЫХОД: печатает JSON с лучшим θ и маппингом в твои переменные окружения:
# MOMENTUM_TP_PCT, MOMENTUM_SL_PCT, DEFAULT_TTL_DAYS, а также вспомогательные.

def build_matrix(rows: List[dict], feature_names: List[str]) -> np.ndarray:
    mat = []
    for r in rows:
        v = [float(r.get(k, 0.0)) for k in feature_names]
        mat.append(v)
    return np.asarray(mat, dtype=np.float32)

def extract_feature_names(rows: List[dict]) -> List[str]:
    # Объединяем ключи, фиксируем порядок по имени
    keys = set()
    for r in rows:
        keys.update(r.keys())
    cols = sorted(list(keys))
    return cols

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--space-yaml", required=True)
    ap.add_argument("--context-json", required=True)   # dict рыночных фич (без префиксов)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, meta = load_model(args.models_dir, device=args.device)
    space = load_space(args.space_yaml)

    # текущий рыночный контекст
    with open(args.context_json, "r", encoding="utf-8") as f:
        ctx = json.load(f)   # {"vol_4h":..., "atr_24h":..., "spread_bp":..., ...}
    fx = encode_fx(ctx)

    # кандидаты θ
    thetas = sample_thetas(space, limit=4000)

    # строим строки признаков (fx + th)
    rows = []
    meta_th = []  # сырой θ рядом, чтобы вернуть пользователю
    for th in thetas:
        th_enc = encode_theta(th)
        row = {**fx, **th_enc}
        rows.append(row)
        meta_th.append(th)

    feat_names = extract_feature_names(rows)
    X_fx = np.asarray([[row.get(k, 0.0) for k in feat_names if k.startswith("fx__")] for row in rows], dtype=np.float32)
    X_th = np.asarray([[row.get(k, 0.0) for k in feat_names if k.startswith("th__")] for row in rows], dtype=np.float32)

    y_pred = predict(model, X_fx, X_th, device=args.device, seq_len=int(meta["seq_len"]))
    order = np.argsort(-y_pred)

    out = []
    for idx in order[:args.topk]:
        th = meta_th[idx]
        # маппинг в окружение твоего конфига:
        tp_pct = float(th["tp_rr"])
        sl_pct = float(th["sl_rr"])
        ttl_days = float(th["ttl_hours"]) / 24.0

        out.append({
            "theta": th,
            "pred_pnl_pct_after_cost": float(y_pred[idx]),
            "env_map": {
                "MOMENTUM_TP_PCT": tp_pct,
                "MOMENTUM_SL_PCT": sl_pct,
                "DEFAULT_TTL_DAYS": ttl_days
            }
        })

    print(json.dumps({"candidates": out}, ensure_ascii=False, indent=2))