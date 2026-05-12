# predict/tp_entry/score_full.py
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="parquet с фичами для скоринга (например, *_feats.parquet)")
    ap.add_argument("--model-dir", required=True, help="директория с артефактами модели (model.pkl|model.txt, feature_names.json)")
    ap.add_argument("--symbol", required=False, default=None, help="фильтр по символу (например, BTCUSDT)")
    ap.add_argument("--side", required=False, default=None, help="фильтр по стороне (BUY/SELL)")
    ap.add_argument("--out", required=True, help="куда сохранить parquet с вероятностями")
    ap.add_argument("--apply-calibrator", type=int, default=1,
                    help="1=применять сохранённый калибратор (по умолчанию), 0=игнорировать")
    # опциональные пост-веса (совместимость с train_model, если когда-нибудь понадобятся)
    ap.add_argument("--weight-market-heat", type=float, default=0.0)
    ap.add_argument("--weight-sess-us", type=float, default=0.0)
    return ap.parse_args()


def _post_weighting(p: np.ndarray, df: pd.DataFrame, w_heat: float, w_us: float) -> np.ndarray:
    out = p.copy()
    if w_heat != 0 and "market_heat" in df.columns:
        logit = np.log(np.clip(out, 1e-9, 1-1e-9) / np.clip(1-out, 1e-9, 1))
        logit = logit * (1.0 + w_heat * (df["market_heat"].astype(float).values))
        out = 1.0 / (1.0 + np.exp(-logit))
    if w_us != 0 and "sess_us" in df.columns:
        logit = np.log(np.clip(out, 1e-9, 1-1e-9) / np.clip(1-out, 1e-9, 1))
        logit = logit + w_us * (df["sess_us"].astype(float).values)
        out = 1.0 / (1.0 + np.exp(-logit))
    return out


def _load_artifacts(model_dir: Path):
    """
    Возвращает кортеж:
      kind: "sk" (LGBMClassifier) | "booster" (LightGBM Booster)
      model: объект модели
      calibrator: калибратор или None
      num_iter: int или None
      feat_names: list[str] или None
    """
    feat_path = model_dir / "feature_names.json"
    feat_names = None
    if feat_path.exists():
        feat_names = json.loads(feat_path.read_text())
        print(f"[INFO] feature list loaded: {feat_path} ({len(feat_names)})")

    pkl_path = model_dir / "model.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and "model" in blob:
            model = blob["model"]
            calibrator = blob.get("calibrator", None)
            num_iter = blob.get("num_iter", None)
            # распознаем тип (классик или Booster)
            if isinstance(model, lgb.Booster):
                return "booster", model, calibrator, (int(num_iter) if num_iter else None), feat_names
            else:
                # ожидаем LGBMClassifier-совместимый интерфейс
                return "sk", model, calibrator, (int(num_iter) if num_iter else None), feat_names

    # fallback: Booster из model.txt
    txt_path = model_dir / "model.txt"
    if txt_path.exists():
        booster = lgb.Booster(model_file=str(txt_path))
        return "booster", booster, None, None, feat_names

    raise RuntimeError(f"В {model_dir} не найдено ни model.pkl, ни model.txt")


def _align_features(df: pd.DataFrame, feat_names):
    if not feat_names:
        # если список фич не сохранён — используем все числовые, исключая служебные
        blacklist = {"entry_ts", "symbol", "side", "y"}
        cols = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
        X = df[cols].copy()
        X = X.astype(float)
        return X, cols
    # жёсткое выравнивание по сохранённому списку
    X = pd.DataFrame(index=df.index)
    for c in feat_names:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            X[c] = 0.0
    X = X.astype(float)
    return X, feat_names


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    # 1) модель и артефакты
    kind, model, calibrator, num_iter, feat_names = _load_artifacts(model_dir)

    # 2) данные
    df = pd.read_parquet(args.feats)
    if "entry_ts" not in df.columns:
        raise ValueError("В feats нет 'entry_ts'")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts")

    if args.symbol:
        df = df[df.get("symbol", "").astype(str).str.upper() == args.symbol.upper()]
    if args.side:
        df = df[df.get("side", "").astype(str).str.upper() == args.side.upper()]

    if df.empty:
        raise ValueError("После фильтров нет строк для скоринга.")

    # 3) порядок фич
    X, used_feats = _align_features(df, feat_names)

    # 4) predict сырых вероятностей
    if kind == "sk":
        # LightGBMClassifier API
        if hasattr(model, "predict_proba"):
            p_raw = model.predict_proba(X, num_iteration=num_iter)[:, 1] if num_iter else model.predict_proba(X)[:, 1]
        else:
            # fallback: margin -> sigmoid
            raw = model.predict(X, num_iteration=num_iter) if num_iter else model.predict(X)
            p_raw = 1.0 / (1.0 + np.exp(-raw))
    else:
        # Booster API
        # В бинарной задаче booster.predict(..., raw_score=False) возвращает prob of positive
        p_raw = model.predict(X, num_iteration=num_iter, raw_score=False)
        p_raw = np.asarray(p_raw).reshape(-1)

    # 5) Калибровка (если есть и включена пользователем)
    p = p_raw
    if args.apply_calibrator and calibrator is not None:
        try:
            if hasattr(calibrator, "transform"):
                # IsotonicRegression
                p = calibrator.transform(p_raw)
            elif hasattr(calibrator, "predict_proba"):
                # Platt / LogisticRegression
                p = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
            else:
                print("[WARN] calibrator найден, но не поддерживает transform/predict_proba — пропуск.")
        except Exception as e:
            print(f"[WARN] ошибка при применении calibrator: {e} — используем сырые p_raw.")
            p = p_raw
    else:
        if not args.apply_calibrator:
            print("[INFO] пропуск калибратора (--apply-calibrator=0)")
    # 6) пост-веса (опционально флагами)
    if args.weight_market_heat != 0.0 or args.weight_sess_us != 0.0:
        p = _post_weighting(p, df, args.weight_market_heat, args.weight_sess_us)

    # 7) сохранить
    out = df[["entry_ts"]].copy()
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"]
    if "side" in df.columns:
        out["side"] = df["side"]
    out["p"] = p.astype(float)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[OK] saved: {args.out}  | rows={len(out)}")


if __name__ == "__main__":
    main()