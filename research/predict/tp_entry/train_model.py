# predict/tp_entry/train_model.py
import argparse, os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="reports/features/dataset_all_enhanced.parquet",
                    help="parquet с фичами")
    ap.add_argument("--outdir", default="reports/features/splits", help="куда складывать результаты")
    ap.add_argument("--symbols", default="BTCUSDT", help='символы через запятую или "ALL"')
    ap.add_argument("--side-filter", choices=["BOTH", "BUY", "SELL"], default="BOTH",
                    help="фильтр по направлению для обучения/оценки")
    ap.add_argument("--gap-days", type=int, default=4, help="дни эмбарго между train и test")

    # ----- разбиение -----
    ap.add_argument("--split-mode", choices=["percent", "dates"], default="percent",
                    help="percent: train/test по процентам таймлайна; dates: старые фолды")
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="доля времени до границы train (для split-mode=percent)")
    # временные сплиты (вариант 'dates', совместимость)
    ap.add_argument("--fold1-tr-end", default="2023-12-31")
    ap.add_argument("--fold1-va-end", default="2024-06-30")
    ap.add_argument("--fold2-tr-end", default="2024-06-30")
    ap.add_argument("--fold2-va-end", default="2024-12-31")
    ap.add_argument("--fold3-tr-end", default="2024-12-31")
    ap.add_argument("--fold3-va-end", default="2025-06-30")
    ap.add_argument("--test-start",   default="2025-07-04")
    ap.add_argument("--test-end",     default="2025-10-31")

    # пороги достаточности классов
    ap.add_argument("--min-pos-total", type=int, default=10, help="минимум TP на всём символе/стороне")
    ap.add_argument("--min-pos-test", type=int, default=5, help="минимум TP в тестовом окне")

    # LightGBM классика
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--min-child-samples", type=int, default=40)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--max-bin", type=int, default=255)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0,
                    help="вес положительного класса для LGBM")
    ap.add_argument("--auto-class-weight", action="store_true",
                    help="если включено — игнорирует scale-pos-weight и ставит neg/pos на каждом train-срезе")

    # Режим обучения / «лосс» / бустинг / ES-метрика
    ap.add_argument("--train-mode", choices=["per_symbol", "pooled", "hybrid"], default="hybrid",
                    help="per_symbol: как раньше; pooled: одна общая; hybrid: pooled претрейн + дообучение per-symbol")
    ap.add_argument("--loss", choices=["bce", "focal"], default="focal",
                    help="bce=классический BCE; focal=приближённо через class-weight (совместимо со sklearn API)")
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="для focal-приближения используем усиление класса (см. ниже)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="параметр-гамма (информативен, если перейдёшь на true custom fobj в будущем)")
    ap.add_argument("--boosting", choices=["gbdt", "goss"], default="gbdt",
                    help="GOSS помогает при сильном дисбалансе")
    ap.add_argument("--es-metric", choices=["auc", "ap"], default="ap",
                    help="метрика для ранней остановки на предтестовом окне")

    # ранняя остановка
    ap.add_argument("--early-stopping", type=int, default=200,
                    help="stopping_rounds для финального pre-test окна")
    ap.add_argument("--min-best-iter", type=int, default=50,
                    help="нижняя граница на best_iteration_ (защита от ложной ранней остановки)")
    ap.add_argument("--final-use-pretest-es", action="store_true",
                    help="ранняя остановка в финальном фите на окне перед тестом")
    ap.add_argument("--final-pretest-days", type=int, default=30,
                    help="длина предтестового окна для финального early stopping")

    # калибровка и пост-веса
    ap.add_argument("--calibrate", choices=["none", "platt", "isotonic"], default="none")
    ap.add_argument("--iso-min-pos", type=int, default=100,
                    help="минимум TP на предтестовой валидации для isotonic; иначе фолбэк на platt")
    ap.add_argument("--weight-market-heat", type=float, default=0.0, help="мультипликативный вес по market_heat (>=0)")
    ap.add_argument("--weight-sess-us", type=float, default=0.0, help="добавка к p в US-сессию (логит-сдвиг)")

    # финетюн при hybrid
    ap.add_argument("--finetune-iters", type=int, default=400,
                    help="кол-во деревьев для дообучения per-symbol при hybrid")
    ap.add_argument("--finetune-lr", type=float, default=0.01,
                    help="LR при дообучении per-symbol")

    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# =========================
# Метрики/утилиты
# =========================
def precision_at_k(y_true: np.ndarray, p: np.ndarray, frac: float = 0.05) -> float:
    n = len(p)
    if n == 0:
        return 0.0
    k = max(1, int(frac * n))
    idx = np.argsort(-p)[:k]
    return float(np.mean(y_true[idx]))


def lift_at_k(y_true: np.ndarray, p: np.ndarray, frac: float = 0.05) -> float:
    k = max(1, int(frac * len(p)))
    idx = np.argsort(-p)[:k]
    base = y_true.mean() + 1e-12
    return float(y_true[idx].mean() / base)


def time_slice(df: pd.DataFrame, start, end):
    return df[(df["entry_ts"] >= pd.Timestamp(start)) & (df["entry_ts"] <= pd.Timestamp(end))].copy()


def choose_features(df: pd.DataFrame):
    blacklist_exact = {
        "entry_ts", "symbol", "side", "y",
        "exit_ts", "exit_px", "exit_reason",
        "ttm_min", "y_fast", "tp", "sl", "p", "p_cal"
    }
    bad_substrings = [
        "exit", "future", "label", "target", "ttm", "pnl",
        "y_", "_y", "prob", "proba", "score"
    ]

    def is_bad(col: str) -> bool:
        cl = str(col).lower()
        if cl in blacklist_exact:
            return True
        return any(sub in cl for sub in bad_substrings)

    feats = [
        c for c in df.columns
        if (c not in blacklist_exact)
        and (not is_bad(c))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feats


def _apply_side_filter(df: pd.DataFrame, side_filter: str) -> pd.DataFrame:
    if side_filter == "BOTH" or "side" not in df.columns:
        return df
    s = df["side"]
    if pd.api.types.is_numeric_dtype(s):
        target = 1 if side_filter == "BUY" else -1
        return df[s == target].copy()
    else:
        return df[s.astype(str).str.upper() == side_filter].copy()


def _calibrate_probs(probs: np.ndarray, y: np.ndarray, method: str):
    if method == "none":
        return probs, None
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(probs.reshape(-1, 1), y)
        def cal(p): return lr.predict_proba(p.reshape(-1, 1))[:, 1]
        return cal(probs), lr
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs, y)
        def cal(p): return ir.transform(p)
        return cal(probs), ir
    return probs, None


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


def _save_model_artifacts(dirpath: Path, final_model, calibrator, feature_names, meta, num_iter: int):
    dirpath.mkdir(parents=True, exist_ok=True)

    with open(dirpath / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "calibrator": calibrator,
                "num_iter": int(num_iter),
                "feature_names": list(feature_names),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    booster = getattr(final_model, "booster_", None)
    if booster is None:
        booster = getattr(final_model, "_Booster", None)
    if booster is not None:
        booster.save_model(str(dirpath / "model.txt"))

    (dirpath / "feature_names.json").write_text(json.dumps(list(feature_names)))
    (dirpath / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    idx_names = set([df.index.name] if df.index.name else [])
    if hasattr(df.index, "names") and df.index.names:
        idx_names |= set([n for n in df.index.names if n is not None])

    if ("entry_ts" in idx_names) and ("entry_ts" not in df.columns):
        df = df.reset_index()

    if df.index.name == "entry_ts":
        df.index.name = None

    if isinstance(df.columns, pd.Index) and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if "entry_ts" not in df.columns:
        raise ValueError("Ожидал колонку 'entry_ts' перед сортировкой/трейном")

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True).dt.tz_localize(None)
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("float").fillna(0).astype("int8")

    df = df.dropna(subset=["entry_ts", "y"]).sort_values("entry_ts")
    return df


def _compute_percent_splits(df: pd.DataFrame, train_frac: float, gap_days: int):
    if df.empty:
        return df.iloc[0:0], df.iloc[0:0], (None, None)
    ts = df["entry_ts"]
    q_train = ts.quantile(train_frac)
    if pd.isna(q_train):
        return df.iloc[0:0], df.iloc[0:0], (None, None)
    test_start = (pd.to_datetime(q_train) + pd.Timedelta(days=gap_days))
    test_end = ts.max()
    train_mask = ts <= q_train
    test_mask = (ts >= test_start) & (ts <= test_end)
    return train_mask, test_mask, (test_start, test_end)


# =========================
# AP для sklearn eval_metric (callable)
# =========================
def sklearn_eval_metric_ap(y_true, y_pred_proba):
    # y_pred_proba — уже вероятности из sklearn-обёртки
    try:
        ap = average_precision_score(y_true, y_pred_proba)
    except Exception:
        ap = np.nan
    # LightGBM ожидает (name, value, is_higher_better)
    return ("ap", float(ap), True)


def _predict_proba_general(model, X, num_iter=None):
    """Унификация: поддержать как sklearn-обёртку, так и Booster."""
    booster = getattr(model, "booster_", None)
    if booster is None:
        booster = getattr(model, "_Booster", None)
    if booster is not None:
        return booster.predict(X, num_iteration=num_iter)
    return model.predict_proba(X, num_iteration=num_iter)[:, 1]


# =========================
# Основной пайплайн per-symbol
# =========================
def run_for_symbol(df_sym: pd.DataFrame, sym: str, args, outdir: Path,
                   pooled_clf: lgb.LGBMClassifier = None):
    print(f"\n=== {sym} ===")

    df = _sanitize_df(df_sym)
    df = _apply_side_filter(df, args.side_filter)

    leak_cols = [c for c in df.columns if any(s in c.lower() for s in
                                              ["exit", "ttm", "target", "label", "pnl", "prob", "proba",
                                               "score"]) or c.lower() in
                 {"y_fast", "tp", "sl", "exit_ts", "exit_px", "exit_reason"}]
    if leak_cols:
        df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
    for c in ["y", "entry_ts"]:
        if c not in df.columns:
            raise ValueError(f"{sym}: нет колонки {c}")
    if df["y"].isna().any():
        raise ValueError(f"{sym}: таргет y содержит NaN")

    feats = choose_features(df)
    if not feats:
        raise ValueError(f"{sym}: не нашёл числовых фичей для обучения")

    # ----- разбиение -----
    if args.split_mode == "percent":
        train_mask, test_mask, test_range = _compute_percent_splits(df, args.train_frac, args.gap_days)
        if test_range[0] is None:
            print("[TEST] пропуск: не удалось вычислить границы percent-split")
            return
        train_final = df[train_mask]
        test_final = df[test_mask]
        pos_total = int((df["y"] == 1).sum())
        pos_test = int((test_final["y"] == 1).sum())
        if pos_total < args.min_pos_total or pos_test < args.min_pos_test:
            if args.train_mode in ("pooled", "hybrid") and pooled_clf is not None:
                print(f"[WARN] {sym} {args.side_filter}: few positives, используем pooled-модель без дообучения")
                X_t, y_t = test_final[feats], test_final["y"].astype(int)
                p_test = _predict_proba_general(pooled_clf, X_t)
                p_test_final = _post_weighting(p_test, test_final, args.weight_market_heat, args.weight_sess_us)
                _report_and_save(sym, args, outdir, test_final, y_t, p_test_final, test_range, used_init=False, feats=feats,
                                 model=None, calibrator=None, num_iter=getattr(pooled_clf, "best_iteration_", None) or args.n_estimators)
            else:
                print(f"[SKIP] {sym} {args.side_filter}: too few positives (total={pos_total}, test={pos_test})")
            return
        test_start, test_end = test_range
        folds = []
    else:
        gd = pd.Timedelta(days=args.gap_days)
        folds = [
            (("2021-01-01", args.fold1_tr_end), (pd.Timestamp(args.fold1_tr_end) + gd, args.fold1_va_end)),
            (("2021-01-01", args.fold2_tr_end), (pd.Timestamp(args.fold2_tr_end) + gd, args.fold2_va_end)),
            (("2021-01-01", args.fold3_tr_end), (pd.Timestamp(args.fold3_tr_end) + gd, args.fold3_va_end)),
        ]
        test_start, test_end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
        train_final = df[df["entry_ts"] <= (test_start - pd.Timedelta(days=args.gap_days))]
        test_final = time_slice(df, test_start, test_end)
        pos_total = int((df["y"] == 1).sum())
        pos_test = int((test_final["y"] == 1).sum())
        if pos_total < args.min_pos_total or pos_test < args.min_pos_test:
            if args.train_mode in ("pooled", "hybrid") and pooled_clf is not None:
                print(f"[WARN] {sym} {args.side_filter}: few positives, используем pooled-модель без дообучения")
                X_t, y_t = test_final[feats], test_final["y"].astype(int)
                p_test = _predict_proba_general(pooled_clf, X_t)
                p_test_final = _post_weighting(p_test, test_final, args.weight_market_heat, args.weight_sess_us)
                _report_and_save(sym, args, outdir, test_final, y_t, p_test_final, (test_start, test_end), used_init=False, feats=feats,
                                 model=None, calibrator=None, num_iter=getattr(pooled_clf, "best_iteration_", None) or args.n_estimators)
            else:
                print(f"[SKIP] {sym} {args.side_filter}: too few positives (total={pos_total}, test={pos_test})")
            return

    # ----- базовые LGB параметры
    base_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        subsample=args.subsample if args.boosting == "gbdt" else 1.0,
        colsample_bytree=args.colsample,
        min_child_samples=args.min_child_samples,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        max_bin=args.max_bin,
        boosting_type=args.boosting,
        n_jobs=os.cpu_count(),
        random_state=args.seed,
        verbose=-1,
        objective="binary",
    )

    # ===== CV отключен для percent-режима (как и было)
    best_n = args.n_estimators
    if folds:
        best_iters = []
        for i, (tr, va) in enumerate(folds, 1):
            tr_df = time_slice(df, tr[0], tr[1])
            va_df = time_slice(df, va[0], va[1])
            if len(tr_df) == 0 or len(va_df) == 0:
                print(f"[Fold {i}] пропуск: пустые срезы ({len(tr_df)} / {len(va_df)})")
                continue

            X_tr, y_tr = tr_df[feats], tr_df["y"].astype(int)
            X_va, y_va = va_df[feats], va_df["y"].astype(int)

            params = dict(base_params)
            params["scale_pos_weight"] = _compute_pos_weight(y_tr, args)

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)

            best_iter = getattr(clf, "best_iteration_", None) or args.n_estimators
            best_iters.append(best_iter)

            p_va = clf.predict_proba(X_va, num_iteration=best_iter)[:, 1]
            auc = roc_auc_score(y_va, p_va)
            ap = average_precision_score(y_va, p_va)
            lift5 = lift_at_k(y_va.values, p_va, 0.05)
            print(f"[Fold {i}] AUC={auc:.4f}  AP(PR-AUC)={ap:.4f}  Lift@5%={lift5:.2f}  (best_iter={best_iter})")

        best_n = int(np.median(best_iters)) if best_iters else args.n_estimators
        print(f"[CV] median(best_iteration_) = {best_n}")

    # ===== Финал =====
    if len(train_final) == 0 or len(test_final) == 0:
        print(f"[TEST] пропуск: пустые train/test ({len(train_final)} / {len(test_final)})")
        return

    X_f, y_f = train_final[feats], train_final["y"].astype(int)
    X_t, y_t = test_final[feats], test_final["y"].astype(int)

    # веса класса
    pos_w = _compute_pos_weight(y_f, args)
    sw = np.ones(len(y_f), dtype="float32")
    if pos_w > 1.0:
        sw[y_f.values == 1] = pos_w

    # ES по предтестовому окну (через sklearn eval_set)
    calibrator = None
    num_iter = best_n
    final = None
    used_init = False

    eval_set = None
    eval_metric = "auc"
    if args.final_use_pretest_es:
        val_start = (test_start - pd.Timedelta(days=args.final_pretest_days))
        val_end = (test_start - pd.Timedelta(days=1))
        pre_val = df[(df["entry_ts"] >= val_start) & (df["entry_ts"] <= val_end)]
        if len(pre_val) >= 100:
            X_vf, y_vf = pre_val[feats], pre_val["y"].astype(int)
            eval_set = [(X_vf, y_vf)]
            if args.es_metric == "ap":
                eval_metric = sklearn_eval_metric_ap

    # финальная модель
    if args.train_mode == "hybrid" and pooled_clf is not None:
        # Дообучение от пула
        fin_params = dict(base_params)
        fin_params["n_estimators"] = args.finetune_iters
        fin_params["learning_rate"] = args.finetune_lr
        fin_params["scale_pos_weight"] = pos_w
        final = lgb.LGBMClassifier(**fin_params)

        final.fit(
            X_f, y_f,
            sample_weight=sw,
            init_model=getattr(pooled_clf, "booster_", None),
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)] if eval_set else None,
        )
        used_init = True
        # общее число деревьев = деревья пула + добавленные
        base_it = getattr(pooled_clf, "best_iteration_", None) or pooled_clf.get_params().get("n_estimators", 0)
        num_iter = int(base_it + getattr(final, "best_iteration_", None) or args.finetune_iters)
        num_iter = max(int(num_iter), int(args.min_best_iter))
    else:
        # Классический fit
        fin_params = dict(base_params)
        fin_params["n_estimators"] = best_n
        fin_params["scale_pos_weight"] = pos_w
        final = lgb.LGBMClassifier(**fin_params)
        final.fit(
            X_f, y_f,
            sample_weight=sw,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)] if eval_set else None,
        )
        num_iter = getattr(final, "best_iteration_", None) or best_n
        num_iter = max(int(num_iter), int(args.min_best_iter))

    # Предсказание на тесте
    p_test = _predict_proba_general(final, X_t, num_iter=num_iter)

    # Калибровка
    if calibrator is None and args.calibrate != "none":
        # если была предтест-валидация — можно откалибровать на ней, иначе на train
        if eval_set:
            X_vf, y_vf = eval_set[0]
            p_cal_src = _predict_proba_general(final, X_vf, num_iter=num_iter)
            method = args.calibrate
            if method == "isotonic" and int((np.asarray(y_vf) == 1).sum()) < args.iso_min_pos:
                print(f"[CAL] isotonic→platt (too few positives: {int((np.asarray(y_vf)==1).sum())} < {args.iso_min_pos})")
                method = "platt"
            _, calibrator = _calibrate_probs(p_cal_src, np.asarray(y_vf), method)
        else:
            p_tr = _predict_proba_general(final, X_f, num_iter=num_iter)
            _, calibrator = _calibrate_probs(p_tr, y_f.values, args.calibrate)

        if calibrator is not None:
            if isinstance(calibrator, LogisticRegression):
                p_test = calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]
            else:
                p_test = calibrator.transform(p_test)

    p_test_final = _post_weighting(p_test, test_final, args.weight_market_heat, args.weight_sess_us)

    # Отчёт и сохранение
    _report_and_save(sym, args, outdir, test_final, y_t, p_test_final, (test_start, test_end),
                     used_init=used_init, feats=feats, model=final, calibrator=calibrator, num_iter=num_iter)


def _compute_pos_weight(y: pd.Series, args) -> float:
    """Возврат scale_pos_weight с учётом настроек и 'focal' режима."""
    if args.auto_class_weight:
        pos = y.sum()
        neg = len(y) - pos
        base = float(neg / max(1, pos)) if pos > 0 else 1.0
    else:
        base = float(args.scale_pos_weight)

    if args.loss == "focal":
        # Приближение фокального: ещё немного усилим положительный класс.
        # Коэффициент можно тюнить; берём зависимость от alpha.
        extra = max(1.0, 1.0 / max(1e-6, float(args.focal_alpha)))
        base *= extra
    return float(base)


def _report_and_save(sym, args, outdir, test_final, y_t, p_test_final, test_range,
                     used_init: bool, feats, model, calibrator, num_iter: int):
    auc_t = roc_auc_score(y_t, p_test_final)
    ap_t = average_precision_score(y_t, p_test_final)
    lift5_t = lift_at_k(y_t.values, p_test_final, 0.05)
    p5 = precision_at_k(y_t.values, p_test_final, 0.05)
    p10 = precision_at_k(y_t.values, p_test_final, 0.10)

    print(f"[TEST {pd.to_datetime(test_range[0]).date()}..{pd.to_datetime(test_range[1]).date()}] "
          f"AUC={auc_t:.4f}  AP(PR-AUC)={ap_t:.4f}  Lift@5%={lift5_t:.2f}  "
          f"P@5%={p5:.3f}  P@10%={p10:.3f}  "
          f"{'(hybrid-init)' if used_init else ''}")

    sym_dir = Path(outdir) / (f"{sym}_{args.side_filter}" if args.side_filter != "BOTH" else sym)
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_df = test_final.assign(p=p_test_final)
    out_df.to_parquet(sym_dir / "test_scored.parquet", index=False)
    pd.Series(p_test_final, index=test_final.index).to_csv(sym_dir / "test_proba.csv")
    print(f"[OK] saved: {sym_dir}/test_scored.parquet")

    meta = {
        "symbol": sym,
        "side": args.side_filter,
        "train_mode": args.train_mode,
        "loss": args.loss,
        "boosting": args.boosting,
        "calibrate": args.calibrate,
        "best_iteration": int(num_iter),
        "params": {
            "n_estimators": int(num_iter),
            "learning_rate": float(args.lr),
        },
        "split": {
            "mode": args.split_mode,
            "train_frac": float(args.train_frac) if args.split_mode == "percent" else None,
            "gap_days": int(args.gap_days),
            "test_start": str(pd.to_datetime(test_range[0])),
            "test_end": str(pd.to_datetime(test_range[1])),
        },
        "min_pos": {
            "total": int(args.min_pos_total),
            "test": int(args.min_pos_test),
        },
    }
    if model is not None:
        _save_model_artifacts(
            dirpath=sym_dir,
            final_model=model,
            calibrator=calibrator,
            feature_names=feats,
            meta=meta,
            num_iter=int(num_iter),
        )
        print(f"[OK] model saved -> {sym_dir}/model.pkl (и model.txt при наличии Booster)")
    else:
        # pooled-only режим: сохраняем только мету
        (sym_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.inp)
    for col in ["entry_ts", "y", "symbol"]:
        if col not in df.columns:
            raise ValueError("Входной parquet должен содержать 'entry_ts', 'y', 'symbol'.")

    df = _sanitize_df(df)

    # список символов
    if args.symbols.strip().upper() == "ALL":
        symbols = sorted(df["symbol"].astype(str).unique())
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # ===== pooled-претрейн (для pooled/hybrid) через sklearn API =====
    pooled_clf = None
    if args.train_mode in ("pooled", "hybrid"):
        dfg = _apply_side_filter(df.copy(), args.side_filter)
        train_mask, _, _ = _compute_percent_splits(dfg, args.train_frac, args.gap_days)
        tr = dfg[train_mask]
        if len(tr) > 0:
            featsg = choose_features(tr)
            Xg = tr[featsg]
            yg = tr["y"].astype(int)

            base_params = dict(
                n_estimators=args.n_estimators,
                learning_rate=args.lr,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                subsample=args.subsample if args.boosting == "gbdt" else 1.0,
                colsample_bytree=args.colsample,
                min_child_samples=args.min_child_samples,
                reg_lambda=args.reg_lambda,
                reg_alpha=args.reg_alpha,
                max_bin=args.max_bin,
                boosting_type=args.boosting,
                n_jobs=os.cpu_count(),
                random_state=args.seed,
                verbose=-1,
                objective="binary",
            )

            pos_w = _compute_pos_weight(yg, args)
            sw = np.ones(len(yg), dtype="float32")
            if pos_w > 1.0:
                sw[yg.values == 1] = pos_w

            pooled_clf = lgb.LGBMClassifier(**base_params)
            pooled_clf.fit(Xg, yg, sample_weight=sw)
            print("[POOLED] готова общая модель (sklearn LGBMClassifier)")

            if args.train_mode == "pooled":
                # Оценка и сохранение pooled сразу на общем тест-окне (для всей выборки)
                _, test_mask, test_range = _compute_percent_splits(dfg, args.train_frac, args.gap_days)
                test_final = dfg[test_mask]
                if not test_final.empty:
                    y_t_all = test_final["y"].astype(int)
                    p_all = _predict_proba_general(pooled_clf, test_final[featsg])
                    p_all = _post_weighting(p_all, test_final, args.weight_market_heat, args.weight_sess_us)
                    auc_t = roc_auc_score(y_t_all, p_all)
                    ap_t = average_precision_score(y_t_all, p_all)
                    lift5_t = lift_at_k(y_t_all.values, p_all, 0.05)
                    print(f"[POOLED TEST {pd.to_datetime(test_range[0]).date()}..{pd.to_datetime(test_range[1]).date()}] "
                          f"AUC={auc_t:.4f}  AP={ap_t:.4f}  Lift@5%={lift5_t:.2f}")

    # запуск по каждому символу
    for sym in symbols:
        sub = df[df["symbol"].astype(str).str.upper() == sym].copy()
        if sub.empty:
            print(f"[SKIP] {sym}: нет строк")
            continue
        num = sub.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if num.isna().any().any():
            bad_cols = num.columns[num.isna().any()].tolist()
            raise ValueError(f"{sym}: найдены NaN/inf в числовых колонках: {bad_cols[:10]}")
        run_for_symbol(sub, sym, args, outdir, pooled_clf=pooled_clf)


if __name__ == "__main__":
    main()


