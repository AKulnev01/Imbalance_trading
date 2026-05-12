# tools/make_param_dataset.py
# Наполняет датасет параметров → KPI.
# Режимы:
#  (1) Рандомная генерация (--n)
#  (2) Прямая оптимизация без ML: --optuna-trials N (через Optuna TPE)
#
# Примеры:
#   python tools/make_param_dataset.py \
#       --signals ./data/signals/signals_var_filtered.xlsx \
#       --n 200 \
#       --out ./models/data/params_kpi.parquet
#
#   python tools/make_param_dataset.py \
#       --signals ./data/signals/signals_var_filtered.xlsx \
#       --optuna-trials 100 \
#       --out ./models/data/params_kpi.parquet
#
# Дополнительно:
#   --tmp-eval  путь к временно сохраняемому Excel от evaluate_momentum.py
#   --space     JSON с переопределением пространства параметров

import os
import json
import uuid
import argparse
import subprocess
from pathlib import Path
import random

import numpy as np
import pandas as pd

# --- пространство параметров (совместимо с models/neuro_optimizer.py) ---
DEFAULT_PARAM_SPACE = {
    "USE_FIB_4H":  [0, 1],
    "USE_DIV_4H":  [0, 1],
    "USE_OBOS_4H": [0, 1],

    "FIB_4H_LOOKBACK_BARS": [60, 90, 120, 180],
    "FIB_4H_PIVOT_LEN":     [2, 3, 4, 5],
    "FIB_SET":              ["0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618"],
    "FIB_TP_INDEX":         [2, 3, 4],
    "FIB_TOUCH_MODE":       ["wick", "close"],
    "FIB_SL_MODE":          ["current", "beyond_prev_fib"],

    "DIV4H_TYPE":          ["off", "rsi", "macd"],
    "RSI_PERIOD":          [10, 14, 21],
    "MACD_FAST":           [8, 12],
    "MACD_SLOW":           [24, 26, 30],
    "MACD_SIGNAL":         [9],
    "DIV4H_PIVOT_LEN":     [2, 3, 4],
    "DIV4H_LOOKBACK_BARS": [60, 90, 120, 180],
    "DIV4H_CONFIRM_BARS":  [1, 2, 3],
    "DIV4H_POLICY":        ["tighten_tp", "skip_entry"],
    "DIV_TIGHTEN_STEP":    [0, 1, 2],

    "OBOS_TYPE":     ["off", "rsi", "stoch", "wpr", "cci"],
    "OB_RSI_OB":     [70.0, 75.0], "OB_RSI_OS": [25.0, 30.0],
    "STO_K":         [14], "STO_D": [3], "STO_SMA": [3],
    "OB_STOCH_OB":   [80.0], "OB_STOCH_OS": [20.0],
    "OB_WPR_OB":     [-20.0], "OB_WPR_OS": [-80.0],
    "OB_CCI_OB":     [100.0, 150.0], "OB_CCI_OS": [-100.0, -150.0],
    "OBOS_POLICY":   ["tp_bias", "filter_entry"],

    "MOMENTUM_TP_PCT": [0.02, 0.03, 0.04],
    "MOMENTUM_SL_PCT": [0.01, 0.015, 0.02],
    "FEE_TAKER":       [0.000, 0.0007],
    "ENTRY_SLIPPAGE_PCT": [0.001, 0.002, 0.003],
    "EXIT_SLIPPAGE_PCT":  [0.001, 0.002, 0.003],
    "STOP_SLIPPAGE_PCT":  [0.001, 0.002, 0.003],
}

PENALTY_SCORE = -1e6   # Явный штраф для невалидных/мусорных прогонов
MIN_TRADES    = 10     # Минимум сделок, чтобы считать прогон валидным
EPS_SCORE     = 1e-9   # Нулевой KPI считаем мусором

# ----------------- утилиты -----------------

def _read_space(path: str = None):
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PARAM_SPACE

def _sample(space: dict, n: int):
    keys = list(space.keys())
    out = []
    for _ in range(n):
        d = {}
        for k in keys:
            d[k] = random.choice(space[k])
        out.append(d)
    return out

def _env_from_dict(d: dict) -> dict:
    env = os.environ.copy()
    for k, v in d.items():
        env[k] = str(v)
    return env

def _parse_eval_xlsx(path: str) -> dict:
    """Читает результат evaluate_momentum Excel и возвращает KPI-словарь."""
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return {}
    kpi = {}

    # Сводка по варианту
    if "summary_by_variant" in xl.sheet_names and "summary" in xl.sheet_names:
        s = pd.read_excel(xl, "summary")
        s = pd.read_excel(xl, "summary_by_variant")
        s.columns = [c.strip().lower() for c in s.columns]

        # строка MOMENTUM если есть, иначе первая
        if "variant" in s.columns:
            mask = s["variant"].astype(str).str.upper() == "MOMENTUM"
            row = s[mask].iloc[0] if mask.any() else s.iloc[0]
        else:
            row = s.iloc[0]

        def _get(col, default=0.0):
            return float(row[col]) if col in s.columns and pd.notna(row[col]) else default

        kpi["kpi_trades"]       = _get("trades", 0.0)
        kpi["kpi_wins"]         = _get("wins", 0.0)
        kpi["kpi_winrate_pct"]  = _get("winrate_pct", 0.0)
        kpi["kpi_pnl_pct"]      = _get("pnl_pct", 0.0)
        kpi["kpi_pnl_usd"]      = _get("pnl_usd", 0.0)
        kpi["kpi_dd_pct"]       = _get("dd_pct", 0.0)  # если появится в отчёте

    # Универсальный kpi_score если нет 'kpi'
    wr = kpi.get("kpi_winrate_pct", 0.0)
    pnl = kpi.get("kpi_pnl_pct", 0.0)
    dd  = kpi.get("kpi_dd_pct", 0.0)
    kpi["kpi_score"] = 0.6*(wr/100.0) + 0.4*(pnl/100.0) - 0.5*(dd/100.0)

    return kpi

def _run_eval(signals: str, out_xlsx: str, env: dict) -> bool:
    base_env = os.environ.copy()
    base_env.setdefault("EVAL_AS_OF_OVERRIDE", "now")
    base_env.setdefault("DISABLE_MINUTE_FALLBACK", "1")  # ← ключевое
    base_env.setdefault("MINUTE_EXIT_FOR_SINGLE", "0")  # ← ключевое
    base_env.setdefault("FEE_TAKER", "0.0000")
    base_env.setdefault("ENTRY_SLIPPAGE_PCT", "0.002")
    base_env.setdefault("EXIT_SLIPPAGE_PCT", "0.002")
    base_env.setdefault("STOP_SLIPPAGE_PCT", "0.002")
    base_env.update(env)

    cmd = [
        "python", "models/evaluate_momentum.py",
        signals, "--out", out_xlsx,
        "--ttl-days", "30",              # как ты и хотел
        # без sequential здесь: это оффлайн-оценка variant’а, а не реальная аллокация
    ]
    print("▶", " ".join(cmd))
    res = subprocess.run(cmd, env=base_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return Path(out_xlsx).exists()

def _row_from_params_kpi(p: dict, kpi: dict) -> dict:
    now = pd.Timestamp.utcnow()
    row = {**p}
    row.update(kpi)
    row["as_of"] = now
    row["uuid"] = str(uuid.uuid4())
    return row

def _append_parquet(df_row: dict, parquet_path: str):
    Path(Path(parquet_path).parent).mkdir(parents=True, exist_ok=True)
    if Path(parquet_path).exists():
        base = pd.read_parquet(parquet_path)
        out = pd.concat([base, pd.DataFrame([df_row])], ignore_index=True)
    else:
        out = pd.DataFrame([df_row])
    out.to_parquet(parquet_path, index=False)

# ----------------- Optuna objective -----------------

def _optuna_objective(trial, signals, tmp_out):
    # Мягко ограничим пространство (совместимо с DEFAULT_PARAM_SPACE)
    param = {
        "USE_FIB_4H": trial.suggest_categorical("USE_FIB_4H", [0, 1]),
        "USE_DIV_4H": trial.suggest_categorical("USE_DIV_4H", [0, 1]),
        "USE_OBOS_4H": trial.suggest_categorical("USE_OBOS_4H", [0, 1]),

        "FIB_4H_LOOKBACK_BARS": trial.suggest_categorical("FIB_4H_LOOKBACK_BARS", [60, 90, 120, 180]),
        "FIB_4H_PIVOT_LEN": trial.suggest_categorical("FIB_4H_PIVOT_LEN", [2, 3, 4]),
        "FIB_SET": trial.suggest_categorical("FIB_SET", ["0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.618"]),
        "FIB_TP_INDEX": trial.suggest_categorical("FIB_TP_INDEX", [2, 3, 4]),
        "FIB_TOUCH_MODE": trial.suggest_categorical("FIB_TOUCH_MODE", ["wick", "close"]),
        "FIB_SL_MODE": trial.suggest_categorical("FIB_SL_MODE", ["current", "beyond_prev_fib"]),

        "DIV4H_TYPE": trial.suggest_categorical("DIV4H_TYPE", ["off", "rsi", "macd"]),
        "RSI_PERIOD": trial.suggest_categorical("RSI_PERIOD", [10, 14, 21]),
        "MACD_FAST":  trial.suggest_categorical("MACD_FAST", [8, 12]),
        "MACD_SLOW":  trial.suggest_categorical("MACD_SLOW", [24, 26, 30]),
        "MACD_SIGNAL": trial.suggest_categorical("MACD_SIGNAL", [9]),
        "DIV4H_PIVOT_LEN": trial.suggest_categorical("DIV4H_PIVOT_LEN", [2, 3, 4]),
        "DIV4H_LOOKBACK_BARS": trial.suggest_categorical("DIV4H_LOOKBACK_BARS", [60, 90, 120, 180]),
        "DIV4H_CONFIRM_BARS": trial.suggest_categorical("DIV4H_CONFIRM_BARS", [1, 2, 3]),
        "DIV4H_POLICY": trial.suggest_categorical("DIV4H_POLICY", ["tighten_tp", "skip_entry"]),
        "DIV_TIGHTEN_STEP": trial.suggest_categorical("DIV_TIGHTEN_STEP", [0, 1, 2]),

        "OBOS_TYPE": trial.suggest_categorical("OBOS_TYPE", ["off", "rsi", "stoch", "wpr", "cci"]),
        "OB_RSI_OB": trial.suggest_categorical("OB_RSI_OB", [70.0, 75.0]),
        "OB_RSI_OS": trial.suggest_categorical("OB_RSI_OS", [25.0, 30.0]),
        "STO_K": trial.suggest_categorical("STO_K", [14]),
        "STO_D": trial.suggest_categorical("STO_D", [3]),
        "STO_SMA": trial.suggest_categorical("STO_SMA", [3]),
        "OB_STOCH_OB": trial.suggest_categorical("OB_STOCH_OB", [80.0]),
        "OB_STOCH_OS": trial.suggest_categorical("OB_STOCH_OS", [20.0]),
        "OB_WPR_OB": trial.suggest_categorical("OB_WPR_OB", [-20.0]),
        "OB_WPR_OS": trial.suggest_categorical("OB_WPR_OS", [-80.0]),
        "OB_CCI_OB": trial.suggest_categorical("OB_CCI_OB", [100.0, 150.0]),
        "OB_CCI_OS": trial.suggest_categorical("OB_CCI_OS", [-100.0, -150.0]),
        "OBOS_POLICY": trial.suggest_categorical("OBOS_POLICY", ["tp_bias", "filter_entry"]),

        "MOMENTUM_TP_PCT": trial.suggest_categorical("MOMENTUM_TP_PCT", [0.02, 0.03, 0.04]),
        "MOMENTUM_SL_PCT": trial.suggest_categorical("MOMENTUM_SL_PCT", [0.01, 0.015, 0.02]),
        "FEE_TAKER": trial.suggest_categorical("FEE_TAKER", [0.0, 0.0007]),
        "ENTRY_SLIPPAGE_PCT": trial.suggest_categorical("ENTRY_SLIPPAGE_PCT", [0.001, 0.002, 0.003]),
        "EXIT_SLIPPAGE_PCT":  trial.suggest_categorical("EXIT_SLIPPAGE_PCT",  [0.001, 0.002, 0.003]),
        "STOP_SLIPPAGE_PCT":  trial.suggest_categorical("STOP_SLIPPAGE_PCT",  [0.001, 0.002, 0.003]),
    }

    env = _env_from_dict(param)
    ok = _run_eval(signals, tmp_out, env)
    if not ok:
        return PENALTY_SCORE

    kpi = _parse_eval_xlsx(tmp_out)
    trades = float(kpi.get("kpi_trades", 0.0))
    score  = float(kpi.get("kpi_score", PENALTY_SCORE))

    # Фильтр мусора: мало сделок / NaN / практически ноль
    if (trades < MIN_TRADES) or (not np.isfinite(score)) or (abs(score) < EPS_SCORE):
        print(f"⚠️ skip trial (penalty): trades={trades}, score={score}")
        return PENALTY_SCORE

    return float(score)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Build/extend params→KPI dataset; supports random and Optuna modes.")
    ap.add_argument("--signals", required=True, help="Путь к Excel/CSV сигналов")
    ap.add_argument("--n", type=int, default=0, help="Сколько случайных комбо добавить")
    ap.add_argument("--out", default="./models/data/params_kpi.parquet")
    ap.add_argument("--space", default=None, help="JSON с PARAM_SPACE (опционально)")
    ap.add_argument("--optuna-trials", type=int, default=0, help="Сколько итераций прямого TPE-поиска (без ML)")
    ap.add_argument("--tmp-eval", default="./models/data/_fast_tmp.xlsx")
    ap.add_argument("--seed", type=int, default=42, help="Сид для рандома (повторяемость)")
    args = ap.parse_args()

    # Стабильность выборок
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    space = _read_space(args.space)

    # (1) Рандомные комбы
    if args.n and args.n > 0:
        combos = _sample(space, int(args.n))
        for i, p in enumerate(combos, 1):
            print(f"\n[{i}/{len(combos)}] run params: {p}")
            env = _env_from_dict(p)
            ok = _run_eval(args.signals, args.tmp_eval, env)
            if not ok:
                print("⚠️ eval failed, skip")
                continue

            kpi = _parse_eval_xlsx(args.tmp_eval)
            trades = float(kpi.get("kpi_trades", 0.0))
            score  = float(kpi.get("kpi_score", np.nan))
            if (trades < MIN_TRADES) or (not np.isfinite(score)) or (abs(score) < EPS_SCORE):
                print(f"⚠️ skip append: trades={trades}, score={score}")
                continue

            row = _row_from_params_kpi(p, kpi)
            added = 0
            _append_parquet(row, args.out)
            added += 1
        if added > 0:
            print(f"\n✅ dataset updated: {args.out} (+{added} rows)")
        else:
            print("\n⚠️ no rows appended — all trials had trades=0 or kpi_score≈0; check ENV/TTL/minute data")

    # (2) Прямая оптимизация (TPE)
    if args.optuna_trials and args.optuna_trials > 0:
        try:
            import optuna
        except Exception:
            print("❌ optuna не установлена: pip install optuna")
            return

        study = optuna.create_study(direction="maximize")

        def _obj(trial):
            # уникальный tmp для параллельности/отладки
            tmp = args.tmp_eval.replace(".xlsx", f"_{trial.number}.xlsx")
            score = _optuna_objective(trial, args.signals, tmp)

            # Append в датасет даже для плохих — но со здравым фильтром
            params = trial.params
            kpi = {}
            if Path(tmp).exists():
                kpi = _parse_eval_xlsx(tmp)

            trades = float(kpi.get("kpi_trades", 0.0))
            sc     = float(kpi.get("kpi_score", np.nan))
            if (trades >= MIN_TRADES) and np.isfinite(sc) and (abs(sc) >= EPS_SCORE):
                row = _row_from_params_kpi(params, kpi)
                _append_parquet(row, args.out)
            else:
                print(f"ℹ️ not appended (trial={trial.number}): trades={trades}, score={sc}")

            return float(score)

        study.optimize(_obj, n_trials=int(args.optuna_trials))
        print(f"🏁 Optuna best: value={study.best_value:.6f}\nparams={study.best_params}")

if __name__ == "__main__":
    main()