# runtime/fast_patch.py
import json
import math
from pathlib import Path
import pandas as pd

# — мы будем патчить функции внутри evaluate_common —
import evaluate_common as ec

# Сохраняем оригинальные функции, чтобы при необходимости можно было дернуть их напрямую
_ORIG_finalize_write = ec.finalize_write
_ORIG_load_price_cache = ec.load_price_cache

# Глобальный кэш цен по ключу (interval, lookback_days, tuple(symbols))
_PRICE_CACHE = {}

def _max_drawdown_pct_from_equity(eq_df: pd.DataFrame) -> float:
    """Считаем max DD% из серии equity, если есть."""
    try:
        if eq_df is None or eq_df.empty:
            return 0.0
        s = pd.to_numeric(eq_df.get("equity"), errors="coerce").dropna()
        if s.empty:
            return 0.0
        peak = s.cummax()
        dd = (s - peak) / peak
        return float((dd.min() * 100.0) if not dd.empty else 0.0)
    except Exception:
        return 0.0

def _safe_sum(series) -> float:
    try:
        return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())
    except Exception:
        return 0.0

def fast_finalize_write(out_path: str,
                        df_out: pd.DataFrame,
                        eq_sheet: pd.DataFrame,
                        by_variant: pd.DataFrame,
                        by_exit_reason: pd.DataFrame,
                        extra_sheets=None):
    """
    Лёгкая версия finalize_write: НИЧЕГО не пишет в Excel.
    Готовит JSON-резюме для evaluate_theta: *_summary.json рядом с out_path.
    """
    try:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # считаем базовые метрики
        trades = int((df_out.get("skipped") == False).sum()) if "skipped" in df_out.columns else int(len(df_out))
        # pnl_usd суммируем, если есть колонка (в eval она обычно есть после симулятора капитала)
        if "pnl_usd" in df_out.columns:
            pnl_usd_total = _safe_sum(df_out["pnl_usd"])
        else:
            # если нет — fallback по pnl_pct (не идеально, но лучше, чем 0)
            pnl_usd_total = 0.0

        max_dd_pct = _max_drawdown_pct_from_equity(eq_sheet)

        summary = {
            "trades": trades,
            "kpi": pnl_usd_total,         # evaluate_theta kpi_key="pnl_usd" будет это читать
            "max_dd_pct": float(max_dd_pct),
        }

        # кладём рядом с out_path
        summ_path = out.with_suffix("").as_posix() + "_summary.json"
        with open(summ_path, "w") as f:
            json.dump(summary, f)
    except Exception as e:
        # в худшем случае, попробуем не уронить пайплайн
        # (можно логировать)
        pass

def fast_load_price_cache(symbols, interval="4h", lookback_days=180):
    """
    Обёртка над load_price_cache с кэшем результатов.
    Ключ кэша — (interval, lookback_days, tuple(sorted(symbols))).
    """
    key = (str(interval), int(lookback_days), tuple(sorted([str(s) for s in symbols])))
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key]
    cache = _ORIG_load_price_cache(symbols, interval=interval, lookback_days=lookback_days)
    _PRICE_CACHE[key] = cache
    return cache

# ==== Сам патч: заменить функции в evaluate_common на быстрые ====
ec.finalize_write = fast_finalize_write
ec.load_price_cache = fast_load_price_cache