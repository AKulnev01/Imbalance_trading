# runtime/evaluate_theta.py
import os
import json
import hashlib
import time
from pathlib import Path

# Опциональный быстрый патч (не обязателен, просто не падаем, если файла нет)
if os.getenv("FAST_EVAL", "0").lower() in ("1","true","yes","y","on"):
    try:
        import runtime.fast_patch  # noqa: F401
    except Exception:
        pass

from evaluate_momentum import evaluate_momentum
from evaluate_common import load_signals, load_price_cache  # для кэша цен

# Директория для мемо-кэша результатов по theta
CACHE_DIR = Path(".cache/eval")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Общий кэш свечей на процесс: ключ = (interval, lookback_days, tuple(symbols))
_price_cache_global = {}


def _hash_theta(theta: dict, split_tag: str) -> str:
    s = json.dumps({"theta": theta, "split": split_tag}, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()


def evaluate_theta(
    signals_path: str,
    out_path: str,
    theta: dict,
    split: dict,
    kpi_key: str = "pnl_usd",
):
    """
    split ожидается такого вида:
      {
        "lookback_days": 180,
        "interval": "4h",
        "ttl_days": 3,
        "capital_aware": True,
        "initial_capital": 10000,
        "only_filled": False,
        "dedup": True
      }
    """
    # FAST-режим
    FAST = os.getenv("FAST_EVAL", "0").lower() in ("1", "true", "yes", "y", "on")

    # Пробрасываем ENV (именно те ключи, которые читает get_cfg в evaluate_momentum)
    for k, v in (theta or {}).items():
        os.environ[str(k)] = str(v)

    # Жёсткие выключатели дорогих веток в FAST
    if FAST:
        os.environ["DISABLE_MINUTE_FALLBACK"] = "1"              # совместимый ключ
        os.environ["MOMENTUM_DISABLE_MINUTE_FALLBACK"] = "1"     # и такой тоже (обратная совмест.)
        os.environ["MINUTE_EXIT_FOR_SINGLE"] = "0"
        os.environ["MOMENTUM_MINUTE_EXIT_FOR_SINGLE_HIT"] = "0"
        os.environ["ENABLE_EARLY_CHECK"] = "0"
        os.environ["EVAL_ENFORCE_ONE_AT_A_TIME"] = "0"
        os.environ["MAX_CONCURRENT_POSITIONS"] = "0"
        os.environ["MOMENTUM_TP_SL_MODE"] = "entry"
        # Excel в FAST не пишем — экономим I/O
        out_path = None

    # Сплит-параметры
    lookback_days = int(split.get("lookback_days", 180))
    interval = str(split.get("interval", "4h")).lower()
    ttl_days = int(split.get("ttl_days", 3))
    capital_aware = bool(split.get("capital_aware", True) and not FAST)
    initial_capital = float(split.get("initial_capital", 10000.0))
    only_filled = bool(split.get("only_filled", False))
    dedup = bool(split.get("dedup", True))

    # Мемоизация по theta+split
    h = _hash_theta(theta or {}, json.dumps(split, sort_keys=True, default=str))
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    # Соберём список символов из сигналов, чтобы прогреть общий кэш свечей
    df_sig = load_signals(signals_path, only_filled=only_filled, dedup=dedup, require_entry=False)
    symbols = df_sig["symbol"].dropna().unique().tolist() if not df_sig.empty else []

    cache_key = (interval, lookback_days, tuple(symbols))
    if cache_key in _price_cache_global:
        price_cache = _price_cache_global[cache_key]
    else:
        price_cache = load_price_cache(symbols, interval=interval, lookback_days=lookback_days)
        _price_cache_global[cache_key] = price_cache

    # Куда писать результат (для summary.json)
    # Если FAST — формально дадим путь, но сам XLSX не создаём (finalize_write обязан писать summary рядом)
    result_path = out_path if (out_path and not FAST) else "./models/data/_fast_tmp.xlsx"

    # Запускаем оценку
    t0 = time.time()
    evaluate_momentum(
        signals_path=signals_path,
        result_path=result_path,
        lookback_days=lookback_days,
        interval=interval,
        max_days=ttl_days,
        only_filled=only_filled,
        dedup=dedup,
        initial_capital=initial_capital,
        capital_aware=capital_aware,
        price_cache=price_cache,  # ← важный кэш, чтобы не грузить свечи заново
    )
    dt = time.time() - t0

    # Читаем summary.json, который пишет finalize_write рядом с result_path
    summ_path = result_path.replace(".xlsx", "_summary.json")
    with open(summ_path, "r") as f:
        summary = json.load(f)

    # KPI
    if summary.get("by_variant"):
        # Берём первую строку by_variant, если она одна (как у нас)
        top = summary["by_variant"][0]
        kpi = float(top.get(kpi_key, 0.0))
        trades = int(top.get("trades", summary.get("trades_total", 0)))
    else:
        kpi = 0.0
        trades = int(summary.get("trades_total", 0))

    dd = float(summary.get("max_dd_pct", 0.0))

    result = {
        "kpi": kpi,
        "trades": trades,
        "max_dd_pct": dd,
        "runtime_sec": dt,
        "theta": theta,
        "split": split,
    }

    cache_file.write_text(json.dumps(result))
    return result