# utils/detect_fvg_close.py
import os
import pandas as pd

# ВКЛ/ВЫКЛ отладочного вывода
FVG_DEBUG = str(os.getenv("FVG_DEBUG", "0")).lower() in ("1","true","yes","y","on")

def _as_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else float(default)
    except Exception:
        return float(default)

def _as_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and str(v).strip() != "" else int(default)
    except Exception:
        return int(default)

def detect_fvg_imbalances_close(
    df: pd.DataFrame,
    *,
    volume_multiplier: float = None,
    max_days_to_fill: int = None,
    tolerance_pct: float = None,
    min_strength_pct: float = None
):
    """
    ЖИВОЙ вариант детектора FVG:
    - Возвращает 'time' как ВРЕМЯ ЗАКРЫТИЯ свечи (t_close = t_open + 4h).
    - Дополнительно кладёт 'time_open' (исходный индекс бара).
    - Остальная логика совпадает с исходным детектором.

    Возвращает список словарей:
      {
        type: 'BUY'|'SELL',
        price: float,                  # уровень FVG
        time: pd.Timestamp (UTC),      # ВРЕМЯ ЗАКРЫТИЯ БАРА
        time_open: pd.Timestamp (UTC), # ВРЕМЯ ОТКРЫТИЯ БАРА (для справки)
        strength: float,
        filled: bool,
        days_to_fill: int|None,
        filled_at: pd.Timestamp|None,
        open2, high2, low2, close2,
        next_open
      }
    """
    # читаем дефолты из ENV (совместимо с текущими переменными)
    volume_multiplier = float(volume_multiplier if volume_multiplier is not None else _as_float("FVG_VOL_MULT", 1.5))
    max_days_to_fill  = int(max_days_to_fill  if max_days_to_fill  is not None else _as_int("MAX_FILL_DAYS", 30))
    tolerance_pct     = float(tolerance_pct   if tolerance_pct     is not None else _as_float("FVG_TOLERANCE_PCT", 0.1))
    min_strength_pct  = float(min_strength_pct if min_strength_pct is not None else _as_float("DEFAULT_MIN_STRENGTH", 3.0))

    if df is None or df.empty:
        return []

    # Копия и нормализация индекса -> UTC
    x = df.copy()
    if isinstance(x, pd.DataFrame) and "timestamp" in x.columns:
        x.set_index("timestamp", inplace=True)

    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")

    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")

    # скользящий средний объём
    avg_vol = x["volume"].rolling(window=20, min_periods=1).mean()

    out = []
    # i — индекс "c2" (третий бар в тройке c0,c1,c2)
    for i in range(2, len(x)):
        c0, c1, c2 = x.iloc[i-2], x.iloc[i-1], x.iloc[i]
        t_open = x.index[i]                   # время ОТКРЫТИЯ бара c2
        t_close = t_open + pd.Timedelta(hours=4)  # ВРЕМЯ ЗАКРЫТИЯ для этого бара

        # полезные поля для downstream
        candle_info = {
            "open2":  float(c2["open"]),
            "high2":  float(c2["high"]),
            "low2":   float(c2["low"]),
            "close2": float(c2["close"]),
        }
        next_open = float(x["open"].iloc[i+1]) if i+1 < len(x) else None

        # условия FVG: разрыв между c0 и c2 + всплеск объёма
        gap_up   = float(c2["low"])  > float(c0["high"])
        gap_down = float(c2["high"]) < float(c0["low"])
        vol_spike = (
            float(c1["volume"]) > float(volume_multiplier) * float(avg_vol.iloc[i-1]) or
            float(c2["volume"]) > float(volume_multiplier) * float(avg_vol.iloc[i])
        )
        if not (vol_spike and (gap_up or gap_down)):
            continue

        # сила и тип
        if gap_up:
            strength  = (float(c2["low"]) - float(c0["high"])) / max(float(c0["high"]), 1e-12) * 100.0
            price_lvl = float(c2["low"])
            imb_type  = "BUY"
        else:
            strength  = (float(c0["low"]) - float(c2["high"])) / max(float(c0["low"]), 1e-12) * 100.0
            price_lvl = float(c2["high"])
            imb_type  = "SELL"

        if strength < float(min_strength_pct):
            continue

        # проверка возврата в зону до дедлайна (как и в исходнике — от t_open)
        tol      = price_lvl * float(tolerance_pct) / 100.0
        deadline = t_open + pd.Timedelta(days=int(max_days_to_fill))
        future   = x[(x.index > t_open) & (x.index <= deadline)]
        filled, fill_time = False, None
        for ts, candle in future.iterrows():
            if imb_type == "BUY" and float(candle["low"])  <= price_lvl + tol:
                filled, fill_time = True, ts
                break
            if imb_type == "SELL" and float(candle["high"]) >= price_lvl - tol:
                filled, fill_time = True, ts
                break
        days_to_fill = (fill_time - t_open).days if fill_time is not None else None

        out.append({
            "type":         imb_type,
            "price":        price_lvl,
            "time":         t_close,     # КЛЮЧ: time = ВРЕМЯ ЗАКРЫТИЯ
            "time_open":    t_open,      # (на всякий случай для логов/диагностики)
            "strength":     float(strength),
            "filled":       bool(filled),
            "days_to_fill": days_to_fill,
            "filled_at":    fill_time,
            **candle_info,
            "next_open":    next_open,
        })

    if FVG_DEBUG:
        print(f"🔍 detect_fvg_imbalances_close: найдено {len(out)} сигналов (≥{min_strength_pct}%), "
              f"time = bar_close")

    return out

__all__ = ["detect_fvg_imbalances_close"]