# utils/detect_fvg.py

import os
import pandas as pd

FVG_DEBUG = str(os.getenv("FVG_DEBUG", "0")).lower() in ("1","true","yes","y","on")

def detect_fvg_imbalances(
    df,
    volume_multiplier=None,
    max_days_to_fill=None,
    tolerance_pct=None,
    min_strength_pct=None
):
    # читаем из env, если не передано вручную
    volume_multiplier = float(volume_multiplier if volume_multiplier is not None else os.getenv("FVG_VOL_MULT", 1.5))
    max_days_to_fill = int(max_days_to_fill if max_days_to_fill is not None else os.getenv("MAX_FILL_DAYS", 30))
    tolerance_pct = float(tolerance_pct if tolerance_pct is not None else os.getenv("FVG_TOLERANCE_PCT", 0.1))
    min_strength_pct = float(
        min_strength_pct if min_strength_pct is not None else os.getenv("DEFAULT_MIN_STRENGTH", 3.0))
    """
    Ищет FVG-имбалансы с фильтрацией по силе.
    Возвращает список словарей с полями:
      - type, price, time, strength, filled, days_to_fill, filled_at
      - open2, high2, low2, close2, next_open для дальнейших методов входа
    """
    if FVG_DEBUG:
        print("DEBUG detect_fvg_imbalances — входные колонки df:", getattr(df, "columns", []))

    imbalances = []
    df = df.copy()

    # Если кто-то не поставил индекс, делаем его из timestamp
    if isinstance(df, pd.DataFrame) and 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    # теперь индекс точно datetime (UTC-aware)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # скользящий средний объёма
    avg_vol = df['volume'].rolling(window=20, min_periods=1).mean()

    for i in range(2, len(df)):
        c0, c1, c2 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        t2 = df.index[i]

        # сохраним данные 3-й свечи
        candle_info = {
            'open2':  float(c2['open']),
            'high2':  float(c2['high']),
            'low2':   float(c2['low']),
            'close2': float(c2['close']),
        }
        next_open = float(df['open'].iloc[i+1]) if i+1 < len(df) else None

        gap_up   = float(c2['low'])  > float(c0['high'])
        gap_down = float(c2['high']) < float(c0['low'])
        vol_spike = (
            float(c1['volume']) > float(volume_multiplier) * float(avg_vol.iloc[i-1]) or
            float(c2['volume']) > float(volume_multiplier) * float(avg_vol.iloc[i])
        )
        if not (vol_spike and (gap_up or gap_down)):
            continue

        # вычисляем силу
        if gap_up:
            strength  = (float(c2['low'])  - float(c0['high'])) / max(float(c0['high']), 1e-12) * 100.0
            price_lvl = float(c2['low'])
            imb_type  = 'BUY'
        else:
            strength  = (float(c0['low'])  - float(c2['high'])) / max(float(c0['low']), 1e-12) * 100.0
            price_lvl = float(c2['high'])
            imb_type  = 'SELL'

        if strength < float(min_strength_pct):
            continue

        # проверка возврата
        tol      = price_lvl * float(tolerance_pct) / 100.0
        deadline = t2 + pd.Timedelta(days=int(max_days_to_fill))
        future   = df[(df.index > t2) & (df.index <= deadline)]
        filled, fill_time = False, None
        for ts, candle in future.iterrows():
            if imb_type=='BUY' and float(candle['low'])  <= price_lvl + tol:
                filled, fill_time = True, ts
                break
            if imb_type=='SELL' and float(candle['high']) >= price_lvl - tol:
                filled, fill_time = True, ts
                break
        days_to_fill = (fill_time - t2).days if fill_time is not None else None

        imbalances.append({
            'type':        imb_type,
            'price':       price_lvl,
            'time':        t2,
            'strength':    float(strength),
            'filled':      bool(filled),
            'days_to_fill':days_to_fill,
            'filled_at':   fill_time,
            **candle_info,
            'next_open':   next_open
        })

    if FVG_DEBUG:
        print(f"🔍 detect_fvg_imbalances: найдено {len(imbalances)} сильных имбалансов (≥{min_strength_pct}%)")
    return imbalances