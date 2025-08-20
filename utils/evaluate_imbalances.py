import pandas as pd
from datetime import timedelta


def evaluate_imbalances(df, imbalances, max_days=14, tolerance_pct=0.1):
    """
    Оценивает, был ли перекрыт каждый имбаланс в течение max_days.

    :param df: DataFrame с историей цены (OHLCV)
    :param imbalances: список имбалансов
    :param max_days: сколько дней дается на возврат
    :param tolerance_pct: допустимое отклонение от уровня
    :return: DataFrame с оценкой для каждого имбаланса
    """
    results = []

    # Приводим индекс к datetime (на всякий случай)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    for imb in imbalances:
        # ⛏ Правильное приведение времени к datetime
        raw_time = imb["time"]
        if isinstance(raw_time, (int, float)):
            # Временные метки часто в миллисекундах
            time_of_imb = pd.to_datetime(raw_time, unit='ms' if raw_time > 1e12 else 's')
        else:
            time_of_imb = pd.to_datetime(raw_time)

        price = imb["price"]
        imb_type = imb["type"]
        tolerance = price * tolerance_pct / 100

        cutoff_time = time_of_imb + timedelta(days=max_days)
        future_df = df[(df.index > time_of_imb) & (df.index <= cutoff_time)]

        filled = False
        fill_time = None

        for timestamp, candle in future_df.iterrows():
            if imb_type == "BUY" and candle["low"] <= price + tolerance:
                filled = True
                fill_time = timestamp
                break
            elif imb_type == "SELL" and candle["high"] >= price - tolerance:
                filled = True
                fill_time = timestamp
                break

        results.append({
            **imb,
            "filled": filled,
            "days_to_fill": (fill_time - time_of_imb).days if filled else None,
            "filled_at": fill_time
        })

    return pd.DataFrame(results)