import os
import pandas as pd
from datetime import datetime


def export_to_excel(imbalances, symbol: str, lookback_days: int) -> str:
    """
    Сохраняет список имбалансов в Excel-файл в папку ~/Documents/отчеты
    Название файла: {symbol}_{lookback_days}d_{YYYY-MM-DD}.xlsx

    :param imbalances: список или DataFrame с данными об имбалансах
    :param symbol: торговая пара, например 'BTCUSDT'
    :param lookback_days: период анализа в днях
    :return: путь к сохранённому файлу
    """
    # Преобразуем список имбалансов в DataFrame, если нужно
    df = imbalances if isinstance(imbalances, pd.DataFrame) else pd.DataFrame(imbalances)

    # Приводим time и filled_at к строковому представлению для корректного отображения в Excel
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'filled_at' in df.columns:
        df['filled_at'] = pd.to_datetime(df['filled_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Убедимся, что папка для отчётов существует
    dir_path = os.path.expanduser('~/Documents/отчеты')
    os.makedirs(dir_path, exist_ok=True)

    # Формируем имя файла с датой
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_name = f"{symbol}_{lookback_days}d_{date_str}.xlsx"
    file_path = os.path.join(dir_path, file_name)

    # Сохраняем в Excel
    df.to_excel(file_path, index=False)
    print(f"✅ Excel-отчет сохранен как {file_path}")

    return file_path