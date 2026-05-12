# utils/symbols.py
import requests
import logging
from config import MIN_DAILY_VOLUME

logger = logging.getLogger(__name__)

def fetch_top_symbols(limit: int = 100):
    """
    Возвращает список из limit символов Futures USDT-пар Bybit,
    отсортированных по 24h объёму.
    При наличии порога MIN_DAILY_VOLUME отбирает сначала тех, кто выше порога,
    затем дополняет до limit самых ликвидных.
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}  # USDT-фьючерсы

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()

        # Проверяем код возврата Bybit
        if data.get("retCode") != 0:
            logger.error(f"Bybit API error: {data.get('retCode')} {data.get('retMsg')}")
            return []

        result = data.get("result", {})
        items = result.get("list", [])

        if not items:
            logger.warning("Bybit API вернул пустой список тикеров.")
            return []

        def get_vol(item):
            # Обычно ключ 'volume24h', но возможны вариации
            return float(
                item.get("volume24h")
                or item.get("turnover24h")
                or item.get("volume")
                or 0
            )

        # Сортируем всех по объёму по убыванию
        all_sorted = sorted(items, key=get_vol, reverse=True)

        # Если задан порог, сначала отбираем по нему
        selected = []
        if MIN_DAILY_VOLUME and MIN_DAILY_VOLUME > 0:
            for it in all_sorted:
                if get_vol(it) >= MIN_DAILY_VOLUME:
                    selected.append(it)
                if len(selected) >= limit:
                    break

        # Если после фильтрации по порогу недостаточно, дополняем до limit
        if len(selected) < limit:
            seen = {it['symbol'] for it in selected}
            for it in all_sorted:
                if it['symbol'] in seen:
                    continue
                selected.append(it)
                if len(selected) >= limit:
                    break

        # Возвращаем только символы
        symbols = [it['symbol'] for it in selected[:limit]]
        logger.info(f"Выбрано {len(symbols)} символов для анализа.")
        return symbols

    except requests.RequestException as e:
        logger.error(f"Ошибка запроса к Bybit API: {e}")
        return []
    except Exception as e:
        logger.error(f"Ошибка в fetch_top_symbols: {e}")
        return []