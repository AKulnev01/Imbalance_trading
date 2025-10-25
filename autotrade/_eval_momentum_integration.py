# autotrade/_eval_momentum_integration.py  (например так; или прямо в файле, где _cmd_eval_mom)

import os
import io
import tempfile

# важно: чтобы импорт совпал с твоей структурой проекта
# если модуль у тебя: models/evaluate_momentum.py — оставь как есть,
# иначе поправь импорт под своё дерево
from models.evaluate_momentum import evaluate_momentum as _eval_momentum

def _eval_and_pack(signals_path: str, hours: int) -> bytes:
    """
    Делает eval «как в models/evaluate_momentum.py» и возвращает XLSX-байты для TG.
    """
    # --- выравниваем среду под оффлайн-минутки и гибридный выход ---
    os.environ["USE_LOCAL_MINUTES"] = "1"
    os.environ["USE_LOCAL_4H"]      = "0"
    os.environ.setdefault("LTF_ROOT", "./data/m1")
    os.environ["DISABLE_MINUTE_FALLBACK"] = "0"
    os.environ["MINUTE_EXIT_FOR_SINGLE"]  = "1"

    # (опционально) если хочешь 1 слот капитала глобально, оставь:
    # os.environ["MAX_CONCURRENT_POSITIONS"] = "1"
    # иначе убери строку выше или поставь "0"

    # эти параметры будут взяты из config/evaluate_common.get_cfg, но
    # при желании можно принудительно прокинуть:
    # os.environ["MOMENTUM_TP_PCT"]  = "0.03"
    # os.environ["MOMENTUM_SL_PCT"]  = "0.01"
    # os.environ["FEE_TAKER"]        = "0.0006"
    # os.environ["ENTRY_SLIPPAGE_PCT"] = "0.003"
    # os.environ["EXIT_SLIPPAGE_PCT"]  = "0.003"
    # os.environ["STOP_SLIPPAGE_PCT"]  = "0.003"
    # os.environ["EVAL_MAX_HOURS"]     = "80"
    # os.environ["EVAL_PICK_STRONGEST"]= "1"

    # --- готовим временный выходной файл ---
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    out_path = tmp.name
    tmp.close()

    # lookback_days/interval/прочие — как в evaluate_momentum.py по умолчанию
    _eval_momentum(
        signals_path=signals_path,
        result_path=out_path,
        lookback_days=360,        # как в твоём скрипте
        interval="4h",
        max_days=None,            # TTL берётся из DEFAULT_TTL_DAYS/EVAL_MAX_HOURS
        only_filled=False,
        dedup=False,
        initial_capital=None,     # если хочешь прогон с капиталом — поставь число
        capital_aware=True,
    )

    # читаем XLSX в память — для отправки в Telegram
    with open(out_path, "rb") as f:
        data = f.read()

    try:
        os.remove(out_path)
    except Exception:
        pass
    return data