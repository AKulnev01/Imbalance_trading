# config.py
import os
from dotenv import load_dotenv


def _as_bool(x, default=False):
    s = str(os.getenv(x, str(default))).strip().lower()
    return s in ("1", "true", "yes", "y")

load_dotenv()

# ===== Bybit keys/env =====
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET = _as_bool("BYBIT_TESTNET", True)
BYBIT_CATEGORY = os.getenv("BYBIT_CATEGORY", "linear").lower()       # 'linear' или 'spot'
# ===== Telegram (уведомления) =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")  # можно строкой

# ===== Стратегия/анализ по реальным данным, а торгуем — где скажем =====
# берём свечи/тикеры для анализа с mainnet (реальные цены)
USE_MAINNET_MARKET_DATA = _as_bool("USE_MAINNET_MARKET_DATA", True)
# куда шлём ордера: "testnet" или "mainnet"
EXECUTION_ENV = os.getenv("EXECUTION_ENV", "testnet").lower()

# ===== Основные настройки анализа =====
SYMBOL = "BTCUSDT"
INTERVAL = "4h"              # 1m, 15m, 1h, 4h, 1d ...
LOOKBACK_DAYS = 30
FVG_STRENGTH_THRESHOLD = 3.0
UNIVERSE = []  # пусто = возьмём динамически топ ликвидных

# ===== Фильтры =====
MIN_DAILY_VOLUME = 0

# ===== Risk / Money management =====
RISK_REWARD_RATIO = 3
RISK_PCT = 1.0
PROFIT_PCT = RISK_PCT * RISK_REWARD_RATIO
MAX_FILL_DAYS = 7
DEFAULT_TTL_DAYS = 7

# ===== Execution / fills =====
ENTRY_DELAY_MINUTES = 2
POSITION_SIZE_USD = 1000  # для ручных тестов; в автоторговле используем аллокатор

# ===== Direction focus =====
ENABLE_BUY = True
ENABLE_SELL = True
BUY_EXTRA_STRENGTH_PCT = 0.5
DEFAULT_MIN_STRENGTH = 3.0

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "4000"))
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "100"))
DEFAULT_TTL_DAYS = int(os.getenv("DEFAULT_TTL_DAYS", "5"))
DEFAULT_MIN_STRENGTH = float(os.getenv("DEFAULT_MIN_STRENGTH", "3.0"))
AUTOTRADE_STATE_PATH = os.getenv("AUTOTRADE_STATE_PATH", "autotrade_state.json")

# ===== Intraminute entry (опционально) =====
USE_INTRAMINUTE_ENTRY = False
INTRAMIN_LOOKBACK_MIN = 60
ENTRY_LOOKAHEAD_MINUTES = 0
INTRAM_VOLUME_MULT = 1.2

# ===== Fees & slippage =====
FEE_MAKER = 0.0002
FEE_TAKER = 0.00055
SLIPPAGE_PCT = 0.0005

# Ранний выход (опционально)
EARLY_EXIT_LAST24H = True
LAST24H_MIN_PROFIT_PCT = 0.0

# ===== Список монет для боевого автотрейда =====
TRADE_UNIVERSE = [
    # --- твои реальные сигнальные (18) ---
    "SIRENUSDT", "RFCUSDT", "ZORAUSDT", "BANANAS31USDT", "GORKUSDT", "EPTUSDT",
    "DOODUSDT", "AGTUSDT", "SOPHUSDT", "SPKUSDT", "HUSDT", "SAHARAUSDT",
    "DMCUSDT", "TACUSDT", "SOONUSDT", "A2ZUSDT",

    # --- мажоры / ликвидные ---
    "BTCUSDT", "ETHUSDT", "APTUSDT", "WLDUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "MATICUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT",

    # --- хайповые L1/L2 ---
    "OPUSDT", "ARBUSDT", "SUIUSDT", "INJUSDT", "NEARUSDT", "RNDRUSDT",
    "SEIUSDT", "TIAUSDT", "STRKUSDT",

    # --- DeFi/инфра ---
    "ATOMUSDT", "FILUSDT", "AAVEUSDT", "LINKUSDT", "UNIUSDT", "LDOUSDT"
]

# === entry-modes ===
ENTRY_MODE         = os.getenv("ENTRY_MODE", "RETEST").upper()          # RETEST | BREAKOUT | MOMENTUM
ENTRY_OFFSET_PCT   = float(os.getenv("ENTRY_OFFSET_PCT", "0.02"))       # 2% (информативный таргет в BREAKOUT)

# === momentum params ===
MOMENTUM_ATR_N     = int(os.getenv("MOMENTUM_ATR_N", "14"))
MOMENTUM_BODY_ATR  = float(os.getenv("MOMENTUM_BODY_ATR", "1.5"))
MOMENTUM_RANGE_ATR = float(os.getenv("MOMENTUM_RANGE_ATR", "2.0"))
MOMENTUM_VOL_SMA   = float(os.getenv("MOMENTUM_VOL_SMA", "2.0"))
MOMENTUM_TP_PCT    = float(os.getenv("MOMENTUM_TP_PCT", "0.02"))        # 2%
MOMENTUM_SL_PCT    = float(os.getenv("MOMENTUM_SL_PCT", "0.01"))        # 1%