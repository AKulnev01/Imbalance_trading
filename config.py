# config.py
import os
from dotenv import load_dotenv

def _as_bool(name: str, default=False) -> bool:
    s = str(os.getenv(name, str(default))).strip().lower()
    return s in ("1", "true", "yes", "y", "on")
def _as_list(name: str) -> list:
    return [x.strip().upper() for x in str(os.getenv(name, "")).split(",") if x.strip()]

load_dotenv()

# ===== Bybit keys/env =====
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_TESTNET    = _as_bool("BYBIT_TESTNET")
BYBIT_CATEGORY   = os.getenv("BYBIT_CATEGORY").lower()   # 'linear' | 'spot'
EXECUTION_ENV    = os.getenv("EXECUTION_ENV").lower()
USE_MAINNET_MARKET_DATA = _as_bool("USE_MAINNET_MARKET_DATA")

# ===== Telegram (уведомления) =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

# ===== Стратегия (Momentum RR 1:5) =====
ENTRY_MODE        = os.getenv("ENTRY_MODE").upper()    # RETEST | BREAKOUT | MOMENTUM
MOMENTUM_TP_PCT   = float(os.getenv("MOMENTUM_TP_PCT"))
MOMENTUM_SL_PCT   = float(os.getenv("MOMENTUM_SL_PCT"))
DEFAULT_TTL_DAYS  = int(os.getenv("DEFAULT_TTL_DAYS"))

# ===== MOMENTUM исполнение =====
MOMENTUM_EXEC            = os.getenv("MOMENTUM_EXEC", "market").lower()      # market | aggr_limit
AGGR_LIMIT_EPS_PCT       = float(os.getenv("AGGR_LIMIT_EPS_PCT", "0.0005"))
MOMENTUM_FALLBACK        = os.getenv("MOMENTUM_FALLBACK", "market").lower()  # market | none
MAX_ACCEPT_SLIPPAGE_PCT  = float(os.getenv("MAX_ACCEPT_SLIPPAGE_PCT", "0.003"))

# ===== Аллокация/комиссии =====
INITIAL_CAPITAL   = float(os.getenv("INITIAL_CAPITAL"))
POSITION_FRACTION = float(os.getenv("POSITION_FRACTION"))
FEE_TAKER         = float(os.getenv("FEE_TAKER"))
SLIPPAGE_PCT      = float(os.getenv("SLIPPAGE_PCT"))

# ===== Датчики FVG =====
DEFAULT_MIN_STRENGTH = float(os.getenv("DEFAULT_MIN_STRENGTH"))
FVG_VOL_MULT         = float(os.getenv("FVG_VOL_MULT"))
FVG_TOLERANCE_PCT    = float(os.getenv("FVG_TOLERANCE_PCT"))

# ===== Риск/менеджмент (совместимость со старым кодом) =====
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO") or (MOMENTUM_TP_PCT / MOMENTUM_SL_PCT))
RISK_PCT          = float(os.getenv("RISK_PCT", "1.0"))
MAX_FILL_DAYS     = int(os.getenv("MAX_FILL_DAYS", "7"))

# ===== Фильтры/дирекшн =====
MIN_DAILY_VOLUME        = float(os.getenv("MIN_DAILY_VOLUME"))
ENABLE_BUY              = _as_bool("ENABLE_BUY")
ENABLE_SELL             = _as_bool("ENABLE_SELL")
BUY_EXTRA_STRENGTH_PCT  = float(os.getenv("BUY_EXTRA_STRENGTH_PCT"))

# ===== Интраминутный вход =====
USE_INTRAMINUTE_ENTRY   = _as_bool("USE_INTRAMINUTE_ENTRY")
INTRAMIN_LOOKBACK_MIN   = int(os.getenv("INTRAMIN_LOOKBACK_MIN"))
ENTRY_LOOKAHEAD_MINUTES = int(os.getenv("ENTRY_LOOKAHEAD_MINUTES"))
INTRAM_VOLUME_MULT      = float(os.getenv("INTRAM_VOLUME_MULT"))

# ===== Intrabar =====
INTRABAR_INTERVALS               = os.getenv("INTRABAR_INTERVALS").split(",")
INTRABAR_LOOKBACK_DAYS_FALLBACK  = int(os.getenv("INTRABAR_LOOKBACK_DAYS_FALLBACK"))

# ===== Прочее runtime =====
AUTOTRADE_STATE_PATH        = os.getenv("AUTOTRADE_STATE_PATH")
LIMIT_ORDER_MODE            = os.getenv("LIMIT_ORDER_MODE")
LIMIT_POSTONLY_FALLBACK     = os.getenv("LIMIT_POSTONLY_FALLBACK")
BACKFILL_4H_BARS            = int(os.getenv("BACKFILL_4H_BARS"))
ENTRY_TOUCH_LTF             = [x.strip() for x in os.getenv("ENTRY_TOUCH_LTF").split(",") if x.strip()]
POLL_ORDERS_SEC             = int(os.getenv("POLL_ORDERS_SEC"))
ENV_RELOAD_SEC              = int(os.getenv("ENV_RELOAD_SEC"))
BALANCE_SYNC_MIN            = int(os.getenv("BALANCE_SYNC_MIN"))
ENTRY_BACKFILL_LOOKBACK_DAYS= int(os.getenv("ENTRY_BACKFILL_LOOKBACK_DAYS"))
BYBIT_DEBUG                 = int(os.getenv("BYBIT_DEBUG"))

# === DEEP_RETEST ===
DEEP_RETEST_DYNAMIC      = _as_bool("DEEP_RETEST_DYNAMIC", False)
DEEP_RETEST_PCT          = float(os.getenv("DEEP_RETEST_PCT", "0.05"))

DEEP_STRENGTH_MIN        = float(os.getenv("DEEP_STRENGTH_MIN", "2.0"))
DEEP_STRENGTH_MAX        = float(os.getenv("DEEP_STRENGTH_MAX", "6.0"))
DEEP_DEPTH_MIN_PCT       = float(os.getenv("DEEP_DEPTH_MIN_PCT", "0.02"))
DEEP_DEPTH_MAX_PCT       = float(os.getenv("DEEP_DEPTH_MAX_PCT", "0.08"))

DEEP_TP_MODE             = os.getenv("DEEP_TP_MODE", "rr").lower()  # rr | zone_mid | zone_top
DEEP_RR                  = float(os.getenv("DEEP_RR", "3.0"))

# (опц) названия колонок с границами зоны (если присутствуют в сигналах)
FVG_TOP_COL              = os.getenv("FVG_TOP_COL", "fvg_top")
FVG_BOTTOM_COL           = os.getenv("FVG_BOTTOM_COL", "fvg_bottom")

# ===== Universe =====
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE"))

TRADE_UNIVERSE = [
    # --- твои реальные сигнальные (18) ---
    "SIRENUSDT", "RFCUSDT", "ZORAUSDT", "BANANAS31USDT", "GORKUSDT", "EPTUSDT",
    "DOODUSDT", "AGTUSDT", "SOPHUSDT", "SPKUSDT", "HUSDT", "SAHARAUSDT",
    "DMCUSDT", "TACUSDT", "SOONUSDT", "A2ZUSDT",

    # --- мажоры / ликвидные ---
    "BTCUSDT", "ETHUSDT", "APTUSDT", "WLDUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT",

    # --- хайповые L1/L2 ---
    "OPUSDT", "ARBUSDT", "SUIUSDT", "INJUSDT", "NEARUSDT",
    "SEIUSDT", "TIAUSDT", "STRKUSDT",

    # --- DeFi/инфра ---
    "ATOMUSDT", "FILUSDT", "AAVEUSDT", "LINKUSDT", "UNIUSDT", "LDOUSDT",

    # --- свежие (Innovation Zone, ≥90d) ---
    "PROMPTUSDT", "OBTUSDT", "HAEDALUSDT", "TUTUSDT", "GPSUSDT", "MILKUSDT",
    "B2USDT", "AINUSDT", "CROSSUSDT", "SERAPHUSDT",
]

BLACKLIST_SYMBOLS = _as_list("BLACKLIST_SYMBOLS")

def filter_universe(symbols: list) -> list:
    """Удобный хелпер: выкинем из списка символы из чёрного списка."""
    bl = set(BLACKLIST_SYMBOLS or [])
    return [s for s in symbols if s.upper() not in bl]

# === MOMENTUM fill realism ===
MOMENTUM_FILL_WINDOW_MIN   = int(os.getenv("MOMENTUM_FILL_WINDOW_MIN", "5"))
MAX_CONCURRENT_POSITIONS   = int(os.getenv("MAX_CONCURRENT_POSITIONS", "4"))
MOMENTUM_MIN_LTF_BARS      = int(os.getenv("MOMENTUM_MIN_LTF_BARS", "1"))

# ===== MOMENTUM LTF entry =====
MOMENTUM_LTF_ENTRY_OFFSET_MIN = int(os.getenv("MOMENTUM_LTF_ENTRY_OFFSET_MIN", "1"))
MOMENTUM_ENTRY_USE_OPEN       = _as_bool("MOMENTUM_ENTRY_USE_OPEN", True)
