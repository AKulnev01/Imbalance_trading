# config.py — import-safe + .env/.envrc + hot-reload
import os
from dotenv import load_dotenv

# ---------- helpers (должны идти до чтения env) ----------
def _as_bool(name: str, default=False) -> bool:
    s = str(os.getenv(name, str(default))).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _as_list(name: str) -> list:
    return [x.strip().upper() for x in str(os.getenv(name, "")).split(",") if x.strip()]

def _as_float(name: str, default: float = 0.0) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else float(default)
    except Exception:
        return float(default)

def _as_int(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and str(v).strip() != "" else int(default)
    except Exception:
        return int(default)

# ---------- загрузка .env / .envrc (первичный pass) ----------
def _load_env_files(*, override=False):
    # 1) .env (если есть)
    if os.path.exists(".env"):
        load_dotenv(".env", override=override)
    else:
        load_dotenv(override=override)

    # 2) .envrc (поддержка export/кавычек)
    if os.path.exists(".envrc"):
        with open(".envrc", "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if override:
                    os.environ[k] = v
                else:
                    os.environ.setdefault(k, v)

# первичная загрузка без override
_load_env_files(override=False)

# ---------- минимальные дефолты (чтобы импорт не падал) ----------
os.environ.setdefault("BYBIT_CATEGORY", "linear")   # 'linear' | 'spot'
os.environ.setdefault("EXECUTION_ENV", "prod")

# Стратегия / риск
os.environ.setdefault("ENTRY_MODE", "MOMENTUM")     # RETEST | BREAKOUT | MOMENTUM
os.environ.setdefault("MOMENTUM_TP_PCT", "0.135")
os.environ.setdefault("MOMENTUM_SL_PCT", "0.004")   # 0.4%
os.environ.setdefault("MOMENTUM_TTL_HOURS", "80")
os.environ.setdefault("BYBIT_MARGIN_MODE", "isolated")  # или "cross"
os.environ.setdefault("BYBIT_LEVERAGE", "3")            # что используешь реально

# TTL (дней)
os.environ.setdefault("DEFAULT_TTL_DAYS", "3")

# Базовые финпараметры
os.environ.setdefault("INITIAL_CAPITAL", "1000")
os.environ.setdefault("POSITION_FRACTION", "1.0")
os.environ.setdefault("FEE_TAKER", "0.001")         # 0.1%
os.environ.setdefault("SLIPPAGE_PCT", "0.004")      # 0.4%

# Per-leg slippage дефолтами равны общему SLIPPAGE_PCT (если не заданы)
os.environ.setdefault("ENTRY_SLIPPAGE_PCT", os.getenv("SLIPPAGE_PCT", "0.004"))
os.environ.setdefault("EXIT_SLIPPAGE_PCT",  os.getenv("SLIPPAGE_PCT", "0.004"))
os.environ.setdefault("STOP_SLIPPAGE_PCT",  os.getenv("SLIPPAGE_PCT", "0.004"))

# FVG датчики
os.environ.setdefault("DEFAULT_MIN_STRENGTH", "3")
os.environ.setdefault("FVG_VOL_MULT", "1.0")
os.environ.setdefault("FVG_TOLERANCE_PCT", "0.001")

# Прочее
os.environ.setdefault("INTRABAR_INTERVALS", "1,5,15")
os.environ.setdefault("INTRABAR_LOOKBACK_DAYS_FALLBACK", "30")
os.environ.setdefault("BACKFILL_4H_BARS", "50")
os.environ.setdefault("POLL_ORDERS_SEC", "2")
os.environ.setdefault("ENV_RELOAD_SEC", "10")
os.environ.setdefault("BALANCE_SYNC_MIN", "10")
os.environ.setdefault("ENTRY_BACKFILL_LOOKBACK_DAYS", "7")
os.environ.setdefault("BYBIT_DEBUG", "0")
os.environ.setdefault("UNIVERSE_SIZE", "50")
os.environ.setdefault("MOMENTUM_LTF_ENTRY_OFFSET_MIN", "1")
os.environ.setdefault("MAX_ACCEPT_SLIPPAGE_PCT", "0.004")  # 0.4%

# ===== Bybit keys/env =====
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_TESTNET    = _as_bool("BYBIT_TESTNET")
BYBIT_CATEGORY   = (os.getenv("BYBIT_CATEGORY") or "linear").strip().lower()   # 'linear' | 'spot'
EXECUTION_ENV    = (os.getenv("EXECUTION_ENV") or "prod").strip().lower()
USE_MAINNET_MARKET_DATA = _as_bool("USE_MAINNET_MARKET_DATA")

# ===== Telegram (уведомления) =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

# ===== Стратегия (Momentum RR по env) =====
ENTRY_MODE        = (os.getenv("ENTRY_MODE") or "MOMENTUM").strip().upper()
MOMENTUM_TP_PCT   = _as_float("MOMENTUM_TP_PCT", 0.135)
MOMENTUM_SL_PCT   = _as_float("MOMENTUM_SL_PCT", 0.004)
DEFAULT_TTL_DAYS  = _as_int("DEFAULT_TTL_DAYS", 3)

# ===== MOMENTUM исполнение =====
MOMENTUM_EXEC            = os.getenv("MOMENTUM_EXEC", "market").strip().lower()      # market | aggr_limit
AGGR_LIMIT_EPS_PCT       = _as_float("AGGR_LIMIT_EPS_PCT", 0.0005)
MOMENTUM_FALLBACK        = os.getenv("MOMENTUM_FALLBACK", "market").strip().lower()  # market | none
MAX_ACCEPT_SLIPPAGE_PCT  = _as_float("MAX_ACCEPT_SLIPPAGE_PCT", 0.004)

# ===== Аллокация/комиссии =====
INITIAL_CAPITAL   = _as_float("INITIAL_CAPITAL", 1000.0)
POSITION_FRACTION = _as_float("POSITION_FRACTION", 1.0)
FEE_TAKER         = _as_float("FEE_TAKER", 0.001)
SLIPPAGE_PCT      = _as_float("SLIPPAGE_PCT", 0.004)

# Явные per-leg (если заданы — переопределяют общий)
ENTRY_SLIPPAGE_PCT = _as_float("ENTRY_SLIPPAGE_PCT", SLIPPAGE_PCT)
EXIT_SLIPPAGE_PCT  = _as_float("EXIT_SLIPPAGE_PCT",  SLIPPAGE_PCT)
STOP_SLIPPAGE_PCT  = _as_float("STOP_SLIPPAGE_PCT",  SLIPPAGE_PCT)

# ===== Датчики FVG =====
DEFAULT_MIN_STRENGTH = _as_float("DEFAULT_MIN_STRENGTH", 3.0)
FVG_VOL_MULT         = _as_float("FVG_VOL_MULT", 1.0)
FVG_TOLERANCE_PCT    = _as_float("FVG_TOLERANCE_PCT", 0.0)

# ===== Риск/менеджмент (совместимость со старым кодом) =====
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO") or (MOMENTUM_TP_PCT / max(MOMENTUM_SL_PCT, 1e-12)))
RISK_PCT          = _as_float("RISK_PCT", 1.0)
MAX_FILL_DAYS     = _as_int("MAX_FILL_DAYS", 7)

# ===== Фильтры/дирекшн =====
MIN_DAILY_VOLUME        = _as_float("MIN_DAILY_VOLUME", 0.0)
ENABLE_BUY              = _as_bool("ENABLE_BUY", True)
ENABLE_SELL             = _as_bool("ENABLE_SELL", True)
BUY_EXTRA_STRENGTH_PCT  = _as_float("BUY_EXTRA_STRENGTH_PCT", 0.0)

# ===== Интраминутный вход =====
USE_INTRAMINUTE_ENTRY   = _as_bool("USE_INTRAMINUTE_ENTRY", False)
INTRAMIN_LOOKBACK_MIN   = _as_int("INTRAMIN_LOOKBACK_MIN", 15)
ENTRY_LOOKAHEAD_MINUTES = _as_int("ENTRY_LOOKAHEAD_MINUTES", 5)
INTRAM_VOLUME_MULT      = _as_float("INTRAM_VOLUME_MULT", 1.5)

# ===== Intrabar =====
INTRABAR_INTERVALS               = (os.getenv("INTRABAR_INTERVALS") or "1,5,15").split(",")
INTRABAR_LOOKBACK_DAYS_FALLBACK  = _as_int("INTRABAR_LOOKBACK_DAYS_FALLBACK", 30)

# ===== Прочее runtime =====
AUTOTRADE_STATE_PATH        = os.getenv("AUTOTRADE_STATE_PATH")
LIMIT_ORDER_MODE            = os.getenv("LIMIT_ORDER_MODE")
LIMIT_POSTONLY_FALLBACK     = os.getenv("LIMIT_POSTONLY_FALLBACK")
BACKFILL_4H_BARS            = _as_int("BACKFILL_4H_BARS", 50)
ENTRY_TOUCH_LTF             = [x.strip() for x in (os.getenv("ENTRY_TOUCH_LTF") or "").split(",") if x.strip()]
POLL_ORDERS_SEC             = _as_int("POLL_ORDERS_SEC", 2)
ENV_RELOAD_SEC              = _as_int("ENV_RELOAD_SEC", 10)
BALANCE_SYNC_MIN            = _as_int("BALANCE_SYNC_MIN", 10)
ENTRY_BACKFILL_LOOKBACK_DAYS= _as_int("ENTRY_BACKFILL_LOOKBACK_DAYS", 7)
BYBIT_DEBUG                 = _as_int("BYBIT_DEBUG", 0)

# === DEEP_RETEST ===
DEEP_RETEST_DYNAMIC      = _as_bool("DEEP_RETEST_DYNAMIC", False)
DEEP_RETEST_PCT          = _as_float("DEEP_RETEST_PCT", 0.05)
DEEP_STRENGTH_MIN        = _as_float("DEEP_STRENGTH_MIN", 2.0)
DEEP_STRENGTH_MAX        = _as_float("DEEP_STRENGTH_MAX", 6.0)
DEEP_DEPTH_MIN_PCT       = _as_float("DEEP_DEPTH_MIN_PCT", 0.02)
DEEP_DEPTH_MAX_PCT       = _as_float("DEEP_DEPTH_MAX_PCT", 0.08)
DEEP_TP_MODE             = (os.getenv("DEEP_TP_MODE", "rr") or "rr").strip().lower()  # rr | zone_mid | zone_top
DEEP_RR                  = _as_float("DEEP_RR", 3.0)

# (опц) названия колонок с границами зоны (если присутствуют в сигналах)
FVG_TOP_COL              = os.getenv("FVG_TOP_COL", "fvg_top")
FVG_BOTTOM_COL           = os.getenv("FVG_BOTTOM_COL", "fvg_bottom")

# ===== Universe =====
UNIVERSE_SIZE = _as_int("UNIVERSE_SIZE", 50)

_env_universe = [s.strip().upper() for s in (os.getenv("TRADE_UNIVERSE") or "").split(",") if s.strip()]
TRADE_UNIVERSE = [
    # --- твои реальные сигнальные (18) ---
    "SIRENUSDT", "RFCUSDT", "ZORAUSDT", "BANANAS31USDT", "GORKUSDT", "EPTUSDT",
    "DOODUSDT", "AGTUSDT", "SOPHUSDT", "SPKUSDT", "HUSDT", "SAHARAUSDT",
    "DMCUSDT", "TACUSDT", "SOONUSDT", "A2ZUSDT", "TRXUSDT",
    # --- мажоры / ликвидные ---
    "BTCUSDT", "ETHUSDT", "APTUSDT", "WLDUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "LTCUSDT", "ADAUSDT", "XRPUSDT",
    # --- хайповые L1/L2 ---
    "OPUSDT", "ARBUSDT", "SUIUSDT", "INJUSDT", "NEARUSDT",
    "SEIUSDT", "TIAUSDT", "STRKUSDT",
    # --- DeFi/инфра ---
    "ATOMUSDT", "FILUSDT", "AAVEUSDT", "LINKUSDT", "UNIUSDT", "LDOUSDT",
    # --- свежие (Innovation Zone, ≥90d) ---
    "PROMPTUSDT", "OBTUSDT", "HAEDALUSDT", "TUTUSDT", "GPSUSDT", "MILKUSDT",
    "B2USDT", "AINUSDT", "CROSSUSDT", "SERAPHUSDT",
    # --- новые ----
    "DOGEUSDT","XLMUSDT","BCHUSDT","HBARUSDT","SHIB1000USDT","MNTUSDT","XMRUSDT",
    "CROUSDT","TONUSDT","DOTUSDT","TAOUSDT","ZECUSDT","OKBUSDT","BGBUSDT",
    "ENAUSDT","1000PEPEUSDT","ASTERUSDT","ETCUSDT","ONDOUSDT","IPUSDT","POLUSDT",
    "MUSDT","ICPUSDT","KCSUSDT","ALGOUSDT","PIUSDT","VETUSDT","KASUSDT",
    "PENGUUSDT","SKYUSDT","FLRUSDT","RENDERUSDT","PUMPUSDT","1000BONKUSDT",
    "PAXGUSDT","GTUSDT","TRUMPUSDT","JUPUSDT","CAKEUSDT","SPXUSDT",
    "IMXUSDT","XDCUSDT","QNTUSDT","XAUTUSDT","2ZUSDT","XPLUSDT","STXUSDT",
    "CRVUSDT","AEROUSDT","NEXOUSDT","FETUSDT","GRTUSDT","1000FLOKIUSDT",
    "KAIAUSDT","PYTHUSDT","MYXUSDT","XTZUSDT","SNXUSDT","MORPHOUSDT",
    "ETHFIUSDT","ENSUSDT","IOTAUSDT","ABUSDT","CFXUSDT","PENDLEUSDT"
]

BLACKLIST_SYMBOLS = _as_list("BLACKLIST_SYMBOLS")

def filter_universe(symbols: list) -> list:
    """Удобный хелпер: выкинем из списка символы из чёрного списка."""
    bl = set(BLACKLIST_SYMBOLS or [])
    return [s for s in symbols if s.upper() not in bl]

# === MOMENTUM fill realism ===
MOMENTUM_FILL_WINDOW_MIN   = _as_int("MOMENTUM_FILL_WINDOW_MIN", 5)
MAX_CONCURRENT_POSITIONS   = _as_int("MAX_CONCURRENT_POSITIONS", 1)
MOMENTUM_MIN_LTF_BARS      = _as_int("MOMENTUM_MIN_LTF_BARS", 1)

# ===== MOMENTUM LTF entry =====
MOMENTUM_LTF_ENTRY_OFFSET_MIN = _as_int("MOMENTUM_LTF_ENTRY_OFFSET_MIN", 1)
MOMENTUM_ENTRY_USE_OPEN       = _as_bool("MOMENTUM_ENTRY_USE_OPEN", False)

# ---------- hot-reload для .env/.envrc ----------
def hot_reload_env(*, override=True) -> dict:
    """
    Перечитывает .env/.envrc (как в начале), обновляет os.environ.
    Возвращает словарь изменённых/добавленных переменных {name: (old, new)}.
    """
    before = dict(os.environ)
    _load_env_files(override=override)
    # (опционально) перезаполним per-leg слippage, если они всё ещё пусты
    os.environ.setdefault("ENTRY_SLIPPAGE_PCT", os.getenv("SLIPPAGE_PCT", "0.004"))
    os.environ.setdefault("EXIT_SLIPPAGE_PCT",  os.getenv("SLIPPAGE_PCT", "0.004"))
    os.environ.setdefault("STOP_SLIPPAGE_PCT",  os.getenv("SLIPPAGE_PCT", "0.004"))
    after = os.environ
    diff = {}
    for k, v_after in after.items():
        v_before = before.get(k)
        if v_before != v_after:
            diff[k] = (v_before, v_after)
    return diff

ENTRY_DETECT_TOL_SEC = 30   # сколько секунд после закрытия бара даём окну на исполнение