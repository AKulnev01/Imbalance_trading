# real/diag_env.py
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# важно: именно импорт config формирует финальные значения (с .env/.envrc + setdefault)
from config import (
    BYBIT_CATEGORY, USE_MAINNET_MARKET_DATA, EXECUTION_ENV,
    MOMENTUM_TP_PCT, MOMENTUM_SL_PCT, DEFAULT_TTL_DAYS,
    ENABLE_BUY, ENABLE_SELL,
    FEE_TAKER, SLIPPAGE_PCT,
    INITIAL_CAPITAL, POSITION_FRACTION,
    DEFAULT_MIN_STRENGTH, FVG_VOL_MULT, FVG_TOLERANCE_PCT,
    MAX_CONCURRENT_POSITIONS,
    TELEGRAM_TOKEN, CHAT_ID,
    AUTOTRADE_STATE_PATH, ENV_RELOAD_SEC,
)

# значения, которые читаются напрямую через os.getenv/get_cfg в eval/boe
def envf(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None:
        return None
    try:
        return cast(v)
    except Exception:
        return v

ENTRY_SLIPPAGE_PCT = envf("ENTRY_SLIPPAGE_PCT", None, float)
EXIT_SLIPPAGE_PCT  = envf("EXIT_SLIPPAGE_PCT",  None, float)
STOP_SLIPPAGE_PCT  = envf("STOP_SLIPPAGE_PCT",  None, float)
MAX_ACCEPT_SLIPPAGE_PCT = envf("MAX_ACCEPT_SLIPPAGE_PCT", None, float)

OUT_TZ = envf("OUT_TZ", None, str)
REST_ATTEMPTS = envf("REST_ATTEMPTS", None, int)
REST_TIMEOUT_SEC = envf("REST_TIMEOUT_SEC", None, float)
FALLBACK_CONCURRENCY = envf("FALLBACK_CONCURRENCY", None, int)
REST_AFTER_CLOSE_DELAY_SEC = envf("REST_AFTER_CLOSE_DELAY_SEC", None, int)

TRADE_UNIVERSE = envf("TRADE_UNIVERSE", "", str)

print("=== CORE / EXCHANGE ===")
print("BYBIT_CATEGORY            =", BYBIT_CATEGORY)
print("USE_MAINNET_MARKET_DATA   =", USE_MAINNET_MARKET_DATA)
print("EXECUTION_ENV             =", EXECUTION_ENV)
print("OUT_TZ                    =", OUT_TZ)
print("ENV_RELOAD_SEC            =", ENV_RELOAD_SEC)

print("\n=== STRATEGY (MOMENTUM) ===")
print("MOMENTUM_TP_PCT           =", MOMENTUM_TP_PCT)
print("MOMENTUM_SL_PCT           =", MOMENTUM_SL_PCT)
print("DEFAULT_TTL_DAYS          =", DEFAULT_TTL_DAYS)
print("ENABLE_BUY / ENABLE_SELL  =", ENABLE_BUY, ENABLE_SELL)
print("MAX_CONCURRENT_POSITIONS  =", MAX_CONCURRENT_POSITIONS)

print("\n=== ALLOCATION / FEES / SLIPPAGE ===")
print("INITIAL_CAPITAL           =", INITIAL_CAPITAL)
print("POSITION_FRACTION         =", POSITION_FRACTION)
print("FEE_TAKER                 =", FEE_TAKER)
print("SLIPPAGE_PCT (generic)    =", SLIPPAGE_PCT)
print("ENTRY_SLIPPAGE_PCT        =", ENTRY_SLIPPAGE_PCT)
print("EXIT_SLIPPAGE_PCT         =", EXIT_SLIPPAGE_PCT)
print("STOP_SLIPPAGE_PCT         =", STOP_SLIPPAGE_PCT)
print("MAX_ACCEPT_SLIPPAGE_PCT   =", MAX_ACCEPT_SLIPPAGE_PCT)

print("\n=== FVG DETECTOR ===")
print("DEFAULT_MIN_STRENGTH      =", DEFAULT_MIN_STRENGTH)
print("FVG_VOL_MULT              =", FVG_VOL_MULT)
print("FVG_TOLERANCE_PCT         =", FVG_TOLERANCE_PCT)

print("\n=== NETWORK / REST FALLBACK ===")
print("REST_ATTEMPTS             =", REST_ATTEMPTS)
print("REST_TIMEOUT_SEC          =", REST_TIMEOUT_SEC)
print("FALLBACK_CONCURRENCY      =", FALLBACK_CONCURRENCY)
print("REST_AFTER_CLOSE_DELAY_SEC=", REST_AFTER_CLOSE_DELAY_SEC)

print("\n=== TELEGRAM ===")
print("TELEGRAM_TOKEN set?       =", bool(TELEGRAM_TOKEN))
print("CHAT_ID                   =", CHAT_ID)

print("\n=== UNIVERSE ===")
print("TRADE_UNIVERSE            =", TRADE_UNIVERSE[:160] + ("…" if TRADE_UNIVERSE and len(TRADE_UNIVERSE) > 160 else ""))

print("\n=== PATHS ===")
print("AUTOTRADE_STATE_PATH      =", AUTOTRADE_STATE_PATH)