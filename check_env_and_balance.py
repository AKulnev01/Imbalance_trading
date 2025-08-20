# check_env_and_balance.py
from config import BYBIT_TESTNET, EXECUTION_ENV
from utils.bybit_trade import get_wallet_balance, _client

print("=== ENVIRONMENT CHECK ===")
print(f"BYBIT_TESTNET   = {BYBIT_TESTNET}")
print(f"EXECUTION_ENV   = {EXECUTION_ENV}")
print(f"API Key (head)  = {_client.api_key[:4]}… len={len(_client.api_key)}")
print(f"API Secret len  = {len(_client.api_secret)}")

try:
    bal = get_wallet_balance("USDT")
    print(f"USDT Balance    = {bal:.2f}")
except Exception as e:
    print(f"[ERR] Can't get balance: {e}")

print("=========================")
if BYBIT_TESTNET and bal >= 4000:
    print("✅ Testnet ON, баланс есть — можно запускать бота!")
else:
    print("⚠️ Проверь BYBIT_TESTNET или баланс.")