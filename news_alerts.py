# news_alerts.py
import os, re, json, time, datetime as dt
from typing import List, Dict
import requests
import pandas as pd

# pip install feedparser openpyxl python-telegram-bot==13.*
import feedparser

from config import TELEGRAM_TOKEN, CHAT_ID, TRADE_UNIVERSE  # уже есть у тебя
#python news_alerts.py --mode alerts (срочные новости)
#python news_alerts.py --mode daily (формирование отчета инфо по юниверсу)
# -------- Настройки --------
# RSS-источники без API-ключей (можно расширять)
RSS_FEEDS = [
    "https://www.binance.com/en/support/announcement/rss",          # Binance announcements
    "https://announcements.bybit.com/rss/announcements_en.xml",     # Bybit announcements
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
    "https://www.kraken.com/learn/feed",                             # Kraken Learn/updates (иногда важное)
]

# Триггеры для срочных алертов
ALERT_PATTERNS = [
    r"\bdelist|\bdelisting|\bdelisted",
    r"suspend(ed)? trading|trading halt|halt(ed)?",
    r"exploit|hack(ed)?|breach|security incident",
    r"contract upgrade|token migration|smart contract change",
    r"chain halt|reorg|re-?org|consensus issue",
    r"bankrupt|insolvency",
]

# Куда класть отчёты
REPORTS_DIR = os.path.expanduser("~/Documents/отчеты/news")

# Память, чтобы не спамить дублями
STATE_PATH = os.path.join(REPORTS_DIR, "news_state.json")

# -------- Хэлперы --------
def _load_state() -> Dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH, "r"))
        except Exception:
            pass
    return {"seen_ids": []}

def _save_state(state: Dict):
    json.dump(state, open(STATE_PATH, "w"))

def _normalize_symbols(universe: List[str]) -> Dict[str, List[str]]:
    """
    Из тикера (ETHUSDT) строим список ключевых слов для матчинга в заголовках/описании.
    Добавляем human-name для некоторых популярных монет.
    """
    name_map = {
        "BTC": ["Bitcoin", "BTC"],
        "ETH": ["Ethereum", "ETH"],
        "SOL": ["Solana", "SOL"],
        "BNB": ["BNB", "BNB Chain"],
        "ADA": ["Cardano", "ADA"],
        "XRP": ["XRP", "Ripple"],
        "LTC": ["Litecoin", "LTC"],
        "AVAX": ["Avalanche", "AVAX"],
        "NEAR": ["NEAR Protocol", "NEAR"],
        "ATOM": ["Cosmos", "ATOM"],
        "LINK": ["Chainlink", "LINK"],
        "OP": ["Optimism", "OP"],
        "ARB": ["Arbitrum", "ARB"],
        "APT": ["Aptos", "APT"],
        "SUI": ["Sui", "SUI"],
        "INJ": ["Injective", "INJ"],
        "TIA": ["Celestia", "TIA"],
        "SEI": ["Sei", "SEI"],
        "RNDR": ["Render", "RNDR"],
        "WLD": ["Worldcoin", "WLD"],
        "UNI": ["Uniswap", "UNI"],
        "AAVE": ["Aave", "AAVE"],
        "LDO": ["Lido", "LDO"],
        "MATIC": ["Polygon", "MATIC"],
        # добавятся fallback-правила ниже
    }

    res: Dict[str, List[str]] = {}
    for sym in universe:
        base = sym.upper().replace("USDT", "").replace("USD", "")
        keys = name_map.get(base, [base])
        # fallback: если не знаем human-name, всё равно матчим по base (и по тикеру целиком)
        keys = list(dict.fromkeys(keys + [base, sym.upper()]))
        res[sym.upper()] = keys
    return res

def _fetch_all_feeds() -> List[dict]:
    out = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                # унифицируем
                out.append({
                    "feed": url,
                    "id": e.get("id") or e.get("link") or (e.get("title") + e.get("published", "")),
                    "title": e.get("title", ""),
                    "summary": re.sub("<.*?>", "", e.get("summary", "") or "", flags=re.S),
                    "link": e.get("link", ""),
                    "published": e.get("published", "") or e.get("updated", ""),
                })
        except Exception:
            continue
    return out

def _matches_any(text: str, keys: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keys)

def _filter_by_universe(items: List[dict], keymap: Dict[str, List[str]]) -> List[dict]:
    rows = []
    for it in items:
        blob = f"{it['title']} || {it['summary']}"
        matched_syms = [sym for sym, keys in keymap.items() if _matches_any(blob, keys)]
        if matched_syms:
            it2 = dict(it)
            it2["symbols"] = ",".join(sorted(set(matched_syms)))
            rows.append(it2)
    return rows

def _is_alert(item: dict) -> bool:
    text = f"{item.get('title','')} || {item.get('summary','')}"
    return any(re.search(p, text, flags=re.I) for p in ALERT_PATTERNS)

def _send_tg_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("TG not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("TG sendMessage error:", e)

def _send_tg_file(path: str, caption: str = ""):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("TG not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"document": f}, timeout=20)
    except Exception as e:
        print("TG sendDocument error:", e)

def _build_excel(rows: List[dict]) -> str:
    if not rows:
        # создадим пустой отчёт, чтобы в ТГ было видно «0 новостей»
        df = pd.DataFrame(columns=["published","symbols","title","summary","link","feed"])
    else:
        df = pd.DataFrame(rows)
        # колонок может не быть — приведём
        for c in ["published","symbols","title","summary","link","feed"]:
            if c not in df.columns: df[c] = ""

        # сортировка по дате, если есть
        def _ts(x):
            try:
                return pd.to_datetime(x)
            except Exception:
                return pd.NaT
        df["published_ts"] = df["published"].map(_ts)
        df = df.sort_values("published_ts", na_position="last", ascending=False).drop(columns=["published_ts"])

    os.makedirs(REPORTS_DIR, exist_ok=True)
    name = f"news_summary_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M')}.xlsx"
    path = os.path.join(REPORTS_DIR, name)
    df.to_excel(path, index=False)
    return path

# -------- Режимы --------
def run_daily():
    keymap = _normalize_symbols(list(dict.fromkeys(TRADE_UNIVERSE)) or [])
    items = _fetch_all_feeds()
    rows = _filter_by_universe(items, keymap)
    path = _build_excel(rows)
    _send_tg_file(path, caption=f"🗞️ Дайджест новостей по универсу ({len(rows)} записей)")

def run_watch():
    state = _load_state()
    seen = set(state.get("seen_ids") or [])
    keymap = _normalize_symbols(list(dict.fromkeys(TRADE_UNIVERSE)) or [])
    items = _fetch_all_feeds()
    rows = _filter_by_universe(items, keymap)

    alerts_sent = 0
    for it in rows:
        _id = it.get("id") or it.get("link")
        if _id in seen:
            continue
        if _is_alert(it):
            title = it.get("title", "")
            syms  = it.get("symbols", "")
            link  = it.get("link", "")
            _send_tg_message(f"⚠️ ALERT [{syms}]\n{title}\n{link}")
            alerts_sent += 1
        seen.add(_id)

    # обновим память
    state["seen_ids"] = list(seen)[-5000:]  # не раздувать файл
    _save_state(state)
    print(f"watch: alerts={alerts_sent}, scanned={len(rows)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily","watch"], required=True)
    args = ap.parse_args()

    if args.mode == "daily":
        run_daily()
    else:
        run_watch()