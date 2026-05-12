import os
import pandas as pd

# Базовый журнал (совместимость со старым пайплайном)
COLUMNS = [
    "trade_id","symbol","side","detect_ts","entry_ts","entry_fill","notional_usd",
    "rr_tp","rr_sl","ttl_hours","fees_bps","slip_bps","spread_bps"
]

def log_trade_open(path_csv: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    else:
        df = pd.DataFrame(columns=COLUMNS)
    # гарантируем все колонки
    out = {}
    for c in COLUMNS:
        out[c] = row.get(c, 0)
    df = pd.concat([df, pd.DataFrame([out])], ignore_index=True)
    df.to_csv(path_csv, index=False)

# Расширенный журнал диагностики лайва
ADV_COLUMNS = [
    "trade_id","symbol","side","category",
    "equity_before","free_cash_before","usd_alloc",
    "detect_px","fill_px","drift_pct",
    "tp_pre","sl_pre","tp_final","sl_final",
    "ttl_hours","fees_bps","slip_bps","ts_utc"
]

def log_trade_open_adv(path_csv: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    else:
        df = pd.DataFrame(columns=ADV_COLUMNS)
    out = {}
    for c in ADV_COLUMNS:
        out[c] = row.get(c, None)
    df = pd.concat([df, pd.DataFrame([out])], ignore_index=True)
    df.to_csv(path_csv, index=False)