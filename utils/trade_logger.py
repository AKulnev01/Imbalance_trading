import os
import pandas as pd

COLUMNS = [
    "trade_id","symbol","side","detect_ts","entry_ts","entry_fill","notional_usd",
    "rr_tp","rr_sl","ttl_hours","fees_bps","slip_bps","spread_bps"
]

def log_trade_open(path_csv: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    else:
        df = pd.DataFrame(columns=COLUMNS)
    # гарантируем все колонки
    for c in COLUMNS:
        if c not in row:
            row[c] = 0
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path_csv, index=False)