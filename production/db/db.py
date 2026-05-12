from sqlalchemy import create_engine
import pandas as pd

ENGINE = create_engine(
    "postgresql+psycopg2://art:1234@localhost:5432/imb_traid"
)

def load_last_n_bars(symbol: str, n: int) -> pd.DataFrame:
    query = """
        SELECT entry_ts, open, high, low, close, volume
        FROM candles_4h
        WHERE symbol = %s
        ORDER BY entry_ts DESC
        LIMIT %s
    """

    df = pd.read_sql(query, ENGINE, params=(symbol, n))

    df = df.sort_values("entry_ts").reset_index(drop=True)

    return df