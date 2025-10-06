import os, pandas as pd
class M1Store:
    def __init__(self, parquet_dir: str):
        self.dir = parquet_dir
        os.makedirs(self.dir, exist_ok=True)
    def _path(self, symbol: str) -> str:
        return os.path.join(self.dir, f"{symbol}_m1.parquet")
    def append_klines(self, symbol: str, rows: list):
        df_new = pd.DataFrame(rows)
        if df_new.empty: return
        df_new = df_new.drop_duplicates(subset=["ts"]).sort_values("ts")
        path = self._path(symbol)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = pd.concat([df, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        else:
            df = df_new.reset_index(drop=True)
        df.to_parquet(path, index=False)
    def load(self, symbol: str, ts_from_ms: int, ts_to_ms: int) -> pd.DataFrame:
        path = self._path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        df = pd.read_parquet(path)
        return df[(df["ts"]>=ts_from_ms) & (df["ts"]<=ts_to_ms)].reset_index(drop=True)
