import pandas as pd
def merge(trades_csv: str, counterfacts_csv: str, out_parquet: str):
    tr = pd.read_csv(trades_csv)
    cf = pd.read_csv(counterfacts_csv)
    df = cf.merge(tr[["trade_id","symbol","detect_ts"]], on=["trade_id","symbol"], how="left")
    df.to_parquet(out_parquet, index=False)
if __name__=="__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--counterfacts-csv", required=True)
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()
    merge(args.trades_csv, args.counterfacts_csv, args.out_parquet)
