import os
import glob
import argparse
import pandas as pd

def resample_tf(df, tf):
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    ohlcv = df.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    ohlcv = ohlcv.dropna().reset_index()
    return ohlcv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--tf", dest="tf", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.in_dir, "*.parquet"))
    print(f"found {len(files)} files in {args.in_dir}")

    for f in files:
        try:
            df = pd.read_parquet(f)
            if "timestamp" not in df.columns and "ts" in df.columns:
                df = df.rename(columns={"ts": "timestamp"})
            out = resample_tf(df, args.tf)
            sym = os.path.basename(f).split("_")[0]
            out_path = os.path.join(args.out_dir, f"{sym}_{args.tf}.parquet")
            out.to_parquet(out_path, index=False)
            print(f"[OK] {sym} rows={len(out)}")
        except Exception as e:
            print(f"[ERR] {f}: {e}")

if __name__ == "__main__":
    main()