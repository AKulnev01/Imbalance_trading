# tools/plot_candles.py
import os, math, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _read_ohlcv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file: {ext}")
    # try common column names
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time") or cols.get("datetime") or cols.get("date")
    o_col  = cols.get("open") or "open"
    h_col  = cols.get("high") or "high"
    l_col  = cols.get("low")  or "low"
    c_col  = cols.get("close") or "close"
    v_col  = cols.get("volume") or cols.get("vol") or None
    if ts_col is None:
        raise ValueError("No time column found (timestamp/time/datetime/date)")
    out = pd.DataFrame({
        "time": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
        "open": pd.to_numeric(df[o_col], errors="coerce"),
        "high": pd.to_numeric(df[h_col], errors="coerce"),
        "low":  pd.to_numeric(df[l_col], errors="coerce"),
        "close":pd.to_numeric(df[c_col], errors="coerce")
    })
    if v_col and v_col in df.columns:
        out["volume"] = pd.to_numeric(df[v_col], errors="coerce")
    out = out.dropna().sort_values("time").set_index("time")
    return out

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf.lower() in ("", "raw", "same"):
        return df
    rule = tf.lower()
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    out = pd.concat([o,h,l,c], axis=1).dropna()
    if "volume" in df.columns:
        out["volume"] = df["volume"].resample(rule).sum().fillna(0)
    return out

def _plot_candles(ax, x, o, h, l, c, width=0.6):
    up = c >= o
    dn = ~up
    # wicks
    ax.vlines(x, l, h, linewidth=1)
    # bodies up
    ax.bar(x[up], (c[up]-o[up]), bottom=o[up], width=width, align="center")
    # bodies down
    ax.bar(x[dn], (c[dn]-o[dn]), bottom=o[dn], width=width, align="center")
    ax.margins(x=0.01)

def render_pdf(df: pd.DataFrame, out_path: str, title: str, tz: str, per_page: int):
    if tz:
        df = df.tz_convert(tz)
    n = len(df)
    pages = max(1, math.ceil(n / per_page))
    with PdfPages(out_path) as pdf:
        for p in range(pages):
            lo = p*per_page
            hi = min((p+1)*per_page, n)
            chunk = df.iloc[lo:hi]
            if chunk.empty: continue
            fig, ax = plt.subplots(figsize=(12, 6))
            xs = np.arange(len(chunk))
            _plot_candles(ax, xs,
                          chunk["open"].to_numpy(),
                          chunk["high"].to_numpy(),
                          chunk["low"].to_numpy(),
                          chunk["close"].to_numpy(),
                          width=0.6)
            ax.set_title(f"{title}  |  {chunk.index[0]} → {chunk.index[-1]}  |  candles: {len(chunk)}")
            ax.set_xlabel("bar")
            ax.set_ylabel("price")
            ax.grid(True, linewidth=0.5, alpha=0.4)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Render OHLC candles to a multi-page PDF (no extra deps).")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV/Parquet with columns: time/open/high/low/close[/volume]")
    ap.add_argument("--out", dest="out", required=True, help="Output PDF path")
    ap.add_argument("--tf", default="raw", help="Resample rule (e.g. 1min, 5min, 15min, 1H, 4H, 1D). Use 'raw' to keep as-is.")
    ap.add_argument("--tz", default="UTC", help="Target timezone label (e.g. UTC, Europe/Moscow)")
    ap.add_argument("--per-page", type=int, default=600, help="Candles per page")
    ap.add_argument("--title", default=None, help="Custom title")
    args = ap.parse_args()

    df = _read_ohlcv(args.inp)
    df = _resample(df, args.tf)
    title = args.title or os.path.basename(args.inp)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    render_pdf(df, args.out, title, args.tz, int(args.per_page))
    print(f"✅ saved → {args.out}  | candles={len(df)}  pages≈{math.ceil(len(df)/int(args.per_page))}")

if __name__ == "__main__":
    main()