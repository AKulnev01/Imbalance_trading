import os, json, yaml, pandas as pd
from data_feed.min_kline_store import M1Store
from evaluator.counterfactual import evaluate_counterfactuals
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    store = M1Store(cfg["marketdata"]["parquet_dir"])
    trades = pd.read_csv(args.trades_csv)
    rows = []
    for _, t in trades.iterrows():
        symbol = t["symbol"]
        entry_ts = int(t["entry_ts"])
        ttl_max_h = max(cfg["evaluation"]["ttl_hours_list"])
        m1 = store.load(symbol, entry_ts, entry_ts + int((ttl_max_h+1)*3600*1000))
        if m1.empty: continue
        m1["ret_abs_bp"] = (m1["close"].pct_change().abs().fillna(0.0) * 1e4)
        vol1m_bp = float(m1["ret_abs_bp"].rolling(20, min_periods=1).mean().iloc[0])
        notional = float(t.get("notional_usd", 1000.0))
        liq = max(cfg["slippage_model"]["liq_floor_usd"], cfg["slippage_model"]["liq_floor_usd"])
        slip_bps = (cfg["slippage_model"]["a_bps"]
                    + cfg["slippage_model"]["b_bps_per_vol"] * vol1m_bp
                    + cfg["slippage_model"]["c_bps_per_liq_ratio"] * (notional / liq))
        res = evaluate_counterfactuals(
            m1=m1, entry_ts_ms=entry_ts, side=t["side"].lower(), entry_price=float(t["entry_fill"]),
            rr_tp_list=cfg["evaluation"]["rr_tp_list"], rr_sl_list=cfg["evaluation"]["rr_sl_list"],
            ttl_hours_list=cfg["evaluation"]["ttl_hours_list"],
            fees_bps=float(t.get("fees_bps", cfg["costs"]["taker_fee_bps"])),
            half_spread_bps=float(t.get("spread_bps", cfg["costs"]["half_spread_bps_default"])),
            slip_bps=float(t.get("slip_bps", slip_bps)),
            intra_bar_policy=cfg["evaluation"]["intra_bar_policy"]
        )
        for r in res:
            r.update({
                "trade_id": t["trade_id"], "symbol": symbol, "side": t["side"], "entry_ts": int(entry_ts),
                "rr_tp_real": float(t.get("rr_tp", 0)), "rr_sl_real": float(t.get("rr_sl", 0)),
                "ttl_hours_real": float(t.get("ttl_hours", 0))
            })
            rows.append(r)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
