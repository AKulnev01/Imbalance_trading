# runtime/optimize_theta.py
import random, joblib, numpy as np, pandas as pd
from pathlib import Path
from runtime.evaluate_theta import evaluate_theta
from models.neuro_optimizer import CONT, BIN

def suggest(pipe_dict, K=20):
    pipe = pipe_dict["pipe"]
    cols = pipe_dict["cols"]
    cand=[]
    for _ in range(2000):
        th = {
            "MOMENTUM_TP_PCT": random.uniform(0.005,0.08),
            "MOMENTUM_SL_PCT": random.uniform(0.002,0.04),
            "ENTRY_SLIPPAGE_PCT": random.uniform(0.0,0.01),
            "EXIT_SLIPPAGE_PCT":  random.uniform(0.0,0.005),
            "STOP_SLIPPAGE_PCT":  random.uniform(0.0,0.01),
            "ENABLE_EARLY_CHECK": random.choice([0,1]),
            "EARLY_CHECK_MIN_BEFORE_CLOSE": random.randint(1,30),
            "EARLY_MOVE_PCT": random.uniform(0.0,0.01),
            "EARLY_VOL_MULT": random.uniform(0.0,3.0),
            "EARLY_REQUIRE_BOTH": random.choice([0,1]),
            "MAX_CONCURRENT_POSITIONS": random.randint(5,40),
            "DEFAULT_TTL_DAYS": random.randint(1,5),
        }
        x = np.array([[th[c] for c in cols]], dtype=float)
        pred = pipe.predict(x)[0]
        cand.append((pred, th))
    cand.sort(key=lambda x: x[0], reverse=True)
    return [th for _, th in cand[:K]]

def main():
    signals="./data/signals/signals.parquet"
    split = {"lookback_days":180, "interval":"4h", "ttl_days":3,
             "capital_aware": True, "initial_capital": 10000, "only_filled": False, "dedup": True}
    ds_path="./models/data/params_kpi.parquet"
    model_path="./models/neuro_opt.joblib"

    df = pd.read_parquet(ds_path)
    pipe = joblib.load(model_path)
    best_real = df.sort_values("kpi", ascending=False).head(1).iloc[0].to_dict()
    print(f"seed best kpi={best_real['kpi']:.2f}")

    for it in range(1,11):
        batch = suggest(pipe, K=8)
        rows=[]
        for i, theta in enumerate(batch, 1):
            res = evaluate_theta(signals, f"./models/data/active_{it}_{i}.xlsx", theta, split, kpi_key="pnl_usd")
            kpi = res["kpi"] - 0.0*max(0, 50-res["trades"]) - 0.0*max(0.0, res["max_dd_pct"]-30.0)
            rows.append({**theta, **{"kpi":kpi, "trades":res["trades"], "dd":res["max_dd_pct"]}})
            if kpi > best_real["kpi"]:
                best_real = rows[-1]
                print(f"[it{it}] NEW BEST kpi={kpi:.2f} θ={theta}")
        if rows:
            df_new = pd.DataFrame(rows)
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_parquet(ds_path)
            # дообучение
            from models.neuro_optimizer import train
            train(ds_path, model_out=model_path)
            pipe = joblib.load(model_path)

    Path("./models").mkdir(exist_ok=True)
    pd.Series(best_real).to_json("./models/theta_best.json")
    print("best saved → ./models/theta_best.json")

if __name__ == "__main__":
    main()