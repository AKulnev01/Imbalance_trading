# tools/eval_param_candidates.py
# Прогон предложенных наборов параметров и добавление результатов в датасет.

import os
import json
import argparse
from pathlib import Path
import subprocess
import uuid
import pandas as pd

def _env_from_dict(d: dict) -> dict:
    env = os.environ.copy()
    for k, v in d.items():
        env[k] = str(v)
    return env

def _run_eval(signals: str, out_xlsx: str, env: dict) -> bool:
    cmd = ["python", "models/evaluate_momentum.py", signals, "--out", out_xlsx]
    print("▶", " ".join(cmd))
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return Path(out_xlsx).exists()

def _parse_eval_xlsx(path: str) -> dict:
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return {}
    kpi = {}
    if "summary_by_variant" in xl.sheet_names:
        s = pd.read_excel(xl, "summary_by_variant")
        s.columns = [c.strip().lower() for c in s.columns]
        row = None
        if "variant" in s.columns:
            m = s[s["variant"] == "MOMENTUM"]
            row = m.iloc[0] if not m.empty else s.iloc[0]
        else:
            row = s.iloc[0]
        for c in ("trades","wins","winrate_pct","pnl_pct","pnl_usd"):
            if c in s.columns:
                kpi["kpi_" + c] = float(row[c])
    if "kpi_score" not in kpi:
        wr, pnl, dd = kpi.get("kpi_winrate_pct", 0.0), kpi.get("kpi_pnl_pct", 0.0), 0.0
        kpi["kpi_score"] = 0.6*(wr/100.0) + 0.4*(pnl/100.0) - 0.5*(dd/100.0)
    return kpi

def _append_parquet(df_row: dict, parquet_path: str):
    Path(Path(parquet_path).parent).mkdir(parents=True, exist_ok=True)
    if Path(parquet_path).exists():
        base = pd.read_parquet(parquet_path)
        out = pd.concat([base, pd.DataFrame([df_row])], ignore_index=True)
    else:
        out = pd.DataFrame([df_row])
    out.to_parquet(parquet_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Evaluate suggested parameter candidates and append to dataset.")
    ap.add_argument("--candidates", required=True, help="JSON файл со списком {param:value} наборов")
    ap.add_argument("--signals", required=True, help="Путь к файлу сигналов")
    ap.add_argument("--append-to", default="./models/data/params_kpi.parquet")
    ap.add_argument("--tmp-eval", default="./models/data/_fast_tmp.xlsx")
    args = ap.parse_args()

    with open(args.candidates, "r", encoding="utf-8") as f:
        cands = json.load(f)
    if not isinstance(cands, list):
        raise ValueError("candidates JSON должен быть списком объектов параметров")

    for i, p in enumerate(cands, 1):
        print(f"\n[{i}/{len(cands)}] run candidate: {p}")
        env = _env_from_dict(p)
        ok = _run_eval(args.signals, args.tmp_eval, env)
        if not ok:
            print("⚠️ eval failed, skip")
            continue
        kpi = _parse_eval_xlsx(args.tmp_eval)
        row = {**p, **kpi, "uuid": str(uuid.uuid4()), "as_of": pd.Timestamp.utcnow()}
        _append_parquet(row, args.append_to)

    print(f"\n✅ dataset updated: {args.append_to}")

if __name__ == "__main__":
    main()