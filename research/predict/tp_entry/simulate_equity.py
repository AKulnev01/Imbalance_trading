from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# попытка использовать нативный загрузчик минуток, если он есть
try:
    from predict.tp_entry.data_utils import load_m1 as _load_m1_native
except Exception:
    _load_m1_native = None


# ========= метрики =========
def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def sharpe_per_trade(returns: pd.Series, eps=1e-12) -> float:
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    return float(mu / (sd + eps))


# ========= I/O & время =========
def _read_parquet_safe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["__src"] = str(path)
    return df


def _to_naive_utc_from_any(x) -> pd.Series:
    """Автодетект эпохи (ms/s/ns) и приведение к naive UTC."""
    s = pd.Series(x)
    if pd.api.types.is_numeric_dtype(s):
        vals = s.astype("float64")
        vmax = float(np.nanmax(vals.values)) if len(vals) else 0.0
        if vmax > 1e14:
            out = pd.to_datetime(vals, unit="ns", utc=True, errors="coerce")
        elif vmax > 1e11:
            out = pd.to_datetime(vals, unit="ms", utc=True, errors="coerce")
        else:
            out = pd.to_datetime(vals, unit="s", utc=True, errors="coerce")
    else:
        out = pd.to_datetime(s, errors="coerce", utc=True)
    return out.dt.tz_localize(None)


def _ensure_datetime_naive_utc(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = _to_naive_utc_from_any(df[c])


def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ========= best_ks.csv =========
def load_best_ks_csv(path: str) -> pd.DataFrame:
    ks = pd.read_csv(path)
    need = {"symbol", "side", "ttl_hours", "k_tp_abs", "k_sl_abs"}
    miss = need - set(ks.columns)
    if miss:
        raise ValueError(f"best_ks.csv missing columns: {miss}")
    ks["symbol"] = ks["symbol"].astype(str)
    ks["side"] = ks["side"].astype(str).str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
    return ks


def attach_best_ks(picks: pd.DataFrame, best_ks_csv: str) -> pd.DataFrame:
    ks = load_best_ks_csv(best_ks_csv)
    out = picks.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["side"] = out["side"].astype(str).str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
    out = out.merge(
        ks[["symbol", "side", "ttl_hours", "k_tp_abs", "k_sl_abs"]],
        on=["symbol", "side"],
        how="left",
    )
    if out["k_tp_abs"].isna().any() or out["k_sl_abs"].isna().any() or out["ttl_hours"].isna().any():
        bad = out[out["k_tp_abs"].isna() | out["k_sl_abs"].isna() | out["ttl_hours"].isna()][["symbol", "side"]].drop_duplicates()
        raise ValueError(f"no baseline KS for some symbol/side:\n{bad.to_string(index=False)}")

    # нормальные имена в стиле датасета
    out = out.rename(columns={"ttl_hours": "ks_ttl_hours", "k_tp_abs": "ks_tp_abs", "k_sl_abs": "ks_sl_abs"})
    return out


# ========= минутки =========
def _load_m1(symbol: str, m1_dir: str) -> pd.DataFrame:
    """Минутки с DatetimeIndex (naive UTC) и колонками open/high/low/close."""
    if _load_m1_native is not None:
        m1 = _load_m1_native(symbol, m1_dir).copy()
        if isinstance(m1.index, pd.DatetimeIndex):
            m1.index = pd.to_datetime(m1.index, utc=True, errors="coerce").tz_localize(None)
        else:
            for col in ("ts", "time", "datetime", "bar_ts"):
                if col in m1.columns:
                    m1.index = _to_naive_utc_from_any(m1[col])
                    break
    else:
        path = Path(m1_dir) / f"{symbol}_m1.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Не найден {path}")
        m1 = pd.read_parquet(path)
        if "ts" in m1.columns:
            m1.index = _to_naive_utc_from_any(m1["ts"])
        elif "time" in m1.columns:
            m1.index = _to_naive_utc_from_any(m1["time"])
        elif "datetime" in m1.columns:
            m1.index = _to_naive_utc_from_any(m1["datetime"])
        elif "bar_ts" in m1.columns:
            m1.index = _to_naive_utc_from_any(m1["bar_ts"])
        elif isinstance(m1.index, pd.DatetimeIndex):
            m1.index = pd.to_datetime(m1.index, utc=True, errors="coerce").tz_localize(None)
        else:
            raise ValueError("В минутках нет DatetimeIndex и нет ts/time/datetime/bar_ts.")

    m1 = m1.sort_index()
    m1 = m1[~m1.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close"):
        if c not in m1.columns:
            raise ValueError(f"В минутках нет колонки '{c}'.")
    return m1


def _slice_m1(m1: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # inclusive
    w = m1.loc[start:end]
    return w


def _calc_tp_sl(entry_px: float, side: str, k_tp_abs: float, k_sl_abs: float) -> tuple[float, float]:
    side = str(side).upper()
    if side == "SELL":
        tp = entry_px * (1.0 - float(k_tp_abs))
        sl = entry_px * (1.0 + float(k_sl_abs))
    else:
        tp = entry_px * (1.0 + float(k_tp_abs))
        sl = entry_px * (1.0 - float(k_sl_abs))
    return float(tp), float(sl)


def _first_hit_tp_sl_ttl(m1: pd.DataFrame,
                         entry_ts: pd.Timestamp,
                         ttl_ts: pd.Timestamp,
                         side: str,
                         tp_px: float,
                         sl_px: float) -> tuple[pd.Timestamp, float, str]:
    """
    Ищем первое срабатывание TP/SL по минуткам (high/low), иначе TTL по close на ttl_ts (или последнему <= ttl_ts).
    """
    side = str(side).upper()
    w = _slice_m1(m1, entry_ts, ttl_ts)
    if w.empty:
        # нет минуток - TTL без цены
        return ttl_ts, np.nan, "TTL"

    hi = w["high"].values
    lo = w["low"].values
    idx = w.index.values

    if side == "SELL":
        hit_tp = np.where(lo <= tp_px)[0]
        hit_sl = np.where(hi >= sl_px)[0]
    else:
        hit_tp = np.where(hi >= tp_px)[0]
        hit_sl = np.where(lo <= sl_px)[0]

    tp_i = hit_tp[0] if len(hit_tp) else None
    sl_i = hit_sl[0] if len(hit_sl) else None

    if tp_i is None and sl_i is None:
        # TTL: close последней минуты
        exit_ts = w.index[-1]
        exit_px = float(w.iloc[-1]["close"])
        return exit_ts, exit_px, "TTL"

    if tp_i is not None and (sl_i is None or tp_i <= sl_i):
        return pd.Timestamp(idx[tp_i]), float(tp_px), "TP"
    return pd.Timestamp(idx[sl_i]), float(sl_px), "SL"


# ========= издержки =========
def _compute_costs(entry_px: float,
                   fee_pct: float,
                   slip_entry_pct: float,
                   slip_exit_pct: float,
                   extra_fee_pct: float) -> dict:
    """
    В долях от notional:
      fee_in = fee_pct
      fee_out = fee_pct
      slip_in = slip_entry_pct
      slip_out = slip_exit_pct
      extra_fee = extra_fee_pct
    """
    return {
        "fee_in": float(fee_pct),
        "fee_out": float(fee_pct),
        "slip_in": float(slip_entry_pct),
        "slip_out": float(slip_exit_pct),
        "extra_fee": float(extra_fee_pct),
        "cost_total": float(2.0 * fee_pct + slip_entry_pct + slip_exit_pct + extra_fee_pct),
    }


def _gross_return(entry_px: float, exit_px: float, side: str) -> float:
    side = str(side).upper()
    if side == "SELL":
        return float((entry_px - exit_px) / entry_px)
    return float((exit_px - entry_px) / entry_px)


def _net_return(entry_px: float, exit_px: float, side: str,
                fee_pct: float,
                slip_entry_pct: float,
                slip_exit_pct: float,
                extra_fee_pct: float) -> tuple[float, float, dict]:
    """
    Возвращает (ret_gross, ret_net, costs_dict) в долях.
    """
    g = _gross_return(entry_px, exit_px, side)
    costs = _compute_costs(entry_px, fee_pct, slip_entry_pct, slip_exit_pct, extra_fee_pct)
    net = g - costs["cost_total"]
    return float(g), float(net), costs


# ========= picks loader =========
def load_picks(pick_paths, feat_paths=None) -> pd.DataFrame:
    picks = pd.concat([_read_parquet_safe(p) for p in pick_paths], ignore_index=True)

    # normalize
    if "side" in picks.columns:
        picks["side"] = picks["side"].astype(str).str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
    else:
        picks["side"] = np.where(picks["__src"].str.contains("SELL", case=False), "SELL", "BUY")

    if "symbol" not in picks.columns:
        raise ValueError("В picks нет 'symbol'")

    if "entry_ts" not in picks.columns:
        raise ValueError("В picks нет 'entry_ts'")

    _ensure_datetime_naive_utc(picks, ["entry_ts"])

    # optional: stitch from feats if asked (не ломаем старую совместимость)
    if feat_paths:
        feats = pd.concat([_read_parquet_safe(p) for p in feat_paths], ignore_index=True)
        if "entry_ts" not in feats.columns:
            if "bar_ts" in feats.columns:
                feats = feats.rename(columns={"bar_ts": "entry_ts"})
            elif isinstance(feats.index, pd.DatetimeIndex):
                feats = feats.reset_index().rename(columns={"index": "entry_ts"})
            else:
                raise ValueError("feats не содержат entry_ts")

        _ensure_datetime_naive_utc(feats, ["entry_ts", "exit_ts"])
        for c in ("symbol", "side"):
            if c in feats.columns:
                feats[c] = feats[c].astype(str).str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})

        # merge by entry_ts,symbol,side first
        keys_list = [["entry_ts", "symbol", "side"], ["entry_ts", "symbol"], ["entry_ts"]]
        merged = None
        for keys in keys_list:
            if all(k in picks.columns for k in keys) and all(k in feats.columns for k in keys):
                right = feats.sort_values(keys + (["exit_ts"] if "exit_ts" in feats.columns else [])).drop_duplicates(keys, keep="last")
                tmp = picks.merge(right, on=keys, how="left", suffixes=("", "_fe"))
                hit = int(tmp.filter(like="_fe").notna().any(axis=1).sum()) if any(c.endswith("_fe") for c in tmp.columns) else 0
                if hit / max(1, len(tmp)) >= 0.5 or keys == ["entry_ts"]:
                    merged = tmp
                    print(f"[INFO] feats merge by {keys} → hits={hit}/{len(tmp)}")
                    break
        if merged is not None:
            picks = merged
            # pour *_fe
            for c in list(picks.columns):
                if c.endswith("_fe"):
                    base = c[:-3]
                    if base in picks.columns:
                        picks[base] = picks[base].where(picks[base].notna(), picks[c])
                    else:
                        picks = picks.rename(columns={c: base})
            picks = picks.drop(columns=[c for c in picks.columns if c.endswith("_fe")], errors="ignore")

    return picks


# ========= overlap policy =========
def pick_resolve_policy(df_at_same_time: pd.DataFrame, policy: str) -> int:
    if policy == "maxp" and "p" in df_at_same_time.columns:
        return df_at_same_time["p"].astype(float).idxmax()
    if policy == "maxpmid" and "p" in df_at_same_time.columns:
        return (df_at_same_time["p"].astype(float) - 0.5).abs().idxmax()
    if policy == "random":
        return df_at_same_time.sample(1, random_state=42).index[0]
    return df_at_same_time.index[0]


# ========= симуляторы =========
def simulate_raw(df: pd.DataFrame, start_cap: float) -> tuple[pd.DataFrame, pd.Series]:
    if not {"entry_ts", "pnl_net"}.issubset(df.columns):
        raise ValueError("Для raw нужны entry_ts и pnl_net")
    out = df.sort_values("entry_ts").copy()
    out["cum_cap"] = float(start_cap) + out["pnl_net"].cumsum()
    eq = out.set_index("entry_ts")["cum_cap"]
    return out, eq


def simulate_cap(df: pd.DataFrame,
                 start_cap: float,
                 slots: int,
                 min_gap_minutes: int,
                 overlap_policy: str,
                 max_per_ts: int) -> tuple[pd.DataFrame, pd.Series]:
    need = {"entry_ts", "exit_ts", "ret_adj"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[cap] missing columns: {miss}")

    df = df.sort_values(["entry_ts", "exit_ts", "side"]).reset_index(drop=True)

    if max_per_ts > 0:
        df = df.groupby("entry_ts", group_keys=False).head(max_per_ts).reset_index(drop=True)

    if max_per_ts == 1:
        keep_idx = []
        for _, g in df.groupby("entry_ts", sort=False):
            if len(g) == 1:
                keep_idx.append(g.index[0])
            else:
                keep_idx.append(pick_resolve_policy(g, overlap_policy))
        df = df.loc[sorted(set(keep_idx))].copy().reset_index(drop=True)

    risk_cap = float(start_cap)
    realized_pnl = 0.0
    gap = pd.Timedelta(minutes=int(min_gap_minutes))
    per_trade_frac = 1.0 / max(1, int(slots))
    last_exit = pd.Timestamp.min

    active: list[tuple[pd.Timestamp, float, float, dict]] = []  # (exit_ts, alloc, ret_adj, meta)
    trades_out = []
    eq_points = []

    for _, r in df.iterrows():
        et, xt = r["entry_ts"], r["exit_ts"]
        if pd.isna(xt):
            continue

        # close matured before next entry
        if active:
            still = []
            for (t_exit, alloc, ret_adj, meta) in active:
                if et >= t_exit:
                    pnl_usd = alloc * ret_adj
                    risk_cap += (alloc + pnl_usd)
                    realized_pnl += pnl_usd
                    eq_points.append((t_exit, start_cap + realized_pnl))
                else:
                    still.append((t_exit, alloc, ret_adj, meta))
            active = still

        if len(active) >= slots:
            continue
        if et < (last_exit + gap):
            continue
        if risk_cap <= 0:
            continue

        alloc = risk_cap * per_trade_frac
        risk_cap -= alloc
        if risk_cap < 0:
            risk_cap = 0.0

        meta = {
            "symbol": r.get("symbol", None),
            "side": r.get("side", None),
            "p": float(r.get("p", np.nan)),
            "entry_px": float(r.get("entry_px", np.nan)),
            "exit_px": float(r.get("exit_px", np.nan)),
            "exit_reason": r.get("exit_reason", None),
            "ret_gross": float(r.get("ret_gross", np.nan)),
            "cost_total": float(r.get("cost_total", np.nan)),
        }

        active.append((xt, alloc, float(r["ret_adj"]), meta))
        trades_out.append({
            "entry_ts": et,
            "exit_ts": xt,
            "symbol": meta["symbol"],
            "side": meta["side"],
            "p": meta["p"],
            "alloc_usd": float(alloc),
            "ret_adj": float(r["ret_adj"]),
            "ret_gross": meta["ret_gross"],
            "cost_total": meta["cost_total"],
            "entry_px": meta["entry_px"],
            "exit_px": meta["exit_px"],
            "exit_reason": meta["exit_reason"],
        })

        last_exit = max(last_exit, xt)

    # close tail
    for (t_exit, alloc, ret_adj, meta) in sorted(active, key=lambda x: x[0]):
        pnl_usd = alloc * ret_adj
        risk_cap += (alloc + pnl_usd)
        realized_pnl += pnl_usd
        eq_points.append((t_exit, start_cap + realized_pnl))

    exec_df = pd.DataFrame(trades_out).sort_values("exit_ts").reset_index(drop=True)
    eq = pd.Series([v for (_, v) in sorted(eq_points, key=lambda x: x[0])],
                   index=[t for (t, _) in sorted(eq_points, key=lambda x: x[0])],
                   name="equity")
    return exec_df, eq


def simulate_bank(df: pd.DataFrame,
                  start_cap: float,
                  slots: int,
                  min_gap_minutes: int,
                  overlap_policy: str,
                  max_per_ts: int) -> tuple[pd.DataFrame, pd.Series]:
    need = {"entry_ts", "exit_ts", "ret_adj"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[bank] missing columns: {miss}")

    df = df.sort_values(["entry_ts", "exit_ts", "side"]).reset_index(drop=True)

    if max_per_ts > 0:
        df = df.groupby("entry_ts", group_keys=False).head(max_per_ts).reset_index(drop=True)

    if max_per_ts == 1:
        keep_idx = []
        for _, g in df.groupby("entry_ts", sort=False):
            if len(g) == 1:
                keep_idx.append(g.index[0])
            else:
                keep_idx.append(pick_resolve_policy(g, overlap_policy))
        df = df.loc[sorted(set(keep_idx))].copy().reset_index(drop=True)

    risk_cap = float(start_cap)
    bank_cap = 0.0
    gap = pd.Timedelta(minutes=int(min_gap_minutes))
    per_trade_frac = 1.0 / max(1, int(slots))
    last_exit = pd.Timestamp.min

    active: list[tuple[pd.Timestamp, float, float]] = []  # (exit_ts, alloc, ret_adj)
    trades_out = []
    eq_points = []

    for _, r in df.iterrows():
        et, xt = r["entry_ts"], r["exit_ts"]
        if pd.isna(xt):
            continue

        # close matured before next entry
        if active:
            still = []
            for (t_exit, alloc, ret_adj) in active:
                if et >= t_exit:
                    pnl_usd = alloc * ret_adj
                    if pnl_usd >= 0:
                        bank_cap += pnl_usd
                        risk_cap += alloc
                    else:
                        risk_cap += (alloc + pnl_usd)
                        if risk_cap < 0:
                            risk_cap = 0.0
                    eq_points.append((t_exit, bank_cap + risk_cap))
                else:
                    still.append((t_exit, alloc, ret_adj))
            active = still

        if len(active) >= slots:
            continue
        if et < (last_exit + gap):
            continue
        if risk_cap <= 0:
            continue

        alloc = risk_cap * per_trade_frac
        risk_cap -= alloc
        if risk_cap < 0:
            risk_cap = 0.0

        active.append((xt, alloc, float(r["ret_adj"])))
        trades_out.append({
            "entry_ts": et,
            "exit_ts": xt,
            "symbol": r.get("symbol", None),
            "side": r.get("side", None),
            "p": float(r.get("p", np.nan)),
            "alloc_usd": float(alloc),
            "ret_adj": float(r["ret_adj"]),
            "ret_gross": float(r.get("ret_gross", np.nan)),
            "cost_total": float(r.get("cost_total", np.nan)),
            "entry_px": float(r.get("entry_px", np.nan)),
            "exit_px": float(r.get("exit_px", np.nan)),
            "exit_reason": r.get("exit_reason", None),
            "bank_cap": float(bank_cap),
        })

        last_exit = max(last_exit, xt)

    # close tail
    for (t_exit, alloc, ret_adj) in sorted(active, key=lambda x: x[0]):
        pnl_usd = alloc * ret_adj
        if pnl_usd >= 0:
            bank_cap += pnl_usd
            risk_cap += alloc
        else:
            risk_cap += (alloc + pnl_usd)
            if risk_cap < 0:
                risk_cap = 0.0
        eq_points.append((t_exit, bank_cap + risk_cap))

    exec_df = pd.DataFrame(trades_out).sort_values("exit_ts").reset_index(drop=True)
    eq = pd.Series([v for (_, v) in sorted(eq_points, key=lambda x: x[0])],
                   index=[t for (t, _) in sorted(eq_points, key=lambda x: x[0])],
                   name="equity")
    return exec_df, eq


# ========= m1 outcomes builder =========
def build_outcomes_m1(picks: pd.DataFrame,
                      m1_dir: str,
                      fee_pct: float,
                      slip_entry_pct: float,
                      slip_exit_pct: float,
                      extra_fee_pct: float) -> pd.DataFrame:
    """
    Требует:
      entry_ts, side, symbol, ref_close, ks_tp_abs, ks_sl_abs, ks_ttl_hours
    Возвращает таблицу с:
      entry_px, exit_px, exit_ts, exit_reason, ret_gross, ret_adj, cost_total
    """
    need = {"entry_ts", "side", "symbol", "ref_close", "ks_tp_abs", "ks_sl_abs", "ks_ttl_hours"}
    miss = need - set(picks.columns)
    if miss:
        raise ValueError(f"Для m1-исходов не хватает: {miss}")

    out_rows = []
    cache = {}

    for _, r in picks.sort_values("entry_ts").iterrows():
        sym = str(r["symbol"])
        if sym not in cache:
            cache[sym] = _load_m1(sym, m1_dir)

        m1 = cache[sym]
        entry_ts = pd.Timestamp(r["entry_ts"])
        side = str(r.get("side", "BUY")).upper()

        if side in ("LONG",):

            side = "BUY"

        elif side in ("SHORT",):

            side = "SELL"

        elif side not in ("BUY","SELL"):

            side = "BUY"

        entry_px = float(r["ref_close"])

        ttl_h = float(r["ks_ttl_hours"])
        ttl_ts = entry_ts + pd.to_timedelta(ttl_h, unit="h")

        tp_px, sl_px = _calc_tp_sl(entry_px, side, float(r["ks_tp_abs"]), float(r["ks_sl_abs"]))
        exit_ts, exit_px, reason = _first_hit_tp_sl_ttl(m1, entry_ts, ttl_ts, side, tp_px, sl_px)

        ret_gross, ret_net, costs = _net_return(
            entry_px, float(exit_px), side,
            fee_pct=float(fee_pct),
            slip_entry_pct=float(slip_entry_pct),
            slip_exit_pct=float(slip_exit_pct),
            extra_fee_pct=float(extra_fee_pct),
        )

        out_rows.append({
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "symbol": sym,
            "side": side,
            "p": float(r.get("p", np.nan)),
            "entry_px": entry_px,
            "exit_px": float(exit_px),
            "exit_reason": reason,
            "ret_gross": float(ret_gross),
            "ret_adj": float(ret_net),
            "cost_total": float(costs["cost_total"]),
        })

    return pd.DataFrame(out_rows).sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)


# ========= CLI =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", nargs="+", required=True, help="parquet со сделками (BUY/SELL)")
    ap.add_argument("--feats", nargs="*", default=[], help="опц.: *_feats.parquet для досшивки (legacy)")

    ap.add_argument("--outdir", required=True, help="куда сохранять")
    ap.add_argument("--pnl-basis", choices=["raw", "cap", "bank"], default="cap",
                    help="raw=cumsum(pnl_net), cap=компаунд, bank=чулан")
    ap.add_argument("--pnl-source", choices=["pnl_net", "m1"], default="m1",
                    help="pnl_net=использовать pnl_net из picks/feats, m1=считать исходы по минуткам (TP/SL/TTL)")

    ap.add_argument("--start-cap", type=float, default=1000.0)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--max-per-timestamp", type=int, default=1, help="макс. сделок на один entry_ts (0=без огр.)")
    ap.add_argument("--min-gap-minutes", type=int, default=0)
    ap.add_argument("--overlap-policy", choices=["first", "maxp", "maxpmid", "random"], default="maxp")

    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)

    # costs & m1
    ap.add_argument("--m1-dir", type=str, default="./data/m1")
    ap.add_argument("--fee-pct", type=float, default=0.001)
    ap.add_argument("--slip-entry-pct", type=float, default=0.0)
    ap.add_argument("--slip-exit-pct", type=float, default=0.004)
    ap.add_argument("--extra-fee-pct", type=float, default=0.0)

    ap.add_argument("--best-ks-csv", type=str, default=None,
                    help="reports/features/best_ks.csv: symbol,side,ttl_hours,k_tp_abs,k_sl_abs")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    picks = load_picks(args.picks, feat_paths=args.feats)
    _ensure_datetime_naive_utc(picks, ["entry_ts", "exit_ts"])
    if args.start:
        picks = picks[picks["entry_ts"] >= pd.Timestamp(args.start)]
    if args.end:
        picks = picks[picks["entry_ts"] <= pd.Timestamp(args.end)]

    # best ks from csv
    if args.best_ks_csv:
        picks = attach_best_ks(picks, args.best_ks_csv)

    # pnl source
    if args.pnl_source == "m1":
        # required: ref_close + ks_*
        if "ref_close" not in picks.columns:
            raise ValueError("Для pnl-source=m1 в picks нужен ref_close (entry_px).")
        picks = build_outcomes_m1(
            picks,
            m1_dir=args.m1_dir,
            fee_pct=args.fee_pct,
            slip_entry_pct=args.slip_entry_pct,
            slip_exit_pct=args.slip_exit_pct,
            extra_fee_pct=args.extra_fee_pct,
        )
    else:
        # legacy pnl_net path
        if "pnl_net" not in picks.columns:
            raise ValueError("pnl-source=pnl_net требует pnl_net в picks/feats.")
        # pnl_net должен быть в долях (если у тебя проценты - конверти заранее)
        picks["ret_adj"] = pd.to_numeric(picks["pnl_net"], errors="coerce").fillna(0.0)
        if "exit_ts" not in picks.columns:
            raise ValueError("pnl-source=pnl_net требует exit_ts (или передай feats).")

    if picks.empty:
        print("[WARN] No picks after filters.")
        return

    if args.pnl_basis == "raw":
        exec_df, eq = simulate_raw(picks, start_cap=float(args.start_cap))
        total_cap = float(eq.iloc[-1]) if len(eq) else float(args.start_cap)
        print(f"[RAW] rows={len(exec_df)} final_cap={total_cap:.6f}")
    elif args.pnl_basis == "cap":
        exec_df, eq = simulate_cap(
            picks, start_cap=float(args.start_cap), slots=int(args.slots),
            min_gap_minutes=int(args.min_gap_minutes),
            overlap_policy=args.overlap_policy,
            max_per_ts=int(args.max_per_timestamp),
        )
    else:
        exec_df, eq = simulate_bank(
            picks, start_cap=float(args.start_cap), slots=int(args.slots),
            min_gap_minutes=int(args.min_gap_minutes),
            overlap_policy=args.overlap_policy,
            max_per_ts=int(args.max_per_timestamp),
        )

    if exec_df.empty:
        print("[WARN] Ни одной сделки не исполнено (после слотов/гэпов).")
        (outdir / "equity_trades.parquet").write_bytes(b"")
        return

    total_cap = float(eq.iloc[-1]) if len(eq) else float(args.start_cap)
    total_ret = total_cap / float(args.start_cap) - 1.0
    mdd = max_drawdown(eq)
    sh = sharpe_per_trade(exec_df["ret_adj"])

    print(f"Сделок исполнено: {len(exec_df)}  |  mode={args.pnl_basis}  |  slots={args.slots}")
    print(f"Итоговый капитал: ${total_cap:,.2f}  (старт ${args.start_cap:,.2f})  TotalReturn={total_ret:.2%}")
    print(f"MaxDD: {mdd:.2%}  |  Sharpe/Trade: {sh:.2f}")

    # save
    trades_path = outdir / "equity_trades.parquet"
    curve_path = outdir / "equity_curve.parquet"
    exec_df.to_parquet(trades_path, index=False)
    eq.to_frame(name="equity").to_parquet(curve_path)

    print(f"[OK] trades -> {trades_path}")
    print(f"[OK] curve  -> {curve_path}")


if __name__ == "__main__":
    main()