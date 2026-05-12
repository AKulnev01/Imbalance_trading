# predict/tp_entry/add_features_full.py
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

# ===== базовые утилы =====
def _normalize_entry_ts(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    idx = x.index
    idx_names = list(getattr(idx, "names", []))
    has_entry_ts_in_index = False
    if isinstance(idx, pd.MultiIndex):
        has_entry_ts_in_index = "entry_ts" in idx_names
    else:
        has_entry_ts_in_index = (getattr(idx, "name", None) == "entry_ts")

    if has_entry_ts_in_index:
        if "entry_ts" in x.columns:
            x = x.drop(columns=["entry_ts"])
        if isinstance(idx, pd.MultiIndex):
            x = x.reset_index(level="entry_ts")
        else:
            x = x.reset_index()

    dup_like = [c for c in x.columns if str(c).startswith("entry_ts")]
    if len(dup_like) > 1:
        keep = "entry_ts" if "entry_ts" in dup_like else dup_like[0]
        if keep != "entry_ts":
            x = x.rename(columns={keep: "entry_ts"})
        for c in dup_like:
            if c != "entry_ts":
                x.drop(columns=c, inplace=True, errors="ignore")

    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    if "entry_ts" in x.columns:
        x["entry_ts"] = pd.to_datetime(x["entry_ts"], errors="coerce")
    return x

def _sma(s, n): return s.rolling(n, min_periods=n).mean()
def _ema(s, n):
    alpha = 2/(n+1.0)
    return s.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
def _wma(s, n):
    w = np.arange(1, n+1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w)/w.sum()), raw=True)
def _zscore(s, n):
    r = s.rolling(n, min_periods=n)
    mu = r.mean()
    sd = r.std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)
def _rolling_corr(a, b, n): return a.rolling(n, min_periods=n).corr(b)
def _poly_slope(y, n):
    x = np.arange(n, dtype=float)
    def _sl(win):
        if np.any(np.isnan(win)): return np.nan
        X = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(X, win, rcond=None)[0]
        return m
    return y.rolling(n, min_periods=n).apply(_sl, raw=True)

# ===== индикаторы =====
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = _ema(up, n) / (_ema(dn, n) + 1e-12)
    return 100 - 100/(1+rs)

def macd(close, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    m = ema_f - ema_s
    s = _ema(m, signal)
    h = m - s
    return m, s, h

def bollinger(close, n=20, k=2.0):
    ma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k*std
    lower = ma - k*std
    width = (upper - lower) / ma.replace(0, np.nan)
    bbp = (close - lower) / (upper - lower)
    return upper, lower, width, bbp

def adx(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move>down_move) & (up_move>0), up_move, 0.0)
    minus_dm = np.where((down_move>up_move) & (down_move>0), down_move, 0.0)
    atr = tr.rolling(n, min_periods=n).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=high.index).rolling(n, min_periods=n).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(n, min_periods=n).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(n, min_periods=n).mean(), plus_di, minus_di

def obv(close, volume):
    d = np.sign(close.diff().fillna(0.0))
    return (d * volume).fillna(0.0).cumsum()

def mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    pos_mf = np.where(tp > tp.shift(1), mf, 0.0)
    neg_mf = np.where(tp < tp.shift(1), mf, 0.0)
    pos = pd.Series(pos_mf, index=close.index).rolling(n, min_periods=n).sum()
    neg = pd.Series(neg_mf, index=close.index).rolling(n, min_periods=n).sum()
    mr = pos / (neg + 1e-12)
    return 100 - (100/(1+mr))

def vwap(high, low, close, volume, n=20):
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(n, min_periods=n).sum()
    v  = volume.rolling(n, min_periods=n).sum()
    return pv / (v + 1e-12)

# ===== паттерны/структура =====
def candle_bits(df):
    o,h,l,c = df["open"], df["high"], df["low"], df["close"]
    rng = (h-l).replace(0, np.nan)
    body = c - o
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    out = pd.DataFrame(index=df.index)
    out["body"] = body
    out["body_pct_rng"] = (body/rng).clip(-5,5)
    out["upper_wick"] = upper.clip(lower=0, upper=5)
    out["lower_wick"] = lower.clip(lower=0, upper=5)
    out["wick_asymmetry"] = out["upper_wick"] - out["lower_wick"]
    out["rng_pct"] = (rng / c).replace([np.inf,-np.inf], np.nan)
    out["dir_prev"] = np.sign(body.shift(1)).fillna(0).astype(np.int8)
    out["body_to_prev"] = body / body.shift(1).replace(0, np.nan)
    small_body = (out["body_pct_rng"].abs() < 0.15).astype(np.int8)
    long_lower = (out["lower_wick"] > 1.2).astype(np.int8)
    long_upper = (out["upper_wick"] > 1.2).astype(np.int8)
    out["doji_score"] = small_body
    out["hammer_like"] = ((long_lower==1) & (out["body"]>0)).astype(np.int8)
    out["pinbar_like"] = ((long_upper==1) & (out["body"]<0)).astype(np.int8)
    prev_o, prev_c = o.shift(1), c.shift(1)
    out["engulf_bull"] = ((c>o) & (prev_c<prev_o) & (c>=prev_o) & (o<=prev_c)).astype(np.int8)
    out["engulf_bear"] = ((c<o) & (prev_c>prev_o) & (c<=prev_o) & (o>=prev_c)).astype(np.int8)
    prev_h, prev_l = h.shift(1), l.shift(1)
    bull_fvg = (prev_l > h).astype(np.int8)
    bear_fvg = (prev_h < l).astype(np.int8)
    out["fvg_bull"] = bull_fvg
    out["fvg_bear"] = bear_fvg
    out["fvg_size"] = np.where(bull_fvg==1, (prev_l - h), np.where(bear_fvg==1, (l - prev_h), 0.0))
    dir_now = np.sign(body).fillna(0).astype(np.int8)
    out["bar_sequence_len"] = dir_now.groupby((dir_now != dir_now.shift()).cumsum()).cumcount()+1
    return out

def daily_context(df):
    g = df.set_index("entry_ts").sort_index()
    day_close = g["close"].resample("1D").last()
    day_high  = g["high"].resample("1D").max()
    day_low   = g["low"].resample("1D").min()
    day_rng   = (day_high - day_low)
    day_ret   = day_close.pct_change(fill_method=None)
    out = pd.DataFrame({
        "prev_day_close": day_close.shift(1),
        "prev_day_range": day_rng.shift(1),
        "prev_day_ret":   day_ret.shift(1),
    })
    return out.reindex(g.index, method="ffill")

def weekly_monthly_trend(df):
    g = df.set_index("entry_ts").sort_index()
    w_close = g["close"].resample("W-SUN").last()
    m_close = g["close"].resample("ME").last()
    w_ret = w_close.pct_change(fill_method=None).shift(1)
    m_ret = m_close.pct_change(fill_method=None).shift(1)
    ww = w_ret.reindex(g.index, method="ffill").rename("trend_week")
    mm = m_ret.reindex(g.index, method="ffill").rename("trend_month")
    return pd.concat([ww, mm], axis=1)

def session_flags(idx: pd.DatetimeIndex):
    h = idx.hour
    asia = ((h>=0) & (h<8)).astype(np.int8)
    eu   = ((h>=8) & (h<16)).astype(np.int8)
    us   = ((h>=12) & (h<21)).astype(np.int8)
    return pd.Series(asia, index=idx, name="sess_asia"), \
           pd.Series(eu,   index=idx, name="sess_eu"), \
           pd.Series(us,   index=idx, name="sess_us")

# ===== быстрая энтропия =====
def candle_entropy_fast(close: pd.Series, n: int = 12) -> pd.Series:
    d = np.sign(close.diff())
    up = (d > 0).astype(float)
    dn = (d < 0).astype(float)
    z0 = (d == 0).astype(float)
    p_up = up.rolling(n, min_periods=n).mean()
    p_dn = dn.rolling(n, min_periods=n).mean()
    p_0  = z0.rolling(n, min_periods=n).mean()
    def _h(p): return -(p * np.log(p + 1e-12))
    H = _h(p_up).fillna(0) + _h(p_dn).fillna(0) + _h(p_0).fillna(0)
    return H

# --- минимальные обязательные колонки во входе ---
REQUIRED_MIN = ["entry_ts","open","high","low","close","volume"]

def _ensure_minimal_base(df):
    miss = [c for c in REQUIRED_MIN if c not in df.columns]
    if miss:
        raise ValueError(f"Входной датасет не содержит обязательные колонки: {miss}")
    x = df.copy()
    if "symbol"   not in x.columns: x["symbol"] = "SYMBOL"
    if "side"     not in x.columns: x["side"] = "BOTH"
    if "side_num" not in x.columns: x["side_num"] = 0
    if "y"        not in x.columns: x["y"] = 0
    if "ret"      not in x.columns: x["ret"] = x["close"] / x["open"] - 1.0
    if "atr14"    not in x.columns:
        prev_close = x["close"].shift(1)
        tr = pd.concat([
            (x["high"]-x["low"]).abs(),
            (x["high"]-prev_close).abs(),
            (x["low"]-prev_close).abs()
        ], axis=1).max(axis=1)
        x["atr14"] = tr.rolling(14, min_periods=14).mean()
    # опциональные рефы (может прийти ref_close, ref_btc_close, ref_eth_close)
    if "ref_close" not in x.columns: x["ref_close"] = 0.0
    if "ref_btc_close" not in x.columns: x["ref_btc_close"] = 0.0
    if "ref_eth_close" not in x.columns: x["ref_eth_close"] = 0.0
    if "vol_regime" not in x.columns: x["vol_regime"] = 0
    return x

def _amihud_like(ret: pd.Series, vol: pd.Series, n: int = 20) -> pd.Series:
    x = (ret.abs() / (vol.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    return x.rolling(n, min_periods=n).median()

def _rolling_quantile(s: pd.Series, n: int, q: float) -> pd.Series:
    return s.rolling(n, min_periods=n).quantile(q)

def build_features_single_symbol(df_in: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_entry_ts(df_in)
    df = _ensure_minimal_base(df)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    idx = pd.DatetimeIndex(df["entry_ts"])
    o,h,l,c,v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    rng = (h-l).replace(0, np.nan)
    atrN = df["atr14"]

    geo = candle_bits(df)

    # тренд/моментум
    sma5=_sma(c,5); sma10=_sma(c,10); sma20=_sma(c,20); sma50=_sma(c,50); sma100=_sma(c,100)
    ema12=_ema(c,12); ema26=_ema(c,26); wma10=_wma(c,10)
    slope6=_poly_slope(c,6); slope12=_poly_slope(c,12)
    momentum6  = c/_sma(c,6) - 1
    momentum12 = c/_sma(c,12) - 1
    trend_strength   = (sma5 - sma20) / (atrN + 1e-12)
    cross_fast_slow  = np.sign(sma5 - sma20).fillna(0).astype(np.int8)
    rsi14 = rsi(c,14)
    macd_val, macd_sig, macd_hist = macd(c)
    cci20 = (c - _sma(c,20)) / (0.015 * (rng.rolling(20, min_periods=20).mean() + 1e-12))
    roll_max20 = h.rolling(20, min_periods=20).max()
    roll_min20 = l.rolling(20, min_periods=20).min()
    dist_to_high = (roll_max20 - c) / (atrN + 1e-12)
    dist_to_low  = (c - roll_min20) / (atrN + 1e-12)

    # волатильность
    atr_ratio6 = atrN / (_sma(atrN,6) + 1e-12)
    volat_ret12 = df["ret"].rolling(12, min_periods=12).std(ddof=0)
    _,_,bb_w,bbp = bollinger(c,20,2.0)
    range_z = _zscore(rng,20)
    hl_spread_ratio = rng / (c + 1e-12)

    # объёмы
    vol_ratio6 = v / (_sma(v,6) + 1e-12)
    vol_delta  = v.pct_change(fill_method=None)
    vol_z      = _zscore(v,20)
    price_vol_corr12 = _rolling_corr(c,v,12)
    obv_v  = obv(c,v)
    mfi14  = mfi(h,l,c,v,14)
    vwap20 = vwap(h,l,c,v,20)
    price_vs_vwap = (c - vwap20) / (atrN + 1e-12)

    # структура/события
    local_high_break = (c > roll_max20.shift(1)).astype(np.int8)
    local_low_break  = (c < roll_min20.shift(1)).astype(np.int8)
    gap_to_prev_close = (o - c.shift(1)) / (atrN + 1e-12)

    # макро-время
    df["hour_of_day"] = idx.hour
    df["day_of_week"] = idx.weekday
    df["is_weekend"]  = (df["day_of_week"]>=5).astype(np.int8)
    df["is_monday"]   = (df["day_of_week"]==0).astype(np.int8)
    df["hod_sin"] = np.sin(2*np.pi*df["hour_of_day"]/24.0)
    df["hod_cos"] = np.cos(2*np.pi*df["hour_of_day"]/24.0)
    asia, eu, us = session_flags(idx)
    df["sess_asia"]=asia.values; df["sess_eu"]=eu.values; df["sess_us"]=us.values

    # дневной/недельный/месячный контекст
    day_ctx = daily_context(df)
    wm_ctx  = weekly_monthly_trend(df)

    # композиты
    market_heat = _zscore(atrN,48)
    rsi_z = _zscore(rsi14,48)
    atr_slope = _poly_slope(atrN,12)
    body_vs_wick = (geo["body"].abs() / (geo["upper_wick"] + geo["lower_wick"]).replace(0, np.nan))
    c_entropy = candle_entropy_fast(c, n=12)
    price_distance_ma20 = (c - sma20) / (atrN + 1e-12)
    momentum_vol_corr = _rolling_corr(momentum12, vol_ratio6, 12)
    regime_index = (trend_strength*0.5 + bb_w*0.3 + rsi_z*0.2)

    # ===== Ликвидность/спрэд (новые) =====
    vol_med20 = v.rolling(20, min_periods=20).median()
    vol_med48 = v.rolling(48, min_periods=48).median()
    zero_vol_share48 = (v==0).astype(float).rolling(48, min_periods=48).mean()
    hl_spread_med48 = hl_spread_ratio.rolling(48, min_periods=48).median()
    atr_to_price = atrN / (c + 1e-12)
    atr_rank_48 = _rolling_quantile(atrN, 48, 0.8)  # уровень 0.8 в окне (как «высокая вола»)
    amihud20 = _amihud_like(df["ret"], v, 20)

    # ===== Coin strength vs BTC/ETH/REF (новые) =====
    # ожидаемые входы: ref_btc_close / ref_eth_close / ref_close — любые из них
    vs_btc = vs_eth = vs_ref = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    z_vs_btc = z_vs_eth = z_vs_ref = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    if "ref_btc_close" in df.columns and df["ref_btc_close"].abs().sum() > 0:
        r_btc = df["ref_btc_close"].pct_change(fill_method=None)
        vs_btc = (df["ret"] - r_btc).fillna(0.0)
        z_vs_btc = _zscore(vs_btc, 48).fillna(0.0)
    if "ref_eth_close" in df.columns and df["ref_eth_close"].abs().sum() > 0:
        r_eth = df["ref_eth_close"].pct_change(fill_method=None)
        vs_eth = (df["ret"] - r_eth).fillna(0.0)
        z_vs_eth = _zscore(vs_eth, 48).fillna(0.0)
    if "ref_close" in df.columns and df["ref_close"].abs().sum() > 0 and (df["ref_close"] != 0).any():
        r_ref = df["ref_close"].pct_change(fill_method=None)
        vs_ref = (df["ret"] - r_ref).fillna(0.0)
        z_vs_ref = _zscore(vs_ref, 48).fillna(0.0)

    # сборка фич
    feat = pd.DataFrame(index=df.index)
    feat = pd.concat([feat, geo], axis=1)
    feat["sma5"]=sma5; feat["sma10"]=sma10; feat["sma20"]=sma20; feat["sma50"]=sma50; feat["sma100"]=sma100
    feat["ema12"]=ema12; feat["ema26"]=ema26; feat["wma10"]=wma10
    feat["slope6"]=slope6; feat["slope12"]=slope12
    feat["momentum6"]=momentum6; feat["momentum12"]=momentum12
    feat["trend_strength"]=trend_strength
    feat["cross_fast_slow"]=cross_fast_slow
    feat["rsi14"]=rsi14; feat["cci20"]=cci20
    feat["macd"]=macd_val; feat["macd_sig"]=macd_sig; feat["macd_hist"]=macd_hist
    feat["dist_to_high"]=dist_to_high; feat["dist_to_low"]=dist_to_low
    feat["atr_ratio6"]=atr_ratio6; feat["volat_ret12"]=volat_ret12
    feat["bb_width"]=bb_w; feat["bbp"]=bbp
    feat["range_z"]=range_z; feat["hl_spread_ratio"]=hl_spread_ratio
    feat["vol_ratio6"]=vol_ratio6; feat["vol_delta"]=vol_delta; feat["vol_z"]=vol_z
    feat["price_vol_corr12"]=price_vol_corr12
    feat["obv"]=obv_v; feat["mfi14"]=mfi14
    feat["vwap20"]=vwap20; feat["price_vs_vwap"]=price_vs_vwap
    feat["local_high_break"]=local_high_break; feat["local_low_break"]=local_low_break
    feat["gap_to_prev_close"]=gap_to_prev_close
    adx14, plus_di, minus_di = adx(h,l,c,14)
    feat["adx14"]=adx14; feat["plus_di"]=plus_di; feat["minus_di"]=minus_di
    feat[["hour_of_day","day_of_week","is_weekend","is_monday","hod_sin","hod_cos","sess_asia","sess_eu","sess_us"]] = \
        df[["hour_of_day","day_of_week","is_weekend","is_monday","hod_sin","hod_cos","sess_asia","sess_eu","sess_us"]]
    feat[["prev_day_close","prev_day_range","prev_day_ret"]] = daily_context(df)[["prev_day_close","prev_day_range","prev_day_ret"]].to_numpy()
    feat[["trend_week","trend_month"]] = weekly_monthly_trend(df)[["trend_week","trend_month"]].to_numpy()
    feat["market_heat"]=market_heat; feat["rsi_z"]=rsi_z; feat["atr_slope"]=atr_slope
    feat["body_vs_wick"]=body_vs_wick; feat["candle_entropy"]=c_entropy
    feat["price_distance_ma20"]=price_distance_ma20; feat["momentum_vol_corr"]=momentum_vol_corr
    feat["regime_index"]=regime_index

    # новые ликвид/спрэд
    feat["vol_med20"]=vol_med20; feat["vol_med48"]=vol_med48
    feat["zero_vol_share48"]=zero_vol_share48
    feat["hl_spread_med48"]=hl_spread_med48
    feat["atr_to_price"]=atr_to_price
    feat["atr_rank_48"]=atr_rank_48
    feat["amihud20"]=amihud20

    # coin strength
    feat["ret_vs_btc"]=vs_btc; feat["ret_vs_eth"]=vs_eth; feat["ret_vs_ref"]=vs_ref
    feat["ret_vs_btc_z"]=z_vs_btc; feat["ret_vs_eth_z"]=z_vs_eth; feat["ret_vs_ref_z"]=z_vs_ref

    # финал: все исходные + фичи
    out = pd.concat([df, feat], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    # требуем готовность окон и regime_index без «прогрева»
    must_have = [c for c in ["sma20","rsi14","bb_width","adx14","mfi14","vwap20","regime_index"] if c in out.columns]
    if must_have:
        out = out.dropna(subset=must_have, how="any")

    # бесконечности → NaN
    out = out.replace([np.inf, -np.inf], np.nan)

    # контексты ffill/bfill
    context_cols = [c for c in ["trend_week","trend_month","prev_day_close","prev_day_range",
                                "prev_day_ret","candle_entropy","sma50","sma100",
                                "rsi_z","market_heat"] if c in out.columns]
    for c in context_cols:
        if out[c].isna().any():
            out[c] = out[c].ffill().bfill()

    # остаточные NaN → 0 (только где остались)
    num_cols = out.select_dtypes(include=["number", "bool"]).columns
    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(0)

    # sanity по нулевой дисперсии
    numeric = out.select_dtypes(include=["number"]).copy()
    if not numeric.empty:
        var0 = numeric.nunique(dropna=False) <= 1
        n_var0 = int(var0.sum())
        if n_var0 > 0:
            ex = list(var0[var0].index[:8])
            print(f"[WARN] zero-variance columns: {n_var0}. examples: {ex}")

    assert out["entry_ts"].isna().sum() == 0
    assert out.isna().sum().max() == 0
    return out.reset_index(drop=True)

def _add_symbol_encodings(out: pd.DataFrame, sym_topk: int = 24) -> pd.DataFrame:
    """One-hot по топ-K символам (+ other) и числовой symbol_id."""
    if "symbol" not in out.columns:
        return out
    x = out.copy()
    # стабильный id
    uniq = sorted(x["symbol"].astype(str).unique())
    mapping = {s:i for i,s in enumerate(uniq)}
    x["symbol_id"] = x["symbol"].astype(str).map(mapping).astype(np.int32)

    # top-K one-hot
    vc = x["symbol"].astype(str).value_counts()
    top_syms = list(vc.index[:sym_topk])
    x["sym_is_other"] = (~x["symbol"].astype(str).isin(top_syms)).astype(np.int8)
    for s in top_syms:
        col = f"sym_{s}"
        x[col] = (x["symbol"].astype(str) == s).astype(np.int8)
    return x

def build_features(df: pd.DataFrame, sym_topk: int = 24) -> pd.DataFrame:
    df = _normalize_entry_ts(df)
    df = _ensure_minimal_base(df.copy())
    if "symbol" in df.columns:
        parts = []
        for sym, g in df.groupby("symbol", sort=False, group_keys=False):
            g = _normalize_entry_ts(g)
            part = build_features_single_symbol(g)
            if part is not None and len(part):
                parts.append(part)
        if not parts:
            raise ValueError("После сборки по группам пусто (parts is empty).")
        out = pd.concat(parts, axis=0, ignore_index=True)
        out = out.sort_values(["symbol","entry_ts"]).reset_index(drop=True)
        out = _add_symbol_encodings(out, sym_topk=sym_topk)
        return out
    else:
        out = build_features_single_symbol(df)
        out = _add_symbol_encodings(out, sym_topk=sym_topk)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="входной parquet (per-symbol/per-side или общий)")
    ap.add_argument("--out", type=str, required=True,
                    help="куда сохранить parquet с расширенными фичами")
    ap.add_argument("--sym-topk", type=int, default=24,
                    help="кол-во символов для one-hot кодирования (+ other)")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    if "entry_ts" not in df.columns and getattr(df.index, "name", None) != "entry_ts" and "entry_ts" not in list(getattr(df.index, "names", [])):
        raise ValueError("Входной parquet должен содержать колонку или индекс 'entry_ts'.")

    df = _normalize_entry_ts(df)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")

    out = build_features(df, sym_topk=args.sym_topk)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    core_nan = {
        "entry_ts": float(out["entry_ts"].isna().mean()),
        "close":    float(out["close"].isna().mean()),
        "y":        float(out.get("y").isna().mean() if "y" in out.columns else 0.0),
    }
    n_feats = len([c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])])
    print(f"[SANITY] rows={len(out)} | NaN rates {core_nan} | n_feats≈{n_feats}")
    out.to_parquet(args.out, index=False)
    print(f"[OK] saved → {args.out}")

if __name__ == "__main__":
    main()