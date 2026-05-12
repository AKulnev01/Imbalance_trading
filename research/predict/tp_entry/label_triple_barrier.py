# predict/tp_entry/label_triple_barrier.py
import numpy as np
import pandas as pd

MS = 60_000

def ensure_idx(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "ts" in x.columns:
        x["ts"] = pd.to_datetime(x["ts"], unit="ms", utc=True)
        x = x.set_index("ts")
    x.index = pd.to_datetime(x.index, utc=True)
    return x.sort_index()

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = (h - l).abs()
    tr = pd.concat([tr, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def minute_iter(df_m1: pd.DataFrame, start_ts, limit_minutes: int):
    # генератор минут от start_ts (включая) вперёд
    idx = df_m1.index
    pos = idx.searchsorted(start_ts)  # первая минута >= start_ts
    end_pos = min(pos + limit_minutes + 1, len(idx))
    for i in range(pos, end_pos):
        yield idx[i]

def price_hits(m1: pd.DataFrame, start_ts: pd.Timestamp, side: int,
               tp_px: float, sl_px: float, max_minutes: int):
    """
    Быстрый поиск первого срабатывания TP/SL в окне [start_ts, start_ts + max_minutes).
    Векторно (без поколоночных .at[...] в цикле).
    Возвращает: (reason: str, exit_ts: pd.Timestamp, exit_px: float)
    reason ∈ {"tp","sl","timeout"}.
    """
    # границы окна
    end_ts = start_ts + pd.Timedelta(minutes=int(max_minutes))
    # минутки окна
    try:
        window = m1.loc[start_ts:end_ts, ["high", "low", "close"]]
    except KeyError:
        window = m1.loc[start_ts:end_ts]

    if window.empty:
        # нет данных — считаем таймаутом у старта
        ref = float(m1.loc[start_ts, "close"]) if start_ts in m1.index else float(m1["close"].iloc[0])
        return "timeout", start_ts, ref

    # булевы маски срабатываний
    h = window["high"].to_numpy(copy=False)
    l = window["low"].to_numpy(copy=False)
    idx = window.index

    if side > 0:
        tp_mask = (h >= tp_px)
        sl_mask = (l <= sl_px)
        tp_price = tp_px
        sl_price = sl_px
    else:
        # шорт: TP — цена вниз до tp_px (low <= tp), SL — вверх до sl_px (high >= sl)
        tp_mask = (l <= tp_px)
        sl_mask = (h >= sl_px)
        tp_price = tp_px
        sl_price = sl_px

    # первые истинные позиции (если есть)
    tp_pos = np.flatnonzero(tp_mask)
    sl_pos = np.flatnonzero(sl_mask)

    has_tp = tp_pos.size > 0
    has_sl = sl_pos.size > 0

    if not has_tp and not has_sl:
        # не задели уровни — таймаут в конце окна
        return "timeout", idx[-1], float(window["close"].iloc[-1])

    if has_tp and has_sl:
        i_tp = int(tp_pos[0])
        i_sl = int(sl_pos[0])
        if i_tp < i_sl:
            ts = idx[i_tp]
            return "tp", ts, float(tp_price)
        elif i_sl < i_tp:
            ts = idx[i_sl]
            return "sl", ts, float(sl_price)
        else:
            # одновременно — честно отдать «хужее» для стороны (консерватизм)
            ts = idx[i_tp]
            if side > 0:
                return "sl", ts, float(sl_price)
            else:
                return "sl", ts, float(sl_price)
    elif has_tp:
        ts = idx[int(tp_pos[0])]
        return "tp", ts, float(tp_price)
    else:
        ts = idx[int(sl_pos[0])]
        return "sl", ts, float(sl_price)

def label_entries(
    m1: pd.DataFrame,
    entries_4h: pd.DataFrame,
    side_col: str,
    k_tp: float,
    k_sl: float,
    tmax_hours: int = 80,
    fee_pct: float = 0.001,
    slip_exit_pct: float = 0.004,
    atr_col: str = None,
    atr_n: int = 14,
):
    """
    m1: минутки (index=UTC datetime, cols=open,high,low,close,volume)
    entries_4h: index=UTC datetime (вход по market на открытии следующей 4h минуты entries_4h.index + 1мин), колонки: ['open','high','low','close', ... , side_col ∈ {+1,-1}]
    k_tp,k_sl: мультипликаторы от ATR4h(t) в ценах (TP = close_t ± k_tp*ATR; SL = close_t ∓ k_sl*ATR)
    Возвращает DataFrame с колонками: ['entry_ts','side','tp_px','sl_px','exit_ts','exit_px','reason','pnl_net']
    """
    m1 = ensure_idx(m1)[["open","high","low","close","volume"]].copy()
    e4 = entries_4h.copy()
    e4.index = pd.to_datetime(e4.index, utc=True)
    if atr_col is None:
        # ATR по 4h — допустимо приблизить из 1m, ресемплируя
        ohlc = m1[["open","high","low","close"]].resample("4H", label="right", closed="right").agg({
            "open":"first","high":"max","low":"min","close":"last"
        }).dropna()
        atr4h = compute_atr(ohlc, n=atr_n)
    else:
        atr4h = e4[atr_col]

    out = []
    max_minutes = int(tmax_hours*60)
    for t, row in e4.iterrows():
        if side_col not in row or pd.isna(row[side_col]):
            continue
        side = int(row[side_col])
        px0 = float(row["close"])  # ориентир для ATR-шагов
        atr = float(atr4h.loc[t])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if side > 0:
            tp_px = px0 * (1 + (k_tp*atr)/px0)
            sl_px = px0 * (1 - (k_sl*atr)/px0)
        else:
            tp_px = px0 * (1 - (k_tp*atr)/px0)
            sl_px = px0 * (1 + (k_sl*atr)/px0)

        # вход «после закрытия 4h» — следующая минута
        start_ts = t + pd.Timedelta(minutes=1)

        reason, exit_ts, exit_px = price_hits(m1, start_ts, side, tp_px, sl_px, max_minutes)

        # комиссии и проскальзывание: считаем вход по close_t (или можно open t+1, но единообразно)
        fee_in  = fee_pct * px0
        # выход — рыночный, добавляем проскальзывание:
        exit_px_eff = exit_px * (1 - slip_exit_pct if side>0 else 1 + slip_exit_pct)
        fee_out = fee_pct * exit_px_eff
        pnl = (exit_px_eff - px0) * (1 if side>0 else -1) - fee_in - fee_out

        out.append({
            "entry_ts": t, "side": side, "tp_px": tp_px, "sl_px": sl_px,
            "exit_ts": exit_ts, "exit_px": exit_px, "reason": reason, "pnl_net": pnl,
            "atr4h": atr, "ref_close": px0
        })
    df = pd.DataFrame(out)

    # --- самовосстановление entry_ts, если его нет в out ---
    if "entry_ts" not in df.columns:
        # 1) попробуем типичные альтернативные имена
        rename_map = {}
        if "ts" in df.columns:
            rename_map["ts"] = "entry_ts"
        if "bar_ts" in df.columns:
            rename_map["bar_ts"] = "entry_ts"
        if rename_map:
            df = df.rename(columns=rename_map)

    # 2) если всё ещё нет — возьмём из entries_4h индекса
    if "entry_ts" not in df.columns:
        try:
            idx = entries_4h.index
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx, errors="coerce")
            # обрежем/соотнесём длину на случай рассинхрона
            if len(idx) >= len(df):
                df["entry_ts"] = idx.values[:len(df)]
            else:
                # если out длиннее — дублируем последний ts (крайне редко)
                fill_idx = np.full(len(df), pd.to_datetime(idx[-1], errors="coerce"))
                fill_idx[:len(idx)] = idx.values
                df["entry_ts"] = fill_idx
        except Exception:
            pass

    # 3) если и это не помогло — действительно ошибка входных данных
    if "entry_ts" not in df.columns:
        raise KeyError("label_entries: cannot determine 'entry_ts' in result; ensure entries_4h index is DatetimeIndex or provide 'entry_ts' in out dicts")

    # нормализуем таймзону
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    try:
        df["entry_ts"] = df["entry_ts"].dt.tz_localize(None)
    except Exception:
        pass

    if df["entry_ts"].isna().all():
        raise KeyError("label_entries: 'entry_ts' is NaT for all rows after reconstruction")

    return df.set_index("entry_ts").sort_index()