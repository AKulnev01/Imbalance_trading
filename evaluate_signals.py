# evaluate_signals.py

import sys
import os
import math
import pandas as pd
from datetime import timedelta, datetime, timezone
from typing import Tuple
from utils.fetch_data import get_bybit_klines
from config import (
    RISK_PCT,            # оставляем для совместимости (в расчётах SL/TP не используется)
    RISK_REWARD_RATIO,   # оставляем для совместимости
    POSITION_SIZE_USD,   # не используется в этой версии (заменён на долю)
    FEE_TAKER,
    SLIPPAGE_PCT,
    MAX_FILL_DAYS,
    INITIAL_CAPITAL,
)

# ===== ENV (можно переопределить без правки кода) =====
MOMENTUM_TP_PCT = float(os.getenv("MOMENTUM_TP_PCT", "0.02"))  # +2% (если нужно для MOMENTUM)
MOMENTUM_SL_PCT = float(os.getenv("MOMENTUM_SL_PCT", "0.01"))  # -1%

# Реальный «боевой» режим
ENTRY_MODE = os.getenv("ENTRY_MODE", "RETEST").upper()            # RETEST | BREAKOUT | MOMENTUM
DEFAULT_TTL_DAYS = int(os.getenv("DEFAULT_TTL_DAYS", "5"))        # срок жизни лимитки (для RETEST)
BACKFILL_4H_BARS = int(os.getenv("BACKFILL_4H_BARS", "24"))       # не критично здесь

# Наша модель капитала
POSITION_FRACTION = float(os.getenv("POSITION_FRACTION", "0.25")) # 25% от equity на вход
STOP_PCT = float(os.getenv("STOP_PCT", "0.01"))                   # 1% от цены входа (потеря по позиции)
TAKE_PCT = float(os.getenv("TAKE_PCT", "0.03"))                   # 3% от цены входа (прибыль по позиции)

VARIANT_COL_CANDIDATES = ["variant", "mode", "entry_mode", "strategy"]

# --- Intrabar (для разрешения порядка TP/SL в спорном 4h-баре) ---
INTRABAR_INTERVALS = os.getenv("INTRABAR_INTERVALS", "1m,5m").split(",")  # порядок приоритета
INTRABAR_LOOKBACK_DAYS_FALLBACK = int(os.getenv("INTRABAR_LOOKBACK_DAYS_FALLBACK", "14"))

# ===================== Утилиты =====================
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_convert('UTC')
    return df

def _to_num(x):
    return pd.to_numeric(str(x).replace(',', '.').strip(), errors='coerce')

def _to_bool_filled(x):
    s = str(x).strip().lower()
    return s in ('true', 'истина', '1', 'yes', 'y', 'да')

def _find_variant_col(df: pd.DataFrame) -> str:
    for c in VARIANT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _row_variant(row, variant_col: str) -> str:
    if variant_col and pd.notna(row.get(variant_col)):
        return str(row[variant_col]).strip().upper()
    return "RETEST"

def _calc_sl_tp(entry: float, side: str, risk_pct_price: float, rr: float) -> Tuple[float, float]:
    """
    SL/TP как доля от цены входа (по цене), RR = take/stop.
    Здесь risk_pct_price — относительный шаг от цены входа (например 0.01 = 1%).
    """
    k = float(risk_pct_price)
    if side == "SELL":
        sl = entry * (1.0 + k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

def _first_touch_after(df: pd.DataFrame, entry: float, t0: pd.Timestamp) -> pd.Timestamp:
    """Первое касание entry строго ПОСЛЕ бара имбаланса t0 (для RETEST). Возвращает NaT, если не было касания."""
    if df is None or df.empty or pd.isna(entry) or pd.isna(t0):
        return pd.NaT
    win = df[(df.index > t0)]
    for ts, row in win.iterrows():
        if float(row['low']) <= entry <= float(row['high']):
            return ts
    return pd.NaT

def _repair_levels(side, entry, stop_eval, tp_eval) -> Tuple[float, float, bool]:
    """Проверяем корректность расположения уровней относительно entry; если что — чиним."""
    ok = True
    if side == 'BUY':
        ok = (stop_eval < entry) and (tp_eval > entry)
    else:  # SELL
        ok = (tp_eval < entry) and (stop_eval > entry)
    if ok:
        return float(stop_eval), float(tp_eval), False
    # если криво — пересчёт как 1:3 по цене
    s, t = _calc_sl_tp(float(entry), side, STOP_PCT, TAKE_PCT/STOP_PCT)
    return float(s), float(t), True

def _safe_group_exit_reason(df_res: pd.DataFrame) -> pd.DataFrame:
    df = df_res.copy()
    if 'skipped' in df.columns:
        df = df[df['skipped'] == False].copy()

    for col, default in [
        ('exit_reason', 'unknown'),
        ('win', False),
        ('pnl_pct', 0.0),
        ('pnl_usd', 0.0),
        ('exit_days', pd.NA),
    ]:
        if col not in df.columns:
            df[col] = default

    df['win'] = df['win'].astype('bool')
    df['pnl_pct'] = pd.to_numeric(df['pnl_pct'], errors='coerce').fillna(0.0)
    df['pnl_usd'] = pd.to_numeric(df['pnl_usd'], errors='coerce').fillna(0.0)
    df['exit_days'] = pd.to_numeric(df['exit_days'], errors='coerce')

    def _winrate_safe(s):
        n = int(s.size) if s is not None else 0
        return round(100.0 * float(s.sum()) / float(n), 2) if n > 0 else 0.0

    if df.empty:
        return pd.DataFrame(columns=[
            'exit_reason','trades','wins','winrate_pct','pnl_pct','pnl_usd',
            'avg_exit_days','med_exit_days'
        ])

    try:
        by_exit_reason = (
            df.groupby('exit_reason', dropna=False)
              .agg(
                  trades=('win', 'size'),
                  wins=('win', 'sum'),
                  winrate_pct=('win', _winrate_safe),
                  pnl_pct=('pnl_pct', 'sum'),
                  pnl_usd=('pnl_usd', 'sum'),
                  avg_exit_days=('exit_days', 'mean'),
                  med_exit_days=('exit_days', 'median'),
              )
              .reset_index()
              .sort_values(['pnl_usd', 'winrate_pct'], ascending=[False, False])
        )
        return by_exit_reason
    except TypeError:
        grouped = (
            df.groupby('exit_reason', dropna=False)
              .agg({
                  'win': ['size', 'sum'],
                  'pnl_pct': ['sum'],
                  'pnl_usd': ['sum'],
                  'exit_days': ['mean', 'median'],
              })
        )
        grouped.columns = ['_'.join([c for c in col if c]) for c in grouped.columns.to_flat_index()]
        rename_map = {
            'win_size': 'trades',
            'win_sum': 'wins',
            'pnl_pct_sum': 'pnl_pct',
            'pnl_usd_sum': 'pnl_usd',
            'exit_days_mean': 'avg_exit_days',
            'exit_days_median': 'med_exit_days',
        }
        by_exit_reason = grouped.rename(columns=rename_map).reset_index()
        by_exit_reason['winrate_pct'] = (
            (by_exit_reason['wins'] / by_exit_reason['trades'])
            .replace([pd.NA, pd.NaT], 0).fillna(0).astype(float) * 100.0
        ).round(2)
        by_exit_reason = by_exit_reason.sort_values(['pnl_usd', 'winrate_pct'], ascending=[False, False])
        return by_exit_reason


# ===== UTC helper =====
def _to_utc_safe(ts):
    """Безопасно привести к UTC: если naive — локализуем, если tz-aware — конвертим."""
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors='coerce')
    if t is pd.NaT:
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')


# ===================== Реалистичная симуляция капитала =====================
def _simulate_capital_notional(
        df_res: pd.DataFrame,
        initial_capital: float,
        position_fraction: float,
        stop_pct: float,
        take_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Реалистичная симуляция:
      - Сделка занимает деньги на входе: size = position_fraction * equity (на момент входа).
      - Если свободного кеша не хватает → skipped (без влияния на суммы).
      - Equity меняется ТОЛЬКО при закрытии сделки.
      - PnL в $:
          * tp → +take_pct * size
          * sl → -stop_pct * size
          * timeout_last_close → (pnl_pct (от цены) / 100) * size
    Возвращает: (df_out, equity_curve)
    """
    if df_res is None or df_res.empty:
        return df_res, pd.DataFrame()

    df = df_res.copy()
    for c in ("t_start", "close_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    # порядок исполнения
    df = df.sort_values(["t_start", "symbol", "close_time"], kind="mergesort").reset_index(drop=True)

    equity = float(initial_capital)
    free_cash = equity
    active = []  # [{idx, symbol, size, close_time, pnl_usd, exit_reason}]
    out_rows = []
    eq_rows = []

    fees_slip_pos = (float(FEE_TAKER) * 2.0) + (float(SLIPPAGE_PCT) * 2.0)

    def _close_until(ts):
        nonlocal equity, free_cash, active, eq_rows
        still = []
        for pos in active:
            ctime = pos["close_time"]
            if pd.notna(ctime) and ctime <= ts:
                pnl = float(pos["pnl_usd"])
                eq_before = equity
                equity += pnl
                free_cash += pos["size"] + pnl
                eq_rows.append({
                    "i": pos["idx"] + 1,
                    "time": ctime,
                    "symbol": pos["symbol"],
                    "alloc_usd": round(pos["size"], 2),
                    "pnl_usd_comp": round(pnl, 2),
                    "equity_before": round(eq_before, 2),
                    "equity_after": round(equity, 2),
                    "exit_reason": pos["exit_reason"],
                })
            else:
                still.append(pos)
        active = still

    for i, r in df.iterrows():
        t_start = r.get("t_start")
        t_close = r.get("close_time")
        symbol  = r.get("symbol")

        # Сначала закрыть те, кто уже закрылся к моменту входа этой сделки
        if pd.notna(t_start):
            _close_until(t_start)

        # Если входа не было (timeout_no_fill) — это скип (нет блокировки денег)
        if pd.isna(t_start):
            row = r.to_dict()
            row.update({
                "skipped": True,
                "alloc_usd_comp": pd.NA,
                "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA,
                "pnl_usd": pd.NA,
                "pnl_pct": pd.NA,
            })
            out_rows.append(row)
            continue

        # Размер позиции = доля от equity на момент входа
        size = max(0.0, float(position_fraction) * float(equity))

        # Денег не хватает → skip
        if free_cash < size:
            row = r.to_dict()
            row.update({
                "skipped": True,
                "alloc_usd_comp": pd.NA,
                "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA,
                "pnl_usd": pd.NA,
                "pnl_pct": pd.NA,
                "exit_reason": "skipped_no_capital"
            })
            out_rows.append(row)
            continue

        # Резервируем кэш
        free_cash -= size

        # Рассчитываем итог сделки
        er = str(r.get("exit_reason") or "").lower()
        price_pnl_pct = r.get("pnl_pct")

        if er == "tp":
            # прибыль по позиции с учётом комиссий/проскальзывания
            pnl_usd = (float(take_pct) - fees_slip_pos) * size
        elif er == "sl":
            # убыток по позиции с учётом комиссий/проскальзывания
            pnl_usd = -(float(stop_pct) + fees_slip_pos) * size
        else:
            try:
                pnl_usd = (float(price_pnl_pct) / 100.0) * size if pd.notna(price_pnl_pct) else 0.0
            except Exception:
                pnl_usd = 0.0

        # Регистрируем открытую позицию
        active.append({
            "idx": i,
            "symbol": symbol,
            "size": size,
            "close_time": t_close if pd.notna(t_close) else t_start,
            "exit_reason": r.get("exit_reason"),
            "pnl_usd": float(pnl_usd),
        })

        # В result запишем pnl_usd «по позиции» сразу, alloc — заполним после закрытия
        row = r.to_dict()
        # Для наглядности pnl_pct оставим как проценты от цены (информативно)
        row.update({
            "skipped": False,
            "pnl_usd": round(pnl_usd, 2),
            "alloc_usd_comp": pd.NA,   # ← теперь заполняем ТОЛЬКО при закрытии
            "pnl_usd_comp":   pd.NA,
            "equity_after":   pd.NA,
        })
        out_rows.append(row)

    # Закрываем всё, что осталось
    active.sort(key=lambda x: x["close_time"] if pd.notna(x["close_time"]) else pd.Timestamp.max.tz_localize("UTC"))
    for pos in active:
        pnl = float(pos["pnl_usd"])
        eq_before = equity
        equity += pnl
        free_cash += pos["size"] + pnl
        eq_rows.append({
            "i": pos["idx"] + 1,
            "time": pos["close_time"],
            "symbol": pos["symbol"],
            "alloc_usd": round(pos["size"], 2),
            "pnl_usd_comp": round(pnl, 2),
            "equity_before": round(eq_before, 2),
            "equity_after": round(equity, 2),
            "exit_reason": pos["exit_reason"],
        })

    df_out = pd.DataFrame(out_rows).reset_index(drop=True)
    for e in eq_rows:
        idx = e["i"] - 1
        if 0 <= idx < len(df_out):
            df_out.at[idx, "alloc_usd_comp"] = e["alloc_usd"]
            df_out.at[idx, "pnl_usd_comp"]   = e["pnl_usd_comp"]
            df_out.at[idx, "equity_after"]   = e["equity_after"]

    eq_curve = pd.DataFrame(eq_rows).sort_values("time").reset_index(drop=True)
    return df_out, eq_curve

def _norm_ts_utc(x):
    try:
        return pd.to_datetime(x, utc=True, errors='coerce')
    except Exception:
        return pd.NaT

def _fetch_ltf_window(symbol: str,
                      t_start: pd.Timestamp,
                      t_end: pd.Timestamp,
                      candidates=None) -> pd.DataFrame:
    """
    Пытаемся достать LTF бары (1m/5m) в интервале [t_start, t_end].
    Если utils.get_bybit_klines не умеет по диапазону — берём lookback по дням и фильтруем по времени.
    Возвращает DataFrame c индексом-временем (UTC) или пустой DF.
    """
    if candidates is None:
        candidates = INTRABAR_INTERVALS
    # грубый lookback по дням (с запасом)
    days = max(1, int((t_end - t_start).total_seconds() // 86400) + 2)
    days = max(days, INTRABAR_LOOKBACK_DAYS_FALLBACK)

    for iv in [c.strip() for c in candidates if c.strip()]:
        try:
            df_ltf = get_bybit_klines(symbol=symbol, interval=iv, lookback_days=days)
            df_ltf = _ensure_dt_index(df_ltf)
            if df_ltf is None or df_ltf.empty:
                continue
            win = df_ltf[(df_ltf.index >= t_start) & (df_ltf.index <= t_end)].copy()
            if not win.empty:
                return win
        except Exception:
            continue
    return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

def _resolve_tp_sl_order_ltf(symbol: str,
                             side: str,
                             entry_at: pd.Timestamp,
                             bar_close_time: pd.Timestamp,
                             stop_eval: float,
                             tp_eval: float) -> tuple[bool, pd.Timestamp, float, str]:
    """
    Пытаемся на LTF определить, что сработало первым в спорном 4h-баре.
    Возвращает (win, close_time, close_price, exit_reason['tp'|'sl'|'uncertain']).
    Если LTF недоступен/неоднозначно -> ('sl' приоритетно, консервативно).
    """
    t0 = _to_utc_safe(entry_at)
    t1 = _to_utc_safe(bar_close_time)
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        # страховка: консервативно SL
        if side == 'BUY':
            return (False, t1, float(stop_eval), 'sl')
        else:
            return (False, t1, float(stop_eval), 'sl')

    # 1) пробуем 1m, затем 5m (или как задано в ENV)
    ltf = _fetch_ltf_window(symbol, t0, t1, candidates=INTRABAR_INTERVALS)

    if ltf.empty:
        # нет данных — консервативно SL
        return (False, t1, float(stop_eval), 'sl')

    # 2) проходим бары LTF по времени
    for ts, c in ltf.iterrows():
        hi, lo = float(c['high']), float(c['low'])
        if side == 'BUY':
            hit_tp = (hi >= tp_eval)
            hit_sl = (lo <= stop_eval)
        else:
            hit_tp = (lo <= tp_eval)
            hit_sl = (hi >= stop_eval)

        if hit_tp and not hit_sl:
            return (True, ts, float(tp_eval), 'tp')
        if hit_sl and not hit_tp:
            return (False, ts, float(stop_eval), 'sl')
        if hit_tp and hit_sl:
            # в пределах одного LTF-бара всё равно не знаем порядок → консервативно SL
            return (False, ts, float(stop_eval), 'sl')

    # если до закрытия 4h так ничего и не зацепили — пусть решит внешний код (обычно timeout_last_close)
    return (False, t1, float(stop_eval), 'uncertain')

# ===================== Основной пайплайн =====================
def evaluate_signals(
    signals_path: str,
    result_path: str,
    lookback_days: int = 360,
    interval: str = '4h',
    max_days: int = None,
    include_open: bool = False,   # не используем — оценка исхода внутри TTL
    compounding: bool = True,     # для совместимости
    initial_capital: float = None,
    capital_aware: bool = True,   # включено по умолчанию (реалистично)
):
    # as_of — по mtime файла
    try:
        mtime = os.path.getmtime(signals_path)
        as_of = datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        as_of = datetime.now(tz=timezone.utc)

    # 1) читаем сигналы
    head = pd.read_excel(signals_path, nrows=0)
    parse_dates = [c for c in ['imb_time', 'entry_at'] if c in head.columns]
    df_sig = pd.read_excel(signals_path, parse_dates=parse_dates)
    if df_sig.empty:
        print("⚠️ Нет сигналов для оценки — выходим.")
        return

    # нормализация
    for c in ['entry', 'stop', 'tp', 'strength']:
        if c in df_sig.columns:
            df_sig[c] = df_sig[c].map(_to_num)

    if 'filled' in df_sig.columns:
        df_sig['filled'] = df_sig['filled'].map(_to_bool_filled)

    if 'imb_time' in df_sig.columns:
        df_sig['imb_time'] = df_sig['imb_time'].map(_to_utc_safe)
        df_sig = df_sig[df_sig['imb_time'].notna()]

    if df_sig.empty:
        print("⚠️ После базовой фильтрации сигналов не осталось.")
        return

    max_fill_days_used = MAX_FILL_DAYS if max_days is None else int(max_days)

    # 2) данные цен
    results = []
    price_cache = {}
    symbols = df_sig['symbol'].dropna().unique().tolist()
    for symbol in symbols:
        df_hist = get_bybit_klines(symbol=symbol, interval=interval, lookback_days=lookback_days)
        df_hist = _ensure_dt_index(df_hist)
        price_cache[symbol] = df_hist

    variant_col = _find_variant_col(df_sig)

    # 3) оценка каждого сигнала (entry detection + исход)
    for _, row in df_sig.iterrows():
        symbol   = row['symbol']
        t0       = _to_utc_safe(row['imb_time'])      # время имбаланса (UTC-safe)
        side     = str(row['type']).upper().strip()
        entry    = _to_num(row['entry'])
        df       = price_cache.get(symbol)
        variant  = _row_variant(row, variant_col)

        if df is None or df.empty or pd.isna(entry) or side not in ('BUY', 'SELL'):
            continue

        # --- определяем entry_at, stop_eval, tp_eval (как в «бою») ---
        entry_at = None
        entry_px = None

        if ENTRY_MODE == "BREAKOUT":
            # вход на закрытии бара t0
            bar = df[df.index == t0]
            if bar.empty:
                # nearest
                try:
                    nearest_idx = df.index.get_indexer([t0], method="nearest")[0]
                    bar = df.iloc[[nearest_idx]]
                except Exception:
                    continue
            entry_px = float(bar.iloc[0]['close'])
            stop_eval, tp_eval = _calc_sl_tp(entry_px, side, STOP_PCT, TAKE_PCT/STOP_PCT)
            entry_at = t0

        elif ENTRY_MODE == "MOMENTUM":
            # упрощённо — вход в момент t0 по entry из файла
            entry_px = float(entry)
            stop_eval, tp_eval = _calc_sl_tp(entry_px, side, STOP_PCT, TAKE_PCT/STOP_PCT)
            entry_at = t0

        else:
            # RETEST: ждём первого касания entry ПОСЛЕ бара t0 в пределах TTL
            entry_px = float(entry)
            t0_utc = t0
            first_touch = _first_touch_after(df, entry_px, t0_utc)
            if pd.notna(first_touch) and first_touch <= (t0_utc + pd.Timedelta(days=DEFAULT_TTL_DAYS)):
                entry_at = first_touch
            else:
                entry_at = pd.NaT  # не исполнилось в TTL
            stop_eval, tp_eval = _calc_sl_tp(entry_px, side, STOP_PCT, TAKE_PCT/STOP_PCT)

        # sanity-fix
        stop_eval, tp_eval, _ = _repair_levels(side, float(entry_px), float(stop_eval), float(tp_eval))

        # --- окно оценки исхода (TP/SL/timeout) ---
        t_start = entry_at
        t0_utc = t0
        window_end = min(t0_utc + pd.Timedelta(days=DEFAULT_TTL_DAYS), as_of)

        if pd.notna(t_start):
            window = df[(df.index > t_start) & (df.index <= window_end)]
        else:
            window = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

        win = False
        close_time = None
        close_price = None
        exit_reason = None

        if pd.notna(entry_at) and not window.empty:
            last_checked = t_start  # будем шагать бар за баром
            for ts, c in window.iterrows():
                hi, lo = float(c['high']), float(c['low'])

                if side == 'BUY':
                    hit_tp = (hi >= tp_eval)
                    hit_sl = (lo <= stop_eval)
                else:
                    hit_tp = (lo <= tp_eval)
                    hit_sl = (hi >= stop_eval)

                if hit_tp and hit_sl:
                    # спорный 4h-бар → уходим в LTF и выясняем порядок
                    w, ct, cp, er = _resolve_tp_sl_order_ltf(
                        symbol=symbol,
                        side=side,
                        entry_at=last_checked,
                        bar_close_time=ts,
                        stop_eval=float(stop_eval),
                        tp_eval=float(tp_eval),
                    )
                    if er in ('tp', 'sl'):
                        win = w
                        close_time = ct
                        close_price = cp
                        exit_reason = er
                        break
                    else:
                        # LTF не дал однозначного ответа (не зацепили ни то ни это) — двигаемся дальше
                        last_checked = ts
                        continue

                # обычные (не спорные) случаи:
                if hit_tp:
                    win = True;
                    close_time = ts;
                    close_price = tp_eval;
                    exit_reason = 'tp';
                    break
                if hit_sl:
                    win = False;
                    close_time = ts;
                    close_price = stop_eval;
                    exit_reason = 'sl';
                    break

                # если в баре ничего не случилось — просто идём дальше
                last_checked = ts
            if close_time is None:
                close_time = window.index[-1]
                close_price = float(window.iloc[-1]['close'])
                exit_reason = 'timeout_last_close'
        else:
            # вход не состоялся в TTL
            close_time = t0_utc + pd.Timedelta(days=DEFAULT_TTL_DAYS)
            close_price = float('nan')
            exit_reason = 'timeout_no_fill'

        # --- диагностический move/pnl в % (по цене) ---
        fee_in  = float(FEE_TAKER)
        fee_out = float(FEE_TAKER)
        if pd.notna(entry_at) and pd.notna(close_time) and not math.isnan(entry_px) and not (isinstance(close_price, float) and math.isnan(close_price)):
            if side == 'BUY':
                move_pct = (float(close_price) - float(entry_px)) / float(entry_px) * 100.0
            else:
                move_pct = (float(entry_px) - float(close_price)) / float(entry_px) * 100.0
            fees_slip_pct = (fee_in + fee_out) * 100.0 + (2.0 * float(SLIPPAGE_PCT) * 100.0)
            pnl_pct_price = float(move_pct) - float(fees_slip_pct)
        else:
            move_pct = float('nan')
            pnl_pct_price = float('nan')

        out = row.to_dict()
        out.update({
            'variant':     variant,
            'as_of':       as_of,
            'stop_eval':   float(stop_eval),
            'tp_eval':     float(tp_eval),
            'win':         True if exit_reason == 'tp' else (False if exit_reason in ('sl','timeout_last_close') else False),
            'risk_pct':    1.0,   # инфо поле (не используется в $PnL)
            'profit_pct':  3.0,
            'move_pct':    move_pct,
            'pnl_pct':     pnl_pct_price,    # информативно, $PnL считаем в симуляции
            'pnl_usd':     pd.NA,
            'close_time':  close_time,
            'close_price': float(close_price) if close_price is not None and not (isinstance(close_price, float) and math.isnan(close_price)) else pd.NA,
            'exit_reason': exit_reason if exit_reason is not None else 'unknown',
            'is_open_mark': False,
            't_start':     entry_at,         # именно момент входа (NaT, если не исполнилось)
        })
        results.append(out)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("⚠️ После оценки сделок нет данных.")
        return

    # 5) метрики времени
    for c in ['close_time','imb_time','t_start']:
        if c in df_res.columns:
            df_res[c] = df_res[c].map(_to_utc_safe)
    df_res['exit_time'] = df_res['close_time']
    df_res['exit_days'] = ((df_res['exit_time'] - df_res['t_start']) / pd.Timedelta(days=1)).round(3)

    # 6) Симуляция капитала (реалистичная)
    init_cap = float(initial_capital) if initial_capital is not None else float(INITIAL_CAPITAL or 0.0)
    eq_sheet = pd.DataFrame()
    if init_cap <= 0:
        print("⚠️ INITIAL_CAPITAL <= 0 — симуляция будет пропущена.")
        df_out = df_res.copy()
        df_out['skipped'] = False
    elif capital_aware:
        df_out, eq_sheet = _simulate_capital_notional(
            df_res,
            init_cap,
            position_fraction=POSITION_FRACTION,
            stop_pct=STOP_PCT,
            take_pct=TAKE_PCT,
        )
    else:
        df_out = df_res.copy()
        df_out['skipped'] = False

    # 7) сводки (только исполненные)
    df_exec = df_out[df_out['skipped'] == False].copy()
    try:
        by_variant = (
            df_exec.groupby('variant')
                  .agg(trades=('win','size'),
                       wins=('win','sum'),
                       winrate_pct=('win', lambda s: round(100.0*float(s.sum())/max(int(s.size),1),2)),
                       pnl_pct=('pnl_pct','sum'),
                       pnl_usd=('pnl_usd','sum'))
                  .reset_index()
                  .sort_values(['pnl_usd','winrate_pct'], ascending=[False, False])
        )
    except Exception:
        by_variant = pd.DataFrame()

    by_exit_reason = _safe_group_exit_reason(df_out)

    # equity summary
    equity_summary = pd.DataFrame()
    if not eq_sheet.empty:
        start_eq = float(eq_sheet['equity_before'].iloc[0])
        end_eq   = float(eq_sheet['equity_after'].iloc[-1])
        total_ret_pct = (end_eq / start_eq - 1.0) * 100.0 if start_eq > 0 else 0.0
        equity_summary = pd.DataFrame({
            'metric': ['start_equity','end_equity','total_return_pct','closed_trades'],
            'value':  [round(start_eq,2), round(end_eq,2), round(total_ret_pct,2), int(len(eq_sheet))]
        })

    # 8) вывод в Excel — снимаем tz для совместимости с Excel
    for col in ['imb_time', 'close_time', 'exit_time', 'as_of', 't_start']:
        if col in df_out.columns:
            ser = pd.to_datetime(df_out[col], errors='coerce')
            if getattr(ser.dt, 'tz', None) is not None:
                df_out[col] = ser.dt.tz_convert(None)
            else:
                df_out[col] = ser
    if not eq_sheet.empty:
        ser = pd.to_datetime(eq_sheet['time'], errors='coerce')
        if getattr(ser.dt, 'tz', None) is not None:
            eq_sheet['time'] = ser.dt.tz_convert(None)
        else:
            eq_sheet['time'] = ser

    if 'skipped' in df_out.columns:
        for c in ('move_pct','pnl_pct','pnl_usd','alloc_usd_comp','pnl_usd_comp','equity_after'):
            if c in df_out.columns:
                df_out.loc[df_out['skipped'] == True, c] = pd.NA

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)

    try:
        with pd.ExcelWriter(result_path, engine='xlsxwriter') as wr:
            df_out.to_excel(wr, sheet_name='results', index=False)
            if not by_exit_reason.empty:
                by_exit_reason.to_excel(wr, sheet_name='by_exit_reason', index=False)
            if not by_variant.empty:
                by_variant.to_excel(wr, sheet_name='by_variant', index=False)
            if not eq_sheet.empty:
                eq_sheet.to_excel(wr, sheet_name='equity_curve', index=False)
            if not equity_summary.empty:
                equity_summary.to_excel(wr, sheet_name='equity_summary', index=False)
        print(f"✅ Результаты сохранены в {result_path}")
    except ModuleNotFoundError:
        csv_fallback = os.path.splitext(result_path)[0] + ".csv"
        df_out.to_csv(csv_fallback, index=False)
        print(f"💾 Сохранил в CSV: {csv_fallback}")


def _default_reports_dir() -> str:
    return os.path.expanduser("~/Documents/отчеты")

def _derive_default_result_path(signals_path: str) -> str:
    reports_dir = _default_reports_dir()
    base = os.path.splitext(os.path.basename(signals_path))[0]
    out_name = f"{base}_eval.xlsx"
    return os.path.join(reports_dir, out_name)


if __name__ == "__main__":
    import argparse

    def _str2bool(v: str) -> bool:
        return str(v).strip().lower() in ("1", "true", "yes", "y", "t", "on")

    p = argparse.ArgumentParser(
        description="Evaluate imbalance signals with capital-aware simulation and intrabar TP/SL resolution."
    )
    p.add_argument("signals", help="Путь к входному Excel с сигналами (лист/структура как раньше).")
    p.add_argument("--out", default=None,
                   help="Путь к результату (.xlsx). По умолчанию: рядом с исходником, *_eval.xlsx.")
    p.add_argument("--lookback-days", type=int, default=360,
                   help="Сколько дней истории тянуть для базового таймфрейма (default: 360).")
    p.add_argument("--ttl-days", type=int, default=None,
                   help="TTL лимитки/оценки в днях; если не задан, берётся из MAX_FILL_DAYS/DEFAULT_TTL_DAYS.")
    p.add_argument("--include-open", type=_str2bool, default=False,
                   help="Учитывать ли незакрытые сделки (0/1). В отчёте мы всё равно закрываем в пределах TTL.")
    p.add_argument("--interval", default="4h",
                   help="Базовый интервал для оценки исходов (default: 4h).")
    p.add_argument("--compounding", type=_str2bool, default=True,
                   help="Флаг совместимости (0/1), на $PnL не влияет (по умолчанию 1).")
    p.add_argument("--initial-capital", type=float, default=None,
                   help="Стартовый капитал в USD. Если не задан — берётся из INITIAL_CAPITAL.")
    p.add_argument("--capital-aware", type=_str2bool, default=True,
                   help="Включить капитал-гейт/пропуски из-за нехватки кэша (0/1, default: 1).")

    # Для удобства — можно пробросить LTF настройки прямо флагами (они пишутся в ENV на время запуска)
    p.add_argument("--intrabar", default=None,
                   help='Список LTF интервалов через запятую (например: "1m,5m"). Если не задан — берётся из ENV INTRABAR_INTERVALS.')
    p.add_argument("--intrabar-lookback-days", type=int, default=None,
                   help="Сколько дней тянуть для LTF при спорном баре (override ENV INTRABAR_LOOKBACK_DAYS_FALLBACK).")

    args = p.parse_args()

    sig_path = os.path.expanduser(args.signals)
    if args.out:
        res_path = os.path.expanduser(args.out)
    else:
        # как и раньше: ~/Documents/отчеты/<basename>_eval.xlsx
        res_path = _derive_default_result_path(sig_path)
        os.makedirs(os.path.dirname(res_path) or ".", exist_ok=True)

    # опционально прокинем LTF настройки в ENV для текущего процесса
    if args.intrabar:
        os.environ["INTRABAR_INTERVALS"] = args.intrabar
    if args.intrabar_lookback_days is not None:
        os.environ["INTRABAR_LOOKBACK_DAYS_FALLBACK"] = str(int(args.intrabar_lookback_days))

    # TTL (max_days) — если не передан, оставляем None, и внутри возьмётся дефолт
    max_days_arg = int(args.ttl_days) if args.ttl_days is not None else None

    evaluate_signals(
        signals_path=sig_path,
        result_path=res_path,
        lookback_days=int(args.lookback_days),
        interval=str(args.interval),
        max_days=max_days_arg,
        include_open=bool(args.include_open),
        compounding=bool(args.compounding),
        initial_capital=(float(args.initial_capital) if args.initial_capital is not None else None),
        capital_aware=bool(args.capital_aware),
    )