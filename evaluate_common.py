# evaluate_common.py
import os
import math
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, List, Dict

# === проектные зависимости ===
import config as CFG
from utils.fetch_data import get_bybit_klines, get_bybit_klines_range

# ---------------------------------------------------------------------
# Конфиг-хелперы
# ---------------------------------------------------------------------
def get_cfg(name, *, required=True, cast=None, default=None):
    """
    Берём значение из config.py, затем из ENV. Поддерживаем приведение типа.
    """
    val = getattr(CFG, name, None)
    if val is None:
        val = os.getenv(name, None)
    if val is None:
        if required and default is None:
            raise RuntimeError(f"Missing required setting '{name}' in config.py or .env.")
        val = default
    if cast is not None and val is not None:
        try:
            if cast is bool:
                s = str(val).strip().lower()
                return s in ("1", "true", "yes", "y", "on")
            if cast is list:
                return [x.strip() for x in str(val).split(",") if x.strip()]
            return cast(val)
        except Exception:
            raise RuntimeError(f"Bad value for '{name}': {val!r} (expected {cast.__name__}).")
    return val

# ---------------------------------------------------------------------
# Глобальные параметры (общие)
# ---------------------------------------------------------------------
INITIAL_CAPITAL   = float(get_cfg("INITIAL_CAPITAL",   cast=float))
POSITION_FRACTION = float(get_cfg("POSITION_FRACTION", cast=float))
FEE_TAKER         = float(get_cfg("FEE_TAKER",         cast=float))
SLIPPAGE_PCT      = float(get_cfg("SLIPPAGE_PCT",      cast=float))
DEFAULT_TTL_DAYS  = int(get_cfg("DEFAULT_TTL_DAYS",    cast=int))

INTRABAR_INTERVALS              = get_cfg("INTRABAR_INTERVALS", cast=list) or []
INTRABAR_LOOKBACK_DAYS_FALLBACK = int(get_cfg("INTRABAR_LOOKBACK_DAYS_FALLBACK", cast=int, default=14))
INTRABAR_MAX_LOOKBACK_DAYS      = int(get_cfg("INTRABAR_MAX_LOOKBACK_DAYS",      cast=int, default=720))
MAX_CONCURRENT_POSITIONS        = int(get_cfg("MAX_CONCURRENT_POSITIONS",        cast=int, default=0))  # 0 = без лимита

# (опционально, если где-то нужно минимум 1m-баров)
MOMENTUM_MIN_LTF_BARS           = int(get_cfg("MOMENTUM_MIN_LTF_BARS",       cast=int, default=1))

# ---------------------------------------------------------------------
# Утилиты дат и цен
# ---------------------------------------------------------------------
def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делаем DatetimeIndex(UTC), поддерживаем входные форматы из fetch_data.
    """
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

def to_num(x):
    return pd.to_numeric(str(x).replace(',', '.').strip(), errors='coerce')

def to_utc_safe(ts):
    """
    Любое в pandas.Timestamp(UTC) без смещения — для Bybit/TV это даёт то же «настенное» время.
    """
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors='coerce')
    if t is pd.NaT:
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')

def calc_sl_tp(entry: float, side: str, risk_pct_price: float, rr: float) -> Tuple[float, float]:
    """
    «Честные» уровни: от цены входа, без фокусов.
    SELL: SL = entry*(1+k); TP = entry - (SL-entry)*rr
    BUY : SL = entry*(1-k); TP = entry + (entry-SL)*rr
    """
    k = float(risk_pct_price)
    if side == "SELL":
        sl = entry * (1.0 + k)
        tp = entry - (sl - entry) * rr
    else:
        sl = entry * (1.0 - k)
        tp = entry + (entry - sl) * rr
    return float(sl), float(tp)

# ---------------------------------------------------------------------
# Загрузка LTF/HTF истории
# ---------------------------------------------------------------------
def fetch_ltf_window(symbol: str, t_start: pd.Timestamp, t_end: pd.Timestamp, candidates=None) -> pd.DataFrame:
    """
    Тянем 1m/… строго по диапазону (с маленьким паддингом), возвращаем окно [t_start, t_end].
    Никаких TZ-сдвигов — работаем в UTC (как Bybit).
    """
    t_start = to_utc_safe(t_start); t_end = to_utc_safe(t_end)
    if candidates is None:
        candidates = INTRABAR_INTERVALS or ["1m"]

    pad = pd.Timedelta(minutes=1)
    s = t_start - pad
    e = t_end   + pad

    for iv in [c.strip() for c in candidates if c and c.strip()]:
        try:
            df_ltf = get_bybit_klines_range(
                symbol=symbol, interval=iv,
                start_dt=s.to_pydatetime(), end_dt=e.to_pydatetime()
            )
        except Exception:
            continue
        df_ltf = ensure_dt_index(df_ltf)
        if df_ltf is None or df_ltf.empty:
            continue
        win = df_ltf[(df_ltf.index >= t_start) & (df_ltf.index <= t_end)].copy()
        if not win.empty:
            return win
    # пустая, но с tz, чтобы не ломать down-stream код
    return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))

def load_price_cache(symbols: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """
    История HTF (например, 4h) на каждый символ.
    """
    cache = {}
    for sym in symbols:
        try:
            df_hist = get_bybit_klines(symbol=sym, interval=interval, lookback_days=lookback_days)
        except Exception:
            df_hist = pd.DataFrame()
        cache[sym] = ensure_dt_index(df_hist)
    return cache

# ---------------------------------------------------------------------
# Загрузка исходного файла сигналов
# ---------------------------------------------------------------------
def load_signals(signals_path: str, *, only_filled=False, dedup=False, require_entry: bool = True) -> pd.DataFrame:
    """
    Универсальная загрузка xlsx сигналов:
      - нормализация символов,
      - imb_time → UTC,
      - опциональная фильтрация filled,
      - опциональный dedup по ['symbol','imb_time'].
    """
    head = pd.read_excel(signals_path, nrows=0)
    parse_dates = [c for c in ["imb_time", "entry_at"] if c in head.columns]
    df = pd.read_excel(signals_path, parse_dates=parse_dates)

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df.get("symbol").notna()]

    if "imb_time" in df.columns:
        df["imb_time"] = pd.to_datetime(df["imb_time"], utc=True, errors="coerce")
        df = df[df["imb_time"].notna()]

    if require_entry:
        if "entry" in df.columns:
            df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
            df = df[df["entry"].notna()]
        else:
            return pd.DataFrame()

    if "filled" in df.columns and only_filled:
        df["filled"] = df["filled"].map(lambda x: str(x).strip().lower() in ("1","true","yes","y","да","истина"))
        df = df[df["filled"] == True]

    if dedup and set(["symbol","imb_time"]).issubset(df.columns):
        df = (df
              .sort_values(["symbol","imb_time"])
              .drop_duplicates(subset=["symbol","imb_time"], keep="first")
              .reset_index(drop=True))
    return df

# ---------------------------------------------------------------------
# Пост-обработка результатов
# ---------------------------------------------------------------------
def enforce_one_at_a_time_per_symbol(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Запрет перекрытий ПО СИМВОЛУ. Внутри по символу сортируем по времени,
    но исходный глобальный порядок для капитала будет обеспечен позже.
    """
    if df_res is None or df_res.empty:
        return df_res
    df = df_res.copy()
    for c in ["imb_time", "t_start", "close_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    # локальная сортировка — только для детерминизма внутри символа
    df = df.sort_values(["symbol","imb_time","t_start","close_time"], kind="mergesort").reset_index(drop=True)

    out_rows = []
    last_close_by_sym = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol"))
        t_start = r.get("t_start")
        t_close = r.get("close_time")
        prev_close = last_close_by_sym.get(sym, pd.Timestamp.min.tz_localize("UTC"))

        if pd.isna(t_start):
            if pd.notna(t_close) and t_close > prev_close:
                last_close_by_sym[sym] = t_close
            out_rows.append(r.to_dict()); continue

        if t_start < prev_close:
            rr = r.to_dict()
            rr.update({
                "skipped": True, "exit_reason": "skipped_overlap",
                "pnl_usd": pd.NA, "pnl_pct": pd.NA,
                "alloc_usd_comp": pd.NA, "pnl_usd_comp": pd.NA, "equity_after": pd.NA
            })
            out_rows.append(rr); continue

        if pd.notna(t_close) and t_close > prev_close:
            last_close_by_sym[sym] = t_close
        out_rows.append(r.to_dict())
    return pd.DataFrame(out_rows).reset_index(drop=True)

def simulate_capital_notional(df_res: pd.DataFrame,
                              initial_capital: float,
                              position_fraction: float,
                              stop_pct: float,
                              take_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    «Честная» капитал-симуляция:
      - глобальная хронология: сортируем по t_start, затем close_time (НЕ по символу),
      - открытие позы резервирует кэш; закрытие возвращает,
      - если free_cash < size → сделка помечается skipped_no_capital (реалистично),
      - pnl_usd для 'price' считается по net pnl_pct, который приходит из evaluate_momentum.
    """
    if df_res is None or df_res.empty:
        return df_res, pd.DataFrame()

    df = df_res.copy()
    for c in ("t_start","close_time","imb_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')

    # ключевая правка: ГЛОБАЛЬНЫЙ порядок по времени
    df = df.sort_values(
        ["t_start", "close_time", "symbol"],
        kind="mergesort",
        na_position="last"
    ).reset_index(drop=True)

    equity = float(initial_capital)
    free_cash = equity
    active: List[Dict] = []
    out_rows, eq_rows = [], []
    fees_slip_pos = (float(FEE_TAKER)*2.0) + (float(SLIPPAGE_PCT)*2.0)

    def _close_until(ts):
        nonlocal equity, free_cash, active, eq_rows
        still=[]
        for pos in active:
            ctime = pos["close_time"]
            if pd.notna(ctime) and ctime <= ts:
                pnl = float(pos["pnl_usd"])
                eq_before = equity
                equity += pnl
                free_cash += pos["size"] + pnl
                eq_rows.append({
                    "i": pos["idx"]+1, "time": ctime, "symbol": pos["symbol"],
                    "alloc_usd": round(pos["size"],2), "pnl_usd_comp": round(pnl,2),
                    "equity_before": round(eq_before,2), "equity_after": round(equity,2),
                    "exit_reason": pos["exit_reason"], "size_weight": pos.get("size_weight", pd.NA)
                })
            else:
                still.append(pos)
        active = still

    for i, r in df.iterrows():
        t_start = r.get("t_start")
        t_close = r.get("close_time")
        symbol  = r.get("symbol")

        # закрываем всё, что успело закрыться до старта текущей
        if pd.notna(t_start):
            _close_until(t_start)

        # нечем торговать — пропуск
        if pd.isna(t_start):
            row = r.to_dict(); row.update({
                "skipped": True, "alloc_usd_comp": pd.NA, "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA, "pnl_usd": pd.NA, "pnl_pct": pd.NA
            })
            out_rows.append(row); continue

        # лимит на число одновременных позиций (если задан)
        if MAX_CONCURRENT_POSITIONS and len(active) >= int(MAX_CONCURRENT_POSITIONS):
            row = r.to_dict(); row.update({
                "skipped": True, "exit_reason": "skipped_slots_full",
                "alloc_usd_comp": pd.NA, "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA, "pnl_usd": pd.NA, "pnl_pct": pd.NA
            })
            out_rows.append(row); continue

        # вес позиции
        size_weight = r.get("size_weight")
        try: size_weight = float(size_weight) if pd.notna(size_weight) else 1.0
        except Exception: size_weight = 1.0
        size_weight = max(0.0, min(1.0, size_weight))

        size = max(0.0, float(position_fraction) * float(equity) * size_weight)
        if free_cash < size or size <= 0:
            row = r.to_dict(); row.update({
                "skipped": True, "exit_reason":"skipped_no_capital",
                "alloc_usd_comp": pd.NA, "pnl_usd_comp": pd.NA,
                "equity_after": pd.NA, "pnl_usd": pd.NA, "pnl_pct": pd.NA
            })
            out_rows.append(row); continue

        # резервируем
        free_cash -= size

        # PnL: если 'price' — использовать уже посчитанный net pnl_pct;
        # поддерживаем tp/sl на случай других вариантов
        er = str(r.get("exit_reason") or "").lower()
        price_pnl_pct = r.get("pnl_pct")
        if er == "tp":
            pnl_usd = (float(take_pct) - fees_slip_pos) * size
        elif er == "sl":
            pnl_usd = -(float(stop_pct) + fees_slip_pos) * size
        else:
            try:
                pnl_usd = (float(price_pnl_pct)/100.0) * size if pd.notna(price_pnl_pct) else 0.0
            except Exception:
                pnl_usd = 0.0

        active.append({
            "idx": i, "symbol": symbol, "size": size, "size_weight": size_weight,
            "close_time": t_close if pd.notna(t_close) else t_start,
            "exit_reason": r.get("exit_reason"), "pnl_usd": float(pnl_usd)
        })

        row = r.to_dict(); row.update({
            "skipped": False, "pnl_usd": round(pnl_usd,2),
            "alloc_usd_comp": pd.NA, "pnl_usd_comp": pd.NA, "equity_after": pd.NA
        })
        out_rows.append(row)

    # закрываем хвост
    active.sort(key=lambda x: x["close_time"] if pd.notna(x["close_time"]) else pd.Timestamp.max.tz_localize("UTC"))
    for pos in active:
        pnl = float(pos["pnl_usd"]); eq_before = float(equity)
        equity += pnl; free_cash += pos["size"] + pnl
        eq_rows.append({
            "i": pos["idx"]+1, "time": pos["close_time"], "symbol": pos["symbol"],
            "alloc_usd": round(pos["size"],2), "pnl_usd_comp": round(pnl,2),
            "equity_before": round(eq_before,2), "equity_after": round(equity,2),
            "exit_reason": pos["exit_reason"], "size_weight": pos.get("size_weight", pd.NA)
        })

    df_out = pd.DataFrame(out_rows).reset_index(drop=True)

    # проталкиваем alloc/pnl/equity_after из eq_rows обратно в строки
    for e in eq_rows:
        idx = e["i"] - 1
        if 0 <= idx < len(df_out):
            df_out.at[idx,"alloc_usd_comp"] = e["alloc_usd"]
            df_out.at[idx,"pnl_usd_comp"]   = e["pnl_usd_comp"]
            df_out.at[idx,"equity_after"]   = e["equity_after"]

    if not eq_rows:
        eq_curve = pd.DataFrame(columns=["i","time","symbol","alloc_usd","pnl_usd_comp","equity_before","equity_after","exit_reason","size_weight"])
    else:
        eq_curve = pd.DataFrame(eq_rows).sort_values("time").reset_index(drop=True)

    return df_out, eq_curve

def safe_group_exit_reason(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация метрик по причинам выхода (по «исполненным» сделкам).
    """
    df = df_res.copy()
    if 'skipped' in df.columns:
        df = df[df['skipped'] == False].copy()
    for col, default in [
        ('exit_reason','unknown'), ('win',False), ('pnl_pct',0.0), ('pnl_usd',0.0), ('exit_days', pd.NA)
    ]:
        if col not in df.columns:
            df[col] = default
    df['win'] = df['win'].astype('bool')
    df['pnl_pct'] = pd.to_numeric(df['pnl_pct'], errors='coerce').fillna(0.0)
    df['pnl_usd'] = pd.to_numeric(df['pnl_usd'], errors='coerce').fillna(0.0)
    df['exit_days'] = pd.to_numeric(df['exit_days'], errors='coerce')

    if df.empty:
        return pd.DataFrame(columns=['exit_reason','trades','wins','winrate_pct','pnl_pct','pnl_usd','avg_exit_days','med_exit_days'])
    g = (df.groupby('exit_reason', dropna=False)
           .agg(trades=('win','size'), wins=('win','sum'),
                pnl_pct=('pnl_pct','sum'), pnl_usd=('pnl_usd','sum'),
                avg_exit_days=('exit_days','mean'), med_exit_days=('exit_days','median'))
           .reset_index())
    g['winrate_pct'] = g['wins'].div(g['trades']).fillna(0).astype(float).mul(100).round(2)
    return g.sort_values(['pnl_usd','winrate_pct'], ascending=[False, False])

# ---------------------------------------------------------------------
# Финальная запись в Excel
# ---------------------------------------------------------------------
def finalize_write(result_path: str,
                   df_out: pd.DataFrame,
                   eq_sheet: pd.DataFrame,
                   by_variant: pd.DataFrame,
                   by_exit_reason: pd.DataFrame,
                   extra_sheets: dict = None):
    """
    Пишем Excel:
      - сохраняем «настенное» время (UTC) без конверсий, просто убираем tz,
      - перед записью сортируем results по t_start, close_time, symbol (глобальная хронология!),
      - поддерживаем произвольные доп.листы через extra_sheets.
    """
    # снять tz для Excel (но сами значения времени остаются те же — UTC)
    def _drop_tz_inplace(df: pd.DataFrame, col: str):
        if col in df.columns:
            ser = pd.to_datetime(df[col], errors='coerce', utc=True)
            # просто убираем признак tz, не конвертируя в локаль
            df[col] = ser.dt.tz_convert(None)

    for col in ['imb_time','close_time','as_of','t_start','exit_time']:
        _drop_tz_inplace(df_out, col)

    if not eq_sheet.empty and 'time' in eq_sheet.columns:
        ser = pd.to_datetime(eq_sheet['time'], errors='coerce', utc=True)
        eq_sheet['time'] = ser.dt.tz_convert(None)

    # hide numeric columns for skipped
    if 'skipped' in df_out.columns:
        for c in ('move_pct','pnl_pct','pnl_usd','alloc_usd_comp','pnl_usd_comp','equity_after'):
            if c in df_out.columns:
                df_out.loc[df_out['skipped'] == True, c] = pd.NA

    # важное: глобальная хронология в финальном Excel
    sort_cols = [c for c in ["t_start","close_time","symbol"] if c in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols, kind="mergesort", na_position="last")

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with pd.ExcelWriter(result_path, engine='xlsxwriter') as wr:
        df_out.to_excel(wr, sheet_name='results', index=False)
        if not by_exit_reason.empty:
            by_exit_reason.to_excel(wr, sheet_name='by_exit_reason', index=False)
        if not by_variant.empty:
            by_variant.to_excel(wr, sheet_name='by_variant', index=False)
        if not eq_sheet.empty:
            eq_sheet.to_excel(wr, sheet_name='equity_curve', index=False)
        if extra_sheets:
            for name, df in extra_sheets.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(wr, sheet_name=name[:31], index=False)  # Excel лимит 31