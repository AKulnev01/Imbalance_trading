# utils/strategy.py

import os

# robust bool parser for ENV
def _b(v):
    return str(v).strip().lower() in ('1','true','yes','y','on')
from pathlib import Path
import pandas as pd
import re
import numpy as np

from utils.detect_fvg_close import detect_fvg_imbalances_close as detect_fvg_imbalances
from utils.ta import SMA, RSI, ATR, is_bull_div, is_bear_div  # пусть лежат — не мешают
from config import (
    RISK_REWARD_RATIO,
    ENABLE_BUY, ENABLE_SELL,
    DEFAULT_MIN_STRENGTH, BUY_EXTRA_STRENGTH_PCT,
    USE_INTRAMINUTE_ENTRY, INTRAMIN_LOOKBACK_MIN,
    ENTRY_LOOKAHEAD_MINUTES, INTRAM_VOLUME_MULT,
    SLIPPAGE_PCT,
    BYBIT_CATEGORY,
)

# === локальные источники данных ===
USE_LOCAL_MINUTES = _b(os.getenv("USE_LOCAL_MINUTES", "1"))     # читать минутки <SYMBOL>_m1.parquet
LTF_ROOT          = os.getenv("LTF_ROOT", "./data/m1")                  # ПАПКА минуток
USE_LOCAL_4H      = _b(os.getenv("USE_LOCAL_4H", '0'))          # читать готовые 4h parquet
OHLCV_4H_ROOT     = os.getenv("OHLCV_4H_ROOT", "./data/agg_4h")        # ПАПКА 4h
OHLCV_ROOT = LTF_ROOT
SMART_QUOTES = '“”„‟«»‚‘’'

# ---------- time helpers ----------
def _normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает индекс DatetimeIndex UTC из одной из колонок времени:
    'time' | 'timestamp' | 'open_time' | 'datetime' | 'date' | 'ts' | уже индекс.
    Поддерживает числа-эпоху (сек/мс). Возвращает df с отсортированным индексом.
    """
    if df is None or df.empty:
        return df

    candidates = ["time", "timestamp", "open_time", "datetime", "date", "ts"]
    time_col = next((c for c in candidates if c in df.columns), None)

    if time_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
            return out.sort_index()
        raise ValueError("Не нашёл колонку времени (ожидал одну из: time/timestamp/open_time/datetime/date/ts).")

    t = df[time_col]
    if np.issubdtype(t.dtype, np.number):
        mx = float(t.max()) if len(t) else 0.0
        dt = pd.to_datetime(t, unit=("ms" if mx > 1e12 else "s"), utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(t, utc=True, errors="coerce")

    out = df.copy()
    out = out.assign(__t=dt).dropna(subset=["__t"]).set_index("__t").sort_index()
    out.index.name = "time"
    return out

def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Числовые типы, индексы → UTC, без пустых баров."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ("open","high","low","close","volume","turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df[~df.index.isna()]
    return df

def _clean_symbol(s: str) -> str:
    s = (s or "").strip()
    s = s.replace('"','').replace("'",'')
    for q in SMART_QUOTES:
        s = s.replace(q, '')
    s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
    return s.upper()

def _m1_close_at(symbol: str, t_close_utc: pd.Timestamp) -> float:
    """
    Возвращает close 1m-барa, чей timestamp == t_close_utc.
    Источник минуток — как и раньше: локальные m1 (LTF_ROOT) либо сеть.
    """
    if USE_LOCAL_MINUTES:
        from pathlib import Path
        sym = str(symbol).upper().strip()
        p = Path(LTF_ROOT) / f"{sym}_m1.parquet"
        if not p.exists():
            p_alt = p.with_name(f"{sym}.parquet")
            if not p_alt.exists():
                return None
            p = p_alt
        df1m = pd.read_parquet(p)
        df1m = _normalize_time_index(df1m)
    else:
        from utils.fetch_data import get_bybit_klines
        df1m = get_bybit_klines(symbol=symbol, interval="1m", lookback_days=1, category=BYBIT_CATEGORY)

    df1m = _sanitize_ohlcv(df1m)
    if df1m is None or df1m.empty:
        return None

    # точное совпадение по минуте; если вдруг нет — nearest назад/вперёд
    if t_close_utc in df1m.index:
        return float(df1m.loc[t_close_utc, "close"])
    pos = df1m.index.get_indexer([t_close_utc], method="nearest")[0]
    return float(df1m.iloc[int(pos)]["close"])

# ---------- loaders ----------
def _load_4h_from_local_minutes(symbol: str) -> pd.DataFrame:
    """
    Читает ./data/m1/<SYMBOL>_m1.parquet (или <SYMBOL>.parquet), ресемплит в 4h.
    Ожидаемые колонки цен и (опц.) volume.
    """
    sym = str(symbol).upper().strip()
    p = Path(LTF_ROOT) / f"{sym}_m1.parquet"
    if not p.exists():
        p_alt = p.with_name(f"{sym}.parquet")
        if not p_alt.exists():
            raise FileNotFoundError(f"no local minutes for {symbol}: {p}")
        p = p_alt

    dfm = pd.read_parquet(p)
    dfm = _normalize_time_index(dfm)

    o = dfm["open"].resample("4h", label="left", closed="left").first()
    h = dfm["high"].resample("4h").max()
    l = dfm["low"].resample("4h").min()
    c = dfm["close"].resample("4h").last()
    df4h = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    if "volume" in dfm.columns:
        df4h["volume"] = dfm["volume"].resample("4h").sum()

    df4h = df4h.dropna()
    # Excel-friendly: сделаем индекс naive, подразумевая UTC
    df4h.index = df4h.index.tz_convert(None)
    df4h.index.name = "time"
    return df4h

def _load_4h_from_local_4h(symbol: str, lookback_days: int = 360) -> pd.DataFrame:
    path = os.path.join(OHLCV_4H_ROOT, f"{symbol}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no local 4h parquet for {symbol}: {path}")

    df = pd.read_parquet(path)
    df = _normalize_time_index(df)

    if lookback_days and lookback_days > 0 and not df.empty:
        cutoff = (df.index.max() - pd.Timedelta(days=int(lookback_days)))
        df = df[df.index >= cutoff]

    for c in ("open","high","low","close","volume","turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])

    df.index = df.index.tz_convert(None)
    df.index.name = "time"
    return df

def get_klines_4h(symbol: str, lookback_days: int = 360, interval: str = "4h"):
    if interval.lower() not in ("4h", "4hr", "240m"):
        raise ValueError(f"interval '{interval}' сейчас поддержан только для 4h")

    # приоритет: локальные минутки → 4h
    if USE_LOCAL_MINUTES:
        return _load_4h_from_local_minutes(symbol)

    # иначе — если есть готовые 4h — читаем их
    if USE_LOCAL_4H:
        return _load_4h_from_local_4h(symbol, lookback_days=lookback_days)

    # последний фоллбек — сеть (не обязателен)
    from utils.fetch_data import get_bybit_klines
    return get_bybit_klines(symbol, interval="4h", lookback_days=lookback_days)

# ---------- universe filter ----------
def filter_universe_to_local(universe):
    """
    Отфильтровывает символы, для которых локальных файлов нет.
      - если USE_LOCAL_MINUTES=1 → ищем <LTF_ROOT>/<SYMBOL>_m1.parquet (или <SYMBOL>.parquet)
      - elif USE_LOCAL_4H=1      → ищем <OHLCV_4H_ROOT>/<SYMBOL>.parquet
    """
    if USE_LOCAL_MINUTES:
        root = Path(LTF_ROOT)
        have = set()
        for p in root.glob("*.parquet"):
            name = p.stem.upper()  # EPTUSDT_m1
            have.add(name[:-3] if name.endswith("_M1") else name)
    elif USE_LOCAL_4H:
        root = Path(OHLCV_4H_ROOT)
        have = {p.stem.upper() for p in root.glob("*.parquet")}
    else:
        return universe

    uni = [_clean_symbol(s) for s in universe]
    kept = [s for s in uni if s in have]
    dropped = sorted(set(uni) - set(kept))
    if dropped:
        print(f"⚠️ пропускаю без локальных данных: {', '.join(dropped[:10])}" + (" …" if len(dropped) > 10 else ""))
    return kept

# ---------- entry helpers ----------
def _get_fvg_level(imb: dict) -> float:
    lvl = imb.get("price")
    if lvl is not None:
        try:
            return float(lvl)
        except Exception:
            pass
    return float(imb["low2"]) if imb["type"] == "BUY" else float(imb["high2"])

def _try_intramin_entry(symbol: str, imb: dict) -> float:
    """
    Подтянуть вход на 1m в окне [imb.time - lookback; imb.time + ENTRY_LOOKAHEAD_MINUTES]
    с проверкой всплеска объёма. Источник: LTF_ROOT/<SYMBOL>_m1.parquet.
    """
    try:
        t0   = pd.to_datetime(imb["time"], utc=True)
        side = imb["type"]
        lvl  = _get_fvg_level(imb)

        if USE_LOCAL_MINUTES:
            sym = str(symbol).upper().strip()
            p = Path(LTF_ROOT) / f"{sym}_m1.parquet"
            if not p.exists():
                p_alt = p.with_name(f"{sym}.parquet")
                if not p_alt.exists():
                    raise FileNotFoundError(f"no local minutes for {symbol}: {p}")
                p = p_alt
            df1m = pd.read_parquet(p)
            df1m = _normalize_time_index(df1m)
        else:
            from utils.fetch_data import get_bybit_klines
            df1m = get_bybit_klines(symbol=symbol, interval="1m", lookback_days=1, category=BYBIT_CATEGORY)

        df1m = _sanitize_ohlcv(df1m)
        if df1m is None or df1m.empty:
            return None

        before = t0 - pd.Timedelta(minutes=INTRAMIN_LOOKBACK_MIN)
        after  = t0 + pd.Timedelta(minutes=ENTRY_LOOKAHEAD_MINUTES)
        win = df1m[(df1m.index >= before) & (df1m.index <= after)].copy()
        if win.empty:
            return None

        if "volume" not in win.columns:
            win["volume"] = 0.0

        win["vol_sma20"] = win["volume"].rolling(20, min_periods=1).mean()
        corridor = win[(win.index >= t0) & (win.index <= after)]
        if corridor.empty:
            return None

        for _, c in corridor.iterrows():
            base   = c["vol_sma20"] if pd.notna(c["vol_sma20"]) else c["volume"]
            vol_ok = c["volume"] >= INTRAM_VOLUME_MULT * max(base, 1e-9)
            if side == "BUY":
                if c["low"] <= lvl and vol_ok:
                    return float(lvl * (1 + SLIPPAGE_PCT))
            else:
                if c["high"] >= lvl and vol_ok:
                    return float(lvl * (1 - SLIPPAGE_PCT))
        return None
    except Exception:
        return None

def atr(series_h, series_l, series_c, n=14):
    hl = (series_h - series_l).abs()
    hc = (series_h - series_c.shift(1)).abs()
    lc = (series_l - series_c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def momentum_signal_on_close(
    df: pd.DataFrame,
    atr_n: int = 14,
    body_atr: float = 1.5,
    range_atr: float = 2.0,
    vol_sma_k: float = 2.0,
):
    if df is None or df.empty or len(df) < max(atr_n, 5):
        return None, {}
    a = atr(df["high"], df["low"], df["close"], n=atr_n)
    vol_sma = df["volume"].rolling(atr_n, min_periods=1).mean()
    c = df.iloc[-1]
    rng  = float(c["high"] - c["low"])
    body = float(abs(c["close"] - c["open"]))
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0
    vol  = float(c["volume"])
    vavg = float(vol_sma.iloc[-1]) if pd.notna(vol_sma.iloc[-1]) else vol
    ok_range = (atrv > 0) and (rng >= range_atr * atrv or body >= body_atr * atrv)
    ok_vol   = vol >= vol_sma_k * max(vavg, 1e-9)
    if ok_range and ok_vol:
        side = "BUY" if float(c["close"]) > float(c["open"]) else "SELL"
        return side, {"body": body, "range": rng, "atr": atrv, "vol": vol, "vavg": vavg}
    return None, {"body": body, "range": rng, "atr": atrv, "vol": vol, "vavg": vavg}

def select_entry_price(
    df: pd.DataFrame,
    symbol: str,
    imb: dict,
    volume_mult: float = 1.2,
    avg_window: int = 20,
) -> float:
    """
    ВАЖНО: для согласования с evaluate_momentum:
    entry := CLOSE 4h-свечи ровно в момент imb_time (time = close-время бара).
    Никаких интраминутных подтяжек и уровней FVG — чисто close[T].
    """
    df = _sanitize_ohlcv(df)
    if df is None or df.empty:
        return None

    t0 = pd.to_datetime(imb["time"], utc=True, errors="coerce")
    if pd.isna(t0):
        return None

    bar_open = t0 - pd.Timedelta(hours=4)
    if bar_open in df.index:
        return float(df.loc[bar_open, "close"])

    try:
        loc = df.index.searchsorted(bar_open, side="right") - 1
        if 0 <= loc < len(df):
            return float(df.iloc[loc]["close"])
    except Exception:
        pass
    return None

# ---------- main scan ----------
def scan_universe(
    universe: list = None,
    lookback_days: int = 100,
    mode: str = "live",
    interval: str = "4h",
) -> pd.DataFrame:
    """
    mode:
      - "live" — входы по touched;
      - "all"  — ВСЕ FVG по силе, без filled/touched.
    Источник 4h — см. get_klines_4h().
    """
    if not universe:
        from config import TRADE_UNIVERSE
        from utils.symbols import fetch_top_symbols
        universe = TRADE_UNIVERSE or fetch_top_symbols()

    # если работаем от локальных файлов — заранее отфильтруем отсутствующие
    if USE_LOCAL_MINUTES or USE_LOCAL_4H:
        universe = filter_universe_to_local(universe)

    label = "LIVE: без max_days_to_fill" if mode == "live" else "ALL: все свежие FVG, без filled"
    print(f"🔄 scan_universe: начинаем сканирование {len(universe)} символов за {lookback_days} дней ({label}, TF={interval})")

    rows = []
    for symbol in universe:
        try:
            print(f"⏳ Обрабатываем {symbol}…")
            dfx = get_klines_4h(symbol=symbol, lookback_days=lookback_days, interval=interval)
            dfx = _sanitize_ohlcv(dfx)
            if dfx is None or dfx.empty:
                print(f"   → {symbol}: пустые данные {interval}, пропуск.")
                continue

            imbs_raw = detect_fvg_imbalances(
                dfx,
                volume_multiplier=float(os.getenv("VOLUME_MULTIPLIER", 1.5)),
                tolerance_pct=float(os.getenv("TOLERANCE_PCT", 0.0)),
                min_strength_pct=float(os.getenv("MIN_STRENGTH_PCT", DEFAULT_MIN_STRENGTH)),
                max_days_to_fill=int(os.getenv("MAX_FILL_DAYS", 30)),
            )
            print(f"   → {symbol}: найдено imbalances = {len(imbs_raw)}")

            if mode == "all":
                imbs = [
                    imb for imb in imbs_raw
                    if imb.get("type") in ("BUY","SELL")
                    and pd.notna(imb.get("strength"))
                    and float(imb["strength"]) >= float(DEFAULT_MIN_STRENGTH)
                ]
                print(f"   → {symbol}: ALL (без filled/touched) = {len(imbs)}")

                added = 0
                for imb in imbs:
                    side = imb["type"]
                    if side == "BUY" and not ENABLE_BUY:  continue
                    if side == "SELL" and not ENABLE_SELL: continue

                    entry = select_entry_price(dfx, symbol, imb)
                    if entry is None:
                        continue

                    if side == "BUY":
                        stop_zone = float(imb["low2"]) * 0.998
                        stop = min(stop_zone, float(entry) * 0.999)
                        tp   = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop_zone = float(imb["high2"]) * 1.002
                        stop = max(stop_zone, float(entry) * 1.001)
                        tp   = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)

                    rows.append({
                        "symbol": symbol,
                        "side": side,  # <— ВАЖНО: было "type"
                        "strength": float(imb["strength"]),
                        "imb_time": pd.to_datetime(imb["time"], utc=True),
                        "entry": float(entry),
                        "stop": float(stop),
                        "tp": float(tp),
                        "touched": bool(imb.get("touched", False)),
                    })
                    added += 1
                print(f"   → {symbol}: добавлено сигналов = {added}")

            else:
                imbs = [
                    imb for imb in imbs_raw
                    if bool(imb.get("touched")) and imb.get("type") in ("BUY","SELL")
                ]
                print(f"   → {symbol}: LIVE (touched) = {len(imbs)}")

                added = 0
                for imb in imbs:
                    side = imb["type"]
                    if side == "BUY" and not ENABLE_BUY:  continue
                    if side == "SELL" and not ENABLE_SELL: continue

                    entry = select_entry_price(dfx, symbol, imb)
                    if entry is None:
                        continue

                    if side == "BUY":
                        stop_zone = float(imb["low2"]) * 0.998
                        stop = min(stop_zone, float(entry) * 0.999)
                        tp   = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop_zone = float(imb["high2"]) * 1.002
                        stop = max(stop_zone, float(entry) * 1.001)
                        tp   = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)

                    rows.append({
                        "symbol":   symbol,
                        "type":     side,
                        "strength": float(imb["strength"]),
                        "imb_time": pd.to_datetime(imb["time"], utc=True),
                        "entry":    float(entry),
                        "stop":     float(stop),
                        "tp":       float(tp),
                        "touched":  True,
                    })
                    added += 1
                print(f"   → {symbol}: добавлено сигналов = {added}")

        except Exception as e:
            print(f"⚠️ Ошибка по {symbol}: {e}")

    df_signals = pd.DataFrame(rows)
    print(f"🔍 scan_universe: найдено сигналов = {len(df_signals)}")
    return df_signals