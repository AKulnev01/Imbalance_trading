# utils/strategy.py

import pandas as pd
from utils.fetch_data import get_bybit_klines
from utils.detect_fvg import detect_fvg_imbalances
from utils.ta import SMA, RSI, ATR, is_bull_div, is_bear_div
from config import (
    RISK_REWARD_RATIO,
    ENABLE_BUY, ENABLE_SELL,
    DEFAULT_MIN_STRENGTH, BUY_EXTRA_STRENGTH_PCT,
    USE_INTRAMINUTE_ENTRY, INTRAMIN_LOOKBACK_MIN,
    ENTRY_LOOKAHEAD_MINUTES, INTRAM_VOLUME_MULT,
    SLIPPAGE_PCT,
    BYBIT_CATEGORY,  # ← добавлено: используем ту же category, что и в «бою»
)

def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим данные к числам и убираем пустые бары, чтобы не ловить NoneType."""
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

def _get_fvg_level(imb: dict) -> float:
    lvl = imb.get('price')
    if lvl is not None:
        try:
            return float(lvl)
        except Exception:
            pass
    return float(imb['low2']) if imb['type'] == 'BUY' else float(imb['high2'])

def _try_intramin_entry(symbol: str, imb: dict) -> float:
    try:
        t0 = pd.to_datetime(imb['time'], utc=True)
        side = imb['type']
        lvl = _get_fvg_level(imb)

        # ⬇️ прокинули category=BYBIT_CATEGORY
        df1m = get_bybit_klines(symbol=symbol, interval='1m', lookback_days=1, category=BYBIT_CATEGORY)
        df1m = _sanitize_ohlcv(df1m)
        if df1m is None or df1m.empty:
            return None

        before = t0 - pd.Timedelta(minutes=INTRAMIN_LOOKBACK_MIN)
        after  = t0 + pd.Timedelta(minutes=ENTRY_LOOKAHEAD_MINUTES)
        win = df1m[(df1m.index >= before) & (df1m.index <= after)].copy()
        if win.empty:
            return None

        win['vol_sma20'] = win['volume'].rolling(20, min_periods=1).mean()
        corridor = win[(win.index >= t0) & (win.index <= after)]
        if corridor.empty:
            return None

        for _, c in corridor.iterrows():
            base = c['vol_sma20'] if pd.notna(c['vol_sma20']) else c['volume']
            vol_ok = c['volume'] >= INTRAM_VOLUME_MULT * base
            if side == 'BUY':
                if c['low'] <= lvl and vol_ok:
                    return float(lvl * (1 + SLIPPAGE_PCT))
            else:
                if c['high'] >= lvl and vol_ok:
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

def momentum_signal_on_close(df: pd.DataFrame,
                             atr_n: int = 14,
                             body_atr: float = 1.5,
                             range_atr: float = 2.0,
                             vol_sma_k: float = 2.0):
    """
    На последнем закрытом баре df решаем, есть ли импульс (deep bar).
    Возвращает ('BUY'|'SELL'|None, {'body':..,'range':..,'atr':..,'vol':..,'vavg':..})
    """
    if df is None or df.empty or len(df) < max(atr_n, 5):
        return None, {}

    a = atr(df['high'], df['low'], df['close'], n=atr_n)
    vol_sma = df['volume'].rolling(atr_n, min_periods=1).mean()
    c = df.iloc[-1]
    rng   = float(c['high'] - c['low'])
    body  = float(abs(c['close'] - c['open']))
    atrv  = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else 0.0
    vol   = float(c['volume'])
    vavg  = float(vol_sma.iloc[-1]) if pd.notna(vol_sma.iloc[-1]) else vol

    ok_range = (atrv > 0) and (rng >= range_atr*atrv or body >= body_atr*atrv)
    ok_vol   = (vol >= vol_sma_k*max(vavg, 1e-9))

    if ok_range and ok_vol:
        side = "BUY" if float(c['close']) > float(c['open']) else "SELL"
        return side, {"body":body, "range":rng, "atr":atrv, "vol":vol, "vavg":vavg}
    return None, {"body":body, "range":rng, "atr":atrv, "vol":vol, "vavg":vavg}

def select_entry_price(
    df: pd.DataFrame,
    symbol: str,
    imb: dict,
    volume_mult: float = 1.2,
    avg_window: int = 20
) -> float:
    df = _sanitize_ohlcv(df)
    if df is None or df.empty:
        return None

    if USE_INTRAMINUTE_ENTRY:
        intr = _try_intramin_entry(symbol, imb)
        if intr is not None:
            return float(intr)

    # ближайший бар к времени имба
    t0 = pd.to_datetime(imb['time'], utc=True)
    idx = df.index.get_indexer([t0], method="nearest")
    pos = int(idx[0])

    avg20 = df['volume'].rolling(window=avg_window, min_periods=1).mean()
    base_price = float(imb['low2'] if imb['type'] == 'BUY' else imb['high2'])

    next_pos = pos + 1
    if 0 <= next_pos < len(df):
        nb = df.iloc[next_pos]
        if nb['volume'] >= volume_mult * avg20.iloc[next_pos]:
            return float(nb['low'] if imb['type'] == 'BUY' else nb['high'])

    return float(base_price)

def scan_universe(
    universe: list = None,
    lookback_days: int = 100,
    mode: str = "live",
    interval: str = "4h",
) -> pd.DataFrame:
    """
    mode:
      - "live" — как раньше (входы по 'touched/filled' логике).
      - "all"  — ВСЕ свежие FVG по силе, без требования 'filled'/'touched'.

    interval: "1h" | "2h" | "4h" | "1d"  (или любое, что поддерживает get_bybit_klines)
    """
    if not universe:
        from config import UNIVERSE
        from utils.symbols import fetch_top_symbols
        universe = UNIVERSE or fetch_top_symbols()

    label = "LIVE: без max_days_to_fill" if mode == "live" else "ALL: все свежие FVG, без filled"
    print(f"🔄 scan_universe: начинаем сканирование {len(universe)} символов за {lookback_days} дней ({label}, TF={interval})")

    rows = []

    for symbol in universe:
        try:
            print(f"⏳ Обрабатываем {symbol}…")
            # ⬇️ прокинули category=BYBIT_CATEGORY
            dfx = get_bybit_klines(symbol=symbol, interval=interval, lookback_days=lookback_days, category=BYBIT_CATEGORY)
            dfx = _sanitize_ohlcv(dfx)
            if dfx is None or dfx.empty:
                print(f"   → {symbol}: пустые данные {interval}, пропуск.")
                continue

            imbs_raw = detect_fvg_imbalances(
                dfx,
                volume_multiplier=1.5,
                tolerance_pct=0,
                min_strength_pct=DEFAULT_MIN_STRENGTH,
            )
            print(f"   → {symbol}: найдено imbalances = {len(imbs_raw)}")

            if mode == "all":
                imbs = [
                    imb for imb in imbs_raw
                    if imb.get('type') in ('BUY', 'SELL')
                       and pd.notna(imb.get('strength'))
                       and float(imb['strength']) >= float(DEFAULT_MIN_STRENGTH)
                ]
                print(f"   → {symbol}: ALL (без filled/touched) = {len(imbs)}")

                added = 0
                for imb in imbs:
                    side = imb['type']
                    if side == 'BUY' and not ENABLE_BUY:
                        continue
                    if side == 'SELL' and not ENABLE_SELL:
                        continue

                    entry = select_entry_price(dfx, symbol, imb)
                    if entry is None:
                        continue

                    # ===== SANITY-СТОП/TP: всегда по правильную сторону от entry =====
                    if side == 'BUY':
                        # «логический» стоп под зоной
                        stop_zone = float(imb['low2']) * 0.998
                        # стоп обязан быть ниже entry (хоть чуть-чуть)
                        stop = min(stop_zone, float(entry) * 0.999)
                        # TP от entry и фактического стопа по RR
                        tp = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        # «логический» стоп над зоной
                        stop_zone = float(imb['high2']) * 1.002
                        # стоп обязан быть выше entry (хоть чуть-чуть)
                        stop = max(stop_zone, float(entry) * 1.001)
                        # TP от entry и фактического стопа по RR
                        tp = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)
                    # ===================================================================

                    rows.append({
                        'symbol':    symbol,
                        'type':      side,
                        'strength':  float(imb['strength']),
                        'imb_time':  pd.to_datetime(imb['time'], utc=True),
                        'entry':     float(entry),
                        'stop':      float(stop),
                        'tp':        float(tp),
                        'touched':   bool(imb.get('touched', False)),
                    })
                    added += 1

                print(f"   → {symbol}: добавлено сигналов = {added}")

            else:
                # "live": вход только когда зону коснулись (пример логики)
                imbs = [
                    imb for imb in imbs_raw
                    if bool(imb.get('touched'))
                    and imb.get('type') in ('BUY','SELL')
                ]
                print(f"   → {symbol}: LIVE (touched) = {len(imbs)}")

                added = 0
                for imb in imbs:
                    side = imb['type']
                    if side == 'BUY' and not ENABLE_BUY:
                        continue
                    if side == 'SELL' and not ENABLE_SELL:
                        continue

                    entry = select_entry_price(dfx, symbol, imb)
                    if entry is None:
                        continue

                    # ===== SANITY-СТОП/TP: всегда по правильную сторону от entry =====
                    if side == 'BUY':
                        stop_zone = float(imb['low2']) * 0.998
                        stop = min(stop_zone, float(entry) * 0.999)
                        tp = float(entry) + (float(entry) - float(stop)) * float(RISK_REWARD_RATIO)
                    else:
                        stop_zone = float(imb['high2']) * 1.002
                        stop = max(stop_zone, float(entry) * 1.001)
                        tp = float(entry) - (float(stop) - float(entry)) * float(RISK_REWARD_RATIO)
                    # ===================================================================

                    rows.append({
                        'symbol':    symbol,
                        'type':      side,
                        'strength':  float(imb['strength']),
                        'imb_time':  pd.to_datetime(imb['time'], utc=True),
                        'entry':     float(entry),
                        'stop':      float(stop),
                        'tp':        float(tp),
                        'touched':   True,
                    })
                    added += 1

                print(f"   → {symbol}: добавлено сигналов = {added}")

        except Exception as e:
            print(f"⚠️ Ошибка по {symbol}: {e}")

    df_signals = pd.DataFrame(rows)
    print(f"🔍 scan_universe: найдено сигналов = {len(df_signals)}")
    return df_signals