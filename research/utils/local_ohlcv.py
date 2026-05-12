# utils/local_ohlcv.py
import os, re, sys
import pandas as pd
from pathlib import Path

# где лежат локальные минутки (по тикерам)
LTF_ROOT = os.getenv("LTF_ROOT", os.getenv("OHLCV_ROOT", "./data/ohlcv/1m"))

# допустимые имена колонок в parquet
REQUIRED_COLS = {"time","open","high","low","close"}
OPTIONAL_COLS = {"volume","turnover"}

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # нормализуем время и типы
    if "time" not in df.columns:
        # поддержка случая, когда время уже в индексе
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"time"})
        else:
            return pd.DataFrame()
    df = df[list(col for col in df.columns if col in (REQUIRED_COLS|OPTIONAL_COLS))].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    num_cols = [c for c in ["open","high","low","close","volume","turnover"] if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time","open","high","low","close"]).sort_values("time")
    df = df.set_index("time")
    # делаем индекс tz-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _maybe_pick_files_by_date(files, t_start, t_end):
    """
    Пытаемся отфильтровать партиции по имени (YYYY[-]?MM[-]?DD или YYYYMMDD),
    чтобы не читать всё подряд. Если не получилось — вернём исходный список.
    """
    if not files:
        return files
    pat = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")  # 2024-10-05 / 20241005
    picks = []
    for p in files:
        m = pat.search(p.name)
        if not m:
            continue
        y, mth, d = map(int, m.groups())
        try:
            dt = pd.Timestamp(year=y, month=mth, day=d, tz="UTC")
        except Exception:
            continue
        if (dt >= (t_start - pd.Timedelta(days=1))) and (dt <= (t_end + pd.Timedelta(days=1))):
            picks.append(p)
    return picks if picks else files

def load_local_minutes(symbol: str, t_start: pd.Timestamp, t_end: pd.Timestamp, root: str = None) -> pd.DataFrame:
    """
    Ищет 1m (или любые минутные) parquet для символа в локальном каталоге, читает и
    отдаёт ровно окно [t_start, t_end].
    Поддерживаются варианты:
      • <root>/<SYMBOL>.parquet
      • <root>/<SYMBOL>/*.parquet  (партиционированно по дням/месяцам)
    """
    sym = str(symbol).upper().strip()
    base = Path(root or LTF_ROOT)
    if not base.exists():
        return pd.DataFrame()

    # кейс 1: один файл на инструмент
    one_file = base / f"{sym}.parquet"
    if one_file.exists():
        try:
            df = pd.read_parquet(one_file, columns=list(REQUIRED_COLS|OPTIONAL_COLS))
        except Exception:
            df = pd.read_parquet(one_file)
        df = _sanitize(df)
        if df.empty:
            return df
        return df[(df.index >= t_start) & (df.index <= t_end)].copy()

    # кейс 2: папка с многими parquet-файлами для инструмента
    sym_dir = base / sym
    if sym_dir.exists() and sym_dir.is_dir():
        files = sorted(sym_dir.glob("*.parquet"))
        files = _maybe_pick_files_by_date(files, t_start, t_end)
        parts = []
        for p in files:
            try:
                dfp = pd.read_parquet(p, columns=list(REQUIRED_COLS|OPTIONAL_COLS))
            except Exception:
                dfp = pd.read_parquet(p)
            dfp = _sanitize(dfp)
            if not dfp.empty:
                # быстрый пред-фильтр по окну (если в файле есть время вне окна — не страшно)
                dfp = dfp[(dfp.index >= (t_start - pd.Timedelta(days=1))) &
                          (dfp.index <= (t_end   + pd.Timedelta(days=1)))]
                if not dfp.empty:
                    parts.append(dfp)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).sort_index()
        return df[(df.index >= t_start) & (df.index <= t_end)].copy()

    # fallback: ничего не нашли
    return pd.DataFrame()