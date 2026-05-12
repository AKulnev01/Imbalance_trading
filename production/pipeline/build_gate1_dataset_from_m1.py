from __future__ import annotations

from pathlib import Path
import contextlib
import io
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
N_CONTEXT_BARS = 500  # сколько 4h баров истории подмешиваем при апдейте
# ====== INPUTS (источники) ======
M1_DIR   = ROOT / "data" / "m1_4"                        # minute parquet per symbol    # symbol,side,ttl_hours,k_tp_abs,k_sl_abs ...

# ====== OUTPUTS (прод датасет) ======
DS_DIR = ROOT / "production" / "dataset" / "gate1"

REPORT_CSV  = DS_DIR / "_BUILD_REPORT.csv"
REPORT_JSON = DS_DIR / "_BUILD_REPORT.json"

# ====== SETTINGS ======
RESAMPLE_RULE = "4h"          # 4h bars
LABEL = "right"
CLOSED = "right"
FORCE_REBUILD = True

# вход в позицию: "next open" = close текущей 4h == open следующей 4h.
# мы храним бар как entry_ts = open бара; модель работает на закрытии предыдущего бара, входится на open следующего
# (это "правильная" логика под реал).
ENTRY_PX_MODE = "open"        # open (вход на открытии бара entry_ts)

# строим обе стороны (BUY/SELL), потому что downstream модели/схема часто ожидают side
SIDES = ("BOTH",)

# подавать референсы:
USE_BTC_ETH_REFS = True
BTC_SYMBOL = "BTCUSDT"
ETH_SYMBOL = "ETHUSDT"

# подавлять ворнинги/принты билдера фич (он спамит zero-variance)
SILENCE_FEATURE_BUILDER_STDOUT = True

# ограничение: если в m1 есть хвост <4h, он не даёт полного бара — это ок, ресемплер сам выкинет NaN
# ====== IMPORT feature builder (эталон) ======
# ВАЖНО: этот импорт должен работать при запуске из корня проекта.
# Запускай: python production/pipeline/build_gate1_dataset_from_m1.py
from production.features.build_features_full import build_features_single_symbol


def fail(msg: str):
    raise SystemExit(msg)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_ts_col(df: pd.DataFrame) -> str:
    for c in ["ts", "timestamp", "open_time", "time", "datetime", "dt"]:
        if c in df.columns:
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise RuntimeError(f"cannot find timestamp column; cols={list(df.columns)[:30]}")


def read_m1(sym: str) -> pd.DataFrame:
    p = M1_DIR / f"{sym}.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)

    ts_col = _find_ts_col(df)
    if ts_col == "__index__":
        df = df.reset_index().rename(columns={"index": "ts"})
        ts_col = "ts"

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.set_index(ts_col)

    need = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{sym}: m1 missing cols={miss}; cols={list(df.columns)[:30]}")

    return df[need].copy()


def m1_to_4h(m1: pd.DataFrame) -> pd.DataFrame:
    o = m1["open"].resample(RESAMPLE_RULE, label=LABEL, closed=CLOSED, origin="epoch").first()
    h = m1["high"].resample(RESAMPLE_RULE, label=LABEL, closed=CLOSED, origin="epoch").max()
    l = m1["low"].resample(RESAMPLE_RULE, label=LABEL, closed=CLOSED, origin="epoch").min()
    c = m1["close"].resample(RESAMPLE_RULE, label=LABEL, closed=CLOSED, origin="epoch").last()
    v = m1["volume"].resample(RESAMPLE_RULE, label=LABEL, closed=CLOSED, origin="epoch").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index().rename(columns={out.index.name or "index": "entry_ts"})
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["entry_ts"])
    # === FIX TIME GRID ===
    out = out.sort_values("entry_ts").reset_index(drop=True)

    dt = out["entry_ts"].diff()

    bad = dt[(dt != pd.Timedelta(hours=4)) & (~dt.isna())]
    if len(bad):
        print(f"[WARN] {len(bad)} gaps detected, examples:\n{bad.head()}")

    mask = (dt.isna()) | (dt == pd.Timedelta(hours=4))
    out = out[mask].copy()
    return out

def get_template_schema_file() -> Path | None:
    # берём любой существующий датасет как эталон схемы (398 колонок)
    files = sorted([p for p in DS_DIR.glob("*.parquet") if not p.name.startswith("_")])
    return files[0] if files else None


def align_to_template(df_new: pd.DataFrame, df_tpl: pd.DataFrame) -> pd.DataFrame:
    """
    Выравниваем новую порцию под schema шаблона:
    - колонок ровно как в шаблоне
    - отсутствующие колонки -> NaN (не нули!)
    - типы: числовые -> to_numeric, datetime -> to_datetime(utc), строковые -> object
    """
    tpl_cols = list(df_tpl.columns)
    out = df_new.reindex(columns=tpl_cols)

    for c in tpl_cols:
        dt = df_tpl[c].dtype
        try:
            if pd.api.types.is_datetime64_any_dtype(dt):
                out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
            elif pd.api.types.is_numeric_dtype(dt):
                out[c] = pd.to_numeric(out[c], errors="coerce")
            else:
                # string/object
                out[c] = out[c].astype(object)
        except Exception:
            pass
    if "label_gate1" in out.columns:
        out = out.drop(columns=["label_gate1"])

    return out

def add_refs_to_bars(bars: pd.DataFrame, btc4h: pd.DataFrame, eth4h: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()

    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")

    btc = btc4h.copy()
    eth = eth4h.copy()

    btc["entry_ts"] = pd.to_datetime(btc["entry_ts"], utc=True, errors="coerce")
    eth["entry_ts"] = pd.to_datetime(eth["entry_ts"], utc=True, errors="coerce")

    btc = btc.sort_values("entry_ts").drop_duplicates("entry_ts")
    eth = eth.sort_values("entry_ts").drop_duplicates("entry_ts")

    # === ВАЖНО: merge_asof ===
    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        btc[["entry_ts", "close"]].rename(columns={"close": "ref_btc_close"}),
        on="entry_ts",
        direction="backward"
    )

    x = pd.merge_asof(
        x.sort_values("entry_ts"),
        eth[["entry_ts", "close"]].rename(columns={"close": "ref_eth_close"}),
        on="entry_ts",
        direction="backward"
    )
    # === STRICT ALIGNMENT (NO FUTURE) ===
    x["ref_btc_close"] = x["ref_btc_close"].ffill()
    x["ref_eth_close"] = x["ref_eth_close"].ffill()
    # индекс рынка
    x["ref_close"] = (
                             x["ref_btc_close"].fillna(x["ref_eth_close"]) +
                             x["ref_eth_close"].fillna(x["ref_btc_close"])
                     ) / 2.0

    return x


def build_feature_rows(sym: str, bars4h: pd.DataFrame) -> pd.DataFrame:
    # делаем BUY/SELL записи
    x = bars4h.copy()
    x["symbol"] = sym

    # строим фичи эталонным билдёром
    if SILENCE_FEATURE_BUILDER_STDOUT:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            feat = build_features_single_symbol(x)

            feat = (
                feat
                .sort_values("entry_ts")
                .drop_duplicates(subset=["symbol", "entry_ts"], keep="last")
                .reset_index(drop=True)
            )
            # убираем непрогретые бары (очень важно)
            feat = feat[feat["volat_ret12"].notna()].copy()
            feat = feat[feat["atr_to_price"].notna()].copy()
    else:
        feat = build_features_single_symbol(x)

    feat = feat.drop(columns=["label_gate1"], errors="ignore")
    return feat

def main():
    ensure_dir(DS_DIR)

    # предварительно ресемплим BTC/ETH refs один раз
    btc4h = eth4h = None
    if USE_BTC_ETH_REFS:
        btc_m1 = read_m1(BTC_SYMBOL)
        eth_m1 = read_m1(ETH_SYMBOL)
        btc4h = m1_to_4h(btc_m1)
        eth4h = m1_to_4h(eth_m1)

    tpl_df = None

    # символы берём из доступных m1 файлов
    syms = sorted([p.stem for p in M1_DIR.glob("*.parquet")])

    rows_report = []
    for sym in syms:
        ds_path = DS_DIR / f"{sym}.parquet"
        m1_path = M1_DIR / f"{sym}.parquet"

        r = {
            "symbol": sym,
            "status": "OK",
            "err": "",
            "ds_exists": int(ds_path.exists()),
            "m1_exists": int(m1_path.exists()),
            "old_max_ts": "",
            "new_max_ts": "",
            "new_bars": 0,
            "new_rows": 0,
            "wrote_path": str(ds_path),
        }

        try:
            if not m1_path.exists():
                r["status"] = "NO_M1"
                rows_report.append(r)
                continue

            # старый датасет (если есть)
            df_old = None
            old_max = None
            if ds_path.exists():
                df_old = pd.read_parquet(ds_path)
                if "entry_ts" not in df_old.columns:
                    raise RuntimeError("dataset missing entry_ts")
                df_old["entry_ts"] = pd.to_datetime(df_old["entry_ts"], utc=True, errors="coerce")
                df_old = df_old.dropna(subset=["entry_ts"]).copy()
                old_max = df_old["entry_ts"].max()
                r["old_max_ts"] = str(old_max)

            # бары 4h
            m1 = read_m1(sym)
            bars4h = m1_to_4h(m1)

            # refs
            if USE_BTC_ETH_REFS:
                bars4h = add_refs_to_bars(bars4h, btc4h=btc4h, eth4h=eth4h)

            # инкремент
            # инкремент: выделим новые бары
            bars_new = bars4h.copy()
            if old_max is not None:
                bars_new = bars_new[bars_new["entry_ts"] > old_max].copy()

            if bars_new.empty and not FORCE_REBUILD:
                r["status"] = "NO_NEW_BARS"
                rows_report.append(r)
                continue

            if FORCE_REBUILD:
                bars_new = bars4h.copy()
                old_max = None

            r["new_bars"] = int(len(bars_new))

            # ===== контекст для rolling-фич =====
            # берём последние N_CONTEXT_BARS 4h баров ДО old_max из уже существующего датасета (если есть),
            # чтобы SMA/ATR/rolling48/100 считались корректно на новых барах
            if df_old is not None:
                hist = df_old[["entry_ts", "open", "high", "low", "close", "volume"]].copy()
                # refs тоже, если есть
                for rc in ["ref_btc_close", "ref_eth_close", "ref_close"]:
                    if rc in df_old.columns:
                        hist[rc] = df_old[rc]
                hist["entry_ts"] = pd.to_datetime(hist["entry_ts"], utc=True, errors="coerce")
                hist = hist.dropna(subset=["entry_ts"]).sort_values("entry_ts")
                # только уникальные entry_ts (в df_old их 2 строки на бар из-за side, поэтому дедуп)
                hist = hist.drop_duplicates(subset=["entry_ts"], keep="last")
                hist = hist.tail(N_CONTEXT_BARS)
                bars_ctx = pd.concat([hist, bars_new], ignore_index=True)
                bars_ctx = bars_ctx.sort_values("entry_ts").drop_duplicates(subset=["entry_ts"], keep="last")
            else:
                bars_ctx = bars_new

            # строим фичи на bars_ctx, но на выход берём только новые entry_ts
            df_feat_ctx = build_feature_rows(sym, bars_ctx)

            df_feat_ctx = df_feat_ctx.sort_values("entry_ts").reset_index(drop=True)

            # ===== TARGET (NEXT BAR MOVE ≥ 1%) =====
            X = 0.01

            next_high = df_feat_ctx["high"].shift(-1)
            next_low = df_feat_ctx["low"].shift(-1)

            entry = df_feat_ctx["close"]

            up_move = (next_high - entry) / entry
            down_move = (entry - next_low) / entry

            df_feat_ctx["y"] = (
                    (up_move > X) | (down_move > X)
            ).astype(int)

            # обрезаем последний бар (нет future)
            df_feat_ctx = df_feat_ctx.iloc[:-1]

            # теперь только новые строки
            df_new = df_feat_ctx[df_feat_ctx["entry_ts"] > (
                old_max if old_max is not None else pd.Timestamp.min.tz_localize("UTC")
            )].copy()
            # выравниваем схему под существующий датасет/шаблон
            if df_old is not None and not FORCE_REBUILD:
                df_new_aligned = align_to_template(df_new, df_old)
                out = pd.concat([df_old, df_new_aligned], ignore_index=True)
            else:
                if tpl_df is not None:
                    df_new = align_to_template(df_new, tpl_df)
                out = df_new

            # сортировка каноническая
            if "side" in out.columns:
                out = out.sort_values(["entry_ts", "side"], kind="mergesort").reset_index(drop=True)
            else:
                out = out.sort_values(["entry_ts"], kind="mergesort").reset_index(drop=True)
            out = (
                out
                .sort_values("entry_ts")
                .drop_duplicates(subset=["symbol", "entry_ts"], keep="last")
                .reset_index(drop=True)
            )
            r["new_rows"] = int(len(df_new))
            r["new_max_ts"] = str(out["entry_ts"].max())

            if "label_gate1" in out.columns:
                out = out.drop(columns=["label_gate1"])
            # write
            out.to_parquet(ds_path, index=False)

        except Exception as e:
            r["status"] = "ERR"
            r["err"] = f"{type(e).__name__}: {e}"

        rows_report.append(r)

    rep = pd.DataFrame(rows_report)
    rep.to_csv(REPORT_CSV, index=False)

    summary = {
        "root": str(ROOT),
        "m1_dir": str(M1_DIR),
        "ds_dir": str(DS_DIR),
        "use_btc_eth_refs": USE_BTC_ETH_REFS,
        "btc_symbol": BTC_SYMBOL if USE_BTC_ETH_REFS else "",
        "eth_symbol": ETH_SYMBOL if USE_BTC_ETH_REFS else "",
        "resample_rule": RESAMPLE_RULE,
        "entry_px_mode": ENTRY_PX_MODE,
        "files_total": int(len(rep)),
        "status_counts": rep["status"].value_counts().to_dict(),
        "report_csv": str(REPORT_CSV),
    }
    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE", REPORT_CSV)
    print(rep["status"].value_counts().to_string())
    # топ ошибок
    bad = rep[rep["status"] == "ERR"][["symbol", "err"]].head(20)
    if len(bad):
        print("\nTOP ERR:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()