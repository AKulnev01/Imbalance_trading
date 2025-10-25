import os
os.environ.setdefault("USE_LOCAL_MINUTES","1")
os.environ["DISABLE_MINUTE_FALLBACK"]="0"
os.environ["MINUTE_EXIT_FOR_SINGLE"]="1"
os.environ.setdefault("LTF_ROOT","./data/m1")

try:
    import inspect
    import evaluate_signals as _E
    if hasattr(_E, "evaluate_signals"):
        _orig = _E.evaluate_signals
        def evaluate_signals(*a, **kw):
            sig = inspect.signature(_orig)
            if "intrabar" in sig.parameters:
                kw.setdefault("intrabar", True)
            # если параметра нет — форсим минутки только окружением
            else:
                os.environ["USE_LOCAL_MINUTES"] = os.environ.get("USE_LOCAL_MINUTES", "1")
                os.environ["DISABLE_MINUTE_FALLBACK"] = "0"
                os.environ["MINUTE_EXIT_FOR_SINGLE"] = "1"
            if "intrabar_lookback_days" in sig.parameters:
                kw.setdefault("intrabar_lookback_days", 180)
            return _orig(*a, **kw)
        _E.evaluate_signals = evaluate_signals
except Exception:
    pass
