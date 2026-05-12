# TP-first entry pipeline
1) build dataset:
PYTHONPATH=. python predict/tp_entry/build_tp_dataset.py --m1-dir ./data/m1 --tp-pct 0.135 --sl-pct 0.04 --slippage-pct 0.004 --ttl-hours 80 --lookback-days 720 --out ./predict/tp_entry/tp_dataset.xlsx

2) train model:
PYTHONPATH=. python predict/tp_entry/train_tp_model.py --data ./predict/tp_entry/tp_dataset.xlsx --outdir ./predict/tp_entry/models --min-recall 0.3

3) backtest entries (with dir-rules + AFTER filters):
PYTHONPATH=. python predict/tp_entry/backtest_tp_entry.py --model-dir ./predict/tp_entry/models --m1-dir ./data/m1 --tp-pct 0.135 --sl-pct 0.04 --slippage-pct 0.004 --ttl-hours 80 --apply-dir-rules 1 --after-buy ./after_predict/models/after_buy.pkl --after-sell ./after_predict/models/after_sell.pkl --after-thr-buy 0.48 --after-thr-sell 0.55 --out ./predict/tp_entry/tp_backtest.xlsx
