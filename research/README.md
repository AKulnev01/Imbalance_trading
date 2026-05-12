Эта папка содержит старые исследовательские и экспериментальные материалы проекта **ImbalanceSearcher**.

Важно: `research/` не является текущим production-контуром. Актуальная рабочая логика находится в корневых production-папках проекта: `online/`, `production/`, `pipeline/`, `NIRS/`, `data/`, `db_dumps/`, `models/`.

Содержимое `research/` сохранено как история разработки: старые модели, датасеты, отчёты, backtest-скрипты, эксперименты с признаками, ранние версии торговой логики и вспомогательные утилиты.

## Назначение папки

`research/` нужен для:

- сохранения истории экспериментов;
- понимания, какие подходы уже проверялись;
- восстановления старых идей при необходимости;
- анализа причин, почему часть направлений не попала в текущий production;
- хранения старых исследовательских артефактов отдельно от актуального кода.

Эта папка не должна импортироваться из текущего production-кода.

## Что не является production

Код внутри `research/` может быть:

- устаревшим;
- несовместимым с текущей структурой проекта;
- завязанным на старые пути;
- завязанным на старые форматы данных;
- неполным или черновым;
- содержащим старые assumptions по entry/exit/backtest;
- не проходящим актуальные проверки на phase alignment, leakage и live-совместимость.

Перед любым повторным использованием код из `research/` нужно заново проверять.

## Структура

### `models/`

Старый набор обученных моделей.

По факту содержит сотни CatBoost-моделей `.cbm`, несколько служебных файлов и старые модели по направлениям вроде:

- `ks_v11_state_per_symbol_cat_focus_only`;
- `ks_v11_static_per_symbol_cat`;
- BONK v15 entry/ranker/transformer experiments.

Это архив моделей из ранних веток исследования. В текущем production эти модели напрямую не используются.

Причина выноса в `research`: модели относятся к старым экспериментам, не к текущему multi-gate production pipeline.

### `models_organized/`

Организованный архив старых моделей.

Содержит разложенные по символам и экспериментальным режимам модели:

- `pnl_norm`;
- `rmse_norm`;
- `base_static`;
- `focus_state`;
- `weighted_state`;
- разные размеры моделей: `A_tiny`, `B_small`, `C_mid`, `D_big`, `E_huge`.

Эта структура полезна как история массового перебора моделей по символам, но не является актуальным layout для production.

Причина выноса в `research`: старый формат организации моделей, большой объём, не используется текущим online-контуром.

### `predict/`

Старый исследовательский ML-контур.

Основные направления:

#### `predict/ks/`

Эксперименты вокруг KS/v11-подхода:

- сравнение state models против static models;
- построение state-фичей;
- merge state-фичей в датасеты;
- обучение CatBoost per-symbol моделей;
- восстановление метрик старых моделей.

Примеры файлов:

- `backtest_ks_v11_state_models_vs_static.py`;
- `backtest_ks_v11_state_models_vs_static_ext.py`;
- `build_ks_v11_states_175.py`;
- `merge_ks_v11_states_into_feats_175.py`;
- `train_ks_v11_state_per_symbol_cat_focus_only.py`;
- `train_ks_v11_state_per_symbol_cat_weighted.py`.

Причина выноса в `research`: KS/v11-подход был важным исследовательским этапом, но текущий production перешёл на более строгую gate-архитектуру и отдельные production-dataset/model roots.

#### `predict/ks/bonk/`

Отдельная серия BONK-экспериментов:

- генерация TP/SL grid;
- статический best-KS;
- CatBoost ranker;
- Transformer ranker;
- сравнение static strategy vs CatBoost vs Transformer.

Примеры файлов:

- `generate_bonk_grid_v14.py`;
- `generate_bonk_grid_v15_from_base.py`;
- `eval_bonk_v15_ks_cat_vs_static.py`;
- `eval_bonk_v15_ks_cat_vs_static_vs_trans.py`;
- `train_bonk_entry_bestks_cat.py`;
- `train_bonk_entry_ranker_v2.py`.

Причина выноса в `research`: это отдельная экспериментальная ветка под один/несколько символов, не текущий универсальный production-пайплайн.

#### `predict/tp_entry/`

Старые эксперименты по TP/SL-entry и triple-barrier style датасетам.

Содержит:

- построение фичей;
- анализ timeout;
- анализ качества TP/SL;
- построение датасетов;
- старые модели и промежуточные артефакты.

Примеры файлов:

- `add_features_full.py`;
- `analyze_timeouts.py`;
- `analyze_tp_sl_quality.py`;
- `build_dataset_allbars.py`.

Причина выноса в `research`: часть идей ушла в production Gate5/TP-SL логику, но конкретная реализация здесь старая и не должна смешиваться с текущей цепочкой.

### `real/`

Ранний live/API слой.

Содержит старые скрипты для:

- диагностики окружения;
- быстрого API evaluation;
- старого Telegram control;
- прямого получения market data.

Примеры файлов:

- `diag_env.py`;
- `eval_quick_api.py`;
- `tgctl.py`;
- `utils/market_data_api.py`.

Причина выноса в `research`: текущий live-контур заменён на `online/trading/*`, где есть отдельные `autotrade_service`, `orchestrator`, `execution`, `monitor`, `reconcile`, `service_status`, `tg_control_bot`.

### `reports/`

Большой архив старых отчётов, датасетов, backtest-результатов и model outputs.

Содержит много тяжёлых файлов:

- `.parquet`;
- `.cbm`;
- `.json`;
- `.csv`;
- `.txt`;
- `.xlsx`.

Типовые направления:

- старые backtest reports;
- feature datasets;
- feature importance;
- BONK grid reports;
- Gate-stack diagnostics;
- KS/v11 feature outputs;
- промежуточные датасеты для обучения и сравнения моделей.

Причина выноса в `research`: это исторические артефакты исследований. Они полезны для анализа старых результатов, но не должны лежать рядом с актуальными production-артефактами.

### `scripts/`

Старые standalone-скрипты.

Назначение:

- загрузка m1 данных с Bybit;
- rollup таймфреймов;
- диагностика eval;
- генерация сигналов;
- early momentum backtest;
- реконструкция meta из m1;
- listing report.

Примеры файлов:

- `fetch_m1_bybit.py`;
- `rollup_any_tf.py`;
- `rollup_generic.py`;
- `backtest_early_momentum.py`;
- `diagnose_eval.py`;
- `listing_report.py`;
- `reconstruct_meta_from_m1.py`;
- `quick_eval_probe.py`.

Некоторые скрипты в этой папке имеют parse error по результатам аудита:

- `generate_signals.py`;
- `generate_signals_variant.py`;
- `signals_with_context_and_eval.py`.

Их следует считать черновиками или повреждёнными старыми файлами, пока они не будут вручную восстановлены.

Причина выноса в `research`: это ранние утилиты и прототипы, не часть текущей production-командной цепочки.

### `tools/`

Старые вспомогательные инструменты.

Назначение:

- фильтрация сигналов индикаторами 4h;
- построение параметрических датасетов;
- перебор параметров;
- построение PDF-графиков свечей.

Примеры файлов:

- `filter_signals.py`;
- `make_param_dataset.py`;
- `eval_param_candidates.py`;
- `plot_candles.py`.

Причина выноса в `research`: инструменты полезны как вспомогательные исследования, но не являются обязательной частью текущего production runtime.

### `utils/`

Старая инфраструктурная библиотека.

Содержит ранние реализации:

- Bybit HTTP-клиента;
- загрузки свечей;
- технических индикаторов;
- FVG/imbalance detection;
- стратегии;
- symbol fetch;
- state store;
- trade logger;
- Excel export.

Примеры файлов:

- `bybit_trade.py`;
- `fetch_data.py`;
- `strategy.py`;
- `ta.py`;
- `detect_fvg.py`;
- `detect_fvg_close.py`;
- `evaluate_imbalances.py`;
- `allocator.py`;
- `state_store.py`;
- `trade_logger.py`.

Причина выноса в `research`: текущий production использует новый код в `online/`, `production/` и `pipeline/`. Старый `utils/` не должен быть зависимостью актуального автотрейда.

## Корневые файлы в `research/`

### `autotrade_momentum.py`

Старая версия автоторговой логики по momentum/FVG/imbalance-подходу.

Судя по размеру и старым импортам, это был ранний монолитный автотрейд-скрипт до текущей декомпозиции на:

- `online/trading/autotrade_service.py`;
- `online/trading/orchestrator.py`;
- `online/trading/execution.py`;
- `online/trading/monitor.py`;
- `online/trading/reconcile.py`.

Причина выноса в `research`: монолитная старая логика больше не соответствует текущему production-процессу.

### `main.py`

Старый основной entrypoint проекта.

Вероятно относится к раннему запуску стратегии/сканера до текущей production-архитектуры.

Причина выноса в `research`: текущие production entrypoints находятся в `online/trading/*` и production pipeline scripts.

### `optimize_params.py`

Старый скрипт оптимизации параметров.

Вероятно относится к ранним grid/parameter-search экспериментам.

Причина выноса в `research`: текущие approved-пороги и blacklist-логика живут в production/backtest/online-контуре, а не в этом старом standalone-скрипте.

### `config.py`

Старый конфигурационный файл.

Может содержать старые пути, параметры, ключи окружения или assumptions, не совпадающие с текущим `online/trading/config.py`.

Причина выноса в `research`: нельзя смешивать старый config с production config.

### `.env.test`

Пустой тестовый env-файл.

Не содержит данных, но любые `.env*` файлы перед публикацией всё равно нужно проверять.

## Почему это не вошло в production

Основные причины:

1. **Смена архитектуры.**  
   Проект ушёл от отдельных скриптов и ранних монолитов к gate-based production pipeline.

2. **Смена live-контура.**  
   Старый `real/` и `utils/bybit_trade.py` заменены на `online/trading/bybit_client.py`, `execution.py`, `monitor.py`, `reconcile.py`.

3. **Смена модели данных.**  
   Старые датасеты, KS/v11 outputs, BONK grids и TP-entry датасеты имеют другие форматы и assumptions.

4. **Риск phase/timing mismatch.**  
   Для production критично, чтобы признаки были доступны строго на закрытии H4 свечи, а вход происходил на следующей свече с заданной задержкой. Старые исследования не обязательно соответствуют этой дисциплине.
