# ImbalanceSearcher
Production-проект многоуровневой ML-системы для алгоритмической торговли криптовалютами на Bybit.
Система не пытается напрямую предсказывать цену. Основная задача проекта — оценить вероятность успешной реализации торгового сигнала и принять решение:
```text
входить / не входить → LONG или SHORT → какой TP/SL → отправка market-ордера → контроль позиции

Архитектура построена как каскад моделей:

Gate1 → Gate2 → Gate3 → Gate4 → Gate5.1 → Gate5.2 → Gate5.3 → selector → execution

Каждый gate решает отдельную задачу. Итоговое решение формируется не одной моделью, а последовательностью фильтров, скорингов и проверок.

⸻

1. Главная идея проекта

Криптовалютные временные ряды шумные, нестационарные и плохо подходят для прямого точечного прогноза цены. Поэтому проект использует другую постановку:

не предсказывать точную цену,
а оценивать вероятность того, что после входа сделка достигнет нужного результата

Система работает как многоуровневый decision pipeline:

1. Gate1 отсекает мусорные рыночные состояния.
2. Gate2 проверяет краткосрочную реализуемость движения.
3. Gate3 оценивает структурное качество сетапа.
4. Gate4 выбирает сторону LONG/SHORT.
5. Gate5 оценивает качество сделки и TP/SL-сетки.
6. Selector выбирает один лучший сигнал.
7. Execution открывает позицию market-ордером и ставит TP/SL.
8. Monitor/Reconcile сверяют локальную БД с Bybit.

⸻

2. Базовая trading-логика

Основной рабочий таймфрейм:

H4

Дополнительно используются минутные данные:

m1

Они нужны для:

* точного backtest;
* расчёта forward labels;
* MFE/MAE;
* TP/SL hit logic;
* hit-time;
* проверки реального исполнения.

⸻

3. Критичная временная логика

Production-логика входа:

1. Закрывается H4-свеча T.
2. По этой закрытой свече и истории до неё считаются признаки.
3. Признаки подаются в модели.
4. Если сигнал прошёл pipeline, вход делается на следующей H4-свече.
5. Вход отправляется market-ордером.
6. Фактическая цена исполнения берётся с Bybit.
7. TP/SL считаются от фактической цены входа.

Пример:

signal_ts = 2026-05-12 08:00 UTC
entry_ts_plan = 2026-05-12 12:00 UTC
execution = после завершения pipeline, обычно 1–3 минуты после новой H4

Важный принцип:

features относятся к последней закрытой H4-свече,
execution относится уже к следующей H4-свече

Это должно быть одинаково отражено:

* в dataset builder;
* в offline backtest;
* в live execution;
* в label logic.

⸻

4. Market execution

В production система использует именно market-ордера.

Причина:

* пайплайн может отработать за 1 минуту или за 3 минуты;
* цена может отличаться от h4_close;
* это отличие заранее учитывается в backtest через комиссии и проскальзывание;
* поэтому execution не должен ждать лимитку.

Production-принцип:

после прохождения сигнала отправляется market order независимо от отличия текущей цены от h4_close

После market entry система получает фактическую цену исполнения:

entry_avg_px

И уже от неё считает:

tp_px_plan
sl_px_plan
entry_slippage_abs
entry_slippage_pct

⸻

5. Комиссии и проскальзывание

Зафиксированная taker fee на Bybit:

≈ 0.1% на сторону

Backtest учитывает execution realism отдельно от features.

Проскальзывание:

≈ 0.2%–0.4% на сторону в разных проверках

Важно:

проскальзывание не является feature

Оно используется только для:

* PnL simulation;
* backtest realism;
* сравнения с фактическим исполнением.

⸻

6. Production-ограничения торговли

Текущая схема:

1 слот
1 открытая позиция одновременно
пока позиция открыта — новые сигналы игнорируются
следующая сделка использует обновлённый капитал

Это не портфельная мультисигнальная стратегия. Это односделочный pipeline.

⸻

7. Чулан / рабочий капитал

В production используется режим ограничения рабочего капитала.

Параметры:

CHULAN_ENABLED = 1
CHULAN_BASE_CAPITAL_USDT = 100

Логика:

trade_capital_usdt = min(available_usdt, CHULAN_BASE_CAPITAL_USDT)

Пример:

balance = 90    → trade_capital = 90
balance = 100   → trade_capital = 100
balance = 110   → trade_capital = 100
balance = 1000  → trade_capital = 100

⸻

8. Основные production-папки

Актуальные папки проекта:

online/
production/
pipeline/
models/
data/
db_dumps/
NIRS/

Исторические исследования вынесены в:

research/

⸻

9. online/

Главный production-runtime.

online/

Ключевые части:

online/trading/
online/gate2/
online/gate4/
online/gate5/

Именно online/ использует боевые модели, БД, Bybit API и Telegram-control.

Основная online-цепочка:

sync candles
    ↓
build online features
    ↓
Gate2 predictions
    ↓
Gate4 side prediction
    ↓
Gate5.1 score
    ↓
Gate5.2 ranker
    ↓
Gate5.3 decision
    ↓
selector
    ↓
execution
    ↓
monitor / reconcile

⸻

10. production/

Папка production — рабочая витрина проекта.

Здесь лежат:

* финальные и промежуточные датасеты;
* обученные модели;
* valid predictions;
* threshold reports;
* feature importance;
* audit CSV/JSON;
* manifests;
* результаты сравнений;
* pipeline-артефакты.

Важные поддиректории:

production/dataset/
production/models/
production/pipeline/
production/features/

⸻

11. Данные

Минутные данные:

data/m1_4/

Возможные форматы:

SYMBOL.parquet
SYMBOL_m1.parquet

H4-данные:

data/h4_3/<SYMBOL>.parquet

Исторически также встречались:

data/m1/
data/m1_3/
data/h4_2/

Но актуальные production-проверки ориентируются на новые ветки m1_4 и h4_3, если конкретный модуль не указывает иначе.

⸻

12. База данных

Production-БД:

imb_traid

PostgreSQL DSN обычно задаётся через .env:

IMB_DB_DSN=postgresql://...

Ключевые trading-таблицы:

public.trading_signals
public.trading_positions
public.trading_orders

Ключевые online-таблицы:

public.online_gate2_predictions
public.online_gate4_features
public.online_gate4_predictions_no_raw_refs
public.online_gate5_1_scores
public.online_gate5_2_ranker
public.online_gate5_3_decisions

Аудит:

public.audit_events
public.order_events

⸻

13. Service control

Основной сервис:

online.trading.autotrade_service

Контроллер:

online.trading.service_status

Проверить статус Windows-host:

python -m online.trading.service_status status win

Проверить статус Mac-host:

python -m online.trading.service_status status mac

Запустить на Windows:

python -m online.trading.service_status start win

Остановить на Windows:

python -m online.trading.service_status stop win

История сигналов:

python -m online.trading.service_status history 20 win

Backtest:

python -m online.trading.service_status backtest win

⸻

14. Telegram control bot

Telegram-control:

online/trading/tg_control_bot.py

Команды:

/status
/history
/run
/stop
/backtest
/backtest_wizard
/cancel

Меню:

📊 Статус
📜 История
▶️ Запуск
⏹ Стоп
🧪 Бэктест
⚙️ Команды
❌ Отмена

Telegram-бот является оболочкой над online.trading.service_status.

⸻

15. Описание моделей

⸻

15.1. Gate1 — первичный фильтр мусора

Роль

Gate1 — первый фильтр pipeline.

Задача:

отсеять рыночные состояния, где нет смысла продолжать анализ

Gate1 не принимает финальное торговое решение. Он только пропускает или отбрасывает потенциально бесполезные ситуации.

Типовая постановка

Основная историческая задача:

gate1_impulse_abs_move_atr_16h

Смысл:

будет ли после сигнальной точки значимое движение цены в горизонте 16 часов

Датасеты

Основной источник:

production/dataset/gate1/

Модели

Общий вид пути:

production/models/final_gate1/<SYMBOL>/gate1/gate1_impulse_abs_move_atr_16h.cbm

Пример:

production/models/final_gate1/ICPUSDT/gate1/gate1_impulse_abs_move_atr_16h.cbm

Рядом обычно лежит:

meta.json

Признаки

Gate1 использует H4-признаки:

* OHLCV;
* ATR;
* candle geometry;
* wick/body features;
* RSI/MACD/CCI;
* volatility features;
* volume/liquidity features;
* daily/weekly/monthly context;
* cross-market признаки BTC/ETH.

Важное замечание

Старые ks_v11-датасеты имели риск фазовой неоднозначности. Поэтому актуальная production-логика должна строго соблюдать:

features = только последняя закрытая H4-свеча T
execution = следующая свеча T+1

⸻

15.2. Gate2 — краткосрочная реализуемость движения

Роль

Gate2 отвечает на вопрос:

уйдёт ли цена быстро в нужную сторону после входа

Это слой не про общий “хороший сигнал”, а именно про краткосрочную реализацию движения.

Основная текущая постановка

Текущий основной вариант:

directional reach classification

То есть две отдельные модели:

up_reach_high
dn_reach_high

Смысл:

up_reach_high — вероятность достижения верхнего уровня
dn_reach_high — вероятность достижения нижнего уровня

Основные датасеты

Старый общий reach dataset:

production/dataset/final_gate2_2_directional_reach_all.parquet

Новый актуальный 5features reach dataset:

production/dataset/final_gate2_2_directional_reach_5features_all.parquet

Посимвольная strength-ветка:

production/dataset/final_gate2_3_directional_strength_5features_by_symbol/<SYMBOL>.parquet

Общая strength-ветка:

production/dataset/final_gate2_3_directional_strength_5features_all.parquet

Модели

Старые модели:

production/models/gate2_mod/cls/up_reach_high/up_reach_high.cbm
production/models/gate2_mod/cls/dn_reach_high/dn_reach_high.cbm

Актуальная 5features-ветка:

production/models/gate2_mod_5features/cls/up_reach_high/up_reach_high.cbm
production/models/gate2_mod_5features/cls/dn_reach_high/dn_reach_high.cbm

5 дополнительных признаков

В новой ветке восстановлены 5 признаков:

atr4h
hammer_like
ret_l1
ret_l2
vol_regime

Они были пересчитаны по H4 и проверены аудитом:

exact_match_share = 1.0
max_abs_diff = 0.0

Метрики новой 5features-ветки

up_reach_high:

old_auc = 0.644789
new_auc = 0.651772
delta = +0.006984

dn_reach_high:

old_auc = 0.663919
new_auc = 0.676756
delta = +0.012837

Production threshold selection

Для precision около 0.8:

up_reach_high:

threshold ≈ 0.812714
share ≈ 1.27%
precision ≈ 0.8017

dn_reach_high:

threshold ≈ 0.884121
share ≈ 0.68%
precision ≈ 0.8015

Важные отчёты

production/models/gate2_mod/_COMPARE_OLD_VS_5FEATURES_SUMMARY.csv
production/models/gate2_mod/_COMPARE_OLD_VS_5FEATURES_THRESHOLDS.csv
production/models/gate2_mod/_COMPARE_OLD_VS_5FEATURES_SYMBOLS.csv
production/models/gate2_mod_5features/_PROD_THRESHOLD_SELECTION.csv

⸻

15.3. Gate3 — структурное качество сетапа

Роль

Gate3 проверяет не просто импульс, а структуру сделки.

Смысл:

если price action pattern появился,
то при каких условиях он действительно отрабатывает свою исходную математику

Gate3 оценивает:

* рыночный контекст;
* активные паттерны;
* long/short структуру;
* состояние тренда;
* локальные пробои;
* свечные формации;
* волатильность;
* пригодность сетапа.

Датасеты

Посимвольный dataset:

production/dataset/pa_gate3_v3_long_short_by_symbol/<SYMBOL>.parquet

Общий dataset:

production/dataset/pa_gate3_v3_long_short_all.parquet

Исторические/предыдущие ветки:

production/dataset/pa_gate3_v1_all.parquet
production/dataset/pa_gate3_v1_by_symbol/
production/dataset/pa_gate3_v1_labeled_by_symbol/
production/dataset/pa_gate3_v1_structext_by_symbol/
production/dataset/pa_gate3_v2_stateful_struct3_by_symbol/

Модели

Финальные long/short score-модели:

production/models/final_gate3_score_long_short/<SYMBOL>/<long|short>/gate3_score/gate3_score.cbm

Пример:

production/models/final_gate3_score_long_short/1000PEPEUSDT/long/gate3_score/gate3_score.cbm

Рядом находится threshold grid:

threshold_grid.csv

Отдельная squeeze-ветка

Историческая squeeze-ветка:

production/models/gate3_squeeze/<SYMBOL>/short/gate3_score/gate3_score.cbm

Она не является основной production-веткой, но была полезна как кандидат для отдельных символов.

Важный вывод

Gate3 не выбирает TP/SL. Gate3 оценивает структурное качество long/short-сетапа.

TP/SL выбираются и проверяются уже на Gate5.

⸻

15.4. Gate4 — выбор стороны LONG/SHORT

Роль

Gate4 агрегирует upstream-сигналы и выбирает направление сделки:

LONG или SHORT

Gate4 получает на вход признаки и результаты предыдущих уровней:

* Gate1 score/proba;
* Gate2 up/down probabilities;
* Gate3 long/short score;
* структурные признаки;
* side-related признаки;
* confidence/margin признаки.

Датасеты

Основная папка:

production/dataset/gate4/gate4_1_side_builder/

Ключевые файлы:

production/dataset/gate4/gate4_1_side_builder/gate4_1_candidates_raw.parquet
production/dataset/gate4/gate4_1_side_builder/gate4_1_side_dataset.parquet
production/dataset/gate4/gate4_1_side_builder/_AUDIT.csv
production/dataset/gate4/gate4_1_side_builder/_REPORT.json

Модель

production/models/gate4/gate4_y_side_clean_multiclass/gate4_y_side_clean_multiclass.cbm

Важно:

название multiclass историческое;
фактически текущая задача — бинарный выбор LONG/SHORT

Использование в Gate5

Для Gate5.1 использовались confident Gate4-сигналы.

Ключевой параметр:

GATE4_CONFIDENCE_THRESHOLD = 0.90

Смысл:

оставлять только сигналы, где Gate4 достаточно уверенно выбрал сторону

⸻

16. Gate5

Gate5 — финальный блок оценки качества сделки и TP/SL-логики.

Важно:

Gate5 не заменяет Gate1–Gate4,
а работает поверх уже отфильтрованного и направленного сигнала

Актуальная online-цепочка Gate5:

Gate5.1 → Gate5.2 → Gate5.3

⸻

16.1. Gate5.1 — вероятность TP раньше SL

Роль

Gate5.1 оценивает качество сделки для конкретной TP/SL-сетки.

Главный смысл:

насколько вероятно, что для выбранной стороны TP будет достигнут раньше SL

Датасеты pair/grid

Основная папка pair datasets:

production/dataset/gate5/gate5_pair_datasets/

Пример:

production/dataset/gate5/gate5_pair_datasets/gate5_dataset_tp100_sl075.parquet
production/dataset/gate5/gate5_pair_datasets/gate5_dataset_tp150_sl075.parquet

В датасете могут быть диагностические поля:

g5_mfe_side_atr_<grid>
g5_first_tp_minute_<grid>
g5_tp_before_sl_<grid>

Они нужны для разметки и анализа, но не должны попадать в features модели.

Online runner

Важный файл:

online/gate5/build_online_gate5_1_features.py

Несмотря на название, это не просто feature builder. В текущей production-цепочке он фактически является полноценным Gate5.1 online runner:

online_gate4_predictions + online_gate4_features
    ↓
build_online_gate5_1_features.py
    ↓
online_gate5_1_scores

Поэтому отдельный predict_online_gate5_1.py сейчас не нужен.

Online output

public.online_gate5_1_scores

⸻

16.2. Gate5.2 — grid ranker

Роль

Gate5.2 ранжирует TP/SL-сетки внутри сигнала.

Идея:

для одного сигнала есть несколько возможных TP/SL-сеток,
нужно оценить, какая сетка выглядит лучше

Dataset

production/dataset/gate5/gate5_2/gate5_grid_ranker_dataset.parquet

Что внутри

Для каждого сигнала считаются разные сетки, например:

tp100_sl075
tp120_sl060
tp150_sl075
tp225_sl075

Фичи могут включать:

* proba;
* margin;
* signal-level stats;
* grid-level stats;
* rank внутри сигнала;
* meta-признаки.

Target:

target_score

Online runner

online/gate5/build_online_gate5_2_ranker.py

Online output

public.online_gate5_2_ranker

⸻

16.3. Gate5.3 — pairwise grid decision

Роль

Gate5.3 сравнивает две TP/SL-сетки.

Важно:

Gate5.3 не выбирает произвольную TP/SL-сетку из всех возможных.
Он сравнивает пару grid_A vs grid_B.

Формат пары:

grid_A__vs__grid_B

Пример production-пары:

tp225_sl075__vs__tp100_sl075

Датасеты

production/dataset/gate5/gate5_3/<pair>.parquet

Пример:

production/dataset/gate5/gate5_3/tp225_sl075__vs__tp100_sl075.parquet

Target

Pairwise target строится через разницу качества:

delta_score = target_score_B - target_score_A

И далее:

y = 1, если grid_B лучше grid_A
y = 0, если grid_A лучше grid_B

В части экспериментов использовался фильтр:

|delta_score| > 0.25

Features

Типовые группы признаков:

signal-level:

sig_top1_proba
sig_mean_proba
sig_margin

grid-level:

grid_A_proba
grid_B_proba
grid_A_margin
grid_B_margin

diff features:

proba_diff
margin_diff
rr_diff
tp_diff
sl_diff

Anti-leak logic

Из features должны быть исключены точные target/outcome/future-поля.

Exact exclusions:

y
delta_score
safe_target_score
agg_target_score
target_score

Contains exclusions:

target
label
future
mfe
mae
tp_hit
sl_hit
first_tp
first_sl

Prefix exclusions:

g5_
safe_g5_
agg_g5_

Модели

production/models/gate5/gate5_3/<pair>/

Внутри:

model.cbm
valid_predictions.parquet
features.csv
report.json

Проверки

Для Gate5.3 были сделаны проверки:

pred_label leakage — не попал в модель
permutation test — AUC около 0.5
feature filtering — target/future поля вырезаются

Важное production-замечание

Для текущей пары:

tp225_sl075__vs__tp100_sl075

выбор tp100_sl075 почти всегда является нормой и не считается ошибкой.

Текущая production-сетка:

tp100_sl075

То есть:

TP = 1.00 ATR
SL = 0.75 ATR

Gate5.3 пока сохраняется в pipeline в основном ради совместимости с уже созданными артефактами и online-таблицами.

Online runner

online/gate5/build_online_gate5_3_decision.py

Online output

public.online_gate5_3_decisions

⸻

17. Актуальная online-ветка Gate5

Полная актуальная online-цепочка:

public.online_gate4_predictions_no_raw_refs
public.online_gate4_features
        ↓
online/gate5/build_online_gate5_1_features.py
        ↓
public.online_gate5_1_scores
        ↓
online/gate5/build_online_gate5_2_ranker.py
        ↓
public.online_gate5_2_ranker
        ↓
online/gate5/build_online_gate5_3_decision.py
        ↓
public.online_gate5_3_decisions
        ↓
selector
        ↓
public.trading_signals

⸻

18. Selector

Selector выбирает итоговый сигнал.

Он должен учитывать:

* выбранный symbol;
* сторону LONG/SHORT;
* Gate2 threshold;
* Gate4 confidence;
* Gate5.1 threshold;
* Gate5.3 threshold;
* dynamic symbol filter;
* отсутствие открытой позиции;
* ranking/strength сигнала.

Итог записывается в:

public.trading_signals

Ключевой признак выбранного сигнала:

selected = TRUE
rejected = FALSE

⸻

19. Execution

Файл:

online/trading/execution.py

Основные шаги:

1. Проверить stale signal.
2. Проверить, что сигнал ещё не исполнялся.
3. Проверить Bybit на наличие открытой позиции.
4. Если Bybit пустой, но БД считает позицию активной — синхронизировать БД.
5. Рассчитать капитал.
6. Рассчитать qty.
7. Отправить market entry.
8. Получить фактическую среднюю цену исполнения.
9. Посчитать TP/SL от фактической цены.
10. Записать entry_avg_px, entry_slippage_abs, entry_slippage_pct.
11. Поставить TP/SL reduce-only trigger market orders.
12. Записать ордера в public.trading_orders.
13. Обновить public.trading_positions.

⸻

19.1. Фактическая цена входа

В БД должны сохраняться две цены:

entry_px_plan
entry_avg_px

Смысл:

entry_px_plan — цена, известная/плановая на момент расчёта
entry_avg_px — фактическая средняя цена исполнения market-ордера

Также сохраняется:

entry_slippage_abs = entry_avg_px - entry_px_plan
entry_slippage_pct = entry_slippage_abs / entry_px_plan

Для SHORT знак slippage может быть интерпретирован отдельно при анализе PnL, но как execution-diff хранится простая разница цен.

⸻

19.2. TP/SL от фактической цены

TP/SL считаются от:

entry_avg_px

А не от:

h4_close
entry_px_plan

Формулы:

LONG:

tp = entry_avg_px + TP_ATR * atr14
sl = entry_avg_px - SL_ATR * atr14

SHORT:

tp = entry_avg_px - TP_ATR * atr14
sl = entry_avg_px + SL_ATR * atr14

⸻

20. Bybit client

Файл:

online/trading/bybit_client.py

Основные методы:

get_wallet_balance_usdt()
get_ticker_last_price(symbol)
get_instrument_info(symbol)
place_market_order(...)
place_reduce_only_market_close(...)
place_tp_sl_orders(...)
get_executions(...)
get_avg_fill_price_by_link_id(...)
get_position(symbol)
get_open_positions(symbol=None)
cancel_order(...)

Важный метод для TTL/manual close:

place_reduce_only_market_close(...)

Важный метод для фактической цены входа:

get_avg_fill_price_by_link_id(...)

⸻

21. Monitor / Reconcile

Monitor:

online/trading/monitor.py

Reconcile:

online/trading/reconcile.py

Их задача:

* сверять локальную БД с Bybit;
* отслеживать активные позиции;
* отмечать ручное/внешнее закрытие;
* контролировать TTL;
* не давать БД жить отдельно от реальной биржи.

Главный production-принцип:

источником истины по факту открытой позиции является Bybit

БД хранит состояние системы, но при конфликте должна синхронизироваться с биржей.

⸻

22. Dynamic symbol filter

Фильтр символов должен считаться по закрытым production-сделкам.

Не по valid.

Не по backtest.

Смысл:

если символ в реальной торговле показывает плохие сделки,
он временно уходит в cooldown

Текущие параметры:

enabled = True
lookback_days = 30
min_trades = 4
min_wilson = 0.20
max_bad_streak = 2
base_cooldown_days = 7
max_cooldown_days = 30
return_mode = probation_after_cooldown
probation_success = net_ret > 0
probation_fail = net_ret <= 0

⸻

23. Production baseline

Текущий baseline:

PAIR_MODEL_NAME = tp225_sl075__vs__tp100_sl075
GRID_NAME = tp100_sl075

Пороги:

Gate2 = 0.70
Gate4 = 0.57
Gate5.1 = 0.10
Gate5.3 = 0.54

Сетка:

TP = 1.00 ATR
SL = 0.75 ATR

Капитал:

CHULAN_ENABLED = 1
CHULAN_BASE_CAPITAL_USDT = 100

Entry:

market order после завершения H4 pipeline

⸻

24. Backtest

Основной модуль:

online.trading.backtest_m1_thresholds

Запуск:

python -m online.trading.service_status backtest win

Интерактивно вводятся:

start UTC
end UTC
Gate2 threshold
Gate4 threshold
Gate5.1 threshold
Gate5.3 threshold
chulan
write dynamic blacklist
reset blacklist

Backtest должен учитывать:

* одну позицию одновременно;
* капитал после каждой сделки;
* комиссии;
* проскальзывание;
* entry timing;
* TP/SL/TTL;
* dynamic filter, если включён.

⸻

25. History

Команда:

python -m online.trading.service_status history 20 win

История должна показывать:

close
signal
symbol
decision
side
entry_signal
entry_plan
entry_actual
slip_pct
tp
sl
pos_status
reason
rank

Смысл цен:

entry_signal — цена H4/signal close
entry_plan — плановая цена перед execution
entry_actual — фактическая средняя цена входа
slip_pct — отличие фактического входа от плановой цены

⸻

26. Anti-leak правила

Во всех train/online ветках запрещено использовать future/outcome-поля как features.

Нельзя пускать в model features:

target
label
future
mfe
mae
tp_hit
sl_hit
first_tp
first_sl
entry_avg_px
exit_avg_px
pnl
fee
status
realized
closed

Для Gate5 особенно опасны:

target_score
delta_score
safe_target_score
agg_target_score
g5_*
safe_g5_*
agg_g5_*

Их наличие в features ломает честность модели.

⸻

27. ML stack

Основной production-инструмент:

CatBoost

Также исторически использовались:

LightGBM
PyTorch
sklearn
Optuna

Типовые CatBoost-параметры:

loss_function = Logloss
eval_metric = AUC
iterations = 4000–20000
learning_rate = 0.03
depth = 6/8
l2_leaf_reg = 6.0/10.0
random_seed = 42
od_type = Iter
use_best_model = True

⸻

28. Research

Папка:

research/

Содержит старые эксперименты:

research/models/
research/models_organized/
research/predict/
research/real/
research/reports/
research/scripts/
research/tools/
research/utils/

Это не production runtime.

Сюда вынесены:

* старые CatBoost-модели;
* старые отчёты;
* KS/TP-entry эксперименты;
* старые real/autotrade прототипы;
* старые scripts/tools/utils;
* historical research artifacts.


29. Что считается источником истины

Для online trading:

Bybit = источник истины по фактическим позициям
PostgreSQL = источник истории, сигналов, ордеров и внутреннего состояния

Если Bybit показывает open_positions_count = 0, а БД думает, что позиция активна, production-логика должна синхронизировать БД с биржей.

⸻

30. Критические правила проекта

1. Features считаются только по закрытой H4.
2. Entry происходит на следующей H4 через market order.
3. TP/SL считаются от фактической цены исполнения.
4. Открыта может быть только одна позиция.
5. Bybit важнее БД при проверке факта позиции.
6. Future/outcome поля не должны попадать в features.
7. Gate5 target/outcome поля особенно опасны для leakage.
8. Dynamic blacklist считается по production-сделкам.
9. Research-папка не является частью production runtime.
10. Смена путей/перенос папок требует проверки импортов.

⸻

31. Короткая схема production flow

H4 close
    ↓
sync candles
    ↓
build online features
    ↓
Gate2 inference
    ↓
Gate4 side inference
    ↓
Gate5.1 score
    ↓
Gate5.2 ranker
    ↓
Gate5.3 decision
    ↓
selector chooses one signal
    ↓
execution sends market order
    ↓
fetch actual entry avg price
    ↓
calculate TP/SL from actual entry
    ↓
place reduce-only TP/SL
    ↓
monitor / reconcile

⸻

32. Главный смысл проекта

ImbalanceSearcher — это не одна модель и не простой торговый скрипт.

Это production ML-pipeline, который решает инженерную задачу:

из большого количества шумовых H4-состояний выбрать редкие ситуации,
где вероятность успешной сделки достаточно высока,
выбрать сторону,
оценить TP/SL,
открыть позицию market-ордером,
и контролировать её через Bybit + PostgreSQL.
