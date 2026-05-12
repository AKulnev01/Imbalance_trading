from __future__ import annotations

from online.trading.db import db_cursor


def ensure_trading_schema() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS public.trading_signals (
        signal_key TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        signal_ts TIMESTAMPTZ NOT NULL,
        entry_ts_plan TIMESTAMPTZ NOT NULL,
        side TEXT NOT NULL,

        pair_model_name TEXT NOT NULL,
        grid_name TEXT NOT NULL,
        tp_atr DOUBLE PRECISION NOT NULL,
        sl_atr DOUBLE PRECISION NOT NULL,
        ttl_hours DOUBLE PRECISION NOT NULL,

        gate2_proba DOUBLE PRECISION,
        gate4_confidence DOUBLE PRECISION,
        gate5_1_proba DOUBLE PRECISION,
        gate5_3_proba DOUBLE PRECISION,
        signal_strength DOUBLE PRECISION,

        selected BOOLEAN NOT NULL DEFAULT FALSE,
        rejected BOOLEAN NOT NULL DEFAULT FALSE,
        reject_reason TEXT,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_ts
    ON public.trading_signals (signal_ts DESC);

    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_ts
    ON public.trading_signals (symbol, signal_ts DESC);

    CREATE TABLE IF NOT EXISTS public.trading_positions (
        trade_id BIGSERIAL PRIMARY KEY,
        signal_key TEXT NOT NULL REFERENCES public.trading_signals(signal_key),

        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        status TEXT NOT NULL,

        entry_order_id TEXT,
        tp_order_id TEXT,
        sl_order_id TEXT,

        qty DOUBLE PRECISION,
        entry_avg_px DOUBLE PRECISION,
        entry_filled_at TIMESTAMPTZ,

        tp_px_plan DOUBLE PRECISION,
        sl_px_plan DOUBLE PRECISION,
        ttl_close_ts TIMESTAMPTZ,

        exit_reason TEXT,
        exit_order_id TEXT,
        exit_avg_px DOUBLE PRECISION,
        exit_filled_at TIMESTAMPTZ,

        gross_ret DOUBLE PRECISION,
        net_ret DOUBLE PRECISION,
        pnl_usd DOUBLE PRECISION,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        UNIQUE(signal_key)
    );

    CREATE INDEX IF NOT EXISTS idx_trading_positions_status
    ON public.trading_positions (status);

    CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol_created
    ON public.trading_positions (symbol, created_at DESC);

    CREATE TABLE IF NOT EXISTS public.trading_orders (
        local_order_key TEXT PRIMARY KEY,
        trade_id BIGINT REFERENCES public.trading_positions(trade_id),

        signal_key TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_role TEXT NOT NULL,

        bybit_order_id TEXT,
        bybit_order_link_id TEXT,

        order_type TEXT NOT NULL,
        reduce_only BOOLEAN NOT NULL DEFAULT FALSE,

        qty_plan DOUBLE PRECISION,
        price_plan DOUBLE PRECISION,
        trigger_price_plan DOUBLE PRECISION,

        status TEXT NOT NULL,
        sent_at TIMESTAMPTZ,
        acknowledged_at TIMESTAMPTZ,
        filled_at TIMESTAMPTZ,

        avg_fill_px DOUBLE PRECISION,
        filled_qty DOUBLE PRECISION,
        fee_usd DOUBLE PRECISION,

        error_code TEXT,
        error_message TEXT,

        request_json JSONB,
        response_json JSONB,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        UNIQUE(bybit_order_id),
        UNIQUE(signal_key, order_role)
    );

    CREATE INDEX IF NOT EXISTS idx_trading_orders_trade_role
    ON public.trading_orders (trade_id, order_role);

    CREATE TABLE IF NOT EXISTS public.trading_fills (
        fill_id BIGSERIAL PRIMARY KEY,
        trade_id BIGINT REFERENCES public.trading_positions(trade_id),
        local_order_key TEXT REFERENCES public.trading_orders(local_order_key),

        symbol TEXT NOT NULL,
        bybit_order_id TEXT,
        bybit_exec_id TEXT,

        side TEXT,
        price DOUBLE PRECISION,
        qty DOUBLE PRECISION,
        fee DOUBLE PRECISION,
        fee_currency TEXT,

        executed_at TIMESTAMPTZ,
        raw_json JSONB,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        UNIQUE(bybit_exec_id)
    );

    CREATE TABLE IF NOT EXISTS public.trading_trade_events (
        event_id BIGSERIAL PRIMARY KEY,
        trade_id BIGINT REFERENCES public.trading_positions(trade_id),
        signal_key TEXT,
        symbol TEXT,
        event_type TEXT NOT NULL,
        event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        details JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE INDEX IF NOT EXISTS idx_trading_trade_events_trade_ts
    ON public.trading_trade_events (trade_id, event_ts DESC);

    CREATE TABLE IF NOT EXISTS public.trading_dynamic_symbol_state (
        symbol TEXT PRIMARY KEY,

        is_blocked BOOLEAN NOT NULL DEFAULT FALSE,
        blocked_until TIMESTAMPTZ,
        cooldown_days INTEGER NOT NULL DEFAULT 0,
        bad_streak INTEGER NOT NULL DEFAULT 0,

        lookback_days INTEGER,
        hist_n INTEGER,
        hist_k INTEGER,
        hist_wilson DOUBLE PRECISION,
        hist_sum_net_ret DOUBLE PRECISION,

        probation_active BOOLEAN NOT NULL DEFAULT FALSE,
        last_reason TEXT,

        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS public.trading_runtime_locks (
        lock_name TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS public.trading_backtest_shadow (
        shadow_id BIGSERIAL PRIMARY KEY,
        trade_id BIGINT REFERENCES public.trading_positions(trade_id),
        signal_key TEXT NOT NULL,

        symbol TEXT NOT NULL,
        side TEXT NOT NULL,

        signal_ts TIMESTAMPTZ NOT NULL,
        entry_ts_plan TIMESTAMPTZ NOT NULL,

        entry_px_shadow DOUBLE PRECISION,
        tp_px_shadow DOUBLE PRECISION,
        sl_px_shadow DOUBLE PRECISION,
        exit_ts_shadow TIMESTAMPTZ,
        exit_px_shadow DOUBLE PRECISION,
        exit_reason_shadow TEXT,

        gross_ret_shadow DOUBLE PRECISION,
        net_ret_shadow DOUBLE PRECISION,

        live_entry_avg_px DOUBLE PRECISION,
        live_exit_avg_px DOUBLE PRECISION,

        entry_slippage_vs_shadow_pct DOUBLE PRECISION,
        exit_slippage_vs_shadow_pct DOUBLE PRECISION,
        total_slippage_vs_shadow_pct DOUBLE PRECISION,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        UNIQUE(trade_id)
    );

    CREATE TABLE IF NOT EXISTS public.trading_audit_latency (
        audit_id BIGSERIAL PRIMARY KEY,
        signal_key TEXT,
        symbol TEXT,

        h4_open_ts TIMESTAMPTZ,
        h4_close_ts TIMESTAMPTZ,

        orchestrator_started_at TIMESTAMPTZ,
        selector_finished_at TIMESTAMPTZ,
        order_sent_at TIMESTAMPTZ,
        order_ack_at TIMESTAMPTZ,
        first_fill_at TIMESTAMPTZ,

        seconds_close_to_orchestrator DOUBLE PRECISION,
        seconds_close_to_order_sent DOUBLE PRECISION,
        seconds_order_sent_to_ack DOUBLE PRECISION,
        seconds_order_sent_to_first_fill DOUBLE PRECISION,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    with db_cursor(commit=True) as (_, cur):
        cur.execute(ddl)


def main() -> None:
    ensure_trading_schema()
    print("DONE: trading schema is ready")


if __name__ == "__main__":
    main()


# === AUTO ADDED: trading audit tables ===
if __name__ == "__main__":
    from online.trading.audit_log import ensure_audit_tables
    ensure_audit_tables()
    print("DONE: trading audit tables are ready")
# === END AUTO ADDED: trading audit tables ===
