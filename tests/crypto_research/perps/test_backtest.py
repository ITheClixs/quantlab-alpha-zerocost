from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_research_stack.crypto_research.perps.backtest import (
    PerpBacktestConfig,
    run_event_backtest,
)


def _two_row_frame(*, prediction: float = 1.0) -> pl.DataFrame:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "event_time": [t0, t0 + timedelta(seconds=1)],
            "prediction": [prediction, 0.0],
            "best_bid": [100.0, 101.0],
            "best_ask": [100.2, 101.2],
            "future_best_bid_1": [101.0, None],
            "future_best_ask_1": [101.2, None],
            "relative_spread": [0.002, 0.002],
            "best_bid_size": [10.0, 10.0],
            "best_ask_size": [10.0, 10.0],
        }
    )


def test_event_backtest_long_uses_entry_ask_and_exit_bid() -> None:
    result = run_event_backtest(
        _two_row_frame(),
        config=PerpBacktestConfig(horizon=1, fee_bps=0.0, slippage_bps=0.0),
    )

    trade = result.trades.row(0, named=True)

    assert trade["side"] == "long"
    assert trade["entry_price"] == 100.2
    assert trade["exit_price"] == 101.0
    assert trade["gross_return"] == 101.0 / 100.2 - 1.0


def test_event_backtest_short_uses_entry_bid_and_exit_ask() -> None:
    result = run_event_backtest(
        _two_row_frame(prediction=-1.0),
        config=PerpBacktestConfig(horizon=1, fee_bps=0.0, slippage_bps=0.0),
    )

    trade = result.trades.row(0, named=True)

    assert trade["side"] == "short"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 101.2
    assert trade["gross_return"] == 100.0 / 101.2 - 1.0


def test_event_backtest_cost_multiplier_reduces_net_return() -> None:
    frame = _two_row_frame()

    low_cost = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, fee_bps=1.0, cost_multiplier=1.0))
    high_cost = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, fee_bps=1.0, cost_multiplier=3.0))

    assert high_cost.metrics["net_total_return"] < low_cost.metrics["net_total_return"]
    assert high_cost.trades["cost_return"][0] > low_cost.trades["cost_return"][0]


def test_event_backtest_latency_shifts_signal_forward() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "event_time": [t0 + timedelta(seconds=i) for i in range(3)],
            "prediction": [1.0, 0.0, 0.0],
            "best_bid": [100.0, 101.0, 102.0],
            "best_ask": [100.2, 101.2, 102.2],
            "future_best_bid_1": [101.0, 102.0, None],
            "future_best_ask_1": [101.2, 102.2, None],
            "relative_spread": [0.002, 0.002, 0.002],
            "best_bid_size": [10.0, 10.0, 10.0],
            "best_ask_size": [10.0, 10.0, 10.0],
        }
    )

    result = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, latency_events=1))

    assert result.trades.height == 1
    assert result.trades["event_time"][0] == t0 + timedelta(seconds=1)


def test_event_backtest_filters_wide_spreads_and_shallow_depth() -> None:
    frame = _two_row_frame()

    wide = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, max_relative_spread=0.0001))
    shallow = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, min_top_of_book_depth=11.0))

    assert wide.trades.is_empty()
    assert shallow.trades.is_empty()
