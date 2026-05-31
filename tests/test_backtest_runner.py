from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.backtest.runner import BacktestConfig, BacktestRunner
from quant_research_stack.brokers.fill_model import FillModelConfig
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _tick(price: float, minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


@pytest.mark.asyncio
async def test_buy_and_hold_runs_end_to_end_and_emits_one_fill() -> None:
    events = [_tick(100.0, i) for i in range(10)]

    cfg = BacktestConfig(
        events=events,
        fill_model=FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0),
        starting_cash=100_000.0,
        strategy_name="buy_and_hold",
        strategy_params={"symbol": "BTCUSDT", "quantity": 1.0},
        metrics_horizon_minutes=1,
    )
    runner = BacktestRunner(cfg)
    result = await runner.run()
    assert len(result.fills) == 1
    assert result.equity_curve.height == len(events)


@pytest.mark.asyncio
async def test_unknown_strategy_raises() -> None:
    cfg = BacktestConfig(
        events=[_tick(100.0, 0)],
        fill_model=FillModelConfig(),
        starting_cash=100_000.0,
        strategy_name="does_not_exist",
        strategy_params={},
    )
    runner = BacktestRunner(cfg)
    with pytest.raises(ValueError):
        await runner.run()


@pytest.mark.asyncio
async def test_two_identical_runs_produce_identical_metrics() -> None:
    events = [_tick(100.0 + i * 0.1, i) for i in range(20)]
    cfg = BacktestConfig(
        events=events,
        fill_model=FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0),
        starting_cash=100_000.0,
        strategy_name="moving_average_cross",
        strategy_params={"symbol": "BTCUSDT", "quantity": 1.0, "fast_window": 3, "slow_window": 5},
    )
    a = await BacktestRunner(cfg).run()
    b = await BacktestRunner(cfg).run()
    assert a.metrics == b.metrics
    assert a.equity_curve.to_dicts() == b.equity_curve.to_dicts()
