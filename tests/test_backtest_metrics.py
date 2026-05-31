from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.backtest.metrics import (
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
    turnover,
    value_at_risk,
)
from quant_research_stack.brokers.order_types import Fill, OrderSide


def _eq_curve(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"equity": values})


def test_total_return_basic() -> None:
    assert total_return(_eq_curve([100.0, 110.0])) == pytest.approx(0.10, rel=1e-9)


def test_total_return_zero_initial_returns_zero() -> None:
    assert total_return(_eq_curve([0.0, 100.0])) == 0.0


def test_max_drawdown_zero_when_monotonic_up() -> None:
    assert max_drawdown(_eq_curve([100.0, 101.0, 110.0])) == pytest.approx(0.0)


def test_max_drawdown_negative_value() -> None:
    dd = max_drawdown(_eq_curve([100.0, 110.0, 99.0, 105.0]))
    assert dd == pytest.approx(-(110.0 - 99.0) / 110.0, rel=1e-9)


def test_sharpe_basic_positive() -> None:
    returns = pl.Series("r", [0.001, 0.002, -0.001, 0.0015])
    s = sharpe_ratio(returns, periods_per_year=252)
    assert s > 0


def test_sharpe_zero_volatility_returns_zero() -> None:
    returns = pl.Series("r", [0.001, 0.001, 0.001, 0.001])
    assert sharpe_ratio(returns, periods_per_year=252) == 0.0


def test_hit_rate_alternating() -> None:
    from datetime import UTC, datetime
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="X", side=OrderSide.buy,
             price=100.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC)),
        Fill(client_order_id="b", fill_id="2", symbol="X", side=OrderSide.sell,
             price=110.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC)),
        Fill(client_order_id="c", fill_id="3", symbol="X", side=OrderSide.buy,
             price=120.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 3, tzinfo=UTC)),
        Fill(client_order_id="d", fill_id="4", symbol="X", side=OrderSide.sell,
             price=115.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 4, tzinfo=UTC)),
    ]
    assert hit_rate(fills) == 0.5


def test_turnover_sums_notionals_normalized_by_capital() -> None:
    from datetime import UTC, datetime
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="X", side=OrderSide.buy,
             price=100.0, quantity=2.0, timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    assert turnover(fills, starting_cash=1000.0) == pytest.approx(200.0 / 1000.0)


def test_value_at_risk_returns_left_tail() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    returns = pl.Series("r", rng.normal(size=1000).tolist())
    var5 = value_at_risk(returns, alpha=0.05)
    assert var5 < 0
