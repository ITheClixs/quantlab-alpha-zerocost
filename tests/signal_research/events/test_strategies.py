from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.events.strategies import (
    backtest_positions,
    buy_and_hold,
    daily_returns,
    risk_off_gate,
    risk_on_gate,
    sma_gate,
    vol_target_position,
)


def test_daily_returns_first_is_zero() -> None:
    close = np.array([100.0, 110.0, 99.0])
    r = daily_returns(close)
    assert r[0] == 0.0
    assert abs(r[1] - 0.1) < 1e-12
    assert abs(r[2] - (99.0 / 110.0 - 1.0)) < 1e-12


def test_risk_off_gate_is_flat_in_window() -> None:
    flag = np.array([False, True, True, False])
    pos = risk_off_gate(flag)
    assert pos.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert risk_on_gate(flag).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_risk_off_reduces_drawdown_when_window_is_the_bad_day() -> None:
    # day 2 is a -10% crash; risk-off gate flat that day should beat buy-and-hold
    returns = np.array([0.0, 0.01, -0.10, 0.01, 0.01])
    flag = np.array([False, False, True, False, False])
    bah = backtest_positions(returns, buy_and_hold(len(returns)), cost_oneway_bps=0.0)
    gated = backtest_positions(returns, risk_off_gate(flag), cost_oneway_bps=0.0)
    assert gated.metrics["total_return"] > bah.metrics["total_return"]
    assert gated.metrics["max_drawdown"] >= bah.metrics["max_drawdown"]  # less negative


def test_cost_reduces_returns_with_turnover() -> None:
    returns = np.array([0.0, 0.01, 0.01, 0.01, 0.01])
    flag = np.array([False, True, False, True, False])  # toggling -> turnover
    free = backtest_positions(returns, risk_off_gate(flag), cost_oneway_bps=0.0)
    costed = backtest_positions(returns, risk_off_gate(flag), cost_oneway_bps=10.0)
    assert costed.metrics["total_return"] < free.metrics["total_return"]
    assert costed.metrics["turnover"] > 0.0


def test_vol_target_and_sma_are_causal_shifted() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 300)
    close = 100.0 * np.cumprod(1.0 + returns)
    vt = vol_target_position(returns, lookback=20)
    sma = sma_gate(close, fast=50, slow=200)
    # first values are zero (insufficient lookback) and arrays are full length
    assert vt.shape == returns.shape and sma.shape == close.shape
    assert vt[0] == 0.0 and sma[0] == 0.0
    assert np.all(vt >= 0.0)


def test_delay_shifts_position() -> None:
    returns = np.array([0.0, 0.01, -0.10, 0.01, 0.01])
    flag = np.array([False, False, True, False, False])
    base = backtest_positions(returns, risk_off_gate(flag), cost_oneway_bps=0.0, delay=0)
    delayed = backtest_positions(returns, risk_off_gate(flag), cost_oneway_bps=0.0, delay=1)
    # delaying the gate by a bar makes it miss the crash day -> worse
    assert delayed.metrics["total_return"] < base.metrics["total_return"]
