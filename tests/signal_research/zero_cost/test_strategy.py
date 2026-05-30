from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from quant_research_stack.signal_research.zero_cost.strategy import (
    InstrumentSeries,
    apply_execution_shift,
    backtest_portfolio,
    daily_returns,
    trend_on,
    vol_target_weight,
    weeklyize,
)


def _dates(n: int) -> list[date]:
    d, out = date(2020, 1, 1), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_daily_returns_and_vol_target_causal() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 300)
    w = vol_target_weight(r, target_ann_vol=0.12, lookback=20, cap=1.5)
    assert w.shape == r.shape
    assert w[0] == 0.0  # insufficient lookback
    assert np.all((w >= 0.0) & (w <= 1.5))
    # lower (nonzero) realized vol -> higher target weight
    lo = vol_target_weight(rng.normal(0, 0.003, 300), lookback=20)
    hi = vol_target_weight(rng.normal(0, 0.03, 300), lookback=20)
    assert np.nanmean(lo[20:]) >= np.nanmean(hi[20:])


def test_execution_shift_direction_blocks_lookahead() -> None:
    # returns: day 3 is a -10% crash; a gate that is 0 ON day 3 must, after the
    # t+1 execution shift, only take effect the day AFTER -> it does NOT dodge the crash
    returns = np.array([0.0, 0.01, 0.01, -0.10, 0.01])
    raw_weight = np.array([1.0, 1.0, 1.0, 0.0, 1.0])  # "decided at close t"
    held = apply_execution_shift(raw_weight, delay=1)
    # held[3] should equal raw_weight[2] = 1.0 (still long into the crash) -> no look-ahead
    assert held[3] == 1.0
    assert held[4] == 0.0


def test_weeklyize_holds_within_week() -> None:
    ds = _dates(10)  # two ISO weeks
    w = np.arange(10, dtype=float)
    wk = weeklyize(w, ds)
    # within the first week, the weight stays at the week's first value (0.0)
    assert wk[0] == 0.0 and wk[1] == 0.0
    # a new week resets to that day's value
    assert len(set(wk.tolist())) <= 3  # ~2-3 distinct weekly holds across 10 business days


def test_trend_on_causal() -> None:
    close = np.cumprod(1.0 + np.full(300, 0.001)) * 100
    t = trend_on(close, slow=200)
    assert t[0] == np.False_  # insufficient lookback
    assert t[-1] == np.True_  # uptrend


def test_backtest_portfolio_equal_risk_and_cost() -> None:
    ds = _dates(60)
    n = len(ds)
    rng = np.random.default_rng(1)
    insts = {
        "A": InstrumentSeries("A", ds, np.ones(n), rng.normal(0, 0.01, n), False),
        "B": InstrumentSeries("B", ds, np.ones(n), rng.normal(0, 0.01, n), True),
    }
    tw = {"A": np.ones(n), "B": np.ones(n)}
    free = backtest_portfolio(insts, tw, dates=ds, cost_bps={"A": 0.0, "B": 0.0}, delay=1, weekly=False)
    costed = backtest_portfolio(insts, tw, dates=ds, cost_bps={"A": 50.0, "B": 50.0}, delay=1, weekly=False)
    assert free.daily_returns.shape == (n,)
    # equal-risk split: each instrument weight is 1/2 when both active
    assert abs(free.weights["A"][5] - 0.5) < 1e-9
    assert costed.metrics["total_return"] <= free.metrics["total_return"]
