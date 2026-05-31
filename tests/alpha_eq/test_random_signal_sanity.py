"""Random predictions must not produce stable positive Sharpe (spec §6.2)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    run_backtest,
)


def test_random_signals_no_stable_positive_sharpe() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(25):
            rows.append({
                "execution_date": d,
                "feature_as_of_date": d - timedelta(days=1),
                "symbol": f"S{s}",
                "y_xs_pred": float(rng.standard_normal()),
                "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                "adv_20d_dollar_lag1": 1e8,
                "tradable": True, "in_pit_universe": True,
                "borrow_tier": "general", "roll_spread_bps": 10.0, "sector": "tech",
            })
    df = pl.DataFrame(rows)
    res = run_backtest(
        signals_with_bars=df,
        config=BacktestConfig(
            portfolio=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
            fill_model=FillModel.OPEN, cohort="full_universe",
            borrow_multiplier=1.0, financing_rate_annual=0.0,
        ),
        dividends=None,
    )
    sharpe_proxy = float(res.daily_returns["net_return"].mean() or 0.0) * (252 ** 0.5)
    assert abs(sharpe_proxy) < 1.0
