"""Backtest runner end-to-end on tiny synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)


def _toy_signals_panel(n_days: int = 30, n_symbols: int = 25) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_days):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(n_symbols):
            rows.append(
                {
                    "execution_date": d,
                    "feature_as_of_date": d - timedelta(days=1),
                    "symbol": f"S{s}",
                    "y_xs_pred": float(rng.standard_normal()),
                    "open": 100.0 + float(rng.standard_normal()),
                    "high": 101.0 + float(rng.standard_normal()),
                    "low": 99.0 + float(rng.standard_normal()),
                    "close": 100.0 + float(rng.standard_normal()),
                    "adv_20d_dollar_lag1": 1e8,
                    "tradable": True,
                    "in_pit_universe": True,
                    "borrow_tier": "general",
                    "roll_spread_bps": 10.0,
                    "sector": ["tech", "finance", "energy"][s % 3],
                }
            )
    return pl.DataFrame(rows)


def test_run_backtest_produces_pnl_series() -> None:
    cfg = BacktestConfig(
        portfolio=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
        fill_model=FillModel.OPEN,
        cohort="full_universe",
        borrow_multiplier=1.0,
        financing_rate_annual=0.0,
    )
    res = run_backtest(signals_with_bars=_toy_signals_panel(), config=cfg, dividends=None)
    assert isinstance(res, BacktestResult)
    assert res.daily_returns.height > 0
    assert "net_return" in res.daily_returns.columns
