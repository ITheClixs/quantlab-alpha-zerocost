"""Backtest metrics (spec §5.13)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.metrics import (
    annualized_return,
    annualized_sharpe,
    calmar_ratio,
    max_drawdown,
    monthly_returns,
)


def test_sharpe_of_constant_positive() -> None:
    r = np.full(252, 0.001)
    assert annualized_sharpe(r) > 100.0


def test_max_drawdown_negative() -> None:
    r = np.array([0.01, 0.01, -0.30, 0.05])
    mdd = max_drawdown(r)
    assert mdd < 0


def test_monthly_returns_aggregate() -> None:
    n = 60
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), interval="1d", eager=True).head(n)
    r = np.full(n, 0.001)
    df = pl.DataFrame({"date": dates, "net_return": r})
    m = monthly_returns(df)
    assert "year_month" in m.columns
    assert m.height >= 2


def test_calmar_ratio_uses_max_dd() -> None:
    r = np.array([0.001, 0.001, -0.05, 0.001, 0.001])
    c = calmar_ratio(r)
    # just a numeric finite value
    assert c == c  # not NaN


def test_annualized_return_zero() -> None:
    r = np.full(252, 0.0)
    assert abs(annualized_return(r) - 0.0) < 1e-12
