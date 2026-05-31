"""Backtest metrics (spec §5.13)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray


def annualized_return(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    cum = float(np.prod(1.0 + returns))
    return float(cum ** (252.0 / returns.size) - 1.0)


def annualized_sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return float("inf") if float(np.mean(returns)) > 0 else 0.0
    return float(np.mean(returns)) / sd * np.sqrt(252.0)


def annualized_sortino(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0]
    sd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    if sd == 0.0:
        return float("inf") if float(np.mean(returns)) > 0 else 0.0
    return float(np.mean(returns)) / sd * np.sqrt(252.0)


def max_drawdown(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: NDArray[np.float64]) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return annualized_return(returns) / abs(mdd)


def monthly_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("year_month"))
        .group_by("year_month")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("monthly_return"))
        .sort("year_month")
    )


def annual_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").dt.strftime("%Y").alias("year"))
        .group_by("year")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("annual_return"))
        .sort("year")
    )
