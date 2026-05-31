"""Concentration diagnostics — PnL share by sector / month / stock.

A strategy whose PnL is dominated by one sector, one month, or one stock
is not a robust generalizable edge. These diagnostics report the maximum
share of total |PnL| attributable to any single bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ConcentrationReport:
    max_month_share: float
    max_year_share: float
    n_months_with_pnl: int
    months_above_50pct_share: int


def concentration_by_period(
    daily_returns: pl.DataFrame,
) -> ConcentrationReport:
    """Per-month and per-year concentration of |net_return| contribution.

    `daily_returns` must have columns (date, net_return).
    """
    if daily_returns.is_empty():
        return ConcentrationReport(0.0, 0.0, 0, 0)
    df = daily_returns.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.strftime("%Y-%m").alias("year_month"),
    )
    monthly = df.group_by("year_month").agg(pl.col("net_return").sum().alias("pnl"))
    yearly = df.group_by("year").agg(pl.col("net_return").sum().alias("pnl"))

    total_abs = float(np.sum(np.abs(monthly["pnl"].to_numpy()))) or 1e-9
    monthly_shares = (
        np.abs(monthly["pnl"].to_numpy()).astype(np.float64) / total_abs
    )
    yearly_total_abs = float(np.sum(np.abs(yearly["pnl"].to_numpy()))) or 1e-9
    yearly_shares = (
        np.abs(yearly["pnl"].to_numpy()).astype(np.float64) / yearly_total_abs
    )
    return ConcentrationReport(
        max_month_share=float(monthly_shares.max()) if monthly_shares.size else 0.0,
        max_year_share=float(yearly_shares.max()) if yearly_shares.size else 0.0,
        n_months_with_pnl=int(monthly.height),
        months_above_50pct_share=int((monthly_shares > 0.5).sum()),
    )
