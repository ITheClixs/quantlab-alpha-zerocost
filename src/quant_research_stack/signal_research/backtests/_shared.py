"""Shared helpers for cross-sectional backtests.

Used by triple_barrier_av_lee.py and multi_model_fixture.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha_eq.backtest.costs import CostConfig
from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import BacktestConfig


def to_m4_panel(
    *,
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    sectors: dict[str, str] | None = None,
    spread_bps: float = 1.0,
    borrow_tier: str = "general",
) -> pl.DataFrame:
    sorted_bars = bars.sort(["symbol", "date"]).with_columns(
        (
            pl.col("close").shift(1).over("symbol")
            * pl.col("volume").shift(1).over("symbol")
        )
        .rolling_mean(window_size=20)
        .over("symbol")
        .alias("adv_20d_dollar_lag1"),
    )
    panel = sorted_bars.join(signals, on=["date", "symbol"], how="left").with_columns(
        pl.col("y_xs_pred").fill_null(0.0),
        pl.col("date").alias("execution_date"),
        (pl.col("date") - pl.duration(days=1)).alias("feature_as_of_date"),
        pl.lit(True).alias("tradable"),
        pl.lit(True).alias("in_pit_universe"),
        pl.lit(borrow_tier).alias("borrow_tier"),
        pl.lit(spread_bps).alias("roll_spread_bps"),
    )
    if sectors:
        mapping_df = pl.DataFrame(
            {"symbol": list(sectors.keys()), "sector": list(sectors.values())}
        )
        panel = panel.join(mapping_df, on="symbol", how="left").with_columns(
            pl.col("sector").fill_null("unknown")
        )
    else:
        panel = panel.with_columns(pl.lit("unknown").alias("sector"))
    return panel.drop_nulls(
        subset=["open", "high", "low", "close", "adv_20d_dollar_lag1"]
    )


def sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns)) / sd * float(np.sqrt(252.0))


def equity_metrics(daily: pl.DataFrame) -> dict[str, float]:
    rets = daily["net_return"].to_numpy().astype(np.float64)
    if rets.size == 0:
        return {"sharpe": 0.0, "max_dd": 0.0, "cum_return": 0.0, "n_days": 0}
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0).min()
    return {
        "sharpe": sharpe(rets),
        "max_dd": float(dd),
        "cum_return": float(equity[-1] - 1.0),
        "n_days": int(rets.size),
    }


def build_backtest_config(
    *,
    commission_bps_one_way: float,
    spread_bps_one_way: float,
    cost_stress_mult: float,
    q_quantile: float,
    target_gross: float,
    equity: float,
    cohort: str,
    financing_rate_annual: float = 0.05,
) -> BacktestConfig:
    return BacktestConfig(
        portfolio=PortfolioBuildConfig(
            q_quantile=q_quantile,
            target_gross=target_gross,
            equity=equity,
        ),
        fill_model=FillModel.CLOSE,
        cohort=cohort,
        borrow_multiplier=cost_stress_mult,
        financing_rate_annual=financing_rate_annual,
        cost=CostConfig(
            commission_bps_one_way=commission_bps_one_way * cost_stress_mult,
            tiered_fallback_easy_bps=5.0 * cost_stress_mult,
            tiered_fallback_general_bps=spread_bps_one_way * 10.0 * cost_stress_mult,
            tiered_fallback_hard_bps=spread_bps_one_way * 30.0 * cost_stress_mult,
        ),
    )


def data_quality_banner(
    *, data_quality_label: str, constituent_survivorship_applicable: bool
) -> str:
    return (
        f"DATA QUALITY: data_quality_label={data_quality_label}, "
        f"constituent_survivorship_applicable={constituent_survivorship_applicable}. "
        "Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time "
        "membership feed. Results may overstate alpha due to survivorship bias. "
        "Institutional-grade labels (per spec §5.4) are NOT allowed for this run."
    )
