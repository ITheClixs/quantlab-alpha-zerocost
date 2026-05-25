"""Three price series: tradable_* (split-adjusted execution-consistent),
split_adj_* (alias of tradable_* in v1), and total_return_* (split-
adjusted + dividend reinvested, used only for labels/diagnostics/
benchmarks — never portfolio MTM).  Spec §2.3."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class PriceSeriesBundle:
    tradable: pl.DataFrame
    split_adj: pl.DataFrame
    total_return: pl.DataFrame


_OHLCV_COLS: tuple[str, ...] = ("open", "high", "low", "close")


def build_three_series(
    *,
    panel: pl.DataFrame,
    dividends: pl.DataFrame,
    source_is_total_return: bool,
) -> PriceSeriesBundle:
    """Build the three price series from a daily-bars panel + dividend feed.

    If source_is_total_return=True, the upstream panel is assumed to already
    embed dividend reinvestment; tradable_* is recovered by removing the
    dividend ladder.
    """
    panel = panel.sort(["symbol", "date"])
    if source_is_total_return:
        tradable = _remove_dividend_reinvestment(panel, dividends)
    else:
        tradable = panel.clone()
    split_adj = tradable.clone()  # v1: alias; future raw-lot/share accounting splits this out
    total_return = _apply_dividend_reinvestment(tradable, dividends)
    total_return = total_return.rename({c: f"{c}_tr" for c in _OHLCV_COLS})
    return PriceSeriesBundle(tradable=tradable, split_adj=split_adj, total_return=total_return)


def _apply_dividend_reinvestment(panel: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    if dividends.is_empty():
        return panel.clone()
    panel = panel.sort(["symbol", "date"])
    divs = dividends.rename({"ex_date": "date"}).sort(["symbol", "date"])
    joined = panel.join(divs, on=["symbol", "date"], how="left").with_columns(
        pl.col("dividend_per_share").fill_null(0.0)
    )
    joined = joined.with_columns(
        pl.col("close").shift(1).over("symbol").alias("close_prior")
    ).with_columns(
        pl.when(pl.col("close_prior").is_not_null() & (pl.col("close_prior") > 0))
        .then(1.0 + pl.col("dividend_per_share") / pl.col("close_prior"))
        .otherwise(1.0)
        .alias("reinvest_factor")
    )
    joined = joined.with_columns(
        pl.col("reinvest_factor").cum_prod().over("symbol").alias("cum_factor")
    )
    out = joined.with_columns(
        [(pl.col(c) * pl.col("cum_factor")).alias(c) for c in _OHLCV_COLS]
    ).drop(["dividend_per_share", "close_prior", "reinvest_factor", "cum_factor"])
    return out


def _remove_dividend_reinvestment(panel: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    """Inverse of _apply_dividend_reinvestment."""
    if dividends.is_empty():
        return panel.clone()
    panel = panel.sort(["symbol", "date"])
    divs = dividends.rename({"ex_date": "date"}).sort(["symbol", "date"])
    joined = panel.join(divs, on=["symbol", "date"], how="left").with_columns(
        pl.col("dividend_per_share").fill_null(0.0)
    )
    joined = joined.with_columns(
        pl.col("close").shift(1).over("symbol").alias("close_prior")
    ).with_columns(
        pl.when(pl.col("close_prior").is_not_null() & (pl.col("close_prior") > 0))
        .then(1.0 + pl.col("dividend_per_share") / pl.col("close_prior"))
        .otherwise(1.0)
        .alias("reinvest_factor")
    )
    joined = joined.with_columns(
        pl.col("reinvest_factor").cum_prod().over("symbol").alias("cum_factor")
    )
    out = joined.with_columns(
        [(pl.col(c) / pl.col("cum_factor")).alias(c) for c in _OHLCV_COLS]
    ).drop(["dividend_per_share", "close_prior", "reinvest_factor", "cum_factor"])
    return out


def de_total_return_to_tradable(
    total_return: pl.DataFrame, dividends: pl.DataFrame
) -> pl.DataFrame:
    """Public helper for the prepare-equity-data script when the upstream
    HF dataset is vendor_total_return."""
    panel = total_return.rename({f"{c}_tr": c for c in _OHLCV_COLS})
    return _remove_dividend_reinvestment(panel, dividends)
