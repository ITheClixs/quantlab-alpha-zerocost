"""Exposure diagnostics (spec §5.12)."""

from __future__ import annotations

import polars as pl


def compute_daily_exposures(*, positions: pl.DataFrame) -> pl.DataFrame:
    by_date = positions.group_by("date").agg(
        pl.col("signed_notional").sum().alias("net_exposure"),
        pl.col("signed_notional").abs().sum().alias("gross_exposure"),
        pl.col("signed_notional")
        .filter(pl.col("signed_notional") > 0)
        .sum()
        .alias("long_exposure"),
        pl.col("signed_notional")
        .filter(pl.col("signed_notional") < 0)
        .abs()
        .sum()
        .alias("short_exposure"),
    )
    if "sector" in positions.columns:
        sec = positions.with_columns(
            pl.when(pl.col("signed_notional") > 0)
            .then(pl.col("signed_notional"))
            .otherwise(0.0)
            .alias("_long_sec"),
            pl.when(pl.col("signed_notional") < 0)
            .then(-pl.col("signed_notional"))
            .otherwise(0.0)
            .alias("_short_sec"),
        )
        long_pivot = sec.group_by(["date", "sector"]).agg(
            pl.col("_long_sec").sum().alias("v")
        ).pivot(values="v", index="date", on="sector")
        long_pivot = long_pivot.rename(
            {c: f"sector_long_{c}" for c in long_pivot.columns if c != "date"}
        )
        short_pivot = sec.group_by(["date", "sector"]).agg(
            pl.col("_short_sec").sum().alias("v")
        ).pivot(values="v", index="date", on="sector")
        short_pivot = short_pivot.rename(
            {c: f"sector_short_{c}" for c in short_pivot.columns if c != "date"}
        )
        by_date = (
            by_date.join(long_pivot, on="date", how="left")
            .join(short_pivot, on="date", how="left")
            .fill_null(0.0)
        )
    return by_date


def rolling_spy_beta(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """OLS beta of portfolio_return on spy_return over a rolling window."""
    return df.sort("date").with_columns(
        (
            (
                (pl.col("portfolio_return") * pl.col("spy_return"))
                .rolling_mean(window_size=window, min_samples=window)
                - pl.col("portfolio_return").rolling_mean(window_size=window, min_samples=window)
                * pl.col("spy_return").rolling_mean(window_size=window, min_samples=window)
            )
            / (
                pl.col("spy_return")
                .rolling_var(window_size=window, min_samples=window)
                .clip(lower_bound=1e-18)
            )
        ).alias("rolling_spy_beta")
    )


def top_n_contributors(pnl: pl.DataFrame, *, by: str, n: int) -> pl.DataFrame:
    return (
        pnl.group_by(by)
        .agg(pl.col("net_pnl").sum().alias("total_net_pnl"))
        .sort("total_net_pnl", descending=True)
        .head(n)
        .rename({by: by})
    )
