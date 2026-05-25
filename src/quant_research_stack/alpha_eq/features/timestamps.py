"""Feature/execution timestamp invariants (spec §3.1)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


class TimestampContractError(RuntimeError):
    pass


def attach_feature_as_of_date(df: pl.DataFrame, *, convention: str) -> pl.DataFrame:
    if convention != "after_close_t":
        raise ValueError(f"unsupported convention: {convention!r}")
    return df.with_columns(pl.col("date").alias("feature_as_of_date"))


def attach_execution_date(df: pl.DataFrame, *, convention: str) -> pl.DataFrame:
    """Next trading day; weekends skipped. Holiday handling is deferred."""
    if convention != "next_trading_day":
        raise ValueError(f"unsupported convention: {convention!r}")

    def _bump(d: date) -> date:
        nd = d + timedelta(days=1)
        while nd.weekday() >= 5:
            nd += timedelta(days=1)
        return nd

    return df.with_columns(
        pl.col("date").map_elements(_bump, return_dtype=pl.Date).alias("execution_date")
    )


def assert_feature_before_execution(df: pl.DataFrame) -> None:
    """Hard invariant — runtime assert by training and backtest entry points."""
    if "feature_as_of_date" not in df.columns or "execution_date" not in df.columns:
        raise TimestampContractError("missing feature_as_of_date or execution_date columns")
    bad = df.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise TimestampContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )
