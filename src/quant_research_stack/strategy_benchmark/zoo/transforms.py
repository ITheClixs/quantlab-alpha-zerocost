"""Composable transforms turning a raw family signal into a final position series.
Order: family signal -> position_mode -> vol_target -> holding."""

from __future__ import annotations

import polars as pl

_POSITION_CAP = 3.0  # never lever a single asset beyond 3x in this demo


def apply_position_mode(signal: pl.Series, *, mode: str) -> pl.Series:
    if mode == "long_short":
        return signal
    if mode == "long_only":
        return signal.clip(lower_bound=0.0)
    raise ValueError(f"unknown position mode {mode!r}")


def apply_vol_target(signal: pl.Series, *, vol: pl.Series, target_daily_vol: float) -> pl.Series:
    df = pl.DataFrame({"s": signal, "v": vol}).with_columns(
        pl.when(pl.col("v").is_not_null() & (pl.col("v") > 0.0))
        .then((pl.col("s") * (target_daily_vol / pl.col("v"))).clip(-_POSITION_CAP, _POSITION_CAP))
        .otherwise(0.0)
        .alias("scaled")
    )
    return df["scaled"]


def apply_holding(signal: pl.Series, *, holding: int) -> pl.Series:
    if holding <= 1:
        return signal
    df = pl.DataFrame({"s": signal}).with_columns(
        pl.when(pl.col("s") != 0.0).then(pl.col("s")).otherwise(None).alias("_nz")
    ).with_columns(
        pl.col("_nz").fill_null(strategy="forward", limit=holding - 1).alias("held")
    ).with_columns(pl.col("held").fill_null(0.0))
    return df["held"]
