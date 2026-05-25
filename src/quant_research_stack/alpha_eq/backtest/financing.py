"""Financing drag for gross > 1.0 (spec §5.8)."""

from __future__ import annotations

import polars as pl


def compute_financing_drag(positions: pl.DataFrame, *, rate_annual: float) -> pl.DataFrame:
    excess = (pl.col("gross_notional") - pl.col("equity")).clip(lower_bound=0.0)
    daily = excess * float(rate_annual) / 252.0
    return positions.with_columns(daily.alias("financing_drag"))
