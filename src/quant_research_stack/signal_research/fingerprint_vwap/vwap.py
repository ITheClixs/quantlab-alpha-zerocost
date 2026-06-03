"""VWAP proxy + VWAP-entry primary (spec §6, steps 2-3). Daily bars only:
'VWAP' is a rolling volume-weighted typical price, an as-of proxy for intraday VWAP."""

from __future__ import annotations

import polars as pl


def daily_vwap_proxy(panel: pl.DataFrame, *, window: int = 5) -> pl.DataFrame:
    """Attach `vwap` = rolling volume-weighted typical price over `window` days,
    computed strictly from rows up to and including t (no look-ahead)."""
    tp = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
    return (
        panel.sort(["symbol", "date"])
        .with_columns((tp * pl.col("volume")).alias("_tpv"))
        .with_columns(
            (
                pl.col("_tpv").rolling_sum(window_size=window, min_samples=window).over("symbol")
                / pl.col("volume").rolling_sum(window_size=window, min_samples=window).over("symbol")
            ).alias("vwap")
        )
        .drop("_tpv")
    )


def vwap_primary_position(panel: pl.DataFrame, *, band: float = 0.0) -> pl.DataFrame:
    """Primary entry: long (1.0) when close is at least `band` below vwap (mean
    reversion to VWAP); flat (0.0) otherwise. Long-only v1; requires `vwap` column."""
    if "vwap" not in panel.columns:
        raise ValueError("call daily_vwap_proxy first; missing 'vwap' column")
    return panel.with_columns(
        pl.when(pl.col("vwap").is_not_null() & (pl.col("close") <= pl.col("vwap") * (1.0 - band)))
        .then(1.0)
        .otherwise(0.0)
        .alias("primary_position")
    )
