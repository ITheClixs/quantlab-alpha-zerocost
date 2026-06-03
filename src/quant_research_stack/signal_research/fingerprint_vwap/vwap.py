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
