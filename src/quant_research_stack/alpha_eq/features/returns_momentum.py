"""Returns / momentum features (spec §3.3-1).

After-close_t convention: features at date t may use the complete day-t close.
"""

from __future__ import annotations

import polars as pl


def build_returns_momentum(
    panel: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60, 120, 252),
    include_skip5: tuple[int, ...] = (60, 120, 252),
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel
    for h in horizons:
        out = out.with_columns(
            (pl.col("close").log() - pl.col("close").shift(h).over("symbol").log()).alias(
                f"log_return_{h}"
            )
        )
    for h in include_skip5:
        out = out.with_columns(
            (
                pl.col("close").shift(5).over("symbol").log()
                - pl.col("close").shift(h).over("symbol").log()
            ).alias(f"cumulative_return_{h}_skip5")
        )
    if 5 in horizons:
        out = out.with_columns((-pl.col("log_return_5")).alias("mean_reversion_5"))
    return out
