"""Dollar ADV builder (spec §2.5).

adv_20d_dollar_lag1 = rolling_median_20( close * volume ).shift(1)
"""

from __future__ import annotations

import polars as pl


def build_adv_20d_dollar(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = (
        panel.with_columns(
            (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_dv")
        )
        .with_columns(
            pl.col("_dv")
            .rolling_median(window_size=window, min_samples=window)
            .shift(1)
            .over("symbol")
            .alias("adv_20d_dollar_lag1")
        )
        .drop("_dv")
    )
    return out.select(["date", "symbol", "adv_20d_dollar_lag1"])
