"""Volume / liquidity features (spec §3.3-4).

turnover_proxy_20 is intentionally dropped: synthetic shares-outstanding
estimates from volume history are not financially well-defined.
"""

from __future__ import annotations

import polars as pl


def build_volume_liquidity(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("dollar_volume")
    )
    out = out.with_columns(
        (pl.col("dollar_volume") + 1.0)
        .log()
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
        .alias(f"log_dollar_volume_{window}d")
    )
    out = out.with_columns(
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(window_size=window, min_samples=window).over("symbol"))
            / pl.col("volume").rolling_std(window_size=window, min_samples=window).over("symbol").clip(lower_bound=1e-9)
        ).alias(f"volume_zscore_{window}d")
    )
    return out
