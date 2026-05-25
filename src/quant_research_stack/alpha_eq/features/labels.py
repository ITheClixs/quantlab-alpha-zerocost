"""Forward-return labels (spec §3.2)."""

from __future__ import annotations

import polars as pl


def build_labels(
    panel: pl.DataFrame,
    *,
    close_tr: str,
    vol_col: str,
    universe_col: str,
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        ((pl.col(close_tr).shift(-1).over("symbol") / pl.col(close_tr)) - 1.0).alias("y_raw")
    )
    out = out.with_columns(
        (pl.col("y_raw") / pl.col(vol_col).clip(lower_bound=1e-9)).alias("y_vn")
    )
    inuv_mean = out.filter(pl.col(universe_col)).group_by("date").agg(
        pl.col("y_vn").mean().alias("_xs_mean")
    )
    out = out.join(inuv_mean, on="date", how="left").with_columns(
        pl.when(pl.col(universe_col))
        .then(pl.col("y_vn") - pl.col("_xs_mean"))
        .otherwise(None)
        .alias("y_xs")
    ).drop("_xs_mean")
    return out
