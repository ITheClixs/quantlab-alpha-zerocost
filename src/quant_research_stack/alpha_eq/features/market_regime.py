"""Market / regime context features with mandatory VIX fallback rule (spec §3.3-6)."""

from __future__ import annotations

import polars as pl


def _cross_sectional_vol_20(panel: pl.DataFrame) -> pl.DataFrame:
    rets = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_r1")
    )
    by_date = rets.group_by("date").agg(pl.col("_r1").std().alias("_xs_vol"))
    by_date = by_date.sort("date").with_columns(
        pl.col("_xs_vol")
        .rolling_mean(window_size=20, min_samples=20)
        .alias("cross_sectional_vol_20")
    )
    return by_date.select(["date", "cross_sectional_vol_20"])


def build_market_regime(
    *,
    panel: pl.DataFrame,
    vix: pl.DataFrame | None,
    spy_close: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach broadcast market features. VIX missing → fallback to cross-sectional
    volatility proxy; never silently drops dates."""
    xs_vol = _cross_sectional_vol_20(panel)
    rets = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_r1")
    )
    disp = rets.group_by("date").agg(
        pl.col("_r1").std().alias("cross_sectional_dispersion")
    )

    panel = panel.join(xs_vol, on="date", how="left").join(disp, on="date", how="left")

    if vix is not None and not vix.is_empty():
        panel = panel.join(
            vix.rename({"vix_close": "_vix_external"}), on="date", how="left"
        ).with_columns(
            pl.when(pl.col("_vix_external").is_not_null())
            .then(pl.col("_vix_external"))
            .otherwise(pl.col("cross_sectional_vol_20"))
            .alias("vix_close"),
            pl.col("_vix_external").is_null().alias("vix_is_proxy"),
        ).drop("_vix_external")
    else:
        panel = panel.with_columns(
            pl.col("cross_sectional_vol_20").alias("vix_close"),
            pl.lit(True).alias("vix_is_proxy"),
        )

    if spy_close is not None and not spy_close.is_empty():
        spy_join = spy_close.sort("date").with_columns(
            (pl.col("spy_close").log() - pl.col("spy_close").shift(5).log()).alias("spy_log_return_5"),
            (
                (pl.col("spy_close").log() - pl.col("spy_close").shift(1).log())
                .rolling_std(window_size=20, min_samples=20)
            ).alias("spy_realized_vol_20"),
        ).drop("spy_close")
        panel = panel.join(spy_join, on="date", how="left")
    else:
        panel = panel.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("spy_log_return_5"),
            pl.lit(None, dtype=pl.Float64).alias("spy_realized_vol_20"),
        )

    return panel
