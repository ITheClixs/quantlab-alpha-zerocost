"""Volatility features (spec §3.3-2)."""

from __future__ import annotations

import polars as pl


def build_volatility(
    panel: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    parkinson_window: int = 20,
    gk_window: int = 20,
    vov_window: int = 60,
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_log_ret_1")
    )
    for w in windows:
        out = out.with_columns(
            pl.col("_log_ret_1")
            .rolling_std(window_size=w, min_samples=w)
            .over("symbol")
            .alias(f"realized_vol_{w}")
        )
    # Parkinson: sqrt((1 / (4 ln 2)) * mean((ln(high/low))^2))
    out = out.with_columns(
        ((pl.col("high") / pl.col("low")).log() ** 2)
        .rolling_mean(window_size=parkinson_window, min_samples=parkinson_window)
        .over("symbol")
        .alias("_pk_raw")
    ).with_columns(
        (pl.col("_pk_raw") / (4.0 * pl.lit(0.6931471805599453))).sqrt().alias(
            f"parkinson_vol_{parkinson_window}"
        )
    ).drop("_pk_raw")
    # Garman-Klass
    out = out.with_columns(
        (
            0.5 * ((pl.col("high") / pl.col("low")).log() ** 2)
            - (2.0 * pl.lit(0.6931471805599453) - 1.0)
            * ((pl.col("close") / pl.col("open")).log() ** 2)
        ).alias("_gk_var")
    ).with_columns(
        pl.col("_gk_var")
        .clip(lower_bound=0.0)
        .rolling_mean(window_size=gk_window, min_samples=gk_window)
        .over("symbol")
        .sqrt()
        .alias(f"garman_klass_vol_{gk_window}")
    ).drop("_gk_var")
    if 20 in windows:
        out = out.with_columns(
            pl.col("realized_vol_20")
            .rolling_std(window_size=vov_window, min_samples=vov_window)
            .over("symbol")
            .alias(f"vol_of_vol_{vov_window}")
        )
    return out.drop("_log_ret_1")
