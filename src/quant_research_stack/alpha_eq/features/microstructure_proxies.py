"""Microstructure proxies (spec §3.3-3).

Roll's NaN policy: when the 20-day autocovariance of returns is non-negative,
roll_spread_20 is explicitly null.  Silent zero-fill is forbidden.
"""

from __future__ import annotations

import polars as pl


def build_microstructure_proxies(
    panel: pl.DataFrame, *, window: int = 20
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    df = panel.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_log_ret_1"),
        (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_dv"),
    )
    df = df.with_columns(
        (pl.col("_log_ret_1").abs() / pl.col("_dv").clip(lower_bound=1e-9))
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
        .alias(f"amihud_illiq_{window}")
    )
    # Roll spread: 2 * sqrt(-cov(r_t, r_{t-1})) with NaN policy
    df = df.with_columns(pl.col("_log_ret_1").shift(1).over("symbol").alias("_log_ret_lag1"))
    df = df.with_columns(
        (pl.col("_log_ret_1") * pl.col("_log_ret_lag1"))
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
        .alias("_exy"),
        pl.col("_log_ret_1")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
        .alias("_ex"),
        pl.col("_log_ret_lag1")
        .rolling_mean(window_size=window, min_samples=window)
        .over("symbol")
        .alias("_ey"),
    )
    df = df.with_columns(
        (pl.col("_exy") - pl.col("_ex") * pl.col("_ey")).alias("_autocov")
    ).with_columns(
        pl.when(pl.col("_autocov") < 0)
        .then(2.0 * (-pl.col("_autocov")).sqrt())
        .otherwise(None)
        .alias(f"roll_spread_{window}")
    )
    # Kyle proxy: rolling slope of |log_return| on sign(log_return) * dollar_volume
    df = df.with_columns(
        (pl.col("_log_ret_1").sign() * pl.col("_dv")).alias("_signed_dv"),
        pl.col("_log_ret_1").abs().alias("_abs_ret"),
    )
    df = df.with_columns(
        (
            (pl.col("_abs_ret") * pl.col("_signed_dv"))
            .rolling_mean(window_size=window, min_samples=window)
            .over("symbol")
            - (
                pl.col("_abs_ret")
                .rolling_mean(window_size=window, min_samples=window)
                .over("symbol")
                * pl.col("_signed_dv")
                .rolling_mean(window_size=window, min_samples=window)
                .over("symbol")
            )
        ).alias("_cov_xy"),
        (
            pl.col("_signed_dv")
            .rolling_var(window_size=window, min_samples=window)
            .over("symbol")
        ).alias("_var_x"),
    ).with_columns(
        (pl.col("_cov_xy") / pl.col("_var_x").clip(lower_bound=1e-18)).alias(
            f"kyle_proxy_signed_volume_{window}"
        )
    )
    # overnight_gap, intraday_return
    df = df.with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("symbol")).log().alias("overnight_gap"),
        (pl.col("close") / pl.col("open")).log().alias("intraday_return"),
    )
    # close_location_20
    df = df.with_columns(
        pl.col("high").rolling_max(window_size=window, min_samples=window).over("symbol").alias("_h20"),
        pl.col("low").rolling_min(window_size=window, min_samples=window).over("symbol").alias("_l20"),
    ).with_columns(
        ((pl.col("close") - pl.col("_l20")) / (pl.col("_h20") - pl.col("_l20")).clip(lower_bound=1e-9))
        .alias(f"close_location_{window}")
    )
    return df.drop(
        [
            "_log_ret_1", "_log_ret_lag1", "_dv",
            "_exy", "_ex", "_ey", "_autocov",
            "_signed_dv", "_abs_ret", "_cov_xy", "_var_x",
            "_h20", "_l20",
        ]
    )
