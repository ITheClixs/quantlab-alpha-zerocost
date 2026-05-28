"""VRP strategy variants — predeclared rule grid.

Each function returns a long-form (date, signal) DataFrame where signal
is the target gross exposure ∈ [-1, +1] for one underlying instrument
on day T. The timing-mode backtest converts these to per-bar returns.

All rules are computed past-only: a signal observed at close(T) uses
features observed at close(T) and is applied to the close(T)→close(T+1)
PnL window.
"""

from __future__ import annotations

import polars as pl


def vrp_long_only(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 1: long underlying when VRP > 0; flat otherwise.

    VRP = implied_var_annual - realized_var_annual. When IV² > RV²,
    option market is paying a premium → take the equity exposure.
    """
    return features.with_columns(
        pl.when(pl.col("vrp") > 0.0).then(1.0).otherwise(0.0).alias("signal")
    ).select(["date", "signal"])


def vrp_long_short(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 2: long when VRP > 0; short when VRP < 0."""
    return features.with_columns(
        pl.when(pl.col("vrp") > 0.0)
        .then(1.0)
        .when(pl.col("vrp") < 0.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("signal")
    ).select(["date", "signal"])


def vrp_with_term_structure(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 3: variant 1 AND VIX9D < VIX (contango → risk-on).

    Falls back to variant 1 if `term_9d_30d` is missing.
    """
    if "term_9d_30d" not in features.columns:
        return vrp_long_only(features)
    return features.with_columns(
        pl.when((pl.col("vrp") > 0.0) & (pl.col("term_9d_30d") < 1.0))
        .then(1.0)
        .otherwise(0.0)
        .alias("signal")
    ).select(["date", "signal"])


def vrp_with_vvix(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 4: variant 1 AND VVIX/VIX below trailing-60d median.

    High VVIX/VIX = elevated vol-of-vol = unstable vol regime → step out.
    """
    if "vvix_to_vix" not in features.columns:
        return vrp_long_only(features)
    df = features.with_columns(
        pl.col("vvix_to_vix")
        .rolling_median(window_size=60)
        .alias("vvix_to_vix_med60")
    )
    return df.with_columns(
        pl.when(
            (pl.col("vrp") > 0.0)
            & (pl.col("vvix_to_vix") < pl.col("vvix_to_vix_med60"))
        )
        .then(1.0)
        .otherwise(0.0)
        .alias("signal")
    ).select(["date", "signal"])


def vrp_with_skew(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 5: variant 1 AND SKEW below trailing-60d median.

    Elevated SKEW = priced crash risk → step out.
    """
    if "skew" not in features.columns:
        return vrp_long_only(features)
    df = features.with_columns(
        pl.col("skew").rolling_median(window_size=60).alias("skew_med60")
    )
    return df.with_columns(
        pl.when(
            (pl.col("vrp") > 0.0)
            & (pl.col("skew") < pl.col("skew_med60"))
        )
        .then(1.0)
        .otherwise(0.0)
        .alias("signal")
    ).select(["date", "signal"])


def vrp_combined(features: pl.DataFrame) -> pl.DataFrame:
    """Variant 6: weighted z-score of VRP + conditioning features.

    score = vrp_z60
            - 0.5 * (vvix_to_vix - 60d-mean)/60d-std (penalise high vol-of-vol)
            - 0.5 * skew_z60 (penalise high tail-risk pricing)
    Long if score > 0; flat otherwise. Magnitude not used (continuous gross
    sizing left to subsequent iteration).
    """
    df = features
    expr = pl.col("vrp_z60") if "vrp_z60" in df.columns else pl.col("vrp")
    if "vvix_to_vix" in df.columns:
        df = df.with_columns(
            (
                (pl.col("vvix_to_vix") - pl.col("vvix_to_vix").rolling_mean(window_size=60))
                / pl.col("vvix_to_vix").rolling_std(window_size=60).clip(lower_bound=1e-9)
            ).alias("_vvix_z")
        )
        expr = expr - 0.5 * pl.col("_vvix_z")
    if "skew_z60" in df.columns:
        expr = expr - 0.5 * pl.col("skew_z60")
    df = df.with_columns(expr.alias("combined_score"))
    return df.with_columns(
        pl.when(pl.col("combined_score") > 0.0).then(1.0).otherwise(0.0).alias("signal")
    ).select(["date", "signal"])
