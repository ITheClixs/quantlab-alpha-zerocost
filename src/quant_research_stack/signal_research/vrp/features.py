"""VRP feature engineering — converts raw VIX-family series into the
conditioning features the strategies consume.

All features are computed past-only (no look-ahead). Realized variance
is computed from the underlying's *prior* 21 trading days, then
compared to current VIX² to form the variance risk premium.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _ensure_log_return(underlying_single_symbol: pl.DataFrame) -> pl.DataFrame:
    df = underlying_single_symbol.sort("date")
    if "log_ret" in df.columns:
        return df
    return df.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret")
    )


def compute_realized_variance_annual(
    underlying_single_symbol: pl.DataFrame, *, window: int = 21
) -> pl.DataFrame:
    """Annualised realised variance from trailing-window squared log returns.

    Returns (date, realized_var_annual) where realized_var_annual is in the
    same units as VIX² / 10000 (i.e. fraction-squared, annualized).
    """
    df = _ensure_log_return(underlying_single_symbol)
    return df.with_columns(
        (
            (pl.col("log_ret") ** 2).rolling_sum(window_size=window).over("symbol")
            * (252.0 / window)
        ).alias("realized_var_annual")
    ).select(["date", "realized_var_annual"])


def compute_vrp_features(
    *,
    underlying_single_symbol: pl.DataFrame,
    vol_features: pl.DataFrame,
    realized_window: int = 21,
) -> pl.DataFrame:
    """Join underlying + vol features and compute VRP, term-structure ratio,
    VVIX/VIX, SKEW z-score, VXN ratio.

    Returns long-form (date, ...features...). One row per date in the join.
    """
    realized = compute_realized_variance_annual(
        underlying_single_symbol, window=realized_window,
    )
    df = vol_features.join(realized, on="date", how="left")
    df = df.with_columns(
        # VIX is in vol points (e.g. 20 means 20%); VIX² / 10000 is annualised variance
        ((pl.col("vix") / 100.0) ** 2).alias("implied_var_annual"),
    )
    df = df.with_columns(
        (pl.col("implied_var_annual") - pl.col("realized_var_annual")).alias("vrp"),
    )
    if "vix9d" in df.columns:
        df = df.with_columns((pl.col("vix9d") / pl.col("vix")).alias("term_9d_30d"))
    if "vix3m" in df.columns:
        df = df.with_columns((pl.col("vix3m") / pl.col("vix")).alias("term_30d_3m"))
    if "vvix" in df.columns:
        df = df.with_columns((pl.col("vvix") / pl.col("vix")).alias("vvix_to_vix"))
    if "skew" in df.columns:
        df = df.with_columns(
            (
                (pl.col("skew") - pl.col("skew").rolling_mean(window_size=60))
                / pl.col("skew").rolling_std(window_size=60).clip(lower_bound=1e-6)
            ).alias("skew_z60"),
        )
    if "vxn" in df.columns:
        df = df.with_columns((pl.col("vxn") / pl.col("vix")).alias("vxn_to_vix"))
    return df


def vrp_zscore_60d(features: pl.DataFrame) -> pl.DataFrame:
    """Add a 60-day z-score of VRP for the combined variant rule."""
    return features.with_columns(
        (
            (pl.col("vrp") - pl.col("vrp").rolling_mean(window_size=60))
            / pl.col("vrp").rolling_std(window_size=60).clip(lower_bound=1e-9)
        ).alias("vrp_z60")
    )


def winsorize_columns(
    features: pl.DataFrame, *, columns: list[str], q_low: float = 0.005, q_high: float = 0.995,
) -> pl.DataFrame:
    """Winsorise specified columns at quantile thresholds (computed on the
    full sample; this is a sample-level outlier cap, not a leakage path —
    it's symmetric and doesn't reorder dates).
    """
    df = features
    for c in columns:
        if c not in df.columns:
            continue
        arr = df[c].to_numpy()
        finite = arr[~np.isnan(arr)]
        if finite.size < 10:
            continue
        lo = float(np.quantile(finite, q_low))
        hi = float(np.quantile(finite, q_high))
        df = df.with_columns(pl.col(c).clip(lower_bound=lo, upper_bound=hi))
    return df
