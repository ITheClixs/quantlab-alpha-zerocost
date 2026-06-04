"""Rolling volatility estimators (daily, as-of). Each returns annualised-agnostic
per-day vol (std of log scale). Estimators: close-to-close, Parkinson (1980),
Rogers-Satchell (1991)."""

from __future__ import annotations

import numpy as np
import polars as pl

_ESTIMATORS = ("close_to_close", "parkinson", "rogers_satchell")


def rolling_vol(bars: pl.DataFrame, *, window: int, estimator: str) -> pl.Series:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {_ESTIMATORS}")
    df = bars.sort(["symbol", "date"])
    if estimator == "close_to_close":
        r = (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log())
        vol = r.rolling_std(window_size=window, min_samples=window).over("symbol")
        return df.with_columns(vol.alias("_v"))["_v"]
    if estimator == "parkinson":
        hl = (pl.col("high").log() - pl.col("low").log()) ** 2
        mean_hl = hl.rolling_mean(window_size=window, min_samples=window).over("symbol")
        vol = (mean_hl / (4.0 * np.log(2.0))).sqrt()
        return df.with_columns(vol.alias("_v"))["_v"]
    rs = (
        (pl.col("high").log() - pl.col("close").log()) * (pl.col("high").log() - pl.col("open").log())
        + (pl.col("low").log() - pl.col("close").log()) * (pl.col("low").log() - pl.col("open").log())
    )
    vol = rs.rolling_mean(window_size=window, min_samples=window).over("symbol").clip(lower_bound=0.0).sqrt()
    return df.with_columns(vol.alias("_v"))["_v"]
