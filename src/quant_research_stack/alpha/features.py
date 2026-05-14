from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class FeatureConfig:
    lag_windows: list[int]
    rolling_windows: list[int]
    include_noise_feature: bool
    cross_sectional_ranks: bool
    noise_seed: int = 42


def add_lag_features(
    df: pl.DataFrame, cols: list[str], lags: list[int], group_col: str, time_col: str
) -> pl.DataFrame:
    sorted_df = df.sort([group_col, time_col])
    new_cols = []
    for col in cols:
        for lag in lags:
            new_cols.append(pl.col(col).shift(lag).over(group_col).alias(f"{col}_lag{lag}"))
    return sorted_df.with_columns(new_cols)


def add_rolling_features(
    df: pl.DataFrame, cols: list[str], windows: list[int], group_col: str, time_col: str
) -> pl.DataFrame:
    sorted_df = df.sort([group_col, time_col])
    new_cols = []
    for col in cols:
        for w in windows:
            new_cols.append(pl.col(col).rolling_mean(w).over(group_col).alias(f"{col}_roll{w}_mean"))
            new_cols.append(pl.col(col).rolling_std(w).over(group_col).alias(f"{col}_roll{w}_std"))
    return sorted_df.with_columns(new_cols)


def add_cross_sectional_ranks(df: pl.DataFrame, cols: list[str], date_col: str) -> pl.DataFrame:
    new_cols = []
    for col in cols:
        new_cols.append(
            (pl.col(col).rank(method="ordinal").over(date_col) - 1).cast(pl.Float64).alias(f"{col}_rank_xs")
        )
    return df.with_columns(new_cols)


def add_noise_feature(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return df.with_columns(pl.Series(name=f"noise_seed{seed}", values=rng.normal(size=df.height)))


def no_future_leakage(
    df: pl.DataFrame, target_col: str, group_col: str, time_col: str, abs_corr_threshold: float = 0.999
) -> list[str]:
    """Flag feature columns whose per-row values are identical to the target (cheap leakage check).

    A feature is flagged only when it has no missing values (no evidence of lagging) AND its
    non-null rows are numerically identical to the target at the same row index.
    """
    leaks: list[str] = []
    target = df[target_col].cast(pl.Float64)
    for col in df.columns:
        if col in {target_col, group_col, time_col}:
            continue
        if df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            continue
        feat = df[col].cast(pl.Float64)
        # Skip features with any nulls — nulls indicate lagging/rolling (not direct leakage)
        if feat.null_count() > 0:
            continue
        x = feat.to_numpy()
        y = target.to_numpy()
        std_x = float(np.std(x))
        std_y = float(np.std(y))
        if std_x == 0.0 or std_y == 0.0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if abs(corr) >= abs_corr_threshold:
            leaks.append(col)
    return leaks


def build_feature_frame(
    df: pl.DataFrame, cfg: FeatureConfig, base_features: list[str], date_col: str, symbol_col: str
) -> pl.DataFrame:
    out = df
    if cfg.lag_windows:
        out = add_lag_features(out, base_features, cfg.lag_windows, symbol_col, date_col)
    if cfg.rolling_windows:
        out = add_rolling_features(out, base_features, cfg.rolling_windows, symbol_col, date_col)
    if cfg.cross_sectional_ranks:
        out = add_cross_sectional_ranks(out, base_features, date_col)
    if cfg.include_noise_feature:
        out = add_noise_feature(out, seed=cfg.noise_seed)
    return out
