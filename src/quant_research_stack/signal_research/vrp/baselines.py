"""VRP baselines — comparison strategies that do NOT consume VIX-family.

- spy_buy_and_hold: always long
- hmm_only_gate: HMM regime fit on dev-only underlying returns; long
  when in higher-mean (risk-on) state
- mom_12_1_single_asset: 12-1 momentum applied as a timing signal on the
  single underlying
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def signal_buy_and_hold(underlying_single_symbol: pl.DataFrame) -> pl.DataFrame:
    """Always-long signal at gross 1.0."""
    return (
        underlying_single_symbol.sort("date")
        .select(["date"])
        .with_columns(pl.lit(1.0).alias("signal"))
    )


def signal_hmm_only_gate(
    *,
    underlying_single_symbol: pl.DataFrame,
    dev_end: dt.date,
    n_states: int = 2,
    seed: int = 42,
) -> pl.DataFrame:
    """Long only when HMM (fit on dev returns) classifies the day as
    risk-on (higher-mean state).

    Risk-on identification: state with higher mean dev return → predeclared.
    """
    from hmmlearn.hmm import GaussianHMM

    df = (
        underlying_single_symbol.sort("date")
        .with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret")
        )
        .drop_nulls(subset=["log_ret"])
    )
    dev = df.filter(pl.col("date") <= dev_end)
    if dev.height < 60:
        return df.select(["date"]).with_columns(pl.lit(1.0).alias("signal"))
    dev_r = dev["log_ret"].to_numpy().astype(np.float64)
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=200, random_state=seed,
    )
    model.fit(dev_r.reshape(-1, 1))
    dev_states = model.predict(dev_r.reshape(-1, 1))
    state_means = {
        int(s): float(np.mean(dev_r[dev_states == s])) for s in range(n_states)
    }
    risk_on = max(state_means, key=lambda k: state_means[k])
    full_r = df["log_ret"].to_numpy().astype(np.float64)
    full_states = model.predict(full_r.reshape(-1, 1)).astype(np.int64)
    signal_arr = (full_states == risk_on).astype(np.float64)
    return df.select(["date"]).with_columns(pl.Series("signal", signal_arr))


def signal_mom_12_1_single_asset(
    underlying_single_symbol: pl.DataFrame, *,
    lookback: int = 252,
    skip_recent: int = 21,
) -> pl.DataFrame:
    """12-1 momentum as a single-asset timing signal: long when the
    (lookback - skip_recent)-day return is positive; flat otherwise.
    """
    df = (
        underlying_single_symbol.sort("date")
        .with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret")
        )
    )
    df = df.with_columns(
        (
            pl.col("log_ret").rolling_sum(window_size=lookback)
            - pl.col("log_ret").rolling_sum(window_size=skip_recent)
        ).alias("mom_12_1"),
    )
    return (
        df.with_columns(
            pl.when(pl.col("mom_12_1") > 0.0).then(1.0).otherwise(0.0).alias("signal")
        )
        .drop_nulls(subset=["signal"])
        .select(["date", "signal"])
    )
