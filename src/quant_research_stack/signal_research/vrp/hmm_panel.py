"""Reusable HMM panel — fits a 2-state Gaussian HMM on dev-only underlying
returns and exposes both the discrete risk-on flag and the continuous
posterior probability per date.

Both are needed for the VRP × HMM interaction test (Option γ):
- discrete flag for set-intersection variants (VRP × risk_on)
- continuous prob for size-scaled variants (VRP × p_risk_on)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class HMMPanel:
    panel: pl.DataFrame  # (date, regime_id, risk_on, p_risk_on)
    risk_on_state_id: int
    state_means: dict[int, float]


def fit_hmm_panel(
    *,
    underlying_single_symbol: pl.DataFrame,
    dev_end: dt.date,
    n_states: int = 2,
    seed: int = 42,
) -> HMMPanel:
    """Fit a Gaussian HMM on dev-only log returns; emit per-date regime_id,
    risk_on flag, and p_risk_on posterior on the FULL sample.

    Risk-on is predeclared as the state with higher mean dev return.
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
        # Degenerate fallback: every day risk_on
        n = df.height
        return HMMPanel(
            panel=df.select(["date"]).with_columns(
                pl.Series("regime_id", np.zeros(n, dtype=np.int64)),
                pl.Series("risk_on", np.ones(n, dtype=np.int64)),
                pl.Series("p_risk_on", np.ones(n, dtype=np.float64)),
            ),
            risk_on_state_id=0,
            state_means={0: 0.0},
        )
    dev_r = dev["log_ret"].to_numpy().astype(np.float64).reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=200, random_state=seed,
    )
    model.fit(dev_r)
    dev_states = model.predict(dev_r)
    state_means = {
        int(s): float(np.mean(dev_r.flatten()[dev_states == s]))
        for s in range(n_states)
    }
    risk_on_id = max(state_means, key=lambda k: state_means[k])
    full_r = df["log_ret"].to_numpy().astype(np.float64).reshape(-1, 1)
    full_states = model.predict(full_r).astype(np.int64)
    full_post = model.predict_proba(full_r).astype(np.float64)
    p_risk_on = full_post[:, risk_on_id]
    risk_on_flag = (full_states == risk_on_id).astype(np.int64)
    panel = df.select(["date"]).with_columns(
        pl.Series("regime_id", full_states),
        pl.Series("risk_on", risk_on_flag),
        pl.Series("p_risk_on", p_risk_on),
    )
    return HMMPanel(
        panel=panel,
        risk_on_state_id=risk_on_id,
        state_means=state_means,
    )


def hmm_signal_from_panel(hmm_panel: HMMPanel) -> pl.DataFrame:
    """Discrete risk_on signal as a (date, signal) timing input."""
    return hmm_panel.panel.select(["date"]).with_columns(
        pl.Series("signal", hmm_panel.panel["risk_on"].to_numpy().astype(np.float64))
    )


def signal_to_array(signal_df: pl.DataFrame) -> NDArray[np.float64]:
    return signal_df.sort("date")["signal"].to_numpy().astype(np.float64)
