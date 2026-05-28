"""VRP × HMM interaction variants — Option γ predeclared grid.

Variants in the PBO/DSR pool (plus the two anchor baselines):
1. hmm_only (anchor)
2. vrp_only (anchor)
3. vrp_when_hmm_risk_on  — intersection: long when VRP > 0 AND risk_on
4. vrp_when_hmm_risk_off — intersection: long when VRP > 0 AND risk_off
5. hmm_sized_by_vrp      — hmm signal × clipped-vrp-magnitude
6. vrp_sized_by_hmm_prob — vrp gate × p_risk_on
7. additive_50_50        — 0.5 * hmm_signal + 0.5 * vrp_signal
8. additive_70_30        — 0.7 * hmm_signal + 0.3 * vrp_signal
9. orthogonalized_vrp    — residual of VRP after regressing on HMM (dev only)

All signals are in [0, 1] (long-only) or [-1, +1] (long-short) and trade SPY.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from quant_research_stack.signal_research.vrp.features import vrp_zscore_60d
from quant_research_stack.signal_research.vrp.hmm_panel import HMMPanel


def _vrp_long_only_signal(features: pl.DataFrame) -> pl.DataFrame:
    return features.with_columns(
        pl.when(pl.col("vrp") > 0.0).then(1.0).otherwise(0.0).alias("signal")
    ).select(["date", "signal"])


def signal_vrp_when_hmm_risk_on(
    *, vrp_features: pl.DataFrame, hmm_panel: HMMPanel,
) -> pl.DataFrame:
    """Long SPY when VRP > 0 AND HMM = risk_on."""
    vrp = _vrp_long_only_signal(vrp_features).rename({"signal": "vrp_signal"})
    hmm = hmm_panel.panel.select(["date", "risk_on"])
    return (
        vrp.join(hmm, on="date", how="left")
        .with_columns(
            (pl.col("vrp_signal") * pl.col("risk_on").cast(pl.Float64)).alias("signal")
        )
        .select(["date", "signal"])
    )


def signal_vrp_when_hmm_risk_off(
    *, vrp_features: pl.DataFrame, hmm_panel: HMMPanel,
) -> pl.DataFrame:
    """Long SPY when VRP > 0 AND HMM = risk_off — tests crash-premium hypothesis."""
    vrp = _vrp_long_only_signal(vrp_features).rename({"signal": "vrp_signal"})
    hmm = hmm_panel.panel.select(["date", "risk_on"])
    return (
        vrp.join(hmm, on="date", how="left")
        .with_columns(
            (
                pl.col("vrp_signal") * (1.0 - pl.col("risk_on").cast(pl.Float64))
            ).alias("signal")
        )
        .select(["date", "signal"])
    )


def signal_hmm_sized_by_vrp(
    *, vrp_features: pl.DataFrame, hmm_panel: HMMPanel,
) -> pl.DataFrame:
    """HMM determines whether to be long; VRP z-score (clipped to [0, 1])
    determines exposure size.

    Position = 1[hmm = risk_on] × clip(vrp_z60, 0, 1).
    When VRP is weak/negative, exposure is small even in risk-on regimes.
    """
    feats = vrp_zscore_60d(vrp_features)
    feats = feats.with_columns(
        pl.col("vrp_z60").clip(lower_bound=0.0, upper_bound=1.0).fill_null(0.0)
        .alias("vrp_strength")
    )
    hmm = hmm_panel.panel.select(["date", "risk_on"])
    return (
        feats.select(["date", "vrp_strength"])
        .join(hmm, on="date", how="left")
        .with_columns(
            (
                pl.col("risk_on").cast(pl.Float64) * pl.col("vrp_strength")
            ).alias("signal")
        )
        .select(["date", "signal"])
    )


def signal_vrp_sized_by_hmm_prob(
    *, vrp_features: pl.DataFrame, hmm_panel: HMMPanel,
) -> pl.DataFrame:
    """VRP gate determines whether to be long; HMM posterior p_risk_on
    determines exposure size.

    Position = 1[vrp > 0] × p_risk_on.
    """
    vrp = _vrp_long_only_signal(vrp_features).rename({"signal": "vrp_signal"})
    hmm = hmm_panel.panel.select(["date", "p_risk_on"])
    return (
        vrp.join(hmm, on="date", how="left")
        .with_columns(
            (pl.col("vrp_signal") * pl.col("p_risk_on")).alias("signal")
        )
        .select(["date", "signal"])
    )


def _normalize_expr(expr: pl.Expr) -> pl.Expr:
    """Map a signal expression to [0, 1] for additive ensemble use."""
    return expr.fill_null(0.0).clip(lower_bound=0.0, upper_bound=1.0)


def signal_additive_ensemble(
    *,
    vrp_features: pl.DataFrame,
    hmm_panel: HMMPanel,
    w_hmm: float,
) -> pl.DataFrame:
    """position = w_hmm × hmm_signal + (1 - w_hmm) × vrp_signal.

    Both inputs already in [0, 1] so the ensemble is in [0, 1].
    """
    vrp = _vrp_long_only_signal(vrp_features).rename({"signal": "vrp_signal"})
    hmm = hmm_panel.panel.select(["date", "risk_on"]).rename(
        {"risk_on": "hmm_signal"}
    )
    joined = vrp.join(hmm, on="date", how="full", coalesce=True).with_columns(
        _normalize_expr(pl.col("vrp_signal")).alias("vrp_signal"),
        _normalize_expr(pl.col("hmm_signal").cast(pl.Float64)).alias("hmm_signal"),
    )
    w_vrp = 1.0 - w_hmm
    return joined.with_columns(
        (pl.col("hmm_signal") * w_hmm + pl.col("vrp_signal") * w_vrp).alias("signal")
    ).select(["date", "signal"])


def signal_orthogonalized_vrp(
    *,
    vrp_features: pl.DataFrame,
    hmm_panel: HMMPanel,
    dev_end: dt.date,
) -> pl.DataFrame:
    """Regress dev-period vrp_signal on hmm_signal (with intercept), use the
    fitted coefficients to compute residual VRP on the full sample, trade
    sign(residual).

    Captures VRP information NOT explained by HMM regime.
    """
    vrp = _vrp_long_only_signal(vrp_features).rename({"signal": "vrp_signal"})
    hmm = hmm_panel.panel.select(["date", "risk_on"]).rename(
        {"risk_on": "hmm_signal"}
    )
    joined = vrp.join(hmm, on="date", how="left").drop_nulls()
    dev_mask = (joined["date"].to_numpy() <= np.datetime64(dev_end))
    y_dev = joined["vrp_signal"].to_numpy().astype(np.float64)[dev_mask]
    x_dev = joined["hmm_signal"].cast(pl.Float64).to_numpy().astype(np.float64)[dev_mask]
    # OLS: y = α + β x, fit via numpy
    if x_dev.size > 10 and np.std(x_dev, ddof=1) > 1e-9:
        X = np.column_stack([np.ones_like(x_dev), x_dev])
        coefs, *_ = np.linalg.lstsq(X, y_dev, rcond=None)
        alpha = float(coefs[0])
        beta = float(coefs[1])
    else:
        alpha = 0.0
        beta = 0.0
    full_vrp = joined["vrp_signal"].to_numpy().astype(np.float64)
    full_hmm = joined["hmm_signal"].cast(pl.Float64).to_numpy().astype(np.float64)
    residual = full_vrp - (alpha + beta * full_hmm)
    signal_arr = (residual > 0.0).astype(np.float64)
    return joined.select(["date"]).with_columns(pl.Series("signal", signal_arr))
