"""Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).

Reference: Lopez de Prado 2018, "Advances in Financial Machine Learning",
section 14.3, derived from Bailey & Lopez de Prado 2014.

PSR(SR*) = Φ( (SR_hat - SR*) × √(T-1) / √(1 - γ_3 × SR_hat + (γ_4 - 1)/4 × SR_hat²) )

where Φ is the standard-normal CDF and γ_3, γ_4 are skewness and kurtosis
of the returns.

DSR = PSR(E[max SR | N trials, V[SR]])

E[max SR | N] ≈ √V[SR] × ((1 − γ) × Φ⁻¹(1 − 1/N) + γ × Φ⁻¹(1 − (1/N)/e))

where γ ≈ 0.5772 (Euler-Mascheroni) and V[SR] is the variance of Sharpe
estimates across the N trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329
_E = np.e


@dataclass(frozen=True)
class DSRResult:
    observed_sharpe: float
    expected_max_sharpe_under_null: float
    n_trials: int
    skewness: float
    kurtosis: float
    psr_zero: float       # P(true SR > 0 | observed SR, no trial penalty)
    dsr: float            # PSR after the E[max SR] deflation (multiple-testing penalty)


def _annualise(sr_daily: float) -> float:
    return sr_daily * np.sqrt(252.0)


def _deannualise(sr_annual: float) -> float:
    return sr_annual / np.sqrt(252.0)


def expected_max_sharpe(
    *, n_trials: int, sharpe_variance: float
) -> float:
    """E[max SR] under the null hypothesis that all SRs are i.i.d. normal.

    Returns the value in the same units as the input (daily). Caller decides
    whether to annualise.
    """
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    a = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    b = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * _E))
    return float(np.sqrt(sharpe_variance) * (a + b))


def compute_dsr(
    *,
    returns: NDArray[np.float64],
    sharpe_estimates: NDArray[np.float64],
    selected_idx: int,
) -> DSRResult:
    """Deflate the Sharpe of `selected_idx` by the multiple-testing penalty
    implied by the spread of all `sharpe_estimates`.

    Parameters
    ----------
    returns : (T,) array
        Daily returns of the SELECTED strategy.
    sharpe_estimates : (S,) array
        Annualised Sharpe of every strategy in the multi-test pool.
    selected_idx : int
        Index of the strategy being deflated. Used only for diagnostics; the
        observed Sharpe is taken from `sharpe_estimates[selected_idx]`.
    """
    r = returns[~np.isnan(returns)]
    T = r.size
    if T < 4:
        return DSRResult(
            observed_sharpe=0.0,
            expected_max_sharpe_under_null=0.0,
            n_trials=int(sharpe_estimates.size),
            skewness=0.0,
            kurtosis=3.0,
            psr_zero=0.5,
            dsr=0.5,
        )

    sr_annual = float(sharpe_estimates[selected_idx])
    sr_daily = _deannualise(sr_annual)

    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return DSRResult(
            observed_sharpe=sr_annual,
            expected_max_sharpe_under_null=0.0,
            n_trials=int(sharpe_estimates.size),
            skewness=0.0,
            kurtosis=3.0,
            psr_zero=1.0 if mu > 0 else 0.0,
            dsr=1.0 if mu > 0 else 0.0,
        )

    centred = r - mu
    g3 = float(np.mean(centred**3) / sd**3) if sd > 0 else 0.0  # skewness
    g4 = float(np.mean(centred**4) / sd**4) if sd > 0 else 3.0  # kurtosis (NOT excess)

    # PSR(0): probability observed SR > 0
    denom = np.sqrt(max(1.0 - g3 * sr_daily + (g4 - 1.0) / 4.0 * sr_daily**2, 1e-12))
    psr_zero = float(norm.cdf(sr_daily * np.sqrt(T - 1) / denom))

    # E[max SR] across N i.i.d. trials (daily SRs)
    sr_daily_estimates = sharpe_estimates.astype(np.float64) / np.sqrt(252.0)
    sr_variance = float(np.var(sr_daily_estimates, ddof=1))
    e_max = expected_max_sharpe(
        n_trials=int(sharpe_estimates.size), sharpe_variance=sr_variance
    )
    dsr_z = (sr_daily - e_max) * np.sqrt(T - 1) / denom
    dsr = float(norm.cdf(dsr_z))

    return DSRResult(
        observed_sharpe=sr_annual,
        expected_max_sharpe_under_null=_annualise(e_max),
        n_trials=int(sharpe_estimates.size),
        skewness=g3,
        kurtosis=g4,
        psr_zero=psr_zero,
        dsr=dsr,
    )
