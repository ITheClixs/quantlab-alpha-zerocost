"""Regime-conditional metrics (spec §4.5).

- 2-state Gaussian HMM on broad-market returns.
- Per-strategy Sharpe / DD / PnL by regime.
- Declarations:
  - AGNOSTIC: both regimes not catastrophically negative AND ≥1 materially positive.
  - SPECIFIC: must be PREDECLARED with favorable_regime; retroactive declaration forbidden.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class RegimeDeclaration(enum.StrEnum):
    AGNOSTIC = "agnostic"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class RegimeMetrics:
    sharpe_by_regime: dict[int, float]
    pnl_by_regime: dict[int, float]
    max_dd_by_regime: dict[int, float]
    active_days_by_regime: dict[int, int]
    declaration: RegimeDeclaration
    favorable_regime: int | None
    passes_regime_gate: bool


def fit_hmm_regimes(
    returns: NDArray[np.float64], *, n_states: int = 2, seed: int = 42
) -> NDArray[np.int64]:
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed
    )
    model.fit(returns.reshape(-1, 1))
    return model.predict(returns.reshape(-1, 1)).astype(np.int64)


def _sharpe(r: NDArray[np.float64]) -> float:
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r)) / sd * float(np.sqrt(252.0))


def regime_conditional_metrics(
    *,
    returns: NDArray[np.float64],
    regime_states: NDArray[np.int64],
    declaration: RegimeDeclaration,
    favorable_regime: int | None = None,
    catastrophic_threshold: float = -1.0,
    materially_positive_threshold: float = 0.3,
) -> RegimeMetrics:
    states = sorted(set(regime_states.tolist()))
    by_sharpe: dict[int, float] = {}
    by_pnl: dict[int, float] = {}
    by_dd: dict[int, float] = {}
    by_days: dict[int, int] = {}
    for s in states:
        mask = regime_states == s
        r = returns[mask]
        by_sharpe[s] = _sharpe(r)
        by_pnl[s] = float(np.sum(r))
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity) if equity.size else np.array([1.0])
        by_dd[s] = float((equity / peak - 1.0).min()) if equity.size else 0.0
        by_days[s] = int(mask.sum())

    if declaration == RegimeDeclaration.AGNOSTIC:
        sharpes = list(by_sharpe.values())
        all_not_catastrophic = all(s > catastrophic_threshold for s in sharpes)
        at_least_one_positive = any(s > materially_positive_threshold for s in sharpes)
        passes = all_not_catastrophic and at_least_one_positive
    else:
        if favorable_regime is None:
            passes = False
        else:
            passes = by_sharpe.get(favorable_regime, -1.0) > materially_positive_threshold

    return RegimeMetrics(
        sharpe_by_regime=by_sharpe,
        pnl_by_regime=by_pnl,
        max_dd_by_regime=by_dd,
        active_days_by_regime=by_days,
        declaration=declaration,
        favorable_regime=favorable_regime,
        passes_regime_gate=passes,
    )
