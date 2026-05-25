"""Probability of Backtest Overfitting (PBO) — Bailey & Lopez de Prado 2014.

Algorithm:
1. Take a returns matrix R of shape (T, S) where T=days, S=strategies.
2. Partition the T rows into N non-overlapping submatrices of equal size
   (we drop trailing rows if T is not divisible by N).
3. For each of the C(N, N/2) ways to choose N/2 submatrices as in-sample (J)
   and the other N/2 as out-of-sample (J̄):
     a. Stack J → M_J of shape (T/2, S), stack J̄ → M_J̄
     b. Compute Sharpe per strategy on M_J → SR_J
     c. n* = argmax(SR_J)
     d. Compute Sharpe per strategy on M_J̄ → SR_J̄
     e. r = rank of SR_J̄[n*] among SR_J̄  (1 .. S)
     f. ω = r / (S + 1)  ∈ (0, 1)
     g. λ = log(ω / (1 − ω))
4. PBO = fraction of combinations where λ ≤ 0
        = P(best-IS strategy ranks below median OOS).

A PBO close to 0 means the in-sample winner generalises; close to 1 means
the in-sample winner is a multiple-testing artefact.

Reference: Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of
Backtest Overfitting", J. Computational Finance, 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PBOResult:
    pbo_probability: float
    median_logit: float
    n_partitions: int
    n_combinations: int
    submatrix_size: int
    n_strategies: int
    failure_rate: float  # fraction of combos where best IS had NEGATIVE OOS Sharpe


def _sharpe_per_strategy(rets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Annualised Sharpe of every column."""
    mu = np.mean(rets, axis=0)
    sd = np.std(rets, axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    sr = mu / sd * np.sqrt(252.0)
    sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
    return sr


def compute_pbo(
    *,
    returns: NDArray[np.float64],
    n_partitions: int = 16,
) -> PBOResult:
    """Compute PBO over a returns matrix.

    Parameters
    ----------
    returns : (T, S) array
        Daily NET returns for each strategy.
    n_partitions : even int, default 16
        Number of equal-size time submatrices.
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D (T, S); got shape={returns.shape}")
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even; got {n_partitions}")

    T, S = returns.shape
    sub = T // n_partitions
    if sub < 5:
        raise ValueError(
            f"Submatrix size {sub} too small for {n_partitions} partitions on T={T}."
        )
    usable_T = sub * n_partitions
    R = returns[:usable_T]  # drop trailing odd rows
    # Pre-slice into N submatrices
    submatrices = [R[i * sub : (i + 1) * sub] for i in range(n_partitions)]

    half = n_partitions // 2
    combos = list(combinations(range(n_partitions), half))
    # If there are too many combos, sample for tractability.
    rng = np.random.default_rng(42)
    if len(combos) > 20_000:
        combos = [combos[i] for i in rng.choice(len(combos), size=20_000, replace=False)]

    logits = np.empty(len(combos), dtype=np.float64)
    oos_neg_count = 0
    for k, J in enumerate(combos):
        J_set = set(J)
        Jbar = [i for i in range(n_partitions) if i not in J_set]
        M_J = np.concatenate([submatrices[i] for i in J], axis=0)
        M_Jbar = np.concatenate([submatrices[i] for i in Jbar], axis=0)
        sr_in = _sharpe_per_strategy(M_J)
        sr_out = _sharpe_per_strategy(M_Jbar)
        n_star = int(np.argmax(sr_in))
        # OOS rank of the IS winner (1 = worst, S = best)
        r = 1 + int(np.sum(sr_out < sr_out[n_star]))
        # Handle ties: place at the lower bound (conservative for overfitting)
        omega = r / (S + 1)
        logits[k] = np.log(omega / (1.0 - omega + 1e-300) + 1e-300)
        if sr_out[n_star] < 0:
            oos_neg_count += 1

    return PBOResult(
        pbo_probability=float(np.mean(logits <= 0.0)),
        median_logit=float(np.median(logits)),
        n_partitions=n_partitions,
        n_combinations=len(combos),
        submatrix_size=sub,
        n_strategies=S,
        failure_rate=oos_neg_count / len(combos),
    )
