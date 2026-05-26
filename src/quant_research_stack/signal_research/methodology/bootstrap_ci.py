"""Stationary block bootstrap (Politis & Romano 1994) for Sharpe CIs.

Spec §4.6: mean block length L = T^(1/3); n_resamples = 10000 default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int = 10000
    block_length: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class BootstrapResult:
    point_sharpe: float
    ci_lower_95: float
    ci_upper_95: float


def _sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns)) / sd * float(np.sqrt(252.0))


def _stationary_block_bootstrap(
    returns: NDArray[np.float64],
    *,
    n_resamples: int,
    block_length: int,
    seed: int,
) -> NDArray[np.float64]:
    T = returns.size
    rng = np.random.default_rng(seed)
    p = 1.0 / block_length
    sharpes = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        sample = np.empty(T, dtype=np.float64)
        i = 0
        idx = int(rng.integers(0, T))
        while i < T:
            sample[i] = returns[idx]
            i += 1
            if rng.random() < p:
                idx = int(rng.integers(0, T))
            else:
                idx = (idx + 1) % T
        sharpes[k] = _sharpe(sample)
    return sharpes


def bootstrap_sharpe_ci(
    *,
    returns: NDArray[np.float64],
    config: BootstrapConfig,
) -> BootstrapResult:
    T = returns.size
    block = (
        config.block_length
        if config.block_length is not None
        else max(1, int(round(T ** (1 / 3))))
    )
    sharpes = _stationary_block_bootstrap(
        returns,
        n_resamples=config.n_resamples,
        block_length=block,
        seed=config.seed,
    )
    return BootstrapResult(
        point_sharpe=_sharpe(returns),
        ci_lower_95=float(np.percentile(sharpes, 2.5)),
        ci_upper_95=float(np.percentile(sharpes, 97.5)),
    )
