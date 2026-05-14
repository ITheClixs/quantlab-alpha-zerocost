from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def weighted_zero_mean_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Weighted zero-mean R^2 (Jane Street competition metric).

    R^2 = 1 - sum(w * (y - y_hat)^2) / sum(w * y^2)
    Note: denominator uses y^2 (no mean subtraction) which is the "zero-mean" choice.
    """
    if y_true.shape != y_pred.shape or y_true.shape != weights.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape} w={weights.shape}")
    denom = float(np.sum(weights * y_true * y_true))
    if denom == 0.0:
        return 0.0
    numer = float(np.sum(weights * (y_true - y_pred) ** 2))
    return 1.0 - (numer / denom)


def sharpe_proxy(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Annualized Sharpe-proxy assuming zero risk-free rate."""
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=0))
    if sigma == 0.0:
        return 0.0
    return (mu / sigma) * float(np.sqrt(periods_per_year))
