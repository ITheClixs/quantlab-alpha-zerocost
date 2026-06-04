"""Permutation null control: independently time-shuffle each strategy's IS returns and
recompute the best Sharpe. If the real best ~ the permuted best, the winner is an artifact."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _best_sharpe(mat: NDArray[np.float64]) -> float:
    mu = np.mean(mat, axis=0); sd = np.std(mat, axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    sr = np.nan_to_num(mu / sd * np.sqrt(252.0), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(sr))


def permutation_control(*, is_returns: NDArray[np.float64], seed: int = 42) -> dict[str, Any]:
    r = is_returns.astype(np.float64)
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(r)
    for j in range(r.shape[1]):
        permuted[:, j] = r[rng.permutation(r.shape[0]), j]
    return {"real_best_sharpe": _best_sharpe(r), "permuted_best_sharpe": _best_sharpe(permuted),
            "seed": seed, "n_strategies": int(r.shape[1])}
