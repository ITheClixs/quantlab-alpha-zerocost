"""Multi-objective Pareto front (spec §4.4).

Selection-and-reporting tool only — NOT a promotion criterion (§6.1).
Primary axes (v1): max Sharpe, min |DD|, min turnover, min capacity shrinkage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pareto_front(
    *,
    sharpe: NDArray[np.float64],
    abs_drawdown: NDArray[np.float64],
    turnover: NDArray[np.float64],
    capacity_shrinkage: NDArray[np.float64],
) -> list[int]:
    """Return indices of non-dominated strategies (max sharpe; min the rest)."""
    n = len(sharpe)
    obj = np.column_stack([-sharpe, abs_drawdown, turnover, capacity_shrinkage])
    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front
