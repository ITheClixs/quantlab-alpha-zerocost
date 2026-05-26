"""Multi-objective Pareto front (spec §4.4)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.multi_objective import pareto_front


def test_pareto_front_keeps_non_dominated() -> None:
    sharpe = np.array([1.0, 0.5, 0.7])
    abs_dd = np.array([0.10, 0.10, 0.05])
    turnover = np.array([1.0, 1.0, 1.0])
    capacity_shrinkage = np.array([0.1, 0.1, 0.1])
    front = pareto_front(
        sharpe=sharpe,
        abs_drawdown=abs_dd,
        turnover=turnover,
        capacity_shrinkage=capacity_shrinkage,
    )
    assert 0 in front
    assert 2 in front
    assert 1 not in front
