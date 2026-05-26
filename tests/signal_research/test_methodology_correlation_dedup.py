"""Correlation deduplication (spec §4.3)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.correlation_dedup import (
    DedupConfig,
    deduplicate,
)


def test_dedup_clusters_inverse_strategies_when_using_absolute_correlation() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 1))
    rets = np.hstack([
        base,
        base * 1.01,
        -base,
        rng.standard_normal((500, 1)),
    ])
    sharpe = np.array([0.5, 0.4, 0.3, 0.7])
    turnover = np.array([1.0, 1.0, 1.0, 1.0])
    dsr = np.array([0.6, 0.55, 0.5, 0.4])
    drawdown = np.array([-0.1, -0.11, -0.1, -0.2])
    result = deduplicate(
        net_returns=rets,
        sharpe=sharpe,
        turnover=turnover,
        dsr=dsr,
        drawdown=drawdown,
        config=DedupConfig(absolute_correlation_threshold=0.85),
    )
    assert result.n_clusters == 2


def test_three_representative_rules_reported() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal((500, 3))
    sharpe = np.array([0.5, 0.6, 0.55])
    turnover = np.array([1.0, 4.0, 1.5])
    dsr = np.array([0.4, 0.65, 0.5])
    drawdown = np.array([-0.10, -0.08, -0.05])
    result = deduplicate(
        net_returns=rets,
        sharpe=sharpe, turnover=turnover, dsr=dsr, drawdown=drawdown,
        config=DedupConfig(absolute_correlation_threshold=0.0),
    )
    assert "by_sharpe_per_sqrt_turnover" in result.representatives
    assert "by_dsr" in result.representatives
    assert "by_lowest_drawdown" in result.representatives
