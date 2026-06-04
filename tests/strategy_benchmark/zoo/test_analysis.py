from __future__ import annotations

import numpy as np

from quant_research_stack.strategy_benchmark.zoo.analysis import (
    deflate_best,
    expected_vs_empirical,
)


def test_expected_vs_empirical_rises_with_n() -> None:
    rng = np.random.default_rng(0)
    sharpes = rng.normal(0, 1, 100_000)
    out = expected_vs_empirical(sharpe_estimates=sharpes, tiers=(1_000, 10_000, 100_000))
    e = [r["empirical_max"] for r in out]
    assert e[0] < e[1] < e[2]
    for r in out:
        assert abs(r["empirical_max"] - r["theoretical_max"]) < 0.6


def test_deflate_best_rejects_lucky_winner() -> None:
    rng = np.random.default_rng(1)
    T, N = 500, 5_000
    mat = rng.normal(0, 0.01, (T, N))
    res = deflate_best(is_returns=mat)
    assert res["dsr"] < 0.95
