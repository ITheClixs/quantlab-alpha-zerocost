"""Stationary block bootstrap CIs (spec §4.6)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)


def test_bootstrap_returns_lower_and_upper_bounds() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.01 + 0.0005
    res = bootstrap_sharpe_ci(returns=rets, config=BootstrapConfig(n_resamples=200, seed=0))
    assert res.point_sharpe > 0
    assert res.ci_lower_95 <= res.point_sharpe <= res.ci_upper_95


def test_bootstrap_zero_signal_brackets_zero() -> None:
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(500) * 0.01
    res = bootstrap_sharpe_ci(returns=rets, config=BootstrapConfig(n_resamples=300, seed=1))
    assert res.ci_lower_95 < 0.5 and res.ci_upper_95 > -0.5
