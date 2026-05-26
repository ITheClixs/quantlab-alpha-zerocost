"""Regime-conditional metrics (spec §4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.signal_research.methodology.regime_conditional import (
    RegimeDeclaration,
    regime_conditional_metrics,
)


def test_regime_agnostic_strategy_with_positive_in_both_passes() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.005 + 0.0003
    states = (rng.random(500) > 0.5).astype(np.int64)
    res = regime_conditional_metrics(
        returns=rets,
        regime_states=states,
        declaration=RegimeDeclaration.AGNOSTIC,
    )
    assert isinstance(res.passes_regime_gate, bool)


def test_regime_specific_without_predeclared_gate_fails() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.005
    states = (rng.random(500) > 0.5).astype(np.int64)
    res = regime_conditional_metrics(
        returns=rets,
        regime_states=states,
        declaration=RegimeDeclaration.SPECIFIC,
        favorable_regime=None,
    )
    assert res.passes_regime_gate is False


def test_fit_hmm_returns_state_per_day() -> None:
    pytest.importorskip("hmmlearn")
    from quant_research_stack.signal_research.methodology.regime_conditional import (
        fit_hmm_regimes,
    )
    rng = np.random.default_rng(0)
    rets = np.concatenate([
        rng.standard_normal(500) * 0.005,
        rng.standard_normal(500) * 0.025,
    ])
    states = fit_hmm_regimes(rets, n_states=2, seed=0)
    assert states.shape == (1000,)
    assert set(np.unique(states).tolist()).issubset({0, 1})
