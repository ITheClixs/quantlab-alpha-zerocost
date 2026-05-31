"""Success-gate evaluator (spec §6.4)."""

from __future__ import annotations

from typing import Any

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    evaluate_success_gate,
)


def _good_inputs(**overrides: Any) -> SuccessGateInputs:
    base = SuccessGateInputs(
        data_quality_label=DataQualityLabel.PIT_SAFE,
        holdout_trading_days=800,
        delisting_capture_ratio=0.98,
        delisting_unknown_in_holdout=0,
        s1_eq_net_sharpe=1.0,
        family_b_net_sharpe=0.5,
        spy_sharpe=0.7,
        max_drawdown=-0.10,
        net_sharpe_borrow_2x=0.6,
        net_total_return_borrow_3x=0.05,
        js_overlay_net_sharpe=0.4,
        rolling_window_alpha_consistent=True,
        concentration_stock_violation=False,
        concentration_month_violation=False,
        concentration_sector_violation=False,
        ci_tests_green=True,
        artifacts_complete=True,
    )
    return base.model_copy(update=overrides)


def test_gate_passes_on_good_inputs() -> None:
    res = evaluate_success_gate(_good_inputs())
    assert res.passed is True
    assert res.failures == []


def test_gate_fails_when_holdout_too_short() -> None:
    res = evaluate_success_gate(_good_inputs(holdout_trading_days=500))
    assert res.passed is False
    assert any("holdout" in f for f in res.failures)


def test_gate_two_branch_baseline_negative_family_b() -> None:
    """Family B Sharpe ≤ 0 → S1-EQ must be ≥ 0.7 AND beat Family B by ≥ 0.5."""
    res = evaluate_success_gate(_good_inputs(family_b_net_sharpe=-0.3, s1_eq_net_sharpe=0.8))
    assert res.passed is True


def test_gate_standalone_sharpe_negative_spy_does_not_lower_bar() -> None:
    res = evaluate_success_gate(_good_inputs(spy_sharpe=-0.2, s1_eq_net_sharpe=0.5))
    assert res.passed is False


def test_gate_suspended_for_prototype_only() -> None:
    res = evaluate_success_gate(
        _good_inputs(data_quality_label=DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY)
    )
    assert res.suspended is True
