"""Failure classifier — 13 categories (spec §4.10, §6.3)."""

from __future__ import annotations

from quant_research_stack.signal_research.methodology.failure_classifier import (
    CandidateFailureRecord,
    FailureCategory,
    all_failure_categories,
)


def test_thirteen_categories_present() -> None:
    cats = all_failure_categories()
    assert len(cats) == 13
    expected = {
        "high_pbo", "low_dsr", "cost_failure", "regime_concentration",
        "insufficient_sample", "too_few_trades", "delay_stress_fail",
        "single_period_dominance", "over_correlated_with_baseline",
        "randomization_fail", "data_quality_fail",
        "holdout_failure", "capacity_failure",
    }
    assert {c.value for c in cats} == expected


def test_candidate_failure_record_holds_multiple_categories() -> None:
    rec = CandidateFailureRecord(
        strategy_id="X",
        categories=[FailureCategory.HIGH_PBO, FailureCategory.LOW_DSR],
    )
    assert len(rec.categories) == 2
