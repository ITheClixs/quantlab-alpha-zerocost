"""Three-tier PIT data-quality classifier (spec §2.1)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.delisting_audit import DelistingAuditResult
from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource
from quant_research_stack.alpha_eq.data.pit_quality import (
    PITQualityInputs,
    classify_pit_quality,
)


def _audit(captured: int, missing: int, unknown_in_holdout: int) -> DelistingAuditResult:
    counters = {
        "delisted_captured": captured,
        "delisted_missing": missing,
        "merger_captured": 0,
        "merger_missing": 0,
        "ticker_changed": 0,
        "unknown_exit": unknown_in_holdout,
    }
    return DelistingAuditResult(counters=counters, audit_table=pl.DataFrame())


def test_pit_safe_when_membership_present_and_audit_above_threshold() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=95, missing=5, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PIT_SAFE


def test_partial_when_membership_present_but_audit_below_threshold() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=50, missing=50, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_partial_when_unknown_exit_in_holdout_nonzero() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=100, missing=0, unknown_in_holdout=1),
        unknown_exit_in_holdout=1,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_wikipedia_fallback_caps_at_partial() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.WIKIPEDIA_FALLBACK,
        audit=_audit(captured=100, missing=0, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_prototype_only_when_membership_absent() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.ABSENT_PROTOTYPE_ONLY,
        audit=_audit(captured=0, missing=0, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY
