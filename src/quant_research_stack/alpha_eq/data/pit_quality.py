"""Three-tier PIT data-quality classifier (spec §2.1, §2.9)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.alpha_eq.data.delisting_audit import DelistingAuditResult
from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource

DELISTING_CAPTURE_PIT_SAFE_THRESHOLD = 0.95  # ≥95% captured + zero unknown_exit in holdout


@dataclass(frozen=True)
class PITQualityInputs:
    membership_source: MembershipSource
    audit: DelistingAuditResult
    unknown_exit_in_holdout: int


def _capture_ratio(c: dict[str, int]) -> float:
    captured = c["delisted_captured"] + c["merger_captured"] + c["ticker_changed"]
    total = captured + c["delisted_missing"] + c["merger_missing"] + c["unknown_exit"]
    return 1.0 if total == 0 else captured / total


def classify_pit_quality(inputs: PITQualityInputs) -> DataQualityLabel:
    if inputs.membership_source == MembershipSource.ABSENT_PROTOTYPE_ONLY:
        return DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY

    audit_ok = (
        _capture_ratio(inputs.audit.counters) >= DELISTING_CAPTURE_PIT_SAFE_THRESHOLD
        and inputs.unknown_exit_in_holdout == 0
    )

    if inputs.membership_source == MembershipSource.WIKIPEDIA_FALLBACK:
        # Wikipedia is fallback only — never institutional-grade by itself.
        return DataQualityLabel.PARTIAL_PIT_UNIVERSE

    if audit_ok:
        return DataQualityLabel.PIT_SAFE
    return DataQualityLabel.PARTIAL_PIT_UNIVERSE
