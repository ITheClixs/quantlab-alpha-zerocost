"""Four-tier candidate status taxonomy (spec §6.1)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.status import (
    CandidateStatus,
    promote_if_eligible,
    status_at_least,
)


def test_status_ordering() -> None:
    assert CandidateStatus.NONE < CandidateStatus.RESEARCH_PASS
    assert CandidateStatus.RESEARCH_PASS < CandidateStatus.PROMOTION_ELIGIBLE
    assert CandidateStatus.PROMOTION_ELIGIBLE < CandidateStatus.PAPER_TRADE_CANDIDATE
    assert CandidateStatus.PAPER_TRADE_CANDIDATE < CandidateStatus.PRODUCTION_CANDIDATE


def test_status_string_values() -> None:
    assert CandidateStatus.NONE.name_lower == "none"
    assert CandidateStatus.RESEARCH_PASS.name_lower == "research_pass"
    assert CandidateStatus.PROMOTION_ELIGIBLE.name_lower == "promotion_eligible"
    assert CandidateStatus.PAPER_TRADE_CANDIDATE.name_lower == "paper_trade_candidate"
    assert CandidateStatus.PRODUCTION_CANDIDATE.name_lower == "production_candidate"


def test_status_at_least_returns_true_when_equal_or_higher() -> None:
    assert status_at_least(CandidateStatus.PROMOTION_ELIGIBLE, CandidateStatus.RESEARCH_PASS)
    assert status_at_least(CandidateStatus.PROMOTION_ELIGIBLE, CandidateStatus.PROMOTION_ELIGIBLE)
    assert not status_at_least(CandidateStatus.RESEARCH_PASS, CandidateStatus.PROMOTION_ELIGIBLE)


def test_promote_if_eligible_advances_by_one_tier_only() -> None:
    assert promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=True) == CandidateStatus.PROMOTION_ELIGIBLE
    assert promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=False) == CandidateStatus.RESEARCH_PASS


def test_promote_if_eligible_never_skips_stages() -> None:
    # Cannot jump from RESEARCH_PASS to PAPER_TRADE_CANDIDATE in one call
    promoted = promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=True)
    assert promoted != CandidateStatus.PAPER_TRADE_CANDIDATE


def test_promote_at_top_is_idempotent() -> None:
    assert promote_if_eligible(CandidateStatus.PRODUCTION_CANDIDATE, promoted=True) == CandidateStatus.PRODUCTION_CANDIDATE


def test_unknown_string_raises() -> None:
    with pytest.raises(KeyError):
        CandidateStatus["NOT_A_REAL_STATUS"]
