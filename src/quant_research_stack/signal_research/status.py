"""Five-tier candidate status taxonomy (spec §6.1, §0 non-negotiable #12).

Sequential promotion only — no stage-skipping (§0 non-negotiable #10).

Two parallel mid-ladder tiers reflect the two promotion paths:
- Default path: RESEARCH_PASS → PROMOTION_ELIGIBLE → PAPER_TRADE_CANDIDATE
- Exception path: RESEARCH_PASS → EXCEPTION_REVIEW_REQUIRED → PAPER_TRADE_CANDIDATE

Exception path activates only under the accepted single-index risk-timing
exception policy (commit 74ca502); see ValidationSpec.exception_invoked.
"""

from __future__ import annotations

import enum


class CandidateStatus(enum.IntEnum):
    NONE = 0
    RESEARCH_PASS = 1
    EXCEPTION_REVIEW_REQUIRED = 2  # exception-path tier-2
    PROMOTION_ELIGIBLE = 3  # default-path tier-2
    PAPER_TRADE_CANDIDATE = 4
    PRODUCTION_CANDIDATE = 5

    @property
    def name_lower(self) -> str:
        return self.name.lower()


def status_at_least(actual: CandidateStatus, required: CandidateStatus) -> bool:
    """Returns True iff `actual` has reached `required` or higher.

    Note that EXCEPTION_REVIEW_REQUIRED (2) and PROMOTION_ELIGIBLE (3) are
    parallel tiers on different paths; status_at_least uses integer order
    only and does not enforce path semantics.
    """
    return int(actual) >= int(required)


def promote_if_eligible(
    current: CandidateStatus,
    *,
    promoted: bool,
    target: CandidateStatus | None = None,
) -> CandidateStatus:
    """Advance the candidate by exactly one tier if `promoted=True`.

    `target` may be supplied to choose between EXCEPTION_REVIEW_REQUIRED
    and PROMOTION_ELIGIBLE when current=RESEARCH_PASS. If omitted, the
    function defaults to PROMOTION_ELIGIBLE (the default-path tier-2).

    Sequential promotion only — cannot skip stages, cannot demote.
    """
    if not promoted:
        return current
    if current == CandidateStatus.PRODUCTION_CANDIDATE:
        return current
    if target is None:
        if current == CandidateStatus.RESEARCH_PASS:
            return CandidateStatus.PROMOTION_ELIGIBLE
        if current == CandidateStatus.EXCEPTION_REVIEW_REQUIRED:
            return CandidateStatus.PAPER_TRADE_CANDIDATE
        return CandidateStatus(int(current) + 1)
    if int(target) <= int(current):
        return current  # cannot demote
    # Enforce single-step advances
    diff = int(target) - int(current)
    if current == CandidateStatus.RESEARCH_PASS and target in (
        CandidateStatus.EXCEPTION_REVIEW_REQUIRED,
        CandidateStatus.PROMOTION_ELIGIBLE,
    ):
        return target
    if (
        current in (CandidateStatus.EXCEPTION_REVIEW_REQUIRED, CandidateStatus.PROMOTION_ELIGIBLE)
        and target == CandidateStatus.PAPER_TRADE_CANDIDATE
    ):
        return target
    if diff == 1:
        return target
    return current
