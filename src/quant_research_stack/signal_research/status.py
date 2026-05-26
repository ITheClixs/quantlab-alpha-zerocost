"""Four-tier candidate status taxonomy (spec §6.1, §0 non-negotiable #12).

Sequential promotion only — no stage-skipping (§0 non-negotiable #10).
"""

from __future__ import annotations

import enum


class CandidateStatus(enum.IntEnum):
    NONE = 0
    RESEARCH_PASS = 1
    PROMOTION_ELIGIBLE = 2
    PAPER_TRADE_CANDIDATE = 3
    PRODUCTION_CANDIDATE = 4

    @property
    def name_lower(self) -> str:
        return self.name.lower()


def status_at_least(actual: CandidateStatus, required: CandidateStatus) -> bool:
    """Returns True iff `actual` has reached `required` or higher."""
    return int(actual) >= int(required)


def promote_if_eligible(
    current: CandidateStatus, *, promoted: bool
) -> CandidateStatus:
    """Advance the candidate by exactly one tier if `promoted=True`,
    otherwise return `current` unchanged.

    Sequential promotion only. Cannot skip stages.
    """
    if not promoted:
        return current
    if current == CandidateStatus.PRODUCTION_CANDIDATE:
        return current  # top tier — idempotent
    return CandidateStatus(int(current) + 1)
