from __future__ import annotations

from enum import Enum, StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class Decision(StrEnum):
    pass_ = "pass"
    veto = "veto"
    insufficient_evidence = "insufficient_evidence"


class Direction(int, Enum):
    short = -1
    flat = 0
    long = 1


class RegimeTag(StrEnum):
    trending = "trending"
    mean_reverting = "mean_reverting"
    high_vol = "high_vol"
    low_vol = "low_vol"
    unknown = "unknown"


class GovernorVerdict(BaseModel):
    signal_id: Annotated[str, Field(min_length=8, max_length=64)]
    decision: Decision
    direction: Direction
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    horizon_minutes: Annotated[int, Field(ge=1, le=1440)]
    regime_tag: RegimeTag
    rationale_short: Annotated[str, Field(max_length=200)]
    cited_paper_chunk_ids: Annotated[list[str], Field(min_length=0, max_length=10)]
    contradictions_flagged: Annotated[list[str], Field(max_length=5)]

    @model_validator(mode="after")
    def enforce_citation_invariant(self) -> GovernorVerdict:
        if self.decision == Decision.pass_ and not self.cited_paper_chunk_ids:
            object.__setattr__(self, "decision", Decision.insufficient_evidence)
            object.__setattr__(self, "rationale_short", "no citations provided; auto-downgrade")
        return self
