from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from quant_research_stack.governor.signal_schema import GovernorVerdict


class S1Signal(BaseModel):
    model_config = {"frozen": True}
    signal_id: Annotated[str, Field(min_length=4, max_length=64)]
    symbol: Annotated[str, Field(min_length=1, max_length=32)]
    predicted_score: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    horizon_minutes: Annotated[int, Field(ge=1, le=1440)]
    ts_utc: datetime


class ExecutionTicket(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    signal: S1Signal
    primary_verdict: GovernorVerdict
    tier3_verdict: GovernorVerdict | None
    ingested_at: datetime

    @model_validator(mode="after")
    def _ids_match(self) -> ExecutionTicket:
        if self.signal.signal_id != self.primary_verdict.signal_id:
            raise ValueError("signal_id mismatch between S1 signal and primary verdict")
        if self.tier3_verdict is not None and self.tier3_verdict.signal_id != self.signal.signal_id:
            raise ValueError("signal_id mismatch between S1 signal and tier3 verdict")
        return self
