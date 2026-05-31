from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import GovernorVerdict


def _verdict() -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-00000001",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    })


def test_s1_signal_validates() -> None:
    s = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    assert s.symbol == "BTCUSDT"
    assert 0 <= s.confidence <= 1


def test_s1_signal_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        S1Signal(
            signal_id="sig-00000001",
            symbol="BTCUSDT",
            predicted_score=0.05,
            confidence=2.0,
            horizon_minutes=5,
            ts_utc=datetime.now(UTC),
        )


def test_execution_ticket_pairs_signal_and_verdict() -> None:
    sig = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = _verdict()
    t = ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))
    assert t.signal.signal_id == t.primary_verdict.signal_id


def test_execution_ticket_rejects_mismatched_ids() -> None:
    sig = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = GovernorVerdict.model_validate({
        "signal_id": "sig-00000002",  # different id
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    })
    with pytest.raises(ValueError, match="signal_id mismatch"):
        ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))
