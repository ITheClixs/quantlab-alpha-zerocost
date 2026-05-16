from __future__ import annotations

import pytest

from quant_research_stack.governor.signal_schema import (
    Decision,
    Direction,
    GovernorVerdict,
    RegimeTag,
)


def _valid_payload(**overrides):
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.8,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "trending momentum aligns with cited paper",
        "cited_paper_chunk_ids": ["paper_pdf:foo:0"],
        "contradictions_flagged": [],
    }
    payload.update(overrides)
    return payload


def test_valid_pass_with_citations() -> None:
    v = GovernorVerdict.model_validate(_valid_payload())
    assert v.decision == Decision.pass_
    assert v.cited_paper_chunk_ids == ["paper_pdf:foo:0"]


def test_valid_veto() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(decision="veto", cited_paper_chunk_ids=[]))
    assert v.decision == Decision.veto


def test_valid_insufficient_evidence() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(decision="insufficient_evidence", cited_paper_chunk_ids=[]))
    assert v.decision == Decision.insufficient_evidence


def test_pass_without_citations_is_downgraded() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(cited_paper_chunk_ids=[]))
    assert v.decision == Decision.insufficient_evidence
    assert "no citations" in v.rationale_short


def test_signal_id_too_short_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(signal_id="abc"))


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(confidence=1.5))


def test_horizon_zero_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(horizon_minutes=0))


def test_rationale_too_long_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(rationale_short="x" * 201))


def test_cited_array_too_long_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(cited_paper_chunk_ids=[f"id-{i}" for i in range(11)]))


def test_unknown_decision_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(decision="maybe"))


def test_direction_out_of_set_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(direction=2))


def test_enum_values_have_expected_strings() -> None:
    assert Decision.pass_.value == "pass"
    assert Decision.veto.value == "veto"
    assert Decision.insufficient_evidence.value == "insufficient_evidence"
    assert Direction.short.value == -1
    assert Direction.long.value == 1
    assert RegimeTag.high_vol.value == "high_vol"
