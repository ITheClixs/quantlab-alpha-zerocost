from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.grammar import (
    PACKAGE_GRAMMAR_FULL_PATH,
    PACKAGE_GRAMMAR_TIER1_PATH,
    generate_full_grammar,
    generate_tier1_grammar,
    validate_against_grammar_shape,
)


def test_full_grammar_file_committed() -> None:
    text = Path(PACKAGE_GRAMMAR_FULL_PATH).read_text()
    assert "decision ::=" in text
    assert '"\\"pass\\""' in text
    assert '"\\"veto\\""' in text
    assert '"\\"insufficient_evidence\\""' in text


def test_tier1_grammar_file_committed() -> None:
    text = Path(PACKAGE_GRAMMAR_TIER1_PATH).read_text()
    assert '"\\"pass\\""' in text
    assert '"\\"veto\\""' in text
    assert '"\\"insufficient_evidence\\""' not in text


def test_generated_full_grammar_matches_committed_file() -> None:
    generated = generate_full_grammar()
    committed = Path(PACKAGE_GRAMMAR_FULL_PATH).read_text()
    assert generated.strip() == committed.strip()


def test_generated_tier1_grammar_matches_committed_file() -> None:
    generated = generate_tier1_grammar()
    committed = Path(PACKAGE_GRAMMAR_TIER1_PATH).read_text()
    assert generated.strip() == committed.strip()


def test_validate_against_shape_accepts_valid_payload() -> None:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.85,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:foo:0"],
        "contradictions_flagged": [],
    }
    assert validate_against_grammar_shape(json.dumps(payload)) is True


def test_validate_against_shape_rejects_missing_field() -> None:
    bad = '{"signal_id": "abc"}'
    assert validate_against_grammar_shape(bad) is False


def test_validate_against_shape_rejects_unknown_decision() -> None:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "maybe",
        "direction": 1,
        "confidence": 0.85,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["x"],
        "contradictions_flagged": [],
    }
    assert validate_against_grammar_shape(json.dumps(payload)) is False
