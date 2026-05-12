from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_stack.llm_quant import (
    build_signal_prompt,
    choose_local_model,
    load_research_chunks,
    parse_quant_signal,
    retrieve_chunks,
)


def test_load_and_retrieve_research_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "research.jsonl"
    records = [
        {"id": "paper:1", "text": "Order flow imbalance and microprice improve short horizon prediction.", "source_path": "a.pdf"},
        {"id": "paper:2", "text": "Portfolio theory and long horizon allocation.", "source_path": "b.pdf"},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    chunks = load_research_chunks(corpus)
    retrieved = retrieve_chunks("microprice order flow", chunks, top_k=1)
    assert retrieved[0].id == "paper:1"


def test_build_signal_prompt_contains_context_and_chunk_ids() -> None:
    chunks = load_research_chunks(Path("missing.jsonl"))
    prompt = build_signal_prompt({"feature_00": 1.2, "horizon": 1}, chunks)
    assert "strict JSON object" in prompt
    assert "feature_00" in prompt


def test_parse_quant_signal_accepts_valid_json() -> None:
    raw = json.dumps(
        {
            "signal_direction": "up",
            "confidence": 0.61,
            "horizon": 1,
            "features_used": ["feature_00", "imbalance_l1"],
            "research_support": ["paper:1"],
            "risk_flags": ["low signal-to-noise"],
            "rationale": "Order flow context is positive.",
        }
    )
    signal = parse_quant_signal(raw, {"paper:1"})
    assert signal.signal_direction == "up"
    assert signal.confidence == 0.61


def test_parse_quant_signal_rejects_unknown_research_id() -> None:
    raw = json.dumps(
        {
            "signal_direction": "flat",
            "confidence": 0.3,
            "horizon": 5,
            "features_used": ["feature_00"],
            "research_support": ["invented"],
            "risk_flags": [],
            "rationale": "No edge.",
        }
    )
    with pytest.raises(ValueError, match="unknown chunk"):
        parse_quant_signal(raw, {"paper:1"})


def test_choose_local_model_prefers_available_primary_gguf(tmp_path: Path) -> None:
    primary = tmp_path / "models" / "primary"
    fallback = tmp_path / "models" / "fallback"
    primary.mkdir(parents=True)
    fallback.mkdir(parents=True)
    (fallback / "fallback.Q4_K_M.gguf").write_text("fallback", encoding="utf-8")
    (primary / "primary.Q4_K_M.gguf").write_text("primary", encoding="utf-8")
    choice = choose_local_model(
        {
            "llm_runtime": {
                "primary_model_id": "primary/model",
                "primary_local_dir": "models/primary",
                "fallback_model_id": "fallback/model",
                "fallback_local_dir": "models/fallback",
                "preferred_quant": "Q4_K_M",
            }
        },
        repo_root=tmp_path,
    )
    assert choice.role == "primary"
    assert choice.gguf_file and choice.gguf_file.name == "primary.Q4_K_M.gguf"
