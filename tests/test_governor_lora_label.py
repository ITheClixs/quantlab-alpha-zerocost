from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from governor_lora_label import label_chunk_with_seed  # noqa: E402


def test_labeller_deterministic_across_two_runs() -> None:
    chunk_text = "mean reversion is reliable at 1-min horizon"
    chunk_id = "paper_pdf:foo:0"
    a = label_chunk_with_seed(chunk_id, chunk_text, seed=42)
    b = label_chunk_with_seed(chunk_id, chunk_text, seed=42)
    assert a == b


def test_different_seed_yields_different_label() -> None:
    chunk_text = "mean reversion is reliable at 1-min horizon"
    chunk_id = "paper_pdf:foo:0"
    a = label_chunk_with_seed(chunk_id, chunk_text, seed=1)
    b = label_chunk_with_seed(chunk_id, chunk_text, seed=999)
    # at least one of the synthetic fields must differ
    assert a != b


def test_label_returns_required_keys() -> None:
    out = label_chunk_with_seed("id", "text", seed=0)
    expected = {"signal_id", "decision", "direction", "confidence", "horizon_minutes", "regime_tag", "rationale_short", "cited_paper_chunk_ids", "contradictions_flagged"}
    assert expected.issubset(set(out.keys()))
