from __future__ import annotations

from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _corpus(ids: list[str]) -> CorpusIndex:
    return CorpusIndex([Chunk(id=cid, source_type="t", source_path="p", chunk_index=i, text="t", sha256="x", n_words=1) for i, cid in enumerate(ids)])


def _verdict(**overrides) -> GovernorVerdict:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "x",
        "cited_paper_chunk_ids": ["a"],
        "contradictions_flagged": [],
    }
    payload.update(overrides)
    return GovernorVerdict.model_validate(payload)


def test_all_citations_valid_keeps_pass() -> None:
    corpus = _corpus(["a", "b"])
    v = _verdict(cited_paper_chunk_ids=["a", "b"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.pass_
    assert out.cited_paper_chunk_ids == ["a", "b"]
    assert invalid == []


def test_partial_invalid_drops_them() -> None:
    corpus = _corpus(["a"])
    v = _verdict(cited_paper_chunk_ids=["a", "missing"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.pass_
    assert out.cited_paper_chunk_ids == ["a"]
    assert invalid == ["missing"]


def test_all_invalid_pass_downgrades_to_insufficient() -> None:
    corpus = _corpus(["x"])
    v = _verdict(cited_paper_chunk_ids=["nope1", "nope2"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.insufficient_evidence
    assert out.cited_paper_chunk_ids == []
    assert invalid == ["nope1", "nope2"]


def test_veto_with_invalid_citations_is_kept() -> None:
    corpus = _corpus(["x"])
    v = _verdict(decision="veto", cited_paper_chunk_ids=["nope"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.veto
    assert out.cited_paper_chunk_ids == []
    assert invalid == ["nope"]
