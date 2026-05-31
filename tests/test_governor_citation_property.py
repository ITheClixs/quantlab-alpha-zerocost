from __future__ import annotations

import random

import pytest

from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _corpus() -> CorpusIndex:
    return CorpusIndex([
        Chunk(id=f"valid:{i}", source_type="t", source_path="p", chunk_index=i, text="x", sha256="x", n_words=1)
        for i in range(50)
    ])


def _random_verdict(rng: random.Random) -> GovernorVerdict:
    decisions = ["pass", "veto", "insufficient_evidence"]
    n_valid = rng.randint(0, 5)
    n_invalid = rng.randint(0, 5)
    cited = [f"valid:{rng.randint(0, 49)}" for _ in range(n_valid)] + [f"invalid:{rng.randint(0, 999)}" for _ in range(n_invalid)]
    rng.shuffle(cited)
    payload = {
        "signal_id": f"sig-{rng.randint(10**7, 10**8 - 1):08d}",
        "decision": rng.choice(decisions),
        "direction": rng.choice([-1, 0, 1]),
        "confidence": round(rng.random(), 4),
        "horizon_minutes": rng.choice([1, 5, 15, 60]),
        "regime_tag": rng.choice(["trending", "mean_reverting", "high_vol", "low_vol", "unknown"]),
        "rationale_short": "synthetic",
        "cited_paper_chunk_ids": cited[:10],
        "contradictions_flagged": [],
    }
    return GovernorVerdict.model_validate(payload)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_citation_invariant_across_200_generations(seed: int) -> None:
    rng = random.Random(seed)
    corpus = _corpus()
    for _ in range(200):
        v = _random_verdict(rng)
        out, _ = resolve_citations(v, corpus)
        if out.decision == Decision.pass_:
            assert out.cited_paper_chunk_ids, "pass verdict reaching consumer must have at least one valid citation"
            assert all(cid in corpus for cid in out.cited_paper_chunk_ids), "all citations must resolve"
