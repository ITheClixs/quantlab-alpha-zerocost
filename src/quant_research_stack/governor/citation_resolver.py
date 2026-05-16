from __future__ import annotations

from quant_research_stack.governor.corpus import CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def resolve_citations(verdict: GovernorVerdict, corpus: CorpusIndex) -> tuple[GovernorVerdict, list[str]]:
    valid = [cid for cid in verdict.cited_paper_chunk_ids if cid in corpus]
    invalid = [cid for cid in verdict.cited_paper_chunk_ids if cid not in corpus]
    if not valid and verdict.decision == Decision.pass_:
        verdict = verdict.model_copy(update={
            "decision": Decision.insufficient_evidence,
            "rationale_short": "all citations unresolved; auto-downgrade",
            "cited_paper_chunk_ids": [],
        })
    else:
        verdict = verdict.model_copy(update={"cited_paper_chunk_ids": valid})
    return verdict, invalid
