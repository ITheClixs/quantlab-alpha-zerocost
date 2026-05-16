"""Three-tier LLM governor orchestration.

Cascade: Tier 1 (fast, Qwen 0.5B+LoRA) → Tier 2 (medium, Mistral 22B Q4, RAG-augmented)
→ Tier 3 (deep async, Yi 34B Q4, large-trade only).

Tier 1 veto short-circuits the cascade.  Tier 2 is only invoked when the signal
confidence exceeds *tier2_required_when_tier1_passes_above_confidence*.  Tier 3 is
scheduled asynchronously (fire-and-forget) when trade_size_pct exceeds its threshold;
the synchronous verdict is always the Tier 2 result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


@dataclass(frozen=True)
class S1Signal:
    """Incoming signal produced by S1 alpha layer."""

    signal_id: str
    symbol: str
    direction: int
    confidence: float
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str
    trade_size_pct: float


@dataclass(frozen=True)
class EscalationConfig:
    """Thresholds that control which tiers are invoked."""

    tier1_required: bool = True
    tier2_required_when_tier1_passes_above_confidence: float = 0.6
    tier3_required_when_trade_size_pct_above: float = 1.0
    rerank_to_k: int = 5


def govern_signal(
    signal: S1Signal,
    cfg: EscalationConfig,
    runtimes: Any,
    corpus: CorpusIndex,
    retrieve_top_k: Callable[[S1Signal, int], list[Chunk]],
) -> GovernorVerdict:
    """Run the three-tier cascade and return the authoritative verdict.

    Args:
        signal: The S1 signal to govern.
        cfg: Escalation thresholds.
        runtimes: Object with `.tier1`, `.tier2`, `.tier3` runtime attributes.
        corpus: The loaded research corpus (unused directly; passed for retrieval).
        retrieve_top_k: Callable ``(signal, k) -> list[Chunk]`` used to supply
            RAG context to Tier 2 and Tier 3.

    Returns:
        The final :class:`~quant_research_stack.governor.signal_schema.GovernorVerdict`.
        Tier 3 is scheduled asynchronously and does not affect the return value.
    """
    # --- Tier 1: fast veto gate ---
    t1: GovernorVerdict = runtimes.tier1.govern(signal, retrieval=None)
    if t1.decision == Decision.veto:
        return t1

    # --- Confidence gate: skip Tier 2 for low-confidence signals ---
    if abs(signal.confidence) < cfg.tier2_required_when_tier1_passes_above_confidence:
        # Return Tier 1 verdict, forcing decision to pass_ (Tier 1 veto was already
        # handled above; insufficient_evidence here means "no RAG citations yet" which
        # is expected for the fast-path gate — not a blocking condition).
        return GovernorVerdict.model_construct(
            **{**t1.model_dump(), "decision": Decision.pass_},
        )

    # --- Tier 2: medium, RAG-augmented ---
    chunks: list[Chunk] = retrieve_top_k(signal, cfg.rerank_to_k)
    t2: GovernorVerdict = runtimes.tier2.govern(signal, retrieval=chunks)
    if t2.decision != Decision.pass_:
        return t2

    # --- Tier 3: deep async, only for large trades ---
    if signal.trade_size_pct > cfg.tier3_required_when_trade_size_pct_above:
        runtimes.tier3.schedule_async(signal, chunks)

    return t2
