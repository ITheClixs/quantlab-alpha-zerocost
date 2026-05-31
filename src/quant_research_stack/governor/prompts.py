from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from quant_research_stack.governor.corpus import Chunk


class _SignalShape(Protocol):
    signal_id: str
    symbol: str
    direction: int
    confidence: float
    horizon_minutes: int
    regime_hint: str | None


SYSTEM_PROMPT = """You are QuantLab's signal governor.

You receive an S1 trading signal candidate plus retrieved evidence from the local
research corpus. Decide whether to pass, veto, or return insufficient_evidence.

Rules:
1. Output ONLY valid JSON matching the schema. The grammar will reject anything else.
2. Cite at least one chunk_id you actually used. Do not invent IDs.
3. If the retrieved evidence does not address the signal's regime + horizon + symbol,
   return insufficient_evidence. Do not guess.
4. Veto if the signal contradicts cited evidence (e.g. signal says long-momentum at
   1-min horizon but cited paper shows mean-reversion at that horizon).
5. confidence is your confidence in the verdict, not the trade.
"""


def build_user_message(signal: _SignalShape, retrieved: Iterable[Chunk]) -> str:
    evidence_block = "\n".join(
        f"[{c.id}] ({c.source_path}): {c.text[:600]}..." for c in retrieved
    )
    regime = signal.regime_hint or "unknown"
    return (
        f"Signal:\n"
        f"  signal_id: {signal.signal_id}\n"
        f"  symbol: {signal.symbol}\n"
        f"  direction: {signal.direction}\n"
        f"  confidence: {signal.confidence:.4f}\n"
        f"  horizon_minutes: {signal.horizon_minutes}\n"
        f"  regime_hint: {regime}\n\n"
        f"Retrieved evidence (use these chunk_ids if you cite):\n"
        f"{evidence_block}\n\n"
        f"Emit your verdict as JSON now."
    )
