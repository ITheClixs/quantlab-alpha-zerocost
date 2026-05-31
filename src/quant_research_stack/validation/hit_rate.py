from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredSignal:
    signal_id: str
    predicted_direction: int
    realized_direction: int
    weight: float
    s2_decision: str

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative; got {self.weight}")


@dataclass(frozen=True)
class HitRateResult:
    hit_rate: float
    n_signals: int
    n_hits: int
    governor_block_rate: float


def compute_hit_rate(signals: Iterable[ScoredSignal]) -> HitRateResult:
    """Weighted directional hit-rate plus governor-block-rate.

    Signals with predicted_direction == 0 (vetoed/insufficient_evidence/zero-weight)
    are excluded from the hit_rate numerator and denominator. They DO count toward
    governor_block_rate when s2_decision is veto or insufficient_evidence.
    """
    sigs = list(signals)
    total = len(sigs)
    if total == 0:
        return HitRateResult(hit_rate=0.0, n_signals=0, n_hits=0, governor_block_rate=0.0)

    block_count = sum(1 for s in sigs if s.s2_decision in ("veto", "insufficient_evidence"))
    governor_block_rate = block_count / total

    eligible = [s for s in sigs if s.predicted_direction != 0 and s.weight > 0]
    if not eligible:
        return HitRateResult(
            hit_rate=0.0, n_signals=0, n_hits=0, governor_block_rate=governor_block_rate,
        )

    denom = sum(s.weight for s in eligible)
    numer = sum(
        s.weight for s in eligible
        if s.predicted_direction == s.realized_direction
    )
    n_hits = sum(
        1 for s in eligible
        if s.predicted_direction == s.realized_direction
    )
    hit_rate = numer / denom if denom > 0 else 0.0
    return HitRateResult(
        hit_rate=float(hit_rate),
        n_signals=len(eligible),
        n_hits=n_hits,
        governor_block_rate=float(governor_block_rate),
    )
