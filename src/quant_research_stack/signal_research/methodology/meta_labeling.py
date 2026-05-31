"""Meta-labeling (spec §4.2).

Survivor-only pre-filter:
- Positive validation net Sharpe.
- Positive validation hit rate (>0.5) OR positive expectancy.
- Sufficient event count: ≥200 single-asset / ≥500 cross-sectional.
- No inverted-signal superiority.
- Not a near-duplicate of a stronger primary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimarySignalStats:
    validation_net_sharpe: float
    validation_hit_rate: float
    validation_expectancy: float
    event_count: int
    single_asset_or_cross_sectional: str
    is_inverted_superior: bool
    is_near_duplicate: bool


@dataclass(frozen=True)
class MetaLabelingEligibility:
    eligible: bool
    rejection_reason: str = ""


_MIN_EVENTS = {"single_asset": 200, "cross_sectional": 500}


def check_eligibility(stats: PrimarySignalStats) -> MetaLabelingEligibility:
    if stats.validation_net_sharpe <= 0:
        return MetaLabelingEligibility(False, "validation_net_sharpe <= 0")
    if stats.validation_hit_rate <= 0.5 and stats.validation_expectancy <= 0:
        return MetaLabelingEligibility(
            False, "neither validation hit rate nor expectancy is positive"
        )
    threshold = _MIN_EVENTS.get(stats.single_asset_or_cross_sectional, 200)
    if stats.event_count < threshold:
        return MetaLabelingEligibility(
            False, f"event_count {stats.event_count} < threshold {threshold}"
        )
    if stats.is_inverted_superior:
        return MetaLabelingEligibility(
            False, "inverted signal is superior — sign bug suspect"
        )
    if stats.is_near_duplicate:
        return MetaLabelingEligibility(False, "near-duplicate of a stronger primary")
    return MetaLabelingEligibility(True, "")
