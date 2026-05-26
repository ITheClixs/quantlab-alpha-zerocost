"""Meta-labeling (spec §4.2) — survivor-only pre-filter."""

from __future__ import annotations

from quant_research_stack.signal_research.methodology.meta_labeling import (
    PrimarySignalStats,
    check_eligibility,
)


def _stats(**overrides: object) -> PrimarySignalStats:
    defaults = dict(
        validation_net_sharpe=0.8,
        validation_hit_rate=0.55,
        validation_expectancy=0.001,
        event_count=300,
        single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    defaults.update(overrides)
    return PrimarySignalStats(**defaults)  # type: ignore[arg-type]


def test_eligible_when_all_filters_pass() -> None:
    assert check_eligibility(_stats()).eligible is True


def test_rejects_when_event_count_too_low_single_asset() -> None:
    elig = check_eligibility(_stats(event_count=150))
    assert elig.eligible is False
    assert "event_count" in elig.rejection_reason


def test_rejects_when_event_count_too_low_cross_sectional() -> None:
    elig = check_eligibility(_stats(
        event_count=400, single_asset_or_cross_sectional="cross_sectional"
    ))
    assert elig.eligible is False


def test_rejects_when_negative_sharpe() -> None:
    elig = check_eligibility(_stats(validation_net_sharpe=-0.1))
    assert elig.eligible is False
