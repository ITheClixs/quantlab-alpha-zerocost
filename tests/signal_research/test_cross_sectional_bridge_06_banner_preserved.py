"""Bridge contract #6: data-quality banner preserved when not pit_safe."""

from __future__ import annotations

from quant_research_stack.signal_research.cross_sectional.panel_to_m4 import (
    determine_bridge_metadata,
)
from quant_research_stack.signal_research.data.manifest import DataQualityTier


def test_current_constituents_keep_survivorship_banner() -> None:
    md = determine_bridge_metadata(
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
    )
    assert md.survivorship_banner_required is True
    assert md.institutional_grade_allowed is False


def test_directly_traded_etf_allows_institutional_grade_even_if_label_imperfect() -> None:
    md = determine_bridge_metadata(
        data_quality_tier=DataQualityTier.PARTIAL_PIT_UNIVERSE,
        constituent_survivorship_applicable=False,
    )
    assert md.institutional_grade_allowed is True
