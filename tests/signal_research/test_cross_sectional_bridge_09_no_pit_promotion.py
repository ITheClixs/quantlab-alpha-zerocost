"""Bridge contract #9: current-constituent universe cannot be promoted to pit_safe."""

from __future__ import annotations

from quant_research_stack.signal_research.cross_sectional.panel_to_m4 import (
    determine_bridge_metadata,
)
from quant_research_stack.signal_research.data.manifest import DataQualityTier


def test_no_pit_promotion_for_current_constituents() -> None:
    md = determine_bridge_metadata(
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
    )
    assert md.institutional_grade_allowed is False
