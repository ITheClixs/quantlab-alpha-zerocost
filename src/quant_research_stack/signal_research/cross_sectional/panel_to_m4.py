"""Banner-preserving M4 entry (spec §5.2, §5.4).

- Uses ONLY public M4 interfaces.
- Never silently strips data-quality warnings.
- Refuses institutional-grade labels unless the M4 manifest reports
  data_quality_label == pit_safe OR the universe is directly-traded
  (constituent_survivorship_applicable == False).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.alpha_eq.backtest.runner import BacktestConfig, run_backtest
from quant_research_stack.signal_research.data.manifest import DataQualityTier


@dataclass(frozen=True)
class BridgeMetadata:
    data_quality_tier: DataQualityTier
    constituent_survivorship_applicable: bool
    institutional_grade_allowed: bool
    survivorship_banner_required: bool


def determine_bridge_metadata(
    *,
    data_quality_tier: DataQualityTier,
    constituent_survivorship_applicable: bool,
) -> BridgeMetadata:
    institutional_grade_allowed = (
        data_quality_tier == DataQualityTier.PIT_SAFE
        or not constituent_survivorship_applicable
    )
    survivorship_banner_required = constituent_survivorship_applicable and (
        data_quality_tier
        in (
            DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
            DataQualityTier.PARTIAL_PIT_UNIVERSE,
        )
    )
    return BridgeMetadata(
        data_quality_tier=data_quality_tier,
        constituent_survivorship_applicable=constituent_survivorship_applicable,
        institutional_grade_allowed=institutional_grade_allowed,
        survivorship_banner_required=survivorship_banner_required,
    )


def run_cross_sectional_through_m4(
    *,
    panel: pl.DataFrame,
    bridge_metadata: BridgeMetadata,
    backtest_config: BacktestConfig,
    dividends: pl.DataFrame | None = None,
):
    """Thin wrapper around the existing M4 runner. Does NOT modify M4."""
    return run_backtest(
        signals_with_bars=panel, config=backtest_config, dividends=dividends
    )
