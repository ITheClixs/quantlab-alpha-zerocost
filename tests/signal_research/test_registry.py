"""Strategy registry — mandatory schema (spec §3.6)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.data.manifest import DataQualityTier
from quant_research_stack.signal_research.registry import (
    ModuleType,
    SingleAssetOrCrossSectional,
    StrategyRegistryEntry,
)


def test_module_type_enum_values() -> None:
    assert ModuleType.STANDALONE_STRATEGY.value == "standalone_strategy"
    assert ModuleType.FEATURE_GENERATOR.value == "feature_generator"
    assert ModuleType.WRAPPER.value == "wrapper"
    assert ModuleType.MODEL_FAMILY.value == "model_family"


def test_registry_entry_requires_all_mandatory_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StrategyRegistryEntry(strategy_id="x")  # type: ignore[call-arg]


def test_registry_entry_valid_full_construction() -> None:
    entry = StrategyRegistryEntry(
        strategy_id="AL.SP500.L60.K1.5",
        family="AVELLANEDA_LEE",
        module_type=ModuleType.STANDALONE_STRATEGY,
        paper_source="Avellaneda & Lee 2010",
        asset_class="equity",
        profile="sp500",
        single_asset_or_cross_sectional=SingleAssetOrCrossSectional.CROSS_SECTIONAL,
        required_data=["sp500_current_constituents", "long_history"],
        timestamp_assumptions="after_close_t",
        parameter_grid={"lookback": [60, 120, 252], "k": [1.0, 1.5, 2.0]},
        default_parameters={"lookback": 60, "k": 1.5},
        eligible_for_pbo=True,
        eligible_for_holdout=True,
        eligible_for_cross_sectional_bridge=True,
        data_quality_requirements=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        known_limitations=["current-only constituents", "rolling-PCA approximation"],
    )
    assert entry.strategy_id == "AL.SP500.L60.K1.5"
    assert entry.module_type == ModuleType.STANDALONE_STRATEGY
    assert "current-only" in entry.known_limitations[0]
