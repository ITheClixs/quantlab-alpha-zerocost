"""Strategy registry — mandatory schema (spec §3.6).

All fields listed are mandatory unless explicitly marked optional in the
spec. The registry is consumed by the runner to enumerate trials and by
the PBO/DSR machinery to count effective strategies.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict

from quant_research_stack.signal_research.data.manifest import DataQualityTier


class ModuleType(enum.StrEnum):
    STANDALONE_STRATEGY = "standalone_strategy"
    FEATURE_GENERATOR = "feature_generator"
    WRAPPER = "wrapper"
    MODEL_FAMILY = "model_family"


class SingleAssetOrCrossSectional(enum.StrEnum):
    SINGLE_ASSET = "single_asset"
    CROSS_SECTIONAL = "cross_sectional"


class StrategyRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    family: str
    module_type: ModuleType
    paper_source: str
    asset_class: str
    profile: str
    single_asset_or_cross_sectional: SingleAssetOrCrossSectional
    required_data: list[str]
    timestamp_assumptions: str
    parameter_grid: dict[str, list]
    default_parameters: dict[str, object]
    eligible_for_pbo: bool
    eligible_for_holdout: bool
    eligible_for_cross_sectional_bridge: bool
    data_quality_requirements: DataQualityTier
    known_limitations: list[str]
