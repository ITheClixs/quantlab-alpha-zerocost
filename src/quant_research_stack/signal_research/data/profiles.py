"""Profile + universe configuration loader (spec §2.3, §6.6)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_research_stack.signal_research.data.manifest import DataQualityTier


class UniverseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    name: str
    description: str = ""
    data_quality_label: DataQualityTier
    constituent_survivorship_applicable: bool
    tickers: list[str] = Field(default_factory=list)
    tickers_source: str | None = None
    diagnostic_only: list[str] = Field(default_factory=list)


class CostModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    maker_bps: float | None = None
    taker_bps: float | None = None
    funding_payments: bool = False


class ProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    profile: str
    asset_class: str
    universes: list[UniverseConfig]
    benchmarks: list[str] = Field(default_factory=list)
    context_features: list[str] = Field(default_factory=list)
    cost_model: CostModelConfig


def load_profile(yaml_path: Path) -> ProfileConfig:
    payload = yaml.safe_load(Path(yaml_path).read_text())
    return ProfileConfig.model_validate(payload)


def list_profiles(root: Path) -> list[str]:
    return sorted(p.stem for p in Path(root).glob("*.yaml"))
