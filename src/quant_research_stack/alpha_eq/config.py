"""S1-EQ Pydantic v2 configuration (spec §4)."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingMode(enum.StrEnum):
    FAST_V1 = "fast_v1"
    FULL_V1 = "full_v1"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equity_root: str = "data/processed/equities"
    manifest_path: str = "data/processed/equities/_manifest.json"
    universe: str = "sp500"
    permanent_holdout_fraction: float = Field(default=0.20, gt=0.0, lt=0.4)
    min_holdout_trading_days: int = Field(default=756, ge=252)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_meta_features: bool = False
    noise_seed: int = 42


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout: str = "expanding_window"
    n_folds: int = 5
    label_horizon_days: int = 1
    purge_safety_buffer: int = 2

    @property
    def purge_days(self) -> int:
        return max(5, self.label_horizon_days + self.purge_safety_buffer)

    @property
    def embargo_days(self) -> int:
        return max(5, self.label_horizon_days)


class StackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = 1.0e-3
    prefer_non_negative: bool = True
    flag_large_negative_threshold: float = -0.25


class ReproConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numpy_seed: int = 42
    torch_seed: int = 42
    lightgbm_seed: int = 42
    xgboost_seed: int = 42
    catboost_seed: int = 42


_MODE_MODELS: dict[TrainingMode, tuple[str, ...]] = {
    TrainingMode.FAST_V1: ("ridge", "lightgbm", "xgboost"),
    TrainingMode.FULL_V1: ("ridge", "lightgbm", "xgboost", "catboost", "mlp", "sequence"),
}


class AlphaEqConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: TrainingMode
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    stacker: StackerConfig = Field(default_factory=StackerConfig)
    reproducibility: ReproConfig = Field(default_factory=ReproConfig)

    def active_models(self) -> tuple[str, ...]:
        return _MODE_MODELS[self.mode]

    @model_validator(mode="after")
    def _validate(self) -> AlphaEqConfig:
        if self.cv.n_folds < 3:
            raise ValueError("n_folds must be ≥ 3")
        return self
