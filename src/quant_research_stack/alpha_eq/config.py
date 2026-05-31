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
    rolling_windows: tuple[int, ...] = (5, 20, 60)
    momentum_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60, 120, 252)
    vix_proxy_fallback: str = "cross_sectional_vol_20"


class RollingDiagnosticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    train_years: int = 10
    valid_years: int = 2


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout: str = "expanding_window"
    n_folds: int = 5
    label_horizon_days: int = 1
    purge_safety_buffer: int = 2
    rolling_diagnostic: RollingDiagnosticConfig = Field(default_factory=RollingDiagnosticConfig)

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


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fast_v1: tuple[str, ...] = ("ridge", "lightgbm", "xgboost")
    full_v1: tuple[str, ...] = ("ridge", "lightgbm", "xgboost", "catboost", "mlp", "sequence")


class OptunaTrialsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lightgbm: int = 50
    xgboost: int = 30
    catboost: int = 30
    mlp: int = 20
    sequence: int = 20
    stacker: int = 30


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = True
    trials: OptunaTrialsConfig = Field(default_factory=OptunaTrialsConfig)


class AlphaEqConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: TrainingMode
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    stacker: StackerConfig = Field(default_factory=StackerConfig)
    reproducibility: ReproConfig = Field(default_factory=ReproConfig)

    def active_models(self) -> tuple[str, ...]:
        if self.mode == TrainingMode.FAST_V1:
            return self.models.fast_v1
        return self.models.full_v1

    @model_validator(mode="after")
    def _validate(self) -> AlphaEqConfig:
        if self.cv.n_folds < 3:
            raise ValueError("n_folds must be ≥ 3")
        return self
