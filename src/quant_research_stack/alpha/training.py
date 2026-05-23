"""Unified S1 training pipeline.

Public surface:
    train_s1(config: TrainConfig, registry: RunRegistry) -> RunResult

All Pydantic configs validate at construction time. The public function is pure
in the sense that it takes a config + a registry and returns a RunResult — no
global state, no CLI argument parsing, no console output beyond progress logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Configs (Pydantic v2)
# -----------------------------------------------------------------------------


class DataConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    jane_street_root: str
    synthetic_root: str | None = None
    preprocessed_alt_root: str | None = None
    group_column: str = "date_id"
    target_column: str = "responder_6"
    weight_column: str = "weight"
    max_rows: int = Field(gt=0)
    permanent_holdout_fraction: float = Field(ge=0.05, le=0.5)


class CVConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    n_folds: int = Field(ge=2, le=20)
    purge_days: int = Field(ge=0)
    embargo_days: int = Field(ge=0)
    random_seed: int = 42


class FeatureConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    lag_windows: list[int] = Field(default_factory=lambda: [1, 5])
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 20])
    cross_sectional_ranks: bool = False
    include_noise_feature: bool = True
    noise_seed: int = 42


class RidgeModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    alpha: float = Field(gt=0.0)


class LightGBMModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    num_leaves: int = Field(gt=1)
    max_depth: int
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)
    feature_fraction: float = Field(gt=0.0, le=1.0)
    bagging_fraction: float = Field(gt=0.0, le=1.0)


class XGBoostModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)
    tree_method: Literal["hist", "approx", "exact"] = "hist"


class CatBoostModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    depth: int = Field(gt=0, le=16)
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)


class MLPModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    hidden_dims: list[int]
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    mixed_precision: bool = False


class SequenceModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kernel_sizes: list[int]
    n_filters: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    random_state: int = 0


class ModelsConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    ridge: RidgeModelConfig
    lightgbm: LightGBMModelConfig
    xgboost: XGBoostModelConfig
    catboost: CatBoostModelConfig
    mlp: MLPModelConfig
    sequence: SequenceModelConfig


class TrainConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    data: DataConfig
    cv: CVConfig
    features: FeatureConfig
    models: ModelsConfig
    stacker_alpha: float = Field(gt=0.0)
    streaming: bool = False
    max_rows_streaming: int = Field(gt=0, default=5_000_000)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        return cls.model_validate(payload)


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    fold_metrics: list[dict[str, float]]
    holdout_weighted_zero_mean_r2: float
    n_features_after_adversarial: int
    n_features_after_noise_floor: int
    base_models_persisted: list[str]
    stacker_path: Path
    feature_cols_path: Path


# Placeholder; real implementation lands in Task 13 + 14.
def train_s1(config: TrainConfig, registry: object) -> RunResult:
    raise NotImplementedError("train_s1 is implemented in Task 13 + 14")
