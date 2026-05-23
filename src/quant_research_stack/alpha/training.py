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

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from quant_research_stack.alpha.models.catboost_model import (
    CatBoostAlphaModel,
    CatBoostConfig,
)
from quant_research_stack.alpha.models.lightgbm_model import (
    LightGBMAlphaModel,
    LightGBMConfig,
)
from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig
from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig
from quant_research_stack.alpha.models.xgboost_model import (
    XGBoostAlphaModel,
    XGBoostConfig,
)
from quant_research_stack.alpha.stacking import LinearStacker

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


_BASE_MODEL_NAMES: tuple[str, ...] = ("ridge", "lgb", "xgb", "cat", "mlp", "seq")


def _fit_one_fold(
    *,
    fold_idx: int,
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    w_tr: NDArray[np.float64],
    x_te: NDArray[np.float64],
    y_te: NDArray[np.float64],
    w_te: NDArray[np.float64],
    models_config: ModelsConfig,
) -> dict[str, NDArray[np.float64]]:
    """Fit all 6 base models on (x_tr, y_tr, w_tr); predict each on (x_te) and return OOF.

    Returns a dict keyed by base-model name with each value an (n_te,) float64 array.
    Per-fold models are NOT persisted (consistent with pre-S0 behaviour); only the
    refit-on-full models in phase 4 land on disk.
    """
    out: dict[str, NDArray[np.float64]] = {}

    ridge = RidgeAlphaModel(RidgeConfig(alpha=models_config.ridge.alpha))
    ridge.fit(x_tr, y_tr, w_tr)
    out["ridge"] = ridge.predict(x_te).astype(np.float64)

    lcfg = models_config.lightgbm
    lgb = LightGBMAlphaModel(
        LightGBMConfig(
            num_leaves=lcfg.num_leaves,
            max_depth=lcfg.max_depth,
            learning_rate=lcfg.learning_rate,
            n_estimators=lcfg.n_estimators,
            early_stopping_rounds=lcfg.early_stopping_rounds,
            feature_fraction=lcfg.feature_fraction,
            bagging_fraction=lcfg.bagging_fraction,
        )
    )
    lgb.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["lgb"] = lgb.predict(x_te).astype(np.float64)

    xcfg = models_config.xgboost
    xgb = XGBoostAlphaModel(
        XGBoostConfig(
            max_depth=xcfg.max_depth,
            learning_rate=xcfg.learning_rate,
            n_estimators=xcfg.n_estimators,
            early_stopping_rounds=xcfg.early_stopping_rounds,
            tree_method=xcfg.tree_method,
        )
    )
    xgb.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["xgb"] = xgb.predict(x_te).astype(np.float64)

    ccfg = models_config.catboost
    cat = CatBoostAlphaModel(
        CatBoostConfig(
            depth=ccfg.depth,
            learning_rate=ccfg.learning_rate,
            n_estimators=ccfg.n_estimators,
            early_stopping_rounds=ccfg.early_stopping_rounds,
        )
    )
    cat.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["cat"] = cat.predict(x_te).astype(np.float64)

    mcfg = models_config.mlp
    mlp = MLPAlphaModel(
        MLPConfig(
            hidden_dims=list(mcfg.hidden_dims),
            dropout=mcfg.dropout,
            learning_rate=mcfg.learning_rate,
            batch_size=mcfg.batch_size,
            max_epochs=mcfg.max_epochs,
            patience=mcfg.patience,
            mixed_precision=mcfg.mixed_precision,
        )
    )
    mlp.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["mlp"] = mlp.predict(x_te).astype(np.float64)

    scfg = models_config.sequence
    seq = Conv1DAlphaModel(
        Conv1DConfig(
            kernel_sizes=list(scfg.kernel_sizes),
            n_filters=scfg.n_filters,
            dropout=scfg.dropout,
            learning_rate=scfg.learning_rate,
            batch_size=scfg.batch_size,
            max_epochs=scfg.max_epochs,
            patience=scfg.patience,
            random_state=scfg.random_state,
        )
    )
    seq.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["seq"] = seq.predict(x_te).astype(np.float64)

    return out


def _fit_stacker(
    *,
    oof_by_name: dict[str, NDArray[np.float64]],
    y_full: NDArray[np.float64],
    w_full: NDArray[np.float64],
    stacker_alpha: float,
) -> LinearStacker:
    """Stack 6 OOF columns in the canonical _BASE_MODEL_NAMES order and fit LinearStacker."""
    stack_x = np.column_stack([oof_by_name[n] for n in _BASE_MODEL_NAMES])
    stacker = LinearStacker(alpha=stacker_alpha, feature_order=list(_BASE_MODEL_NAMES))
    stacker.fit(stack_x, y_full, w_full)
    return stacker


# Placeholder; real implementation lands in Task 13 + 14.
def train_s1(config: TrainConfig, registry: object) -> RunResult:
    raise NotImplementedError("train_s1 is implemented in Task 13 + 14")
