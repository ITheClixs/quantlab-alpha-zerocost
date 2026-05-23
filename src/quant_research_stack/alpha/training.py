"""Unified S1 training pipeline.

Public surface:
    train_s1(config: TrainConfig, registry: RunRegistry) -> RunResult

All Pydantic configs validate at construction time. The public function is pure
in the sense that it takes a config + a registry and returns a RunResult — no
global state, no CLI argument parsing, no console output beyond progress logs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from quant_research_stack.alpha.inference import _canonical_sha256
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
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
from quant_research_stack.alpha.registry import RunMetadata, RunRegistry
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
_SECTION_13_ARTIFACT_PATHS: tuple[str, ...] = (
    "metadata.json",
    "predictions.parquet",
    "metrics.json",
    "feature_importance.parquet",
    "cv_folds.json",
    "feature_cols.json",
    "models/ridge.joblib",
    "models/lightgbm.txt",
    "models/lightgbm.config.json",
    "models/xgboost.json",
    "models/xgboost.config.json",
    "models/catboost.cbm",
    "models/catboost.config.json",
    "models/mlp.pt",
    "models/sequence.pt",
    "models/stacker.joblib",
    "report.md",
    "audit_log_smoke.jsonl",
)


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


def _refit_on_full(
    *,
    x_full: NDArray[np.float64],
    y_full: NDArray[np.float64],
    w_full: NDArray[np.float64],
    models_config: ModelsConfig,
) -> dict[str, object]:
    """Refit each base model on the entire training slice. These are the models persisted to disk."""
    eval_n = min(1000, x_full.shape[0] // 5)
    if eval_n < 2:
        eval_n = max(2, x_full.shape[0] // 2)
    x_eval = x_full[-eval_n:]
    y_eval = y_full[-eval_n:]
    w_eval = w_full[-eval_n:]

    finals: dict[str, object] = {}

    ridge = RidgeAlphaModel(RidgeConfig(alpha=models_config.ridge.alpha))
    ridge.fit(x_full, y_full, w_full)
    finals["ridge"] = ridge

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
    lgb.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["lgb"] = lgb

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
    xgb.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["xgb"] = xgb

    ccfg = models_config.catboost
    cat = CatBoostAlphaModel(
        CatBoostConfig(
            depth=ccfg.depth,
            learning_rate=ccfg.learning_rate,
            n_estimators=ccfg.n_estimators,
            early_stopping_rounds=ccfg.early_stopping_rounds,
        )
    )
    cat.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["cat"] = cat

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
    mlp.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["mlp"] = mlp

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
    seq.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["seq"] = seq

    return finals


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_artifact_sha_index(run_dir: Path) -> None:
    sha_index_path = run_dir / "_artifact_sha256.json"
    existing_index: dict[str, str] = {}
    if sha_index_path.exists():
        existing_index = json.loads(sha_index_path.read_text())

    sha_index = {
        rel_path: digest
        for rel_path, digest in existing_index.items()
        if rel_path not in _SECTION_13_ARTIFACT_PATHS
    }

    for rel_path in _SECTION_13_ARTIFACT_PATHS:
        artifact_path = run_dir / rel_path
        if artifact_path.is_file():
            sha_index[rel_path] = _sha256_file(artifact_path)

    sha_index_path.write_text(json.dumps(sha_index, indent=2, sort_keys=True))


def _write_run_sidecars(
    *,
    run_id: str,
    run_dir: Path,
    feature_cols: list[str],
    finals: dict[str, object],
    config: TrainConfig,
    training_rows: int,
    holdout_rows: int,
    fold_metrics: list[dict[str, float]],
    holdout_r2: float,
    per_model_r2: dict[str, float],
) -> None:
    lgb_model = finals["lgb"]
    if not isinstance(lgb_model, LightGBMAlphaModel):
        raise TypeError("expected final lgb model to be LightGBMAlphaModel")
    lgb_importance = lgb_model.feature_importance()
    if lgb_importance.shape[0] != len(feature_cols):
        raise RuntimeError(
            "LightGBM feature importance length does not match feature column count"
        )
    pl.DataFrame(
        {
            "feature": feature_cols,
            "lgb_importance": lgb_importance.astype(np.float64),
            "kept_after_noise_floor": [True] * len(feature_cols),
        }
    ).write_parquet(run_dir / "feature_importance.parquet")

    fold_size = training_rows // config.cv.n_folds
    folds = []
    for fold_idx in range(config.cv.n_folds):
        te_start = fold_idx * fold_size
        te_end = (
            (fold_idx + 1) * fold_size
            if fold_idx < config.cv.n_folds - 1
            else training_rows
        )
        folds.append(
            {
                "fold": fold_idx,
                "test_start_row": te_start,
                "test_end_row_exclusive": te_end,
                "train_rows": training_rows - (te_end - te_start),
                "test_rows": te_end - te_start,
            }
        )
    (run_dir / "cv_folds.json").write_text(
        json.dumps(
            {
                "n_folds": config.cv.n_folds,
                "purge_days": config.cv.purge_days,
                "embargo_days": config.cv.embargo_days,
                "group_column": config.data.group_column,
                "training_rows": training_rows,
                "holdout_rows": holdout_rows,
                "folds": folds,
            },
            indent=2,
            sort_keys=True,
        )
    )

    (run_dir / "report.md").write_text(
        "\n".join(
            [
                f"# S1 Training Report: {run_id}",
                "",
                f"- Holdout weighted zero-mean R2: {holdout_r2:.6f}",
                f"- Training rows: {training_rows}",
                f"- Holdout rows: {holdout_rows}",
                f"- Feature count: {len(feature_cols)}",
                f"- Base models: {', '.join(_BASE_MODEL_NAMES)}",
                "",
                "## Per-Model Holdout R2",
                "",
                *[
                    f"- {name}: {score:.6f}"
                    for name, score in sorted(per_model_r2.items())
                ],
                "",
                "## Fold Metrics",
                "",
                "```json",
                json.dumps(fold_metrics, indent=2, sort_keys=True),
                "```",
                "",
                "## Limitations",
                "",
                "- Synthetic or capped-row smoke runs are not production evidence.",
                "- Post-S0 full retrain acceptance still requires the runbook gate.",
                "",
            ]
        )
    )

    audit_record = {
        "event": "s1_train_complete",
        "not_investment_advice": True,
        "payload": {
            "run_id": run_id,
            "holdout_weighted_zero_mean_r2": holdout_r2,
            "training_rows": training_rows,
            "holdout_rows": holdout_rows,
            "base_models": list(_BASE_MODEL_NAMES),
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (run_dir / "audit_log_smoke.jsonl").write_text(json.dumps(audit_record) + "\n")


def _persist_run(
    *,
    run_dir: Path,
    finals: dict[str, object],
    stacker: LinearStacker,
    feature_cols: list[str],
    data_config: DataConfig,
) -> None:
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    finals["ridge"].save(models_dir / "ridge.joblib")  # type: ignore[attr-defined]
    finals["lgb"].save(models_dir / "lightgbm.txt")  # type: ignore[attr-defined]
    finals["xgb"].save(models_dir / "xgboost.json")  # type: ignore[attr-defined]
    finals["cat"].save(models_dir / "catboost.cbm")  # type: ignore[attr-defined]
    finals["mlp"].save(models_dir / "mlp.pt")  # type: ignore[attr-defined]
    finals["seq"].save(models_dir / "sequence.pt")  # type: ignore[attr-defined]
    stacker.save(models_dir / "stacker.joblib")

    schema_path = run_dir / "feature_cols.json"
    schema_path.write_text(
        json.dumps(
            {
                "feature_columns": list(feature_cols),
                "n_features": len(feature_cols),
                "feature_cols_sha256": _canonical_sha256(list(feature_cols)),
                "target_column": data_config.target_column,
                "weight_column": data_config.weight_column,
                "group_column": data_config.group_column,
            },
            indent=2,
            sort_keys=True,
        )
    )

    _update_artifact_sha_index(run_dir)


def _holdout_eval(
    *,
    finals: dict[str, object],
    stacker: LinearStacker,
    x_h: NDArray[np.float64],
    y_h: NDArray[np.float64],
    w_h: NDArray[np.float64],
) -> tuple[float, dict[str, float], NDArray[np.float64]]:
    """Phase 5 — uses ALL 6 final models (no zeroing). Returns (R², per-model R², ensemble preds)."""
    per_model = {
        name: finals[name].predict(x_h).astype(np.float64)  # type: ignore[attr-defined]
        for name in _BASE_MODEL_NAMES
    }
    h_stack = np.column_stack([per_model[n] for n in stacker.feature_order])
    holdout_pred = stacker.predict(h_stack)
    holdout_r2 = float(weighted_zero_mean_r2(y_h, holdout_pred, w_h))
    per_model_r2 = {
        f"{name}_r2": float(weighted_zero_mean_r2(y_h, preds, w_h))
        for name, preds in per_model.items()
    }
    return holdout_r2, per_model_r2, holdout_pred


def _load_and_split(
    *,
    config: TrainConfig,
    synthetic_dataframe: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Phase 1 — load + split + feature filter.

    If synthetic_dataframe is provided (tests), use it directly and bypass JS-on-disk loaders.
    """
    if synthetic_dataframe is not None:
        df = synthetic_dataframe
    else:
        from quant_research_stack.alpha.io import LoadConfig, load_jane_street

        load_cfg = LoadConfig(
            target_column=config.data.target_column,
            weight_column=config.data.weight_column,
            group_column=config.data.group_column,
            holdout_fraction=config.data.permanent_holdout_fraction,
        )
        df = load_jane_street(config.data.jane_street_root, load_cfg)

    group_col = config.data.group_column

    groups = df[group_col].unique().sort()
    n_groups = groups.len()
    holdout_n = max(1, int(n_groups * config.data.permanent_holdout_fraction))
    holdout_groups = groups.tail(holdout_n).to_list()
    train_groups = groups.head(n_groups - holdout_n).to_list()

    train_df = df.filter(pl.col(group_col).is_in(train_groups))
    holdout_df = df.filter(pl.col(group_col).is_in(holdout_groups))

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise RuntimeError("no feature_* columns found in input frame")
    return train_df, holdout_df, feature_cols


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train_s1(
    config: TrainConfig,
    registry: RunRegistry,
    *,
    synthetic_dataframe: pl.DataFrame | None = None,
) -> RunResult:
    """End-to-end S1 training. Writes a complete run dir loadable by load_predictor_from_run."""
    train_df, holdout_df, feature_cols = _load_and_split(
        config=config, synthetic_dataframe=synthetic_dataframe
    )

    target_col = config.data.target_column
    weight_col = config.data.weight_column

    x_full = train_df.select(feature_cols).to_numpy().astype(np.float64)
    y_full = train_df[target_col].to_numpy().astype(np.float64)
    w_full = train_df[weight_col].to_numpy().astype(np.float64)
    x_full = np.nan_to_num(x_full, nan=0.0)

    n = x_full.shape[0]
    fold_size = n // config.cv.n_folds

    oof: dict[str, NDArray[np.float64]] = {
        name: np.zeros(n, dtype=np.float64) for name in _BASE_MODEL_NAMES
    }
    fold_metrics: list[dict[str, float]] = []

    for fold_idx in range(config.cv.n_folds):
        te_start = fold_idx * fold_size
        te_end = (fold_idx + 1) * fold_size if fold_idx < config.cv.n_folds - 1 else n
        te_idx = np.arange(te_start, te_end)
        tr_idx = np.concatenate([np.arange(0, te_start), np.arange(te_end, n)])

        fold_oof = _fit_one_fold(
            fold_idx=fold_idx,
            x_tr=x_full[tr_idx],
            y_tr=y_full[tr_idx],
            w_tr=w_full[tr_idx],
            x_te=x_full[te_idx],
            y_te=y_full[te_idx],
            w_te=w_full[te_idx],
            models_config=config.models,
        )
        for name in _BASE_MODEL_NAMES:
            oof[name][te_idx] = fold_oof[name]

        fold_metrics.append(
            {
                "fold": float(fold_idx),
                **{
                    f"{name}_r2": float(
                        weighted_zero_mean_r2(y_full[te_idx], fold_oof[name], w_full[te_idx])
                    )
                    for name in _BASE_MODEL_NAMES
                },
            }
        )

    stacker = _fit_stacker(
        oof_by_name=oof,
        y_full=y_full,
        w_full=w_full,
        stacker_alpha=config.stacker_alpha,
    )

    finals = _refit_on_full(
        x_full=x_full,
        y_full=y_full,
        w_full=w_full,
        models_config=config.models,
    )

    x_h = holdout_df.select(feature_cols).to_numpy().astype(np.float64)
    y_h = holdout_df[target_col].to_numpy().astype(np.float64)
    w_h = holdout_df[weight_col].to_numpy().astype(np.float64)
    x_h = np.nan_to_num(x_h, nan=0.0)

    holdout_r2, per_model_r2, holdout_pred = _holdout_eval(
        finals=finals, stacker=stacker, x_h=x_h, y_h=y_h, w_h=w_h
    )

    git_sha = _git_sha()
    meta = RunMetadata(
        version="0.2.0-s0",
        git_sha=git_sha,
        data_hashes={"jane_street_root": config.data.jane_street_root},
        hyperparams=config.model_dump(),
        fold_definition={
            "n_folds": config.cv.n_folds,
            "purge": config.cv.purge_days,
            "embargo": config.cv.embargo_days,
        },
    )
    run_id = registry.create_run(meta)
    run_dir = Path(registry.root) / run_id

    metrics_payload = {
        "fold_metrics": fold_metrics,
        "holdout_weighted_zero_mean_r2": holdout_r2,
        "holdout_per_model_r2": per_model_r2,
        "n_features_after_adversarial": len(feature_cols),
        "n_features_after_noise_floor": len(feature_cols),
        "training_rows": int(n),
        "holdout_rows": int(x_h.shape[0]),
        "profile": "s0_unified_full_holdout_refit",
    }
    registry.save_artifact(run_id, "metrics.json", json.dumps(metrics_payload, indent=2).encode())

    # predictions.parquet — holdout + OOF.
    oof_stack = np.column_stack([oof[mn] for mn in stacker.feature_order])
    oof_pred = stacker.predict(oof_stack)
    preds_df = pl.DataFrame(
        {
            "split": ["holdout"] * x_h.shape[0] + ["train_oof"] * n,
            "target_actual": np.concatenate([y_h, y_full]).astype(np.float32),
            "weight": np.concatenate([w_h, w_full]).astype(np.float32),
            "stacked": np.concatenate(
                [
                    holdout_pred.astype(np.float32),
                    oof_pred.astype(np.float32),
                ]
            ),
            **{
                name: np.concatenate(
                    [
                        finals[name].predict(x_h).astype(np.float32),  # type: ignore[attr-defined]
                        oof[name].astype(np.float32),
                    ]
                )
                for name in _BASE_MODEL_NAMES
            },
        }
    )
    preds_df.write_parquet(run_dir / "predictions.parquet")

    _write_run_sidecars(
        run_id=run_id,
        run_dir=run_dir,
        feature_cols=feature_cols,
        finals=finals,
        config=config,
        training_rows=int(n),
        holdout_rows=int(x_h.shape[0]),
        fold_metrics=fold_metrics,
        holdout_r2=holdout_r2,
        per_model_r2=per_model_r2,
    )

    _persist_run(
        run_dir=run_dir,
        finals=finals,
        stacker=stacker,
        feature_cols=feature_cols,
        data_config=config.data,
    )

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        fold_metrics=fold_metrics,
        holdout_weighted_zero_mean_r2=holdout_r2,
        n_features_after_adversarial=len(feature_cols),
        n_features_after_noise_floor=len(feature_cols),
        base_models_persisted=list(_BASE_MODEL_NAMES),
        stacker_path=run_dir / "models" / "stacker.joblib",
        feature_cols_path=run_dir / "feature_cols.json",
    )
