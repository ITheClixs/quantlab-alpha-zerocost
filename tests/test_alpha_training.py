from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from quant_research_stack.alpha.inference import (
    _EXPECTED_BASE_MODEL_FILES,
    load_predictor_from_run,
)
from quant_research_stack.alpha.registry import RunRegistry
from quant_research_stack.alpha.training import (
    CatBoostModelConfig,
    CVConfig,
    DataConfig,
    FeatureConfig,
    LightGBMModelConfig,
    MLPModelConfig,
    ModelsConfig,
    RidgeModelConfig,
    SequenceModelConfig,
    TrainConfig,
    XGBoostModelConfig,
    _fit_one_fold,
    train_s1,
)


def _minimal_models_config() -> ModelsConfig:
    return ModelsConfig(
        ridge=RidgeModelConfig(alpha=1.0),
        lightgbm=LightGBMModelConfig(
            num_leaves=7,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=20,
            early_stopping_rounds=5,
            feature_fraction=1.0,
            bagging_fraction=1.0,
        ),
        xgboost=XGBoostModelConfig(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=20,
            early_stopping_rounds=5,
            tree_method="hist",
        ),
        catboost=CatBoostModelConfig(
            depth=3,
            learning_rate=0.1,
            n_estimators=20,
            early_stopping_rounds=5,
        ),
        mlp=MLPModelConfig(
            hidden_dims=[8],
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=64,
            max_epochs=2,
            patience=2,
            mixed_precision=False,
        ),
        sequence=SequenceModelConfig(
            kernel_sizes=[3],
            n_filters=8,
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=64,
            max_epochs=2,
            patience=2,
            random_state=0,
        ),
    )


def test_fit_one_fold_returns_six_base_predictions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 8))
    y = x[:, 0] + 0.1 * rng.standard_normal(500)
    w = np.ones(500)
    tr_idx = np.arange(0, 400)
    te_idx = np.arange(400, 500)
    cfg = _minimal_models_config()

    fold_oof = _fit_one_fold(
        fold_idx=0,
        x_tr=x[tr_idx],
        y_tr=y[tr_idx],
        w_tr=w[tr_idx],
        x_te=x[te_idx],
        y_te=y[te_idx],
        w_te=w[te_idx],
        models_config=cfg,
    )

    assert set(fold_oof.keys()) == {"ridge", "lgb", "xgb", "cat", "mlp", "seq"}
    for name, preds in fold_oof.items():
        assert preds.shape == (te_idx.size,), f"{name} returned shape {preds.shape}"
        assert preds.dtype == np.float64


def test_train_config_from_yaml_smoke():
    cfg_dict = {
        "data": {
            "jane_street_root": "data/raw/huggingface/TnnnT0326__Jane_Street_Competition",
            "synthetic_root": "data/raw/kaggle/datasets/christoffer__synthetic-jane-street-dataset",
            "preprocessed_alt_root": "data/raw/kaggle/datasets/saurabhshahane__jane-street-preprocessed-train",
            "group_column": "date_id",
            "target_column": "responder_6",
            "weight_column": "weight",
            "max_rows": 2_000_000,
            "permanent_holdout_fraction": 0.2,
        },
        "cv": {
            "n_folds": 3,
            "purge_days": 5,
            "embargo_days": 5,
            "random_seed": 42,
        },
        "features": {
            "lag_windows": [1, 5],
            "rolling_windows": [5, 20],
            "cross_sectional_ranks": False,
            "include_noise_feature": True,
            "noise_seed": 42,
        },
        "models": {
            "ridge": {"alpha": 1.0},
            "lightgbm": {
                "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05,
                "n_estimators": 200, "early_stopping_rounds": 20,
                "feature_fraction": 0.9, "bagging_fraction": 0.9,
            },
            "xgboost": {
                "max_depth": 6, "learning_rate": 0.05, "n_estimators": 200,
                "early_stopping_rounds": 20, "tree_method": "hist",
            },
            "catboost": {
                "depth": 6, "learning_rate": 0.05, "n_estimators": 200,
                "early_stopping_rounds": 20,
            },
            "mlp": {
                "hidden_dims": [64, 32], "dropout": 0.2, "learning_rate": 1e-3,
                "batch_size": 1024, "max_epochs": 30, "patience": 3,
                "mixed_precision": False,
            },
            "sequence": {
                "kernel_sizes": [3, 5], "n_filters": 16, "dropout": 0.1,
                "learning_rate": 1e-3, "batch_size": 1024, "max_epochs": 30,
                "patience": 3, "random_state": 0,
            },
        },
        "stacker_alpha": 1e-3,
        "streaming": False,
        "max_rows_streaming": 5_000_000,
    }
    cfg = TrainConfig.from_dict(cfg_dict)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.cv, CVConfig)
    assert isinstance(cfg.features, FeatureConfig)
    assert isinstance(cfg.models, ModelsConfig)
    assert cfg.cv.n_folds == 3
    assert cfg.models.ridge.alpha == 1.0
    assert cfg.streaming is False


def test_checked_in_alpha_configs_load():
    for config_path in [
        Path("configs/alpha.yaml"),
        Path("configs/alpha_fast.yaml"),
        Path("configs/alpha_5m.yaml"),
    ]:
        payload = yaml.safe_load(config_path.read_text())
        assert isinstance(payload, dict), f"{config_path} must contain a YAML mapping"
        TrainConfig.from_dict(payload)


def test_train_config_rejects_bad_alpha():
    import pytest
    from pydantic import ValidationError

    from quant_research_stack.alpha.training import RidgeModelConfig

    with pytest.raises(ValidationError):
        RidgeModelConfig(alpha=-1.0)  # alpha must be > 0


def _train_config_for_synthetic(tmp_path: Path) -> dict:
    return {
        "data": {
            "jane_street_root": str(tmp_path),  # ignored when synthetic_dataframe is passed
            "group_column": "date_id",
            "target_column": "responder_6",
            "weight_column": "weight",
            "max_rows": 10_000,
            "permanent_holdout_fraction": 0.2,
        },
        "cv": {"n_folds": 2, "purge_days": 0, "embargo_days": 0, "random_seed": 0},
        "features": {
            "lag_windows": [],
            "rolling_windows": [],
            "cross_sectional_ranks": False,
            "include_noise_feature": False,
            "noise_seed": 0,
        },
        "models": {
            "ridge": {"alpha": 1.0},
            "lightgbm": {
                "num_leaves": 7,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 20,
                "early_stopping_rounds": 5,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
            },
            "xgboost": {
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 20,
                "early_stopping_rounds": 5,
                "tree_method": "hist",
            },
            "catboost": {
                "depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 20,
                "early_stopping_rounds": 5,
            },
            "mlp": {
                "hidden_dims": [8],
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "max_epochs": 2,
                "patience": 2,
                "mixed_precision": False,
            },
            "sequence": {
                "kernel_sizes": [3],
                "n_filters": 8,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "max_epochs": 2,
                "patience": 2,
                "random_state": 0,
            },
        },
        "stacker_alpha": 1e-3,
        "streaming": False,
        "max_rows_streaming": 10_000,
    }


def test_train_s1_end_to_end_on_synthetic(synthetic_js, tmp_path):
    cfg = TrainConfig.from_dict(_train_config_for_synthetic(tmp_path))
    registry = RunRegistry(root=tmp_path / "experiments")
    result = train_s1(cfg, registry, synthetic_dataframe=synthetic_js)

    # Run dir exists with all expected files.
    assert result.run_dir.exists()
    for fname in _EXPECTED_BASE_MODEL_FILES.values():
        assert (result.run_dir / "models" / fname).exists(), f"missing {fname}"
    assert (result.run_dir / "models" / "stacker.joblib").exists()
    assert (result.run_dir / "feature_cols.json").exists()
    assert (result.run_dir / "_artifact_sha256.json").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "predictions.parquet").exists()
    assert (result.run_dir / "feature_importance.parquet").exists()
    assert (result.run_dir / "cv_folds.json").exists()
    assert (result.run_dir / "report.md").exists()
    assert (result.run_dir / "audit_log_smoke.jsonl").exists()
    artifact_index = json.loads((result.run_dir / "_artifact_sha256.json").read_text())
    section_13_artifacts = [
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
    ]
    assert set(section_13_artifacts).issubset(artifact_index)
    for rel_path in section_13_artifacts:
        assert artifact_index[rel_path] == hashlib.sha256(
            (result.run_dir / rel_path).read_bytes()
        ).hexdigest()

    # Loader returns a working predictor.
    predictor = load_predictor_from_run(result.run_dir)
    assert sorted(predictor.expected_feature_columns) == sorted(
        [c for c in synthetic_js.columns if c.startswith("feature_")]
    )

    # And R² is a real number (not NaN/inf), even if synthetic data isn't calibrated to a gate.
    assert np.isfinite(result.holdout_weighted_zero_mean_r2)
