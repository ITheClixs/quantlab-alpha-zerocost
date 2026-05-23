from __future__ import annotations

from quant_research_stack.alpha.training import (
    CVConfig,
    DataConfig,
    FeatureConfig,
    ModelsConfig,
    TrainConfig,
)


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


def test_train_config_rejects_bad_alpha():
    import pytest
    from pydantic import ValidationError

    from quant_research_stack.alpha.training import RidgeModelConfig

    with pytest.raises(ValidationError):
        RidgeModelConfig(alpha=-1.0)  # alpha must be > 0
