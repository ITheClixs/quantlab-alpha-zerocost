"""AlphaEqConfig validates basic invariants."""

from __future__ import annotations

import pytest

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode


def test_config_default_construction() -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    assert cfg.mode == TrainingMode.FAST_V1
    assert cfg.cv.n_folds == 5
    assert cfg.features.enable_meta_features is False
    assert cfg.data.permanent_holdout_fraction == 0.20


def test_config_rejects_invalid_holdout_fraction() -> None:
    with pytest.raises(ValueError):
        AlphaEqConfig(mode=TrainingMode.FAST_V1, data={"permanent_holdout_fraction": 0.5})


def test_mode_full_v1_models() -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FULL_V1)
    assert set(cfg.active_models()) == {"ridge", "lightgbm", "xgboost", "catboost", "mlp", "sequence"}
