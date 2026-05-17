from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.alpha.adversarial import (
    adversarial_drop_features,
    drop_below_noise_floor,
    train_holdout_classifier_auc,
)


def test_train_holdout_auc_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3))
    holdout = rng.normal(size=(200, 3))
    auc = train_holdout_classifier_auc(train, holdout)
    assert 0.45 <= auc <= 0.55


def test_train_holdout_auc_high_for_shifted_distributions() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, size=(200, 3))
    holdout = rng.normal(loc=5.0, size=(200, 3))
    auc = train_holdout_classifier_auc(train, holdout)
    assert auc > 0.9


def test_train_holdout_auc_samples_large_inputs() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, size=(20_000, 2))
    holdout = rng.normal(loc=4.0, size=(20_000, 2))
    auc = train_holdout_classifier_auc(train, holdout, max_rows_per_split=500)
    assert auc > 0.9


def test_adversarial_drop_features_removes_shifted_columns() -> None:
    rng = np.random.default_rng(0)
    train = pl.DataFrame({"good": rng.normal(size=200), "shifted": rng.normal(loc=0.0, size=200)})
    holdout = pl.DataFrame({"good": rng.normal(size=200), "shifted": rng.normal(loc=8.0, size=200)})
    kept = adversarial_drop_features(train, holdout, candidate_cols=["good", "shifted"], auc_threshold=0.6)
    assert "good" in kept
    assert "shifted" not in kept


def test_adversarial_drop_features_keeps_sparse_null_columns() -> None:
    train = pl.DataFrame({"sparse": [None, None, 1.0]})
    holdout = pl.DataFrame({"sparse": [None, None, None]})
    kept = adversarial_drop_features(train, holdout, candidate_cols=["sparse"], auc_threshold=0.6)
    assert kept == ["sparse"]


def test_drop_below_noise_floor() -> None:
    importance = np.array([0.5, 0.2, 0.05])
    feature_names = ["a", "b", "noise_seed42"]
    kept = drop_below_noise_floor(feature_names, importance, noise_feature="noise_seed42")
    assert "a" in kept
    assert "b" in kept
    assert "noise_seed42" not in kept


def test_drop_below_noise_floor_strips_feature_below_noise() -> None:
    importance = np.array([0.5, 0.01, 0.05])
    feature_names = ["a", "b", "noise_seed42"]
    kept = drop_below_noise_floor(feature_names, importance, noise_feature="noise_seed42")
    assert "a" in kept
    assert "b" not in kept
    assert "noise_seed42" not in kept
