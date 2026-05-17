from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_ADVERSARIAL_MAX_ROWS_PER_SPLIT = 10_000


def _sample_rows(values: NDArray[np.float64], max_rows: int, seed: int) -> NDArray[np.float64]:
    if values.shape[0] <= max_rows:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.shape[0], size=max_rows, replace=False)
    return values[idx]


def train_holdout_classifier_auc(
    train: NDArray[np.float64],
    holdout: NDArray[np.float64],
    max_rows_per_split: int = _ADVERSARIAL_MAX_ROWS_PER_SPLIT,
) -> float:
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if holdout.ndim == 1:
        holdout = holdout.reshape(-1, 1)
    train = _sample_rows(train, max_rows_per_split, seed=42)
    holdout = _sample_rows(holdout, max_rows_per_split, seed=43)
    x = np.vstack([train, holdout])
    y = np.concatenate([np.zeros(train.shape[0]), np.ones(holdout.shape[0])])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for fold_train, fold_test in skf.split(x, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[fold_train], y[fold_train])
        prob = clf.predict_proba(x[fold_test])[:, 1]
        aucs.append(roc_auc_score(y[fold_test], prob))
    return float(np.mean(aucs))


def adversarial_drop_features(
    train_df: pl.DataFrame, holdout_df: pl.DataFrame, candidate_cols: list[str], auc_threshold: float = 0.6
) -> list[str]:
    kept: list[str] = []
    for col in candidate_cols:
        train_values = train_df[col].drop_nulls().to_numpy()
        holdout_values = holdout_df[col].drop_nulls().to_numpy()
        if min(train_values.shape[0], holdout_values.shape[0]) < 3:
            kept.append(col)
            continue
        auc = train_holdout_classifier_auc(train_values, holdout_values)
        if auc < auc_threshold:
            kept.append(col)
    return kept


def drop_below_noise_floor(
    feature_names: list[str], importance: NDArray[np.float64], noise_feature: str
) -> list[str]:
    if noise_feature not in feature_names:
        raise ValueError(f"noise feature {noise_feature!r} not in feature_names")
    idx = feature_names.index(noise_feature)
    noise_imp = float(importance[idx])
    return [name for name, imp in zip(feature_names, importance, strict=True) if name != noise_feature and float(imp) > noise_imp]
