from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.alpha.cv import PurgedKFold


def test_purged_kfold_produces_n_folds() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    folds = list(splitter.split(df))
    assert len(folds) == 5


def test_purged_kfold_train_test_disjoint() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    for train_idx, test_idx in splitter.split(df):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_purged_kfold_embargo_gap_respected() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    for train_idx, test_idx in splitter.split(df):
        train_dates = set(df[train_idx]["date_id"].to_list())
        test_dates = set(df[test_idx]["date_id"].to_list())
        for t_test in test_dates:
            for t_train in train_dates:
                if t_train > t_test:
                    assert t_train >= t_test + 5, f"embargo violation: test={t_test} train={t_train}"


def test_purged_kfold_chronological_test_folds() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    test_means = []
    for _, test_idx in splitter.split(df):
        test_dates = df[test_idx]["date_id"].to_numpy()
        test_means.append(float(np.mean(test_dates)))
    assert test_means == sorted(test_means)
