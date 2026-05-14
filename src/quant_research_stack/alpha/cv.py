from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PurgedKFold:
    n_folds: int
    group_column: str
    purge: int = 5
    embargo: int = 5

    def split(self, df: pl.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = df[self.group_column].to_numpy()
        unique_groups = np.unique(groups)
        unique_groups.sort()
        n = unique_groups.size
        fold_size = n // self.n_folds
        for fold_idx in range(self.n_folds):
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_folds - 1 else n
            test_groups = unique_groups[test_start:test_end]
            # purge: drop train groups within `purge` of any test group on the left side
            # embargo: drop train groups within `embargo` of any test group on the right side
            min_test = int(test_groups.min())
            max_test = int(test_groups.max())
            keep_train_mask = (unique_groups < min_test - self.purge) | (unique_groups > max_test + self.embargo)
            train_groups = unique_groups[keep_train_mask]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            train_idx = np.where(np.isin(groups, train_groups))[0]
            yield train_idx, test_idx
