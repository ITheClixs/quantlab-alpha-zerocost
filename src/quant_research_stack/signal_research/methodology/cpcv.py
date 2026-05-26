"""CPCV — Combinatorial Purged Cross-Validation (López de Prado 2018 ch. 12).

Spec §4.1:
- Splits are chronological blocks, not random row splits.
- Purging removes rows whose label horizon overlaps the test block.
- Embargo removes rows immediately after the test block.
- Permanent holdout never touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CPCVConfig:
    n_partitions: int = 8
    test_partitions: int = 2
    label_horizon: int = 10
    embargo: int = 5
    holdout_start: int | None = None


def cpcv_splits(
    *, n_rows: int, config: CPCVConfig
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    end = config.holdout_start if config.holdout_start is not None else n_rows
    block_size = end // config.n_partitions
    blocks = [
        np.arange(i * block_size, (i + 1) * block_size, dtype=np.int64)
        for i in range(config.n_partitions)
    ]
    for test_block_ids in combinations(range(config.n_partitions), config.test_partitions):
        test_idx = np.concatenate([blocks[i] for i in test_block_ids])
        train_idx = np.concatenate(
            [blocks[i] for i in range(config.n_partitions) if i not in test_block_ids]
        )
        train_idx = purge_and_embargo(
            train_idx=train_idx,
            test_idx=test_idx,
            label_horizon=config.label_horizon,
            embargo=config.embargo,
            total_rows=end,
        )
        yield train_idx, test_idx


def purge_and_embargo(
    *,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    label_horizon: int,
    embargo: int,
    total_rows: int,
) -> NDArray[np.int64]:
    """Drop (a) train rows whose [t, t+label_horizon] overlaps test_idx,
    and (b) train rows in [test_max+1, test_max+1+embargo)."""
    test_set = set(test_idx.tolist())
    keep: list[int] = []
    for t in train_idx:
        if any((int(t) + h) in test_set for h in range(label_horizon + 1)):
            continue
        keep.append(int(t))
    if test_idx.size:
        max_test = int(test_idx.max())
        embargo_set = set(range(max_test + 1, max_test + 1 + embargo))
        keep = [t for t in keep if t not in embargo_set]
    return np.asarray(keep, dtype=np.int64)
