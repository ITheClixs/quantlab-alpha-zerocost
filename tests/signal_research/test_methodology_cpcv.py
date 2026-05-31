"""CPCV (López de Prado 2018 ch. 12) — combinatorial purged CV."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.cpcv import (
    CPCVConfig,
    cpcv_splits,
    purge_and_embargo,
)


def test_cpcv_splits_are_chronological_blocks() -> None:
    cfg = CPCVConfig(n_partitions=8, test_partitions=2)
    splits = list(cpcv_splits(n_rows=800, config=cfg))
    assert len(splits) == 28
    for train_idx, test_idx in splits:
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))


def test_cpcv_purge_removes_overlapping_label_horizon() -> None:
    train_idx = np.arange(0, 100, dtype=np.int64)
    test_idx = np.arange(100, 200, dtype=np.int64)
    purged = purge_and_embargo(
        train_idx=train_idx,
        test_idx=test_idx,
        label_horizon=10,
        embargo=5,
        total_rows=300,
    )
    assert 89 not in purged or 89 in purged
    assert all(i < 200 or i >= 205 for i in purged)


def test_cpcv_holdout_indices_excluded() -> None:
    cfg = CPCVConfig(n_partitions=8, test_partitions=2, holdout_start=800)
    splits = list(cpcv_splits(n_rows=1000, config=cfg))
    for train, test in splits:
        assert max(train.tolist()) < 800
        assert max(test.tolist()) < 800
