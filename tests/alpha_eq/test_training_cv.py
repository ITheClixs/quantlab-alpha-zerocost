"""Walk-forward CV with dynamic purge/embargo (spec §4.2)."""

from __future__ import annotations

from datetime import date, timedelta

from quant_research_stack.alpha_eq.config import CVConfig
from quant_research_stack.alpha_eq.training.cv import build_expanding_window_folds


def test_expanding_window_folds_are_chronological() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
    cfg = CVConfig()
    folds = build_expanding_window_folds(dev_window_dates=dates, cv=cfg)
    assert len(folds) == cfg.n_folds
    for f in folds:
        assert max(f.train_dates) < min(f.validation_dates)
        assert (min(f.validation_dates) - max(f.train_dates)).days >= cfg.purge_days
    for prev, nxt in zip(folds[:-1], folds[1:], strict=True):
        assert set(prev.train_dates).issubset(set(nxt.train_dates))


def test_embargo_excludes_post_validation_window_from_next_train() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
    cfg = CVConfig()
    folds = build_expanding_window_folds(dev_window_dates=dates, cv=cfg)
    for prev, nxt in zip(folds[:-1], folds[1:], strict=True):
        embargo_start = max(prev.validation_dates) + timedelta(days=1)
        embargo = {embargo_start + timedelta(days=k) for k in range(cfg.embargo_days)}
        assert embargo.isdisjoint(set(nxt.train_dates))
