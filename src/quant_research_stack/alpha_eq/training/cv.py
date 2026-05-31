"""Expanding-window walk-forward CV with dynamic purge + embargo (spec §4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from quant_research_stack.alpha_eq.config import CVConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]


def build_expanding_window_folds(
    *, dev_window_dates: Sequence[date], cv: CVConfig
) -> list[Fold]:
    sorted_dates = sorted(set(dev_window_dates))
    n = len(sorted_dates)
    n_folds = cv.n_folds
    val_size = max(1, n // (n_folds + 1))
    folds: list[Fold] = []
    for k in range(n_folds):
        val_start_idx = (k + 1) * val_size
        val_end_idx = min(n, val_start_idx + val_size)
        if val_start_idx >= n or val_end_idx <= val_start_idx:
            break
        val_dates = sorted_dates[val_start_idx:val_end_idx]
        purge_cutoff = val_dates[0] - timedelta(days=cv.purge_days)
        train_candidates = [d for d in sorted_dates[:val_start_idx] if d <= purge_cutoff]
        folds.append(
            Fold(
                fold_id=k,
                train_dates=tuple(train_candidates),
                validation_dates=tuple(val_dates),
            )
        )
    cleaned: list[Fold] = []
    for k, f in enumerate(folds):
        if k == 0:
            cleaned.append(f)
            continue
        prev_val_end = max(folds[k - 1].validation_dates)
        embargo = {prev_val_end + timedelta(days=i) for i in range(1, cv.embargo_days + 1)}
        new_train = tuple(d for d in f.train_dates if d not in embargo)
        cleaned.append(
            Fold(
                fold_id=f.fold_id,
                train_dates=new_train,
                validation_dates=f.validation_dates,
            )
        )
    return cleaned
