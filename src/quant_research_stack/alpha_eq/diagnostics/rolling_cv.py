"""Rolling-window CV diagnostic (spec §4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class RollingWindow:
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]


def build_rolling_windows(
    dev_window_dates: Sequence[date],
    *,
    train_years: int,
    valid_years: int,
    step_years: int = 1,
) -> list[RollingWindow]:
    sorted_dates = sorted(set(dev_window_dates))
    if not sorted_dates:
        return []
    start = sorted_dates[0]
    end = sorted_dates[-1]
    windows: list[RollingWindow] = []
    cur_train_end = start + timedelta(days=int(train_years * 365.25))
    while cur_train_end + timedelta(days=int(valid_years * 365.25)) <= end:
        train = tuple(d for d in sorted_dates if start <= d < cur_train_end)
        valid_end = cur_train_end + timedelta(days=int(valid_years * 365.25))
        valid = tuple(d for d in sorted_dates if cur_train_end <= d < valid_end)
        if train and valid:
            windows.append(RollingWindow(train_dates=train, validation_dates=valid))
        cur_train_end += timedelta(days=int(step_years * 365.25))
    return windows
