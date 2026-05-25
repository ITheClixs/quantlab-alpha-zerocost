"""Permanent holdout gate (spec §3.6).

Only `inference.evaluate_holdout` is allowed to read holdout rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import polars as pl

_ALLOWED_CALLERS: Final[frozenset[str]] = frozenset({"inference.evaluate_holdout"})


class HoldoutAccessError(RuntimeError):
    pass


class HoldoutTooShortError(RuntimeError):
    pass


def compute_holdout_dates(
    sorted_unique_dates: Sequence[date], *, fraction: float
) -> tuple[list[date], list[date]]:
    n = len(sorted_unique_dates)
    if n == 0:
        return [], []
    n_hold = max(1, int(round(n * fraction)))
    return list(sorted_unique_dates[: n - n_hold]), list(sorted_unique_dates[n - n_hold :])


@dataclass(frozen=True)
class HoldoutGate:
    holdout_dates: list[date]

    def filter_for_caller(self, panel: pl.DataFrame, *, caller: str) -> pl.DataFrame:
        if caller in _ALLOWED_CALLERS:
            return panel.filter(pl.col("date").is_in(self.holdout_dates))
        if panel.filter(pl.col("date").is_in(self.holdout_dates)).height > 0:
            raise HoldoutAccessError(
                f"caller {caller!r} attempted to access holdout dates; "
                "use inference.evaluate_holdout()"
            )
        return panel


def assert_min_holdout_length(holdout_dates: list[date], *, min_trading_days: int) -> None:
    if len(holdout_dates) < min_trading_days:
        raise HoldoutTooShortError(
            f"holdout has {len(holdout_dates)} trading days, requires ≥ {min_trading_days}"
        )
