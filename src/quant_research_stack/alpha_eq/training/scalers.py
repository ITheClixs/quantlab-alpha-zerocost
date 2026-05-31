"""Per-fold scalers with explicit fit-window metadata (spec §3.5)."""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


class ScalerLeakError(RuntimeError):
    pass


class FoldScaler:
    def __init__(self, *, fold_id: int) -> None:
        self.fold_id = fold_id
        self._scaler = StandardScaler()
        self._fit_start: date | None = None
        self._fit_end: date | None = None
        self._fit_dates: frozenset[date] = frozenset()

    @property
    def fitted_on_start_date(self) -> date:
        if self._fit_start is None:
            raise RuntimeError("scaler not fit yet")
        return self._fit_start

    @property
    def fitted_on_end_date(self) -> date:
        if self._fit_end is None:
            raise RuntimeError("scaler not fit yet")
        return self._fit_end

    def fit(self, *, x: NDArray[np.float64], train_dates: list[date]) -> None:
        self._scaler.fit(x)
        self._fit_dates = frozenset(train_dates)
        self._fit_start = min(self._fit_dates)
        self._fit_end = max(self._fit_dates)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._scaler.transform(x), dtype=np.float64)

    def assert_transform_dates_outside_fit(self, *, transform_dates: list[date]) -> None:
        overlap = self._fit_dates.intersection(transform_dates)
        if overlap:
            raise ScalerLeakError(
                f"fold {self.fold_id}: transform dates overlap fit window: {sorted(overlap)}"
            )
