"""Scaler fit-window object-level contract (spec §3.5)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quant_research_stack.alpha_eq.training.scalers import (
    FoldScaler,
    ScalerLeakError,
)


def test_fold_scaler_records_fit_window() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    assert s.fitted_on_start_date == date(2020, 1, 2)
    assert s.fitted_on_end_date == date(2020, 1, 3)
    assert s.fold_id == 0


def test_fold_scaler_raises_when_transform_includes_validation_window() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    with pytest.raises(ScalerLeakError):
        s.assert_transform_dates_outside_fit(
            transform_dates=[date(2020, 1, 2)]
        )


def test_fold_scaler_transform_validates_distinct_dates() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    s.assert_transform_dates_outside_fit(transform_dates=[date(2020, 1, 10)])
