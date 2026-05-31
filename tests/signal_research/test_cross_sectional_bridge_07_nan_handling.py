"""Bridge contract #7: NaN handling."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    signal_to_panel,
)


def test_nan_predictions_dropped_when_drop_nan_true() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 3,
        "symbol": ["A", "B", "C"],
        "feature_as_of_date": [date(2024, 1, 1)] * 3,
        "execution_date": [date(2024, 1, 2)] * 3,
        "y_xs_pred": [0.1, None, 0.3],
        "tradable": [True] * 3,
        "in_pit_universe": [True] * 3,
    })
    out = signal_to_panel(panel, drop_nan=True)
    assert out.height == 2


def test_nan_predictions_kept_when_drop_nan_false() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 3,
        "symbol": ["A", "B", "C"],
        "feature_as_of_date": [date(2024, 1, 1)] * 3,
        "execution_date": [date(2024, 1, 2)] * 3,
        "y_xs_pred": [0.1, None, 0.3],
        "tradable": [True] * 3,
        "in_pit_universe": [True] * 3,
    })
    out = signal_to_panel(panel, drop_nan=False)
    assert out.height == 3
