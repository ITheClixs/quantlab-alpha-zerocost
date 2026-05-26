"""Bridge contract #8: zero-input equality smoke."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    signal_to_panel,
)


def test_zero_prediction_panel_round_trips_through_bridge_validators() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 4,
        "symbol": ["A", "B", "C", "D"],
        "feature_as_of_date": [date(2024, 1, 1)] * 4,
        "execution_date": [date(2024, 1, 2)] * 4,
        "y_xs_pred": [0.0, 0.0, 0.0, 0.0],
        "tradable": [True] * 4,
        "in_pit_universe": [True] * 4,
    })
    out = signal_to_panel(panel)
    assert out.height == 4
