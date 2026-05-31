"""Bridge contract #4: distinct rows kept; no dedup of legit unique rows."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    signal_to_panel,
)


def test_distinct_rows_pass_through() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3)],
        "symbol": ["A", "B", "A", "B"],
        "feature_as_of_date": [date(2024, 1, 1)] * 4,
        "execution_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3)],
        "y_xs_pred": [0.1, -0.1, 0.2, -0.2],
        "tradable": [True] * 4,
        "in_pit_universe": [True] * 4,
    })
    out = signal_to_panel(panel)
    assert out.height == 4
