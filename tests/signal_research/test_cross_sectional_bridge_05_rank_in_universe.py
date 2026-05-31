"""Bridge contract #5: rank within tradable + in_pit_universe; outside has null rank."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    signal_to_panel,
)


def test_outside_universe_rows_have_null_rank() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 4,
        "symbol": ["A", "B", "C", "D"],
        "feature_as_of_date": [date(2024, 1, 1)] * 4,
        "execution_date": [date(2024, 1, 2)] * 4,
        "y_xs_pred": [0.1, 0.2, 0.3, 0.4],
        "tradable": [True, True, False, True],
        "in_pit_universe": [True, True, True, False],
    })
    out = signal_to_panel(panel)
    out_dict = {r["symbol"]: r["y_xs_pred_rank"] for r in out.iter_rows(named=True)}
    assert out_dict["A"] is not None
    assert out_dict["B"] is not None
    assert out_dict["C"] is None
    assert out_dict["D"] is None
