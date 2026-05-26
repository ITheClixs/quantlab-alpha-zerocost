"""Bridge contract #2: one prediction per (date, symbol)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    BridgeContractError,
    signal_to_panel,
)


def test_duplicate_date_symbol_rows_raise() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 2,
        "symbol": ["A", "A"],
        "feature_as_of_date": [date(2024, 1, 1)] * 2,
        "execution_date": [date(2024, 1, 2)] * 2,
        "y_xs_pred": [0.1, 0.2],
        "tradable": [True, True],
        "in_pit_universe": [True, True],
    })
    with pytest.raises(BridgeContractError):
        signal_to_panel(panel)
