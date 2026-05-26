"""Bridge contract #3: feature_as_of_date < execution_date."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    BridgeContractError,
    signal_to_panel,
)


def test_feature_at_or_after_execution_raises() -> None:
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)],
        "symbol": ["A"],
        "feature_as_of_date": [date(2024, 1, 2)],
        "execution_date": [date(2024, 1, 2)],
        "y_xs_pred": [0.1],
        "tradable": [True],
        "in_pit_universe": [True],
    })
    with pytest.raises(BridgeContractError):
        signal_to_panel(panel)
