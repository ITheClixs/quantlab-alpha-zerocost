"""Bridge contract #1: schema validation."""

from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    BridgeSchemaError,
    signal_to_panel,
)


def test_missing_required_columns_raise_schema_error() -> None:
    bad = pl.DataFrame({"date": [1], "y_xs_pred": [0.0]})
    with pytest.raises(BridgeSchemaError):
        signal_to_panel(bad)
