# tests/signal_research/fingerprint_vwap/test_fingerprint.py
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    window_trend,
)


def test_window_trend_perfect_uptrend() -> None:
    logclose = np.log(np.exp(np.linspace(0.0, 1.0, 60)))  # perfectly linear in log
    direction, strength, r2 = window_trend(logclose)
    assert direction == 1.0
    assert r2 > 0.999
    assert strength > 0.0


def test_build_fingerprint_columns_present_and_asof(panel: pl.DataFrame) -> None:
    out = build_fingerprint_features(panel, windows=(20, 60))
    for w in (20, 60):
        for base in ("trend_direction", "trend_strength", "trend_linearity_r2", "spikiness"):
            assert f"{base}_{w}" in out.columns
    aaa = out.filter(pl.col("symbol") == "AAA").sort("date")
    assert aaa["trend_direction_20"][:19].null_count() == 19
    defined = aaa["trend_direction_60"].drop_nulls()
    assert float((defined == 1.0).mean()) > 0.7
