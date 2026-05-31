"""Options-implied features tests (spec §3.3 #9)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.papers.options_implied import (
    OptionsImpliedConfig,
    OptionsImpliedFeatures,
)


def test_options_implied_features_emit_term_structure() -> None:
    panel = pl.DataFrame({
        "date": [1, 2], "vix": [20.0, 22.0], "vix9d": [18.0, 24.0],
        "vvix": [100.0, 110.0], "skew": [130.0, 132.0],
    })
    out = OptionsImpliedFeatures(OptionsImpliedConfig()).features(panel)
    assert "vix_term_structure" in out.columns
    assert "vol_of_vol_ratio" in out.columns


def test_nasdaq_iv_fallback_marked_explicitly() -> None:
    panel = pl.DataFrame({"date": [1, 2], "vix": [20.0, 22.0]})
    out = OptionsImpliedFeatures(OptionsImpliedConfig()).features(panel)
    assert "nasdaq_iv" in out.columns
    assert "nasdaq_iv_is_vix_fallback" in out.columns
