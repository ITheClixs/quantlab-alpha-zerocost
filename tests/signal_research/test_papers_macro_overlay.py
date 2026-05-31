"""Macro overlay feature tests (spec §3.3 #10)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.papers.macro_overlay import (
    MacroOverlayConfig,
    MacroOverlayFeatures,
)


def test_macro_overlay_features_passthrough_when_series_absent() -> None:
    panel = pl.DataFrame({"date": [1, 2], "close": [100.0, 101.0]})
    out = MacroOverlayFeatures(MacroOverlayConfig()).features(panel)
    assert out.height == 2
