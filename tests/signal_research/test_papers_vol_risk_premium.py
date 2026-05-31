"""Vol-Risk-Premium tests."""

from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.signal_research.papers.vol_risk_premium import (
    VRPFeature,
    VRPFeatureConfig,
    VRPTradableNotConfiguredError,
    VRPTradableStrategy,
)


def test_vrp_feature_emits_vrp_column() -> None:
    panel = pl.DataFrame({
        "date": list(range(30)),
        "close": [100.0 + i * 0.1 for i in range(30)],
        "vix": [20.0] * 30,
    })
    out = VRPFeature(VRPFeatureConfig()).features(panel)
    assert "vrp" in out.columns


def test_vrp_tradable_refuses_without_real_instrument() -> None:
    with pytest.raises(VRPTradableNotConfiguredError):
        VRPTradableStrategy(tradable_instrument=None)
