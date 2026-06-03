# tests/signal_research/fingerprint_vwap/test_eligibility.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.eligibility import (
    primary_signal_stats,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_primary_stats_shape_and_types(panel: pl.DataFrame) -> None:
    p = vwap_primary_position(daily_vwap_proxy(panel, window=5), band=0.0)
    stats = primary_signal_stats(p, horizon_days=3, cost_bps_one_way=1.0)
    assert stats.single_asset_or_cross_sectional == "cross_sectional"
    assert stats.event_count >= 0
    assert isinstance(stats.validation_net_sharpe, float)
    assert isinstance(stats.is_inverted_superior, bool)
