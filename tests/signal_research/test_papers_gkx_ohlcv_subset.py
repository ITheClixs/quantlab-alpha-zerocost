"""GKX-style OHLCV-characteristic subset (spec §3.3 #5, §5.6)."""

from __future__ import annotations

from quant_research_stack.signal_research.papers import gkx_ohlcv_subset as m
from quant_research_stack.signal_research.papers.gkx_ohlcv_subset import (
    GKX_FEATURE_LIST,
    GKXOHLCVSubsetModelFamily,
)


def test_gkx_feature_list_is_explicit_and_complete() -> None:
    assert "momentum_1m" in GKX_FEATURE_LIST
    assert "momentum_12m_skip_1m" in GKX_FEATURE_LIST
    assert "reversal_1d" in GKX_FEATURE_LIST
    assert "realized_vol_20" in GKX_FEATURE_LIST
    assert "beta_to_spy_60" in GKX_FEATURE_LIST
    assert "dollar_volume_20d" in GKX_FEATURE_LIST
    assert "amihud_illiq_20" in GKX_FEATURE_LIST
    assert "close_location_20" in GKX_FEATURE_LIST


def test_naming_does_not_claim_full_gkx_replication() -> None:
    doc = m.__doc__ or ""
    class_doc = GKXOHLCVSubsetModelFamily.__doc__ or ""
    assert "GKX-style" in class_doc or "OHLCV-characteristic" in doc
