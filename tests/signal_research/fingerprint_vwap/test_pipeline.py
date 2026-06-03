# tests/signal_research/fingerprint_vwap/test_pipeline.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.pipeline import (
    FingerprintVwapSpec,
    run_fingerprint_vwap_meta,
)


def test_pipeline_returns_result_with_eligibility_and_lift(panel: pl.DataFrame) -> None:
    spec = FingerprintVwapSpec(
        windows=(20, 60), vwap_window=5, band=0.0, horizon_days=3,
        cost_bps_one_way=1.0, train_window_days=120, test_window_days=30,
        step_days=30, min_train_events=20,
    )
    result = run_fingerprint_vwap_meta(panel=panel, spec=spec)
    assert "eligibility" in result and "eligible" in result["eligibility"]
    if not result["eligibility"]["eligible"]:
        assert result["status"] == "primary_ineligible"
    else:
        assert "meta_net_sharpe" in result and "baseline_net_sharpe" in result
        assert "lift" in result
