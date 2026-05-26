"""Selection funnel (spec §6.4)."""

from __future__ import annotations

from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)


def test_funnel_records_counts_per_filter_in_order() -> None:
    f = SelectionFunnel()
    f.record("total_raw_candidates", 1620)
    f.record("after_data_quality_filter", 1500)
    f.record("after_cost_stress_2x", 980)
    f.record("after_sanity_randomization", 920)
    f.record("after_pbo_profile_threshold", 110)
    f.record("after_dsr_threshold", 45)
    f.record("after_bootstrap_lower_positive", 25)
    f.record("after_regime_concentration", 8)
    f.record("research_pass", 8)
    f.record("promotion_eligible", 0)
    f.record("paper_trade_candidate", 0)
    f.record("production_candidate", 0)
    counts = f.to_ordered_dict()
    assert counts["total_raw_candidates"] == 1620
    assert counts["research_pass"] == 8
    assert counts["production_candidate"] == 0
    keys = list(counts.keys())
    assert keys[0] == "total_raw_candidates"
    assert keys[-1] == "production_candidate"
