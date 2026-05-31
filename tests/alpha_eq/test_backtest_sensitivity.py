"""Standard sensitivity pack expands the headline into a fixed grid."""

from __future__ import annotations

from quant_research_stack.alpha_eq.backtest.sensitivity import (
    enumerate_standard_pack,
)


def test_standard_pack_yields_expected_combinations() -> None:
    runs = list(enumerate_standard_pack())
    # standard pack: borrow {1x, 3x} × fill {open, hlc3_proxy} × q {0.05, 0.10} × gross {1.0} = 8
    assert len(runs) == 8
    seen = {(r.borrow_multiplier, r.fill_model.value, r.q_quantile, r.target_gross) for r in runs}
    assert (1.0, "open", 0.10, 1.0) in seen
    assert (3.0, "vwap_proxy_hlc3", 0.05, 1.0) in seen
