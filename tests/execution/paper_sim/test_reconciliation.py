from __future__ import annotations

import pytest

from quant_research_stack.execution.paper_sim.reconciliation import ReconReport


def test_report_renders_observation_only_and_numbers() -> None:
    rep = ReconReport(
        cycles=10, n_rebalances=4, funding_pnl=1.25,
        basis_samples=[0.0005, -0.0003, 0.0011], equity_start=100000.0, equity_end=100001.25)
    md = rep.render()
    assert "observation-only" in md.lower()
    assert "1.25" in md
    assert "DO_NOT_ADVANCE" in md
    assert rep.basis_mean_pct() == pytest.approx(
        (0.0005 - 0.0003 + 0.0011) / 3 * 100, rel=1e-6)
