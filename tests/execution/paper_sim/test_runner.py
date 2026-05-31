from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.runner import CarryLoop, ensure_paper_stage


def test_ensure_paper_stage_rejects_non_paper(monkeypatch) -> None:
    monkeypatch.setenv("QUANTLAB_STAGE", "live")
    with pytest.raises(SystemExit):
        ensure_paper_stage()
    monkeypatch.setenv("QUANTLAB_STAGE", "paper")
    ensure_paper_stage()  # no raise


@pytest.mark.asyncio
async def test_loop_opens_delta_neutral_and_accrues_funding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTLAB_STAGE", "paper")
    cfg = PaperSimConfig(symbols=["BTCUSDT"], total_notional_usd=10000.0,
                         rebalance_drift_bps=10.0, poll_interval_s=0.1)

    snaps = iter([
        MarketSnapshot("BTCUSDT", 1, 100.0, 100.0, 0.0001, next_funding_ms=8),
        MarketSnapshot("BTCUSDT", 2, 100.0, 100.0, 0.0001, next_funding_ms=16),
    ])

    async def source(symbol: str, now_ms: int) -> MarketSnapshot:
        return next(snaps)

    loop = CarryLoop(cfg, audit_root=tmp_path / "audit", snapshot_root=tmp_path / "book",
                     snapshot_source=source)
    await loop.run(max_cycles=2)

    # leg_notional = 10000/(2*1) = 5000 @100 -> 50 units long spot, 50 short perp
    pos = loop.positions()
    assert abs(pos["BTCUSDT"] - 50.0) < 1.0
    assert abs(pos["BTCUSDTPERP"] + 50.0) < 1.0
    # funding accrued at least once (short 50@100 * 0.0001 = 0.5 per settlement)
    assert loop.funding_pnl() > 0.0
    # audit file written
    assert any((tmp_path / "audit").glob("*.jsonl"))
    # reconciliation report reflects the run (observation-only)
    rep = loop.report()
    assert rep.cycles == 2
    assert rep.n_rebalances >= 2          # at least both legs opened
    assert rep.funding_pnl == loop.funding_pnl()
    assert "DO_NOT_ADVANCE" in rep.render()
