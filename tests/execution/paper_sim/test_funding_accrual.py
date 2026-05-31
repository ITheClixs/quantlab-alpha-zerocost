from __future__ import annotations

from quant_research_stack.execution.paper_sim.funding_accrual import FundingAccrual
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import perp_symbol


def _snap(rate: float, next_ms: int) -> MarketSnapshot:
    return MarketSnapshot(symbol="BTCUSDT", ts_ms=0, spot_price=100.0, perp_mark=100.0,
                          funding_rate=rate, next_funding_ms=next_ms)


def test_short_receives_positive_funding_once_per_settlement() -> None:
    acc = FundingAccrual()
    pos = {perp_symbol("BTCUSDT"): -50.0}  # short 50 @ 100 -> notional 5000
    pnl = acc.maybe_settle(_snap(0.0001, next_ms=8), positions=pos)
    assert abs(pnl - 0.5) < 1e-9            # 0.0001 * 5000
    # same settlement window -> no double-count
    assert acc.maybe_settle(_snap(0.0001, next_ms=8), positions=pos) == 0.0
    # new settlement -> accrues again
    assert abs(acc.maybe_settle(_snap(0.0001, next_ms=16), positions=pos) - 0.5) < 1e-9


def test_no_funding_when_flat() -> None:
    acc = FundingAccrual()
    assert acc.maybe_settle(_snap(0.0001, next_ms=8), positions={}) == 0.0
