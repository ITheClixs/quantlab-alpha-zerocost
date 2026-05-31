from __future__ import annotations

from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import perp_symbol


class FundingAccrual:
    """Credits/debits the short-perp leg at each 8h settlement from the REAL rate.

    Short receives funding when rate > 0: pnl = -perp_qty * perp_mark * rate.
    Dedups by the settlement boundary (`next_funding_ms`) so it accrues once per window.
    """

    def __init__(self) -> None:
        self._settled: set[int] = set()

    def maybe_settle(self, snap: MarketSnapshot, *, positions: dict[str, float]) -> float:
        if snap.next_funding_ms in self._settled:
            return 0.0
        self._settled.add(snap.next_funding_ms)
        perp_qty = positions.get(perp_symbol(snap.symbol), 0.0)
        return -perp_qty * snap.perp_mark * snap.funding_rate
