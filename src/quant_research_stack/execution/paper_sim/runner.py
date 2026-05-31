from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.brokers.order_types import OrderIntent
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.funding_accrual import FundingAccrual
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.reconciliation import ReconReport
from quant_research_stack.execution.paper_sim.strategy import FundingCarryStrategy
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

SnapshotSource = Callable[[str, int], Awaitable[MarketSnapshot]]


def ensure_paper_stage() -> None:
    """Refuse to run unless QUANTLAB_STAGE=paper (observation-only; no live)."""
    stage = os.environ.get("QUANTLAB_STAGE", "paper")
    if stage != "paper":
        sys.stderr.write(f"REFUSING: QUANTLAB_STAGE={stage!r}, paper-sim runs only at 'paper'\n")
        raise SystemExit(2)


def _leg_price(snap: MarketSnapshot, leg_symbol: str) -> float:
    return snap.perp_mark if leg_symbol.endswith("PERP") else snap.spot_price


class CarryLoop:
    """Observation-only funding-carry paper sim. Reuses NullBroker/FillModel + AuditLog.

    NOT validation or promotion; the strategy is DO_NOT_ADVANCE (spec §0). 1x unlevered.
    """

    def __init__(self, cfg: PaperSimConfig, *, audit_root: Path, snapshot_root: Path,
                 snapshot_source: SnapshotSource) -> None:
        ensure_paper_stage()
        self._cfg = cfg
        self._source = snapshot_source
        self._audit = AuditLog(audit_root)
        self._audit.append("paper_sim_start",
                            {"observation_only": True, "strategy": "funding_carry",
                             "verdict": "DO_NOT_ADVANCE", "leverage": cfg.leverage,
                             "symbols": cfg.symbols})
        self._broker = NullBroker(
            fill_model=FillModel(FillModelConfig(
                commission_bps=cfg.commission_bps,
                slippage_bps=cfg.slippage_bps,
                half_spread_bps=cfg.half_spread_bps)),
            starting_cash=cfg.starting_equity_usd)
        self._strategy = FundingCarryStrategy(cfg)
        self._accrual = FundingAccrual()
        self._positions: dict[str, float] = {}
        self._funding_pnl = 0.0
        self._cycles = 0
        self._n_rebalances = 0
        self._basis: list[float] = []
        snapshot_root.mkdir(parents=True, exist_ok=True)

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def funding_pnl(self) -> float:
        return self._funding_pnl

    def report(self) -> ReconReport:
        """Live-vs-model reconciliation summary (observation-only)."""
        start = self._cfg.starting_equity_usd
        return ReconReport(
            cycles=self._cycles, n_rebalances=self._n_rebalances,
            funding_pnl=self._funding_pnl, basis_samples=list(self._basis),
            equity_start=start, equity_end=start + self._funding_pnl)

    async def _place(self, intent: OrderIntent, price: float) -> None:
        # NullBroker fills from the FIRST market event; reset deque so this leg prices right.
        self._broker._market_events.clear()  # noqa: SLF001 - sim-only price injection
        self._broker.push_market_event(Tick(
            venue=Venue.binance, symbol=intent.symbol,
            timestamp_utc=datetime.now(UTC), received_utc=datetime.now(UTC),
            price=price, size=0.0, side=TickSide.unknown))
        order = await self._broker.place_order(intent)
        sign = 1.0 if intent.side.value == "buy" else -1.0
        self._positions[intent.symbol] = self._positions.get(intent.symbol, 0.0) + sign * intent.quantity
        self._audit.append("trade_placed", {
            "order_id": order.client_order_id, "symbol": intent.symbol,
            "side": intent.side.value, "qty": intent.quantity, "price": price})

    async def run(self, *, max_cycles: int | None = None) -> None:
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                for symbol in self._cfg.symbols:
                    snap = await self._source(symbol, cycle)
                    self._basis.append(snap.basis)
                    intents = self._strategy.rebalance_intents(
                        snap, positions=self._positions, cycle=cycle)
                    self._n_rebalances += len(intents)
                    for intent in intents:
                        await self._place(intent, _leg_price(snap, intent.symbol))
                    fpnl = self._accrual.maybe_settle(snap, positions=self._positions)
                    if fpnl != 0.0:
                        self._funding_pnl += fpnl
                        self._audit.append("funding_settled", {
                            "symbol": symbol, "rate": snap.funding_rate,
                            "perp_mark": snap.perp_mark, "funding_pnl": fpnl,
                            "basis": snap.basis})
                cycle += 1
                self._cycles = cycle
        finally:
            self._audit.append("paper_sim_stop",
                               {"cycles": cycle, "funding_pnl": self._funding_pnl,
                                "positions": self._positions})
            await self._broker.close()
