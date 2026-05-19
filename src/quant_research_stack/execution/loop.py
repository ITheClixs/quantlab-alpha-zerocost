from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.base import BrokerAdapter
from quant_research_stack.brokers.order_types import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import ExecConfig, RiskConfig
from quant_research_stack.execution.position_book import PositionBook
from quant_research_stack.execution.risk import RiskGate, RiskState
from quant_research_stack.execution.signals import SignalIngestor
from quant_research_stack.execution.sizing import Sizer, SizerInput
from quant_research_stack.execution.types import ExecutionTicket


class S4Loop:
    """Orchestrates SignalIngestor -> RiskGate -> Sizer -> broker.place_order."""

    def __init__(
        self,
        stage: str,
        risk_cfg: RiskConfig,
        exec_cfg: ExecConfig,
        broker: BrokerAdapter,
        audit: AuditLog,
        starting_equity: Decimal,
        mid_price_lookup: Callable[[str], Decimal],
        is_crypto: Callable[[str], bool],
        tier3_stance_pct: float = 0.20,
    ) -> None:
        self._stage = stage
        self._risk_cfg = risk_cfg
        self._exec_cfg = exec_cfg
        self._broker = broker
        self._audit = audit
        self._mid_lookup = mid_price_lookup
        self._is_crypto = is_crypto
        self._starting_equity = starting_equity
        self._book = PositionBook(
            snapshot_root=Path(exec_cfg.position_book.snapshot_root),
            stage=stage,
            starting_equity=starting_equity,
        )
        self._risk_gate = RiskGate(risk_cfg)
        self._sizer = Sizer(risk_cfg, tier3_stance_pct=tier3_stance_pct)
        self._ingestor = SignalIngestor(
            preds_dir=Path(exec_cfg.ingest.s1_predictions_dir),
            verdicts_dir=Path(exec_cfg.ingest.s2_verdicts_dir),
            poll_interval_s=exec_cfg.ingest.poll_interval_seconds,
            pair_window_s=exec_cfg.ingest.pair_window_seconds,
            audit=audit,
        )
        self._orders_last_minute: list[datetime] = []

    async def run(self, max_tickets: int | None = None) -> None:
        processed = 0
        async for ticket in self._ingestor.stream():
            await self._handle(ticket)
            processed += 1
            if max_tickets is not None and processed >= max_tickets:
                self._ingestor.stop()
                return

    async def _handle(self, ticket: ExecutionTicket) -> None:
        sym = ticket.signal.symbol
        now = datetime.now(UTC)
        cutoff = now.timestamp() - 60.0
        self._orders_last_minute = [t for t in self._orders_last_minute if t.timestamp() >= cutoff]
        mid = self._mid_lookup(sym)
        per_sym = self._book.per_symbol_notional({sym: mid})
        gross = self._book.gross_exposure({sym: mid})
        state = RiskState(
            account_equity=float(self._starting_equity),
            peak_equity=float(self._book.peak_equity),
            daily_realized_pnl=float(self._book.daily_realized_pnl),
            gross_exposure_notional=gross,
            per_symbol_notional=per_sym,
            orders_last_minute=len(self._orders_last_minute),
            last_tick_ts={sym: now},
            kill_flag_path=Path(self._exec_cfg.kill_switch.repo_root_marker),
            is_crypto=self._is_crypto,
            now=now,
        )
        decision = self._risk_gate.evaluate(ticket, state)
        if not decision.allowed:
            self._audit.append(
                "risk_blocked",
                {"signal_id": ticket.signal.signal_id, "gate_name": decision.reason, "kill": decision.kill_trigger},
            )
            return

        qty = self._sizer.size(
            SizerInput(
                ticket=ticket,
                account_equity=float(self._starting_equity),
                mid_price=float(mid),
                lot_size=0.0001,
            )
        )
        if qty == 0:
            self._audit.append("trade_skipped_zero_qty", {"signal_id": ticket.signal.signal_id})
            return

        intent = OrderIntent(
            client_order_id=ticket.signal.signal_id,
            symbol=sym,
            side=OrderSide.buy if qty > 0 else OrderSide.sell,
            type=OrderType.market,
            quantity=abs(qty),
            time_in_force=TimeInForce.ioc,
        )
        order = await self._broker.place_order(intent)
        self._orders_last_minute.append(now)
        self._audit.append(
            "trade_placed",
            {
                "signal_id": ticket.signal.signal_id,
                "order_id": order.client_order_id,
                "symbol": sym,
                "side": intent.side.value,
                "qty": intent.quantity,
                "mid": float(mid),
            },
        )
