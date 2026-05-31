from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from quant_research_stack.brokers.capabilities import BrokerCapabilities
from quant_research_stack.brokers.fill_model import FillModel
from quant_research_stack.brokers.order_types import (
    Account,
    Fill,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from quant_research_stack.feeds.market_types import Bar, Tick

_CAPS = BrokerCapabilities(
    venue="null_broker",
    supported_order_types=frozenset(OrderType),
    supported_time_in_force=frozenset(TimeInForce),
    supports_shorting=True,
    supports_fractional_shares=True,
    supports_extended_hours=True,
    max_orders_per_second=1_000_000,
    paper_only=True,
)


@dataclass
class NullBroker:
    fill_model: FillModel
    starting_cash: float = 100_000.0
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        self._next_id = 0
        self._orders: dict[str, Order] = {}
        self._fills: dict[str, list[Fill]] = defaultdict(list)
        self._positions: dict[str, float] = defaultdict(float)
        self._avg_price: dict[str, float] = defaultdict(float)
        self._cash = float(self.starting_cash)
        self._fill_queue: deque[Fill] = deque()
        self._market_events: deque[Tick | Bar] = deque()

    def push_market_event(self, event: Tick | Bar) -> None:
        self._market_events.append(event)

    def _next_broker_id(self) -> str:
        self._next_id += 1
        return f"null-{self._next_id:07d}"

    async def place_order(self, intent: OrderIntent) -> Order:
        broker_id = self._next_broker_id()
        now = datetime.now(UTC)
        fills = self.fill_model.synthesize(intent, iter(list(self._market_events)))
        filled_qty = sum(f.quantity for f in fills)
        status = OrderStatus.filled if filled_qty >= intent.quantity else OrderStatus.accepted
        order = Order(
            client_order_id=intent.client_order_id,
            broker_order_id=broker_id,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            filled_quantity=filled_qty,
            status=status,
            submitted_utc=now,
            updated_utc=now,
        )
        self._orders[intent.client_order_id] = order
        for f in fills:
            self._fills[intent.client_order_id].append(f)
            self._fill_queue.append(f)
            sign = 1.0 if f.side == OrderSide.buy else -1.0
            self._positions[f.symbol] += sign * f.quantity
            self._avg_price[f.symbol] = f.price
            self._cash -= sign * f.price * f.quantity + f.commission
        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        order = self._orders[client_order_id]
        canceled = order.model_copy(update={"status": OrderStatus.canceled, "updated_utc": datetime.now(UTC)})
        self._orders[client_order_id] = canceled
        return canceled

    async def get_order(self, client_order_id: str) -> Order:
        return self._orders[client_order_id]

    async def positions(self) -> list[Position]:
        out = []
        for sym, qty in self._positions.items():
            if qty == 0.0:
                continue
            entry = self._avg_price[sym]
            market_value = entry * qty
            out.append(Position(
                symbol=sym,
                quantity=qty,
                avg_entry_price=entry,
                market_value=market_value,
                unrealized_pnl=0.0,
            ))
        return out

    async def account(self) -> Account:
        equity = self._cash + sum(self._avg_price[s] * q for s, q in self._positions.items())
        return Account(equity=equity, cash=self._cash, buying_power=equity, currency="USD")

    async def stream_fills(self) -> AsyncIterator[Fill]:
        while self._fill_queue:
            yield self._fill_queue.popleft()
        while True:
            await asyncio.sleep(0.01)
            while self._fill_queue:
                yield self._fill_queue.popleft()

    async def close(self) -> None:
        return None
