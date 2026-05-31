from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from quant_research_stack.brokers.capabilities import BrokerCapabilities
from quant_research_stack.brokers.order_types import Account, Fill, Order, OrderIntent, Position


class BrokerAdapter(Protocol):
    capabilities: BrokerCapabilities

    async def place_order(self, intent: OrderIntent) -> Order: ...

    async def cancel_order(self, client_order_id: str) -> Order: ...

    async def get_order(self, client_order_id: str) -> Order: ...

    async def positions(self) -> list[Position]: ...

    async def account(self) -> Account: ...

    def stream_fills(self) -> AsyncIterator[Fill]: ...

    async def close(self) -> None: ...
