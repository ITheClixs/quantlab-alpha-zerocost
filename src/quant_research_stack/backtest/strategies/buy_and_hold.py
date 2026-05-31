from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide, OrderType
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass
class BuyAndHold:
    symbol: str
    quantity: float
    name: str = "buy_and_hold"
    _fired: bool = field(default=False, init=False)
    _fills: list[Fill] = field(default_factory=list, init=False)

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]:
        if self._fired or event.symbol != self.symbol:
            return []
        self._fired = True
        return [OrderIntent.model_validate({
            "client_order_id": f"bh-{uuid.uuid4().hex[:12]}",
            "symbol": self.symbol,
            "side": OrderSide.buy.value,
            "type": OrderType.market.value,
            "quantity": self.quantity,
        })]

    def on_fill(self, fill: Fill) -> None:
        self._fills.append(fill)

    def snapshot_state(self) -> dict:
        return {"fired": self._fired, "n_fills": len(self._fills)}
