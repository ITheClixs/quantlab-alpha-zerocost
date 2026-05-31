from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide, OrderType
from quant_research_stack.feeds.market_types import Bar, Tick


def _price(event: Tick | Bar) -> float:
    return event.close if isinstance(event, Bar) else event.price


@dataclass
class MovingAverageCross:
    symbol: str
    quantity: float
    fast_window: int
    slow_window: int
    name: str = "moving_average_cross"
    _fast: deque = field(default_factory=lambda: deque(), init=False)
    _slow: deque = field(default_factory=lambda: deque(), init=False)
    _prev_fast_gt_slow: bool | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be < slow_window")
        self._fast = deque(maxlen=self.fast_window)
        self._slow = deque(maxlen=self.slow_window)

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]:
        if event.symbol != self.symbol:
            return []
        price = _price(event)
        self._fast.append(price)
        self._slow.append(price)
        if len(self._slow) < self.slow_window:
            return []
        fast_mean = sum(self._fast) / len(self._fast)
        slow_mean = sum(self._slow) / len(self._slow)
        current = fast_mean > slow_mean
        prev = self._prev_fast_gt_slow
        self._prev_fast_gt_slow = current
        if prev is None or prev == current:
            return []
        side = OrderSide.buy if current else OrderSide.sell
        return [OrderIntent.model_validate({
            "client_order_id": f"ma-{uuid.uuid4().hex[:12]}",
            "symbol": self.symbol,
            "side": side.value,
            "type": OrderType.market.value,
            "quantity": self.quantity,
        })]

    def on_fill(self, fill: Fill) -> None:
        return None

    def snapshot_state(self) -> dict:
        return {
            "fast_len": len(self._fast),
            "slow_len": len(self._slow),
            "prev_fast_gt_slow": self._prev_fast_gt_slow,
        }
