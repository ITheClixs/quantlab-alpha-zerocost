from __future__ import annotations

from typing import Protocol

from quant_research_stack.brokers.order_types import Fill, OrderIntent
from quant_research_stack.feeds.market_types import Bar, Tick


class Strategy(Protocol):
    name: str

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]: ...

    def on_fill(self, fill: Fill) -> None: ...

    def snapshot_state(self) -> dict: ...
