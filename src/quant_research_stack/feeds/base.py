from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from quant_research_stack.feeds.market_types import Bar, Tick, Venue

MarketEvent = Tick | Bar


class FeedConnectionError(RuntimeError):
    pass


def exponential_backoff(attempt: int, base: float, cap: float) -> float:
    if attempt < 1:
        return base
    return min(cap, base * (2 ** (attempt - 1)))


class FeedAdapter(Protocol):
    venue: Venue

    async def subscribe(self, symbols: Iterable[str]) -> None: ...

    def iterate(self) -> AsyncIterator[MarketEvent]: ...

    async def close(self) -> None: ...

    @property
    def is_live(self) -> bool: ...

    @property
    def stats(self) -> dict: ...


class AsyncFeedBase:
    """Mixin providing reconnect/backoff helpers and a bounded ring buffer.

    Concrete adapters subclass this and implement `subscribe`, `iterate`, `close`.
    """

    venue: Venue = Venue.replay

    def __init__(self, buffer_cap: int = 10_000) -> None:
        self._buffer_cap = buffer_cap
        self._buffer: deque = deque()
        self._stats = {
            "events_emitted": 0,
            "last_event_lag_ms": 0.0,
            "reconnects": 0,
            "dropped_count": 0,
        }

    def _enqueue(self, item: object) -> None:
        if len(self._buffer) >= self._buffer_cap:
            self._buffer.popleft()
            self._stats["dropped_count"] += 1
        self._buffer.append(item)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def __aenter__(self) -> AsyncFeedBase:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def subscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def iterate(self) -> AsyncIterator[MarketEvent]:  # pragma: no cover - overridden
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError
