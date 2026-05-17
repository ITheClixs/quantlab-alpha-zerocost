from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def test_exponential_backoff_first_attempt_returns_base() -> None:
    assert exponential_backoff(attempt=1, base=1.0, cap=60.0) == 1.0


def test_exponential_backoff_doubles_each_attempt() -> None:
    assert exponential_backoff(attempt=2, base=1.0, cap=60.0) == 2.0
    assert exponential_backoff(attempt=3, base=1.0, cap=60.0) == 4.0
    assert exponential_backoff(attempt=4, base=1.0, cap=60.0) == 8.0


def test_exponential_backoff_caps() -> None:
    assert exponential_backoff(attempt=20, base=1.0, cap=60.0) == 60.0


class _StubFeed(AsyncFeedBase):
    venue = Venue.replay

    def __init__(self) -> None:
        super().__init__()
        self.subscribed_with: tuple[str, ...] | None = None
        self.closed = False
        self._events = [
            Tick(
                venue=Venue.replay, symbol="X", timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
                received_utc=datetime(2026, 1, 1, tzinfo=UTC), price=1.0, size=1.0, side=TickSide.buy,
            ),
            Tick(
                venue=Venue.replay, symbol="X", timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
                received_utc=datetime(2026, 1, 1, tzinfo=UTC), price=2.0, size=1.0, side=TickSide.sell,
            ),
        ]

    async def subscribe(self, symbols) -> None:
        self.subscribed_with = tuple(symbols)

    async def iterate(self):
        for ev in self._events:
            self._stats["events_emitted"] += 1
            yield ev

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_async_feed_base_iterate_yields_events() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    seen = [ev async for ev in feed.iterate()]
    assert len(seen) == 2
    assert feed.subscribed_with == ("X",)


@pytest.mark.asyncio
async def test_async_feed_base_stats_track_emissions() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    _ = [ev async for ev in feed.iterate()]
    assert feed.stats["events_emitted"] == 2


@pytest.mark.asyncio
async def test_async_feed_base_close_is_called() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    await feed.close()
    assert feed.closed is True


@pytest.mark.asyncio
async def test_async_feed_base_buffer_drops_oldest_on_overflow() -> None:
    feed = _StubFeed()
    feed._buffer_cap = 3
    for i in range(5):
        feed._enqueue(f"event_{i}")
    queued = [feed._buffer.popleft() for _ in range(len(feed._buffer))]
    assert queued == ["event_2", "event_3", "event_4"]
    assert feed.stats["dropped_count"] == 2


def test_feed_connection_error_is_runtime_error_subclass() -> None:
    assert issubclass(FeedConnectionError, RuntimeError)
