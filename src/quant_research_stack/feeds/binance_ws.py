from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import websockets

from quant_research_stack.crypto_research.perps.events import normalize_book_ticker, normalize_depth_update
from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

_PUBLIC_URL = "wss://stream.binance.com:9443/ws"


def parse_aggtrade_event(payload: dict, *, received_utc: datetime) -> Tick:
    """Pure parser for Binance @aggTrade messages. Pure function for fixture tests."""
    side = TickSide.sell if payload.get("m") else TickSide.buy
    return Tick(
        venue=Venue.binance,
        symbol=str(payload["s"]),
        timestamp_utc=datetime.fromtimestamp(int(payload["T"]) / 1000.0, tz=UTC),
        received_utc=received_utc,
        price=float(payload["p"]),
        size=float(payload["q"]),
        side=side,
        sequence=int(payload["a"]),
    )


def parse_book_ticker_event(payload: dict, *, received_utc: datetime) -> dict:
    return normalize_book_ticker(payload, received_utc=received_utc)


def parse_depth_update_event(payload: dict, *, received_utc: datetime) -> dict:
    return normalize_depth_update(payload, received_utc=received_utc)


@dataclass
class BinanceWS(AsyncFeedBase):
    url: str = _PUBLIC_URL
    venue: Venue = Venue.binance

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._ws: Any | None = None
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(s.lower() for s in symbols)

    async def _connect(self) -> None:
        streams = "/".join(f"{s}@aggTrade" for s in self._symbols)
        url = f"{self.url}/{streams}" if streams else self.url
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
                return
            except Exception as exc:
                if attempt >= 10:
                    raise FeedConnectionError(f"binance ws connect failed after {attempt} attempts") from exc
                self._stats["reconnects"] += 1
                await asyncio.sleep(exponential_backoff(attempt, base=1.0, cap=60.0))

    async def iterate(self) -> AsyncIterator[Tick]:
        if self._ws is None:
            await self._connect()
        while not self._closed:
            assert self._ws is not None
            try:
                msg = await self._ws.recv()
            except Exception:
                self._stats["reconnects"] += 1
                await self._connect()
                continue
            payload = json.loads(msg)
            if payload.get("e") != "aggTrade":
                continue
            received = datetime.now(UTC)
            tick = parse_aggtrade_event(payload, received_utc=received)
            self._stats["events_emitted"] += 1
            self._stats["last_event_lag_ms"] = (received - tick.timestamp_utc).total_seconds() * 1000.0
            yield tick

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
