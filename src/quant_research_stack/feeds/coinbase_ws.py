from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import websockets

from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

_PUBLIC_URL = "wss://ws-feed.exchange.coinbase.com"


def parse_match_event(payload: dict, *, received_utc: datetime) -> Tick:
    side_raw = str(payload.get("side", "")).lower()
    side = TickSide.buy if side_raw == "buy" else TickSide.sell if side_raw == "sell" else TickSide.unknown
    iso = str(payload["time"]).replace("Z", "+00:00")
    return Tick(
        venue=Venue.coinbase,
        symbol=str(payload["product_id"]),
        timestamp_utc=datetime.fromisoformat(iso).astimezone(UTC),
        received_utc=received_utc,
        price=float(payload["price"]),
        size=float(payload["size"]),
        side=side,
        sequence=int(payload["sequence"]),
    )


@dataclass
class CoinbaseWS(AsyncFeedBase):
    url: str = _PUBLIC_URL
    venue: Venue = Venue.coinbase

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._ws = None
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(symbols)

    async def _connect(self) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)
                payload = {
                    "type": "subscribe",
                    "channels": [{"name": "matches", "product_ids": list(self._symbols)}],
                }
                await self._ws.send(json.dumps(payload))
                return
            except Exception as exc:
                if attempt >= 10:
                    raise FeedConnectionError(f"coinbase ws connect failed after {attempt} attempts") from exc
                self._stats["reconnects"] += 1
                await asyncio.sleep(exponential_backoff(attempt, base=1.0, cap=60.0))

    async def iterate(self) -> AsyncIterator[Tick]:
        if self._ws is None:
            await self._connect()
        while not self._closed:
            try:
                msg = await self._ws.recv()
            except Exception:
                self._stats["reconnects"] += 1
                await self._connect()
                continue
            payload = json.loads(msg)
            if payload.get("type") != "match":
                continue
            received = datetime.now(UTC)
            tick = parse_match_event(payload, received_utc=received)
            self._stats["events_emitted"] += 1
            self._stats["last_event_lag_ms"] = (received - tick.timestamp_utc).total_seconds() * 1000.0
            yield tick

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
