from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Bar, Venue

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def parse_bars_response(response: dict, *, interval_seconds: int) -> Iterator[Bar]:
    bars = response.get("bars") or {}
    for symbol, rows in bars.items():
        for row in rows:
            iso = str(row["t"]).replace("Z", "+00:00")
            yield Bar(
                venue=Venue.alpaca,
                symbol=symbol,
                timestamp_utc=datetime.fromisoformat(iso).astimezone(UTC),
                interval_seconds=interval_seconds,
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
                n_trades=int(row["n"]) if "n" in row else None,
            )


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class AlpacaREST(AsyncFeedBase):
    credentials_path: str = "~/.alpaca/paper_keys.json"
    interval_seconds: int = 900
    poll_offset_seconds: int = 5
    base_url: str = _BARS_URL
    venue: Venue = Venue.alpaca

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(symbols)

    async def iterate(self) -> AsyncIterator[Bar]:
        api_key, api_secret = _load_credentials(self.credentials_path)
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            while not self._closed:
                params: dict[str, str | int] = {
                    "symbols": ",".join(self._symbols),
                    "timeframe": f"{self.interval_seconds // 60}Min",
                    "limit": 1000,
                }
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                payload = response.json()
                for bar in parse_bars_response(payload, interval_seconds=self.interval_seconds):
                    self._stats["events_emitted"] += 1
                    yield bar
                await asyncio.sleep(self.interval_seconds + self.poll_offset_seconds)

    async def close(self) -> None:
        self._closed = True
