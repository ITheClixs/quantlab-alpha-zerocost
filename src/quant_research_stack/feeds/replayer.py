from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Bar, Tick, Venue


@dataclass(frozen=True)
class ReplayerConfig:
    root: Path
    venue: Venue
    symbols: tuple[str, ...]
    start_utc: datetime
    end_utc: datetime
    speed: float = 1.0


@dataclass
class Replayer(AsyncFeedBase):
    cfg: ReplayerConfig

    def __post_init__(self) -> None:
        super().__init__()
        self.venue = self.cfg.venue
        self._closed = False

    def _shard_paths(self) -> list[Path]:
        out: list[Path] = []
        for symbol in self.cfg.symbols:
            symbol_dir = Path(self.cfg.root) / self.cfg.venue.value / symbol
            if not symbol_dir.exists():
                continue
            for shard in sorted(symbol_dir.rglob("*.parquet")):
                out.append(shard)
        return out

    async def subscribe(self, symbols) -> None:
        return None

    async def iterate(self) -> AsyncIterator[Tick | Bar]:
        prev_ts: datetime | None = None
        for shard in self._shard_paths():
            df = pl.read_parquet(shard).sort("timestamp_utc")
            for row in df.iter_rows(named=True):
                ts = row["timestamp_utc"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts < self.cfg.start_utc or ts >= self.cfg.end_utc:
                    continue
                if self.cfg.speed > 0.0 and prev_ts is not None:
                    delta = (ts - prev_ts).total_seconds() / self.cfg.speed
                    if delta > 0:
                        await asyncio.sleep(delta)
                prev_ts = ts
                kind = row.get("_kind", "tick")
                payload = {k: v for k, v in row.items() if k != "_kind"}
                event: Tick | Bar = Tick.model_validate(payload) if kind == "tick" else Bar.model_validate(payload)
                self._stats["events_emitted"] += 1
                yield event

    async def close(self) -> None:
        self._closed = True
