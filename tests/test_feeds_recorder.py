from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig


def _tick(hour: int, minute: int = 0) -> Tick:
    ts = datetime(2026, 5, 17, hour, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=65000.0, size=0.1, side=TickSide.buy,
    )


@dataclass
class _FixtureFeed(AsyncFeedBase):
    events: list[Tick]
    venue: Venue = Venue.binance

    def __post_init__(self) -> None:
        super().__init__()

    async def subscribe(self, symbols) -> None: ...

    async def iterate(self) -> AsyncIterator[Tick]:
        for ev in self.events:
            self._stats["events_emitted"] += 1
            yield ev

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_recorder_writes_one_file_per_hour(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(10, 30), _tick(11, 5)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    written = sorted(tmp_path.rglob("*.parquet"))
    assert len(written) == 2
    # path scheme: <root>/<venue>/<symbol>/<YYYY-MM-DD>/<HH>.parquet
    assert any(p.name == "10.parquet" for p in written)
    assert any(p.name == "11.parquet" for p in written)


@pytest.mark.asyncio
async def test_recorder_files_are_read_only_after_close(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(11, 0)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    written = sorted(tmp_path.rglob("*.parquet"))
    for p in written:
        assert not (p.stat().st_mode & 0o222), f"file {p} still has write bits"


@pytest.mark.asyncio
async def test_recorder_stats_track_writes(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(10, 5), _tick(11, 0)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    stats = recorder.stats()
    assert stats["events_written"] == 3
    assert stats["files_closed"] == 2
