from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_research_stack.feeds.market_types import Tick, TickSide, Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig


def _tick(minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=65000.0 + minute, size=0.1, side=TickSide.buy,
    )


@dataclass
class _FixtureFeed:
    venue = Venue.binance
    events: list

    async def iterate(self) -> AsyncIterator[Tick]:
        for ev in self.events:
            yield ev


@pytest.fixture
def recorded_dir(tmp_path: Path) -> Path:
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    feed = _FixtureFeed(events=[_tick(0), _tick(10), _tick(20), _tick(30)])
    asyncio.run(recorder.run(feed))
    return tmp_path


@pytest.mark.asyncio
async def test_replayer_yields_events_in_timestamp_order(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    seen = [ev async for ev in rep.iterate()]
    times = [ev.timestamp_utc for ev in seen]
    assert times == sorted(times)
    assert len(seen) == 4


@pytest.mark.asyncio
async def test_replayer_speed_zero_runs_fast(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    started = asyncio.get_event_loop().time()
    _ = [ev async for ev in rep.iterate()]
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_replayer_respects_symbol_filter(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("ETHUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    seen = [ev async for ev in rep.iterate()]
    assert seen == []
