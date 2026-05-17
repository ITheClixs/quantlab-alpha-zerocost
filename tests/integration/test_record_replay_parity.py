from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.feeds.binance_ws import BinanceWS
from quant_research_stack.feeds.market_types import Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_record_then_replay_yields_same_event_count(tmp_path: Path) -> None:
    feed = BinanceWS()
    await feed.subscribe(["BTCUSDT"])
    recorder = Recorder(RecorderConfig(root=tmp_path))
    started = datetime.now(UTC)

    async def record_for_60s() -> None:
        try:
            await asyncio.wait_for(recorder.run(feed), timeout=60.0)
        except TimeoutError:
            pass
        finally:
            await feed.close()

    await record_for_60s()
    recorded = sum(pl.read_parquet(p).height for p in tmp_path.rglob("*.parquet"))
    assert recorded >= 1, "live recording produced no events"

    rep = Replayer(ReplayerConfig(
        root=tmp_path, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=started - timedelta(seconds=5),
        end_utc=started + timedelta(seconds=120),
        speed=0.0,
    ))
    replayed = [ev async for ev in rep.iterate()]
    assert len(replayed) == recorded
    times = [ev.timestamp_utc for ev in replayed]
    assert times == sorted(times)
