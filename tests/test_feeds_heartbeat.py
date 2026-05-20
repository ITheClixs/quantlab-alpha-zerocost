from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.feeds.heartbeat import RecordedFeedHeartbeat


def test_recorded_feed_heartbeat_reads_latest_timestamp_for_symbol(tmp_path: Path) -> None:
    root = tmp_path / "parquet"
    path = root / "binance" / "BTCUSDT" / "2026-05-20" / "13.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "timestamp_utc": [
            "2026-05-20T13:35:00+00:00",
            "2026-05-20T13:36:00+00:00",
        ],
        "price": [50000.0, 50001.0],
    }).write_parquet(path)

    heartbeat = RecordedFeedHeartbeat(root)

    assert heartbeat.last_tick_ts("BTCUSDT") == datetime(2026, 5, 20, 13, 36, tzinfo=UTC)


def test_recorded_feed_heartbeat_returns_none_for_missing_symbol(tmp_path: Path) -> None:
    heartbeat = RecordedFeedHeartbeat(tmp_path / "parquet")

    assert heartbeat.last_tick_ts("ETHUSDT") is None
