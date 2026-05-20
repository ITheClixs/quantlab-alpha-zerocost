from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class RecordedFeedHeartbeat:
    """Reads latest event timestamps from S3 recorder parquet output."""

    root: Path

    def last_tick_ts(self, symbol: str) -> datetime | None:
        if not self.root.exists():
            return None
        latest: datetime | None = None
        for path in self.root.glob(f"*/{symbol}/*/*.parquet"):
            try:
                df = pl.read_parquet(path, columns=["timestamp_utc"])
            except Exception:  # noqa: BLE001
                continue
            if df.height == 0:
                continue
            for value in df["timestamp_utc"].to_list():
                ts = _as_utc(value)
                if latest is None or ts > latest:
                    latest = ts
        return latest
