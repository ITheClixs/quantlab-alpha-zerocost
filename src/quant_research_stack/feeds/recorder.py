from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_stack.feeds.base import FeedAdapter
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass(frozen=True)
class RecorderConfig:
    root: Path
    flush_every_n_events: int = 1024
    flush_every_seconds: float = 5.0
    keep_raw: bool = False


def _row_from_event(ev: Tick | Bar) -> dict:
    payload = ev.model_dump(mode="json")
    payload["_kind"] = "tick" if isinstance(ev, Tick) else "bar"
    return payload


def _key(ev: Tick | Bar) -> tuple[str, str, str, str]:
    ts = ev.timestamp_utc
    return (ev.venue.value, ev.symbol, ts.strftime("%Y-%m-%d"), f"{ts.hour:02d}")


class Recorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        self._cfg = cfg
        self._writers: dict[tuple[str, str, str, str], pq.ParquetWriter] = {}
        self._buffers: dict[tuple[str, str, str, str], list[dict]] = {}
        self._closed_paths: list[Path] = []
        self._events_written = 0

    def _path_for(self, key: tuple[str, str, str, str]) -> Path:
        venue, symbol, date, hh = key
        return Path(self._cfg.root) / venue / symbol / date / f"{hh}.parquet"

    def _flush(self, key: tuple[str, str, str, str]) -> None:
        rows = self._buffers.get(key, [])
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        writer = self._writers.get(key)
        if writer is None:
            path = self._path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            self._writers[key] = writer
        writer.write_table(table)
        self._buffers[key] = []

    def _close_writer(self, key: tuple[str, str, str, str]) -> None:
        writer = self._writers.pop(key, None)
        if writer is None:
            return
        writer.close()
        path = self._path_for(key)
        if path.exists():
            os.chmod(path, path.stat().st_mode & ~0o222)
            self._closed_paths.append(path)

    async def run(self, adapter: FeedAdapter) -> None:
        last_key: tuple[str, str, str, str] | None = None
        async for ev in adapter.iterate():
            key = _key(ev)
            if last_key is not None and key[2:] != last_key[2:]:
                # date or hour rolled over for this venue+symbol stream
                self._flush(last_key)
                self._close_writer(last_key)
            self._buffers.setdefault(key, []).append(_row_from_event(ev))
            self._events_written += 1
            if len(self._buffers[key]) >= self._cfg.flush_every_n_events:
                self._flush(key)
            last_key = key
        for k in list(self._writers.keys()):
            self._flush(k)
            self._close_writer(k)
        for k in list(self._buffers.keys()):
            if self._buffers[k]:
                self._flush(k)
                self._close_writer(k)

    def stats(self) -> dict:
        return {
            "events_written": self._events_written,
            "files_closed": len(self._closed_paths),
        }
