from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.feeds.alpaca_rest import AlpacaREST
from quant_research_stack.feeds.binance_ws import BinanceWS
from quant_research_stack.feeds.coinbase_ws import CoinbaseWS
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3 recorder daemon — tail live feeds and write Parquet shards.")
    p.add_argument("--config", default="configs/feeds.yaml")
    return p.parse_args()


def _build_adapter(spec: dict):
    impl = spec["impl"]
    if impl == "BinanceWS":
        feed = BinanceWS()
    elif impl == "CoinbaseWS":
        feed = CoinbaseWS()
    elif impl == "AlpacaREST":
        feed = AlpacaREST(
            credentials_path=spec.get("credentials_path", "~/.alpaca/paper_keys.json"),
            interval_seconds=int(spec.get("interval_minutes", 15)) * 60,
            poll_offset_seconds=int(spec.get("poll_offset_seconds", 5)),
        )
    else:
        raise ValueError(f"unknown feed impl: {impl}")
    return feed


async def _run(cfg: dict) -> None:
    recorder = Recorder(RecorderConfig(
        root=Path(cfg["recorder"]["root"]),
        flush_every_n_events=int(cfg["recorder"]["flush_every_n_events"]),
        flush_every_seconds=float(cfg["recorder"]["flush_every_seconds"]),
        keep_raw=bool(cfg["recorder"]["keep_raw"]),
    ))
    tasks = []
    for spec in cfg["adapters"]:
        feed = _build_adapter(spec)
        await feed.subscribe(spec["symbols"])
        tasks.append(asyncio.create_task(recorder.run(feed)))
    await asyncio.gather(*tasks)


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        console.print("[yellow]recorder draining on SIGINT[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
