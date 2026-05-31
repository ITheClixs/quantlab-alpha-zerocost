from __future__ import annotations

import asyncio

import pytest

from quant_research_stack.feeds.binance_ws import BinanceWS


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_binance_ws_emits_at_least_one_event_in_60_seconds() -> None:
    feed = BinanceWS()
    await feed.subscribe(["BTCUSDT"])
    seen = 0
    async def consume():
        nonlocal seen
        async for _ in feed.iterate():
            seen += 1
            if seen >= 1:
                return
    try:
        await asyncio.wait_for(consume(), timeout=60.0)
    finally:
        await feed.close()
    assert seen >= 1
