from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from quant_research_stack.brokers.binance_testnet import BinanceTestnet
from quant_research_stack.brokers.order_types import OrderIntent


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_place_cancel_get_roundtrip() -> None:
    if not Path("~/.binance/testnet_keys.json").expanduser().exists():
        pytest.skip("binance testnet credentials not present")
    broker = BinanceTestnet()
    intent = OrderIntent.model_validate({
        "client_order_id": f"it-{uuid.uuid4().hex[:12]}",
        "symbol": "BTCUSDT", "side": "buy", "type": "limit",
        "limit_price": 1.0, "quantity": 0.001, "time_in_force": "gtc",
    })
    try:
        await broker.place_order(intent)
        canceled = await broker.cancel_order(intent.client_order_id)
        assert canceled.status.value == "canceled"
    finally:
        await broker.close()
