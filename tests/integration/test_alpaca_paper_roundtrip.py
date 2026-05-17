from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from quant_research_stack.brokers.alpaca_paper import AlpacaPaper
from quant_research_stack.brokers.order_types import OrderIntent


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_place_cancel_get_roundtrip() -> None:
    if not Path("~/.alpaca/paper_keys.json").expanduser().exists():
        pytest.skip("alpaca paper credentials not present")
    broker = AlpacaPaper()
    intent = OrderIntent.model_validate({
        "client_order_id": f"it-{uuid.uuid4().hex[:12]}",
        "symbol": "SPY", "side": "buy", "type": "limit",
        "limit_price": 1.0, "quantity": 1.0, "time_in_force": "day",
    })
    try:
        order = await broker.place_order(intent)
        assert order.client_order_id == intent.client_order_id
        canceled = await broker.cancel_order(intent.client_order_id)
        assert canceled.status.value == "canceled"
    finally:
        await broker.close()
