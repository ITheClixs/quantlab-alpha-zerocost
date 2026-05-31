from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.brokers.order_types import OrderIntent, OrderStatus
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _intent(symbol: str = "BTCUSDT") -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": symbol,
        "side": "buy", "type": "market", "quantity": 1.0,
    })


def _tick(price: float = 100.0) -> Tick:
    ts = datetime(2026, 5, 17, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


@pytest.mark.asyncio
async def test_place_order_returns_accepted_order_with_deterministic_id() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    order_a = await broker.place_order(_intent())
    broker.push_market_event(_tick())
    order_b = await broker.place_order(_intent())
    assert order_a.broker_order_id == "null-0000001"
    assert order_b.broker_order_id == "null-0000002"


@pytest.mark.asyncio
async def test_place_order_synthesizes_fill_when_market_event_available() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick(price=100.0))
    order = await broker.place_order(_intent())
    assert order.status == OrderStatus.filled
    assert order.filled_quantity == 1.0


@pytest.mark.asyncio
async def test_cancel_order_marks_canceled() -> None:
    fm = FillModel(FillModelConfig())
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    order = await broker.place_order(_intent())
    canceled = await broker.cancel_order(order.client_order_id)
    assert canceled.status == OrderStatus.canceled


@pytest.mark.asyncio
async def test_positions_reflect_fills() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick(price=100.0))
    await broker.place_order(_intent())
    positions = await broker.positions()
    btc = next(p for p in positions if p.symbol == "BTCUSDT")
    assert btc.quantity == 1.0
    assert btc.avg_entry_price == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_stream_fills_yields_each_fill_once() -> None:
    fm = FillModel(FillModelConfig())
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    await broker.place_order(_intent())
    seen = []
    async for fill in broker.stream_fills():
        seen.append(fill)
        if len(seen) == 1:
            break
    assert len(seen) == 1
