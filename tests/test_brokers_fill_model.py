from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.order_types import OrderIntent
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _tick(price: float, ts: datetime) -> Tick:
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


def _intent_buy(qty: float = 1.0) -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "buy", "type": "market", "quantity": qty,
    })


def _intent_sell(qty: float = 1.0) -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "sell", "type": "market", "quantity": qty,
    })


def test_buy_market_fill_includes_half_spread_and_slippage_adverse() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_buy(qty=1.0), market)
    assert len(fills) == 1
    expected_px = 100.0 + 100.0 * (1.0 + 2.0) * 1e-4  # half spread + slippage on buy
    assert fills[0].price == pytest.approx(expected_px, rel=1e-9)


def test_sell_market_fill_is_adverse_in_opposite_direction() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_sell(qty=1.0), market)
    expected_px = 100.0 - 100.0 * (1.0 + 2.0) * 1e-4
    assert fills[0].price == pytest.approx(expected_px, rel=1e-9)


def test_commission_uses_bps_of_notional() -> None:
    cfg = FillModelConfig(commission_bps=2.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_buy(qty=2.0), market)
    notional = fills[0].price * fills[0].quantity
    expected_commission = notional * 2.0 * 1e-4
    assert fills[0].commission == pytest.approx(expected_commission, rel=1e-9)


def test_no_market_events_returns_empty() -> None:
    cfg = FillModelConfig()
    fm = FillModel(cfg)
    fills = fm.synthesize(_intent_buy(), iter([]))
    assert fills == []


def test_deterministic_across_runs() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm1 = FillModel(cfg)
    fm2 = FillModel(cfg)
    market_a = [_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))]
    market_b = [_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))]
    fills_a = fm1.synthesize(_intent_buy(), iter(market_a))
    fills_b = fm2.synthesize(_intent_buy(), iter(market_b))
    assert fills_a[0].price == fills_b[0].price
    assert fills_a[0].commission == fills_b[0].commission
