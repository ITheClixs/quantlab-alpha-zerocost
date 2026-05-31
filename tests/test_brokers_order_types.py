from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.brokers.order_types import (
    Account,
    Fill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


def _intent(**overrides) -> dict:
    base = {
        "client_order_id": "co-12345678",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "quantity": 0.5,
    }
    base.update(overrides)
    return base


def test_market_order_intent_valid() -> None:
    o = OrderIntent.model_validate(_intent())
    assert o.type == OrderType.market
    assert o.side == OrderSide.buy


def test_limit_order_requires_limit_price() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="limit"))


def test_stop_order_requires_stop_price() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop"))


def test_stop_limit_requires_both_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop_limit", limit_price=100.0))
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop_limit", stop_price=100.0))


def test_bracket_requires_three_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="bracket", limit_price=100.0, take_profit_price=110.0))


def test_oco_requires_both_oco_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="oco", oco_limit_price=100.0))


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(quantity=0.0))


def test_client_order_id_min_length() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(client_order_id="short"))


def test_order_status_enum_values() -> None:
    assert OrderStatus.accepted.value == "accepted"
    assert OrderStatus.filled.value == "filled"
    assert OrderStatus.canceled.value == "canceled"


def test_time_in_force_default_day() -> None:
    o = OrderIntent.model_validate(_intent())
    assert o.time_in_force == TimeInForce.day


def test_account_basic() -> None:
    a = Account.model_validate({"equity": 1000.0, "cash": 500.0, "buying_power": 2000.0})
    assert a.currency == "USD"


def test_fill_basic() -> None:
    f = Fill.model_validate({
        "client_order_id": "co-12345678", "fill_id": "f1", "symbol": "BTCUSDT",
        "side": "buy", "price": 100.0, "quantity": 0.5,
        "timestamp_utc": datetime(2026, 5, 17, tzinfo=UTC),
    })
    assert f.commission == 0.0


def test_position_signed_quantity() -> None:
    p = Position.model_validate({
        "symbol": "BTCUSDT", "quantity": -0.5, "avg_entry_price": 100.0,
        "market_value": -50.0, "unrealized_pnl": 5.0,
    })
    assert p.quantity == -0.5
