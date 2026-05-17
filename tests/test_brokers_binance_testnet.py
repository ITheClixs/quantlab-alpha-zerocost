from __future__ import annotations

from quant_research_stack.brokers.binance_testnet import build_order_payload
from quant_research_stack.brokers.order_types import OrderIntent


def _intent(**overrides) -> OrderIntent:
    base = {
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "buy", "type": "market", "quantity": 0.1,
    }
    base.update(overrides)
    return OrderIntent.model_validate(base)


def test_market_payload_uses_uppercase_side() -> None:
    payload = build_order_payload(_intent())
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "BUY"
    assert payload["type"] == "MARKET"
    assert payload["quantity"] == "0.1"


def test_limit_payload_includes_price_and_tif() -> None:
    payload = build_order_payload(_intent(type="limit", limit_price=50000.0, time_in_force="gtc"))
    assert payload["type"] == "LIMIT"
    assert payload["price"] == "50000.0"
    assert payload["timeInForce"] == "GTC"


def test_oco_payload_has_both_legs() -> None:
    payload = build_order_payload(_intent(
        type="oco", oco_limit_price=60000.0, oco_stop_price=40000.0,
    ))
    assert payload["type"] == "OCO"
    assert payload["price"] == "60000.0"
    assert payload["stopPrice"] == "40000.0"


def test_client_order_id_passed_through() -> None:
    payload = build_order_payload(_intent())
    assert payload["newClientOrderId"] == "co-12345678"
