from __future__ import annotations

from quant_research_stack.brokers.alpaca_paper import build_order_payload
from quant_research_stack.brokers.order_types import OrderIntent


def _intent(**overrides) -> OrderIntent:
    base = {
        "client_order_id": "co-12345678", "symbol": "SPY",
        "side": "buy", "type": "market", "quantity": 10.0,
    }
    base.update(overrides)
    return OrderIntent.model_validate(base)


def test_market_payload_has_required_fields() -> None:
    payload = build_order_payload(_intent())
    assert payload["symbol"] == "SPY"
    assert payload["side"] == "buy"
    assert payload["type"] == "market"
    assert payload["qty"] == "10.0"
    assert payload["client_order_id"] == "co-12345678"


def test_limit_payload_includes_limit_price() -> None:
    payload = build_order_payload(_intent(type="limit", limit_price=500.0))
    assert payload["type"] == "limit"
    assert payload["limit_price"] == "500.0"


def test_stop_payload_includes_stop_price() -> None:
    payload = build_order_payload(_intent(type="stop", stop_price=490.0))
    assert payload["type"] == "stop"
    assert payload["stop_price"] == "490.0"


def test_bracket_payload_includes_take_profit_and_stop_loss() -> None:
    payload = build_order_payload(_intent(
        type="bracket", limit_price=500.0, take_profit_price=520.0, stop_loss_price=480.0,
    ))
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"]["limit_price"] == "520.0"
    assert payload["stop_loss"]["stop_price"] == "480.0"
