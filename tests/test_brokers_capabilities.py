from __future__ import annotations

import pytest

from quant_research_stack.brokers.capabilities import BrokerCapabilities, UnsupportedOrderError, ensure_supported
from quant_research_stack.brokers.order_types import OrderType, TimeInForce


def _caps(types: set[OrderType]) -> BrokerCapabilities:
    return BrokerCapabilities(
        venue="x",
        supported_order_types=frozenset(types),
        supported_time_in_force=frozenset({TimeInForce.day, TimeInForce.gtc}),
        supports_shorting=True,
        supports_fractional_shares=True,
        supports_extended_hours=True,
        max_orders_per_second=10,
        paper_only=True,
    )


def test_ensure_supported_passes_when_supported() -> None:
    caps = _caps({OrderType.market, OrderType.limit})
    ensure_supported(caps, OrderType.limit)


def test_ensure_supported_raises_on_unsupported() -> None:
    caps = _caps({OrderType.market})
    with pytest.raises(UnsupportedOrderError) as exc:
        ensure_supported(caps, OrderType.oco)
    assert "x" in str(exc.value)
    assert "oco" in str(exc.value)


def test_unsupported_order_error_includes_suggestion() -> None:
    caps = _caps({OrderType.market, OrderType.limit})
    with pytest.raises(UnsupportedOrderError) as exc:
        ensure_supported(caps, OrderType.bracket)
    assert "market" in str(exc.value).lower() or "limit" in str(exc.value).lower()
