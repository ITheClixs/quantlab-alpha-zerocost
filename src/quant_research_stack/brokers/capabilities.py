from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.brokers.order_types import OrderType, TimeInForce


class UnsupportedOrderError(ValueError):
    pass


@dataclass(frozen=True)
class BrokerCapabilities:
    venue: str
    supported_order_types: frozenset[OrderType]
    supported_time_in_force: frozenset[TimeInForce]
    supports_shorting: bool
    supports_fractional_shares: bool
    supports_extended_hours: bool
    max_orders_per_second: int
    paper_only: bool


def ensure_supported(caps: BrokerCapabilities, order_type: OrderType) -> None:
    if order_type in caps.supported_order_types:
        return
    suggestions = ", ".join(sorted(t.value for t in caps.supported_order_types))
    raise UnsupportedOrderError(
        f"venue {caps.venue!r} does not support {order_type.value!r}; "
        f"supported types: {suggestions}"
    )
