from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.feeds.coinbase_ws import parse_match_event
from quant_research_stack.feeds.market_types import TickSide, Venue


def _payload(**overrides) -> dict:
    base = {
        "type": "match",
        "trade_id": 42,
        "maker_order_id": "abc",
        "taker_order_id": "def",
        "side": "buy",
        "size": "0.10",
        "price": "65000.00",
        "product_id": "BTC-USD",
        "sequence": 1234567890,
        "time": "2026-05-17T12:00:00.123456Z",
    }
    base.update(overrides)
    return base


def test_parse_match_basic() -> None:
    tick = parse_match_event(_payload(), received_utc=datetime(2026, 5, 17, 12, 0, 1, tzinfo=UTC))
    assert tick.venue == Venue.coinbase
    assert tick.symbol == "BTC-USD"
    assert tick.price == 65000.0
    assert tick.size == 0.10


def test_parse_match_side_buy() -> None:
    tick = parse_match_event(_payload(side="buy"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.buy


def test_parse_match_side_sell() -> None:
    tick = parse_match_event(_payload(side="sell"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.sell


def test_parse_match_unknown_side_falls_back_to_unknown() -> None:
    tick = parse_match_event(_payload(side="weird"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.unknown


def test_parse_match_uses_sequence_field() -> None:
    tick = parse_match_event(_payload(sequence=9999), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.sequence == 9999
