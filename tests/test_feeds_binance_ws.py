from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.feeds.binance_ws import parse_aggtrade_event
from quant_research_stack.feeds.market_types import TickSide, Venue


def _payload(**overrides) -> dict:
    base = {
        "e": "aggTrade",
        "E": 1747449600000,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "65000.50",
        "q": "0.125",
        "f": 100,
        "l": 105,
        "T": 1747449599000,
        "m": False,  # buyer is maker → trade direction = buy
        "M": True,
    }
    base.update(overrides)
    return base


def test_parse_aggtrade_basic() -> None:
    tick = parse_aggtrade_event(_payload(), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.venue == Venue.binance
    assert tick.symbol == "BTCUSDT"
    assert tick.price == 65000.50
    assert tick.size == 0.125


def test_parse_aggtrade_buyer_maker_is_sell_side() -> None:
    tick = parse_aggtrade_event(_payload(m=True), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.sell


def test_parse_aggtrade_buyer_not_maker_is_buy_side() -> None:
    tick = parse_aggtrade_event(_payload(m=False), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.buy


def test_parse_aggtrade_uses_T_for_timestamp() -> None:
    tick = parse_aggtrade_event(_payload(T=1747449500000), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.timestamp_utc == datetime.fromtimestamp(1747449500.0, tz=UTC)


def test_parse_aggtrade_sequence_from_a_field() -> None:
    tick = parse_aggtrade_event(_payload(a=987654321), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.sequence == 987654321
