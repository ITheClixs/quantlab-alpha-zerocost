from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.feeds.binance_ws import (
    parse_aggtrade_event,
    parse_book_ticker_event,
    parse_depth_update_event,
)
from quant_research_stack.feeds.market_types import TickSide


def test_parse_aggtrade_event_existing_tick_behavior_is_preserved() -> None:
    payload = {
        "e": "aggTrade",
        "E": 1710000000123,
        "s": "BTCUSDT",
        "a": 101,
        "p": "70000.5",
        "q": "0.25",
        "T": 1710000000100,
        "m": True,
    }

    tick = parse_aggtrade_event(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))

    assert tick.symbol == "BTCUSDT"
    assert tick.timestamp_utc == datetime.fromtimestamp(1710000000.100, tz=UTC)
    assert tick.price == 70000.5
    assert tick.size == 0.25
    assert tick.side == TickSide.sell
    assert tick.sequence == 101


def test_parse_book_ticker_event_wraps_normalizer() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    payload = {
        "e": "bookTicker",
        "E": 1710000000300,
        "u": 400900217,
        "s": "ETHUSDT",
        "b": "3500.10",
        "B": "12.5",
        "a": "3500.20",
        "A": "11.0",
    }

    row = parse_book_ticker_event(payload, received_utc=received_utc)

    assert row["event_type"] == "book_ticker"
    assert row["event_time"] == datetime.fromtimestamp(1710000000.300, tz=UTC)
    assert row["update_id"] == 400900217
    assert row["best_ask"] > row["best_bid"]


def test_parse_depth_update_event_wraps_normalizer() -> None:
    payload = {
        "e": "depthUpdate",
        "E": 1710000000200,
        "s": "BTCUSDT",
        "U": 157,
        "u": 160,
        "b": [["70000.0", "1.0"], ["69999.5", "0.5"]],
        "a": [["70000.5", "0.75"]],
    }

    row = parse_depth_update_event(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))

    assert row["event_type"] == "depth_update"
    assert row["first_update_id"] == 157
    assert row["last_update_id"] == 160
    assert row["bids"] == [[70000.0, 1.0], [69999.5, 0.5]]
    assert row["asks"] == [[70000.5, 0.75]]
