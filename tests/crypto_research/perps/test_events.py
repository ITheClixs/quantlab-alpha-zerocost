from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.crypto_research.perps.events import (
    normalize_agg_trade,
    normalize_book_ticker,
    normalize_depth_update,
)


def test_normalize_agg_trade_preserves_event_time_and_aggressor_side() -> None:
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

    row = normalize_agg_trade(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))

    assert row["event_type"] == "agg_trade"
    assert row["symbol"] == "BTCUSDT"
    assert row["event_time"] == datetime.fromtimestamp(1710000000.100, tz=UTC)
    assert row["exchange_event_time"] == datetime.fromtimestamp(1710000000.123, tz=UTC)
    assert row["price"] == 70000.5
    assert row["size"] == 0.25
    assert row["aggressor_side"] == "sell"
    assert row["trade_id"] == 101


def test_normalize_book_ticker_has_positive_spread() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    payload = {
        "u": 400900217,
        "s": "ETHUSDT",
        "b": "3500.10",
        "B": "12.5",
        "a": "3500.20",
        "A": "11.0",
    }

    row = normalize_book_ticker(payload, received_utc=received_utc)

    assert row["event_type"] == "book_ticker"
    assert row["event_time"] == received_utc
    assert row["best_bid"] == 3500.10
    assert row["best_ask"] == 3500.20
    assert row["best_ask"] > row["best_bid"]


def test_normalize_depth_update_keeps_levels_and_update_ids() -> None:
    payload = {
        "e": "depthUpdate",
        "E": 1710000000200,
        "s": "BTCUSDT",
        "U": 157,
        "u": 160,
        "b": [["70000.0", "1.0"], ["69999.5", "0.5"]],
        "a": [["70000.5", "0.75"]],
    }

    row = normalize_depth_update(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))

    assert row["event_type"] == "depth_update"
    assert row["event_time"] == datetime.fromtimestamp(1710000000.200, tz=UTC)
    assert row["first_update_id"] == 157
    assert row["last_update_id"] == 160
    assert row["bids"][0] == [70000.0, 1.0]
    assert row["asks"][0] == [70000.5, 0.75]
