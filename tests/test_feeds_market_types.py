from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.feeds.market_types import Bar, Tick, TickSide, Venue


def _now() -> datetime:
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)


def _valid_tick(**overrides) -> dict:
    base = {
        "venue": "binance",
        "symbol": "BTCUSDT",
        "timestamp_utc": _now(),
        "received_utc": _now(),
        "price": 100.0,
        "size": 0.5,
        "side": "buy",
    }
    base.update(overrides)
    return base


def test_tick_minimal_valid() -> None:
    t = Tick.model_validate(_valid_tick())
    assert t.venue == Venue.binance
    assert t.side == TickSide.buy
    assert t.price == 100.0


def test_tick_price_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Tick.model_validate(_valid_tick(price=0.0))


def test_tick_size_may_be_zero() -> None:
    t = Tick.model_validate(_valid_tick(size=0.0))
    assert t.size == 0.0


def test_tick_size_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        Tick.model_validate(_valid_tick(size=-1.0))


def test_tick_is_frozen() -> None:
    t = Tick.model_validate(_valid_tick())
    with pytest.raises(ValueError):
        t.price = 999.0


def test_tick_round_trip_json() -> None:
    t = Tick.model_validate(_valid_tick())
    payload = t.model_dump_json()
    restored = Tick.model_validate_json(payload)
    assert restored == t


def test_bar_minimal_valid() -> None:
    b = Bar.model_validate({
        "venue": "alpaca",
        "symbol": "SPY",
        "timestamp_utc": _now(),
        "interval_seconds": 900,
        "open": 500.0,
        "high": 501.0,
        "low": 499.0,
        "close": 500.5,
        "volume": 1000.0,
    })
    assert b.interval_seconds == 900
    assert b.high == 501.0


def test_bar_interval_zero_rejected() -> None:
    with pytest.raises(ValueError):
        Bar.model_validate({
            "venue": "alpaca",
            "symbol": "SPY",
            "timestamp_utc": _now(),
            "interval_seconds": 0,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 0.0,
        })


def test_bar_is_frozen() -> None:
    b = Bar.model_validate({
        "venue": "alpaca",
        "symbol": "SPY",
        "timestamp_utc": _now(),
        "interval_seconds": 900,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 0.0,
    })
    with pytest.raises(ValueError):
        b.close = 2.0


def test_tick_side_enum_values() -> None:
    assert TickSide.buy.value == "buy"
    assert TickSide.sell.value == "sell"
    assert TickSide.unknown.value == "unknown"


def test_venue_enum_values() -> None:
    assert Venue.binance.value == "binance"
    assert Venue.coinbase.value == "coinbase"
    assert Venue.alpaca.value == "alpaca"
    assert Venue.replay.value == "replay"
