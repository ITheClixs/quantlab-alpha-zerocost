from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.backtest.strategies.buy_and_hold import BuyAndHold
from quant_research_stack.backtest.strategies.moving_average_cross import MovingAverageCross
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _tick(price: float, minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


def test_buy_and_hold_emits_one_order_on_first_event_and_none_after() -> None:
    s = BuyAndHold(symbol="BTCUSDT", quantity=1.0)
    first = s.on_event(_tick(100.0, 0))
    second = s.on_event(_tick(101.0, 1))
    assert len(first) == 1
    assert first[0].side.value == "buy"
    assert first[0].quantity == 1.0
    assert second == []


def test_moving_average_cross_does_not_trade_before_window_fills() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    assert s.on_event(_tick(100.0, 0)) == []
    assert s.on_event(_tick(101.0, 1)) == []


def test_moving_average_cross_emits_buy_when_fast_crosses_above_slow() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    s.on_event(_tick(100.0, 0))
    s.on_event(_tick(99.0, 1))
    s.on_event(_tick(98.0, 2))  # slow window now full
    orders = s.on_event(_tick(105.0, 3))  # big jump → fast > slow
    assert orders and orders[0].side.value == "buy"


def test_moving_average_cross_emits_sell_when_fast_crosses_below_slow() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    s.on_event(_tick(100.0, 0))
    s.on_event(_tick(101.0, 1))
    s.on_event(_tick(102.0, 2))
    s.on_event(_tick(103.0, 3))  # uptrend, fast > slow
    orders = s.on_event(_tick(80.0, 4))  # big drop → fast < slow
    assert orders and orders[0].side.value == "sell"
