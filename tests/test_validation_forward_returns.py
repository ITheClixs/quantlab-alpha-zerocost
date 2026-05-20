from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.validation.forward_returns import (
    Bar,
    ForwardReturnRequest,
    align_horizon_to_bar,
    fetch_forward_returns,
)


def _bar(ts: datetime, close: float) -> Bar:
    return Bar(symbol="AAPL", ts_utc=ts, open=close, high=close, low=close, close=close, volume=0)


def test_align_horizon_ceil_to_next_bar_5min() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 42, tzinfo=UTC)
    target = align_horizon_to_bar(
        fill_ts=fill_ts, horizon_minutes=5, bar_interval_minutes=1, mode="ceil_to_next_bar",
    )
    assert target == datetime(2026, 5, 20, 13, 41, 0, tzinfo=UTC)


def test_align_horizon_floor_to_next_bar_5min() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 42, tzinfo=UTC)
    target = align_horizon_to_bar(
        fill_ts=fill_ts, horizon_minutes=5, bar_interval_minutes=1, mode="floor_to_next_bar",
    )
    assert target == datetime(2026, 5, 20, 13, 40, 0, tzinfo=UTC)


def test_align_horizon_zero_seconds_exact_boundary() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 0, tzinfo=UTC)
    target = align_horizon_to_bar(
        fill_ts=fill_ts, horizon_minutes=5, bar_interval_minutes=1, mode="ceil_to_next_bar",
    )
    assert target == datetime(2026, 5, 20, 13, 40, 0, tzinfo=UTC)


def test_fetch_forward_returns_uses_close_diff() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 0, tzinfo=UTC)
    horizon_ts = datetime(2026, 5, 20, 13, 40, 0, tzinfo=UTC)
    fixture_bars = {
        ("AAPL", fill_ts): _bar(fill_ts, close=100.0),
        ("AAPL", horizon_ts): _bar(horizon_ts, close=100.5),
    }

    def stub_loader(symbol: str, ts: datetime) -> Bar | None:
        return fixture_bars.get((symbol, ts))

    req = ForwardReturnRequest(
        signal_id="sig-1", symbol="AAPL", fill_ts_utc=fill_ts, horizon_minutes=5,
    )
    [out] = fetch_forward_returns(
        [req], bar_loader=stub_loader, horizon_alignment="ceil_to_next_bar",
    )
    assert out.signal_id == "sig-1"
    assert out.realized_return == pytest.approx(0.005, abs=1e-9)
    assert out.realized_direction == 1


def test_fetch_forward_returns_returns_nan_when_horizon_bar_missing() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 0, tzinfo=UTC)
    fixture_bars = {("AAPL", fill_ts): _bar(fill_ts, close=100.0)}

    def stub_loader(symbol: str, ts: datetime) -> Bar | None:
        return fixture_bars.get((symbol, ts))

    req = ForwardReturnRequest(
        signal_id="sig-2", symbol="AAPL", fill_ts_utc=fill_ts, horizon_minutes=5,
    )
    [out] = fetch_forward_returns(
        [req], bar_loader=stub_loader, horizon_alignment="ceil_to_next_bar",
    )
    assert out.realized_return != out.realized_return  # NaN check
    assert out.realized_direction == 0


def test_fetch_forward_returns_negative_return_direction_minus_one() -> None:
    fill_ts = datetime(2026, 5, 20, 13, 35, 0, tzinfo=UTC)
    horizon_ts = datetime(2026, 5, 20, 13, 40, 0, tzinfo=UTC)
    fixture_bars = {
        ("AAPL", fill_ts): _bar(fill_ts, close=100.0),
        ("AAPL", horizon_ts): _bar(horizon_ts, close=99.0),
    }

    def stub_loader(symbol: str, ts: datetime) -> Bar | None:
        return fixture_bars.get((symbol, ts))

    req = ForwardReturnRequest(
        signal_id="sig-3", symbol="AAPL", fill_ts_utc=fill_ts, horizon_minutes=5,
    )
    [out] = fetch_forward_returns(
        [req], bar_loader=stub_loader, horizon_alignment="ceil_to_next_bar",
    )
    assert out.realized_return == pytest.approx(-0.01, abs=1e-9)
    assert out.realized_direction == -1


def test_fetch_forward_returns_floor_mode_alignment() -> None:
    """End-to-end: floor_to_next_bar must use the bar BEFORE fill_ts+horizon."""
    fill_ts = datetime(2026, 5, 20, 13, 35, 42, tzinfo=UTC)  # 42 seconds past 13:35
    floor_horizon_ts = datetime(2026, 5, 20, 13, 40, 0, tzinfo=UTC)
    entry_ts = datetime(2026, 5, 20, 13, 35, 0, tzinfo=UTC)
    fixture_bars = {
        ("AAPL", entry_ts): _bar(entry_ts, close=100.0),
        ("AAPL", floor_horizon_ts): _bar(floor_horizon_ts, close=101.0),
    }

    def stub_loader(symbol: str, ts: datetime) -> Bar | None:
        return fixture_bars.get((symbol, ts))

    req = ForwardReturnRequest(
        signal_id="sig-floor", symbol="AAPL", fill_ts_utc=fill_ts, horizon_minutes=5,
    )
    [out] = fetch_forward_returns(
        [req], bar_loader=stub_loader, horizon_alignment="floor_to_next_bar",
    )
    assert out.realized_return == pytest.approx(0.01, abs=1e-9)
    assert out.realized_direction == 1
