from __future__ import annotations

from quant_research_stack.feeds.alpaca_rest import parse_bars_response
from quant_research_stack.feeds.market_types import Venue


def _resp() -> dict:
    return {
        "bars": {
            "SPY": [
                {"t": "2026-05-17T13:30:00Z", "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 1000.0, "n": 50},
                {"t": "2026-05-17T13:45:00Z", "o": 500.5, "h": 502.0, "l": 500.0, "c": 501.5, "v": 1200.0, "n": 60},
            ],
        },
        "next_page_token": None,
    }


def test_parse_bars_yields_bar_per_input() -> None:
    bars = list(parse_bars_response(_resp(), interval_seconds=900))
    assert len(bars) == 2
    assert bars[0].symbol == "SPY"
    assert bars[0].venue == Venue.alpaca
    assert bars[0].interval_seconds == 900
    assert bars[0].open == 500.0


def test_parse_bars_handles_empty_symbol_list() -> None:
    bars = list(parse_bars_response({"bars": {}, "next_page_token": None}, interval_seconds=900))
    assert bars == []


def test_parse_bars_handles_n_trades_missing() -> None:
    resp = {
        "bars": {"AAPL": [{"t": "2026-05-17T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}]},
        "next_page_token": None,
    }
    bars = list(parse_bars_response(resp, interval_seconds=900))
    assert bars[0].n_trades is None
