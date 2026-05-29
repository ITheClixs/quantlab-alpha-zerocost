from __future__ import annotations

import httpx
import pytest

from quant_research_stack.feeds.base import FeedConnectionError
from quant_research_stack.feeds.market_types import Venue
from quant_research_stack.feeds.massive_rest import (
    MassiveREST,
    NotAuthorizedError,
    bars_to_dataframe,
    parse_market_status,
    parse_previous_close,
    upsert_panel,
)

# Polygon-compatible response fixtures (verified shapes from api.massive.com).
_MARKET_STATUS = {
    "afterHours": False,
    "earlyHours": False,
    "market": "open",
    "serverTime": "2026-05-29T15:30:00-04:00",
    "exchanges": {"nyse": "open", "nasdaq": "open", "otc": "open"},
    "currencies": {"crypto": "open", "fx": "open"},
}
_PREV_CLOSE = {
    "ticker": "SPY",
    "status": "OK",
    "resultsCount": 1,
    "adjusted": True,
    "results": [
        {"T": "SPY", "o": 500.0, "h": 503.0, "l": 499.0, "c": 502.5, "v": 1.2e8,
         "vw": 501.7, "t": 1748448000000, "n": 900000},
    ],
}


def _client(handler) -> MassiveREST:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="https://api.massive.com", transport=transport)
    return MassiveREST(api_key="test-key", client=http, sleep=lambda _s: None)


# --- pure parsers -----------------------------------------------------------

def test_parse_market_status_maps_exchange_flags() -> None:
    status = parse_market_status(_MARKET_STATUS)
    assert status.nyse_open is True
    assert status.nasdaq_open is True
    assert status.otc_open is True
    assert status.after_hours is False
    assert status.early_hours is False


def test_parse_market_status_closed_exchange() -> None:
    payload = {**_MARKET_STATUS, "exchanges": {"nyse": "closed", "nasdaq": "closed", "otc": "closed"}}
    status = parse_market_status(payload)
    assert status.nyse_open is False
    assert status.nasdaq_open is False


def test_parse_previous_close_yields_daily_bar() -> None:
    bar = parse_previous_close(_PREV_CLOSE, ticker="SPY")
    assert bar.venue == Venue.massive
    assert bar.symbol == "SPY"
    assert bar.interval_seconds == 86400
    assert bar.open == 500.0
    assert bar.close == 502.5
    assert bar.n_trades == 900000


def test_parse_previous_close_empty_results_raises() -> None:
    with pytest.raises(ValueError):
        parse_previous_close({"ticker": "SPY", "status": "OK", "results": []}, ticker="SPY")


# --- client over MockTransport (no network) ---------------------------------

def test_market_status_request() -> None:
    rest = _client(lambda req: httpx.Response(200, json=_MARKET_STATUS))
    status = rest.market_status()
    assert status.nyse_open is True


def test_previous_close_request_sets_bearer_and_path() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["path"] = req.url.path
        return httpx.Response(200, json=_PREV_CLOSE)

    rest = _client(handler)
    bar = rest.previous_close("SPY")
    assert bar.close == 502.5
    assert seen["auth"] == "Bearer test-key"
    assert seen["path"] == "/v2/aggs/ticker/SPY/prev"


def test_403_raises_not_authorized_and_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, json={"status": "NOT_AUTHORIZED", "message": "upgrade plan"})

    rest = _client(handler)
    with pytest.raises(NotAuthorizedError):
        rest.previous_close("SPY")
    assert calls["n"] == 1  # entitlement failure is terminal, not retried


def test_transient_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_MARKET_STATUS)

    rest = _client(handler)
    status = rest.market_status()
    assert status.nyse_open is True
    assert calls["n"] == 2


def test_persistent_5xx_raises_feed_connection_error() -> None:
    rest = _client(lambda req: httpx.Response(503, text="down"))
    rest.max_retries = 2
    with pytest.raises(FeedConnectionError):
        rest.market_status()


def test_rate_limiter_is_invoked_per_request() -> None:
    acquired = {"n": 0}

    rest = _client(lambda req: httpx.Response(200, json=_MARKET_STATUS))
    original = rest.rate_limiter.acquire

    def counting() -> float:
        acquired["n"] += 1
        return original()

    rest.rate_limiter.acquire = counting  # type: ignore[method-assign]
    rest.market_status()
    rest.market_status()
    assert acquired["n"] == 2


# --- panel helpers ----------------------------------------------------------

def test_bars_to_dataframe_has_expected_columns() -> None:
    bar = parse_previous_close(_PREV_CLOSE, ticker="SPY")
    df = bars_to_dataframe([bar])
    assert set(df.columns) >= {"symbol", "timestamp_utc", "open", "high", "low", "close", "volume"}
    assert df.height == 1
    assert df["symbol"][0] == "SPY"


def test_upsert_panel_dedups_on_symbol_and_timestamp() -> None:
    bar = parse_previous_close(_PREV_CLOSE, ticker="SPY")
    existing = bars_to_dataframe([bar])
    incoming = bars_to_dataframe([bar])  # same bar again
    merged = upsert_panel(existing, incoming)
    assert merged.height == 1  # no duplicate row


def test_upsert_panel_appends_new_rows() -> None:
    bar = parse_previous_close(_PREV_CLOSE, ticker="SPY")
    other = parse_previous_close({**_PREV_CLOSE, "ticker": "QQQ",
                                  "results": [{**_PREV_CLOSE["results"][0], "T": "QQQ"}]}, ticker="QQQ")
    merged = upsert_panel(bars_to_dataframe([bar]), bars_to_dataframe([other]))
    assert merged.height == 2
