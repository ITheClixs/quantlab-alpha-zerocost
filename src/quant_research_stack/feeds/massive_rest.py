from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import polars as pl
from dotenv import load_dotenv

from quant_research_stack.feeds.base import FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Bar, Venue
from quant_research_stack.feeds.rate_limit import RateLimiter

_DEFAULT_BASE_URL = "https://api.massive.com"
_DAY_SECONDS = 86_400
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class NotAuthorizedError(RuntimeError):
    """Raised on HTTP 403 / NOT_AUTHORIZED — the endpoint needs a paid plan tier.

    Verified 2026-05-29: the free tier authorizes ``/v1/marketstatus/now`` and
    ``/v2/aggs/ticker/{T}/prev`` only. Historical range aggregates, live
    snapshots, and S3 flat-file downloads all return 403. This error makes that
    entitlement boundary explicit rather than silently swallowing it.
    """


@dataclass(frozen=True)
class MarketStatus:
    nyse_open: bool
    nasdaq_open: bool
    otc_open: bool
    after_hours: bool
    early_hours: bool
    server_time: str
    raw: dict


def _is_open(value: object) -> bool:
    return str(value).lower() == "open"


def parse_market_status(response: dict) -> MarketStatus:
    exchanges = response.get("exchanges") or {}
    return MarketStatus(
        nyse_open=_is_open(exchanges.get("nyse")),
        nasdaq_open=_is_open(exchanges.get("nasdaq")),
        otc_open=_is_open(exchanges.get("otc")),
        after_hours=bool(response.get("afterHours", False)),
        early_hours=bool(response.get("earlyHours", False)),
        server_time=str(response.get("serverTime", "")),
        raw=dict(response),
    )


def parse_previous_close(response: dict, *, ticker: str) -> Bar:
    """Map a Polygon-compatible ``/prev`` aggregate response to a daily ``Bar``."""
    results = response.get("results") or []
    if not results:
        raise ValueError(
            f"previous-close response for {ticker!r} carried no results "
            f"(status={response.get('status')!r})"
        )
    row = results[0]
    epoch_ms = int(row["t"])
    return Bar(
        venue=Venue.massive,
        symbol=str(row.get("T", ticker)),
        timestamp_utc=datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC),
        interval_seconds=_DAY_SECONDS,
        open=float(row["o"]),
        high=float(row["h"]),
        low=float(row["l"]),
        close=float(row["c"]),
        volume=float(row["v"]),
        n_trades=int(row["n"]) if row.get("n") is not None else None,
    )


def bars_to_dataframe(bars: Iterable[Bar]) -> pl.DataFrame:
    rows = [
        {
            "symbol": b.symbol,
            "timestamp_utc": b.timestamp_utc,
            "interval_seconds": b.interval_seconds,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "n_trades": b.n_trades,
            "venue": str(b.venue),
        }
        for b in bars
    ]
    return pl.DataFrame(rows)


def upsert_panel(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    """Append ``incoming`` to ``existing``, de-duplicating on (symbol, timestamp)."""
    if existing.is_empty():
        combined = incoming
    elif incoming.is_empty():
        combined = existing
    else:
        combined = pl.concat([existing, incoming], how="vertical_relaxed")
    return (
        combined.unique(subset=["symbol", "timestamp_utc"], keep="last")
        .sort(["symbol", "timestamp_utc"])
    )


@dataclass
class MassiveREST:
    """Free-tier-aware REST client for the Massive.com market-data API.

    Synchronous on purpose: at 5 calls/min a request/response client with a
    shared :class:`RateLimiter` is the honest model. Streaming feeds live in the
    async adapters (``binance_ws``, ``coinbase_ws``). Only the two endpoints the
    free tier authorizes are exposed; everything else returns
    :class:`NotAuthorizedError` so callers cannot mistake a paywall for a bug.
    """

    api_key: str
    base_url: str = _DEFAULT_BASE_URL
    max_calls_per_minute: int = 5
    max_retries: int = 3
    timeout: float = 15.0
    client: httpx.Client | None = None
    rate_limiter: RateLimiter | None = field(default=None)
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key is required (set MASSIVE_REST_API_KEY)")
        if self.client is None:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        if self.rate_limiter is None:
            self.rate_limiter = RateLimiter(self.max_calls_per_minute, 60.0)

    @classmethod
    def from_env(cls, **overrides: object) -> MassiveREST:
        load_dotenv()
        api_key = os.environ.get("MASSIVE_REST_API_KEY", "")
        if not api_key:
            raise ValueError(
                "MASSIVE_REST_API_KEY missing — copy .env.example to .env and populate."
            )
        return cls(api_key=api_key, **overrides)  # type: ignore[arg-type]

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        assert self.client is not None and self.rate_limiter is not None
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            response = self.client.get(
                path, params=params, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code == 403:
                raise NotAuthorizedError(
                    f"{path} returned 403 NOT_AUTHORIZED — this endpoint requires a "
                    f"paid Massive.com plan tier."
                )
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code} on {path}"
                if attempt < self.max_retries:
                    self.sleep(exponential_backoff(attempt + 1, base=1.0, cap=30.0))
                    continue
                raise FeedConnectionError(f"{last_error} after {self.max_retries} retries")
            response.raise_for_status()
            return response.json()
        raise FeedConnectionError(last_error)  # pragma: no cover - loop always returns/raises

    def market_status(self) -> MarketStatus:
        return parse_market_status(self._get("/v1/marketstatus/now"))

    def previous_close(self, ticker: str) -> Bar:
        path = f"/v2/aggs/ticker/{ticker}/prev"
        return parse_previous_close(self._get(path, {"adjusted": "true"}), ticker=ticker)

    def previous_closes(self, tickers: Sequence[str]) -> list[Bar]:
        """Fetch the previous-day bar for each ticker, rate-limited (5/min)."""
        return [self.previous_close(t) for t in tickers]

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
