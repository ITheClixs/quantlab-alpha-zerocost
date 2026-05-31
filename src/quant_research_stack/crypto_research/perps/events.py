from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _ms_to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)


def _levels(raw: Any) -> list[list[float]]:
    return [[float(price), float(size)] for price, size in (raw or [])]


def normalize_agg_trade(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "agg_trade",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("T") or payload.get("E")),
        "exchange_event_time": _ms_to_utc(payload.get("E")),
        "received_utc": received_utc,
        "trade_id": int(payload["a"]),
        "price": float(payload["p"]),
        "size": float(payload["q"]),
        "aggressor_side": "sell" if bool(payload.get("m")) else "buy",
    }


def normalize_book_ticker(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "book_ticker",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("E")) or received_utc,
        "received_utc": received_utc,
        "update_id": int(payload["u"]),
        "best_bid": float(payload["b"]),
        "best_bid_size": float(payload["B"]),
        "best_ask": float(payload["a"]),
        "best_ask_size": float(payload["A"]),
    }


def normalize_depth_update(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "depth_update",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("E")) or received_utc,
        "received_utc": received_utc,
        "first_update_id": int(payload["U"]),
        "last_update_id": int(payload["u"]),
        "bids": _levels(payload.get("b")),
        "asks": _levels(payload.get("a")),
    }
