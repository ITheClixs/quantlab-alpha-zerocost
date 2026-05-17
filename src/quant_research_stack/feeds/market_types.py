from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class Venue(StrEnum):
    binance = "binance"
    coinbase = "coinbase"
    alpaca = "alpaca"
    replay = "replay"


class TickSide(StrEnum):
    buy = "buy"
    sell = "sell"
    unknown = "unknown"


class Tick(BaseModel):
    model_config = {"frozen": True}
    venue: Venue
    symbol: str
    timestamp_utc: datetime
    received_utc: datetime
    price: Annotated[float, Field(gt=0.0)]
    size: Annotated[float, Field(ge=0.0)]
    side: TickSide
    sequence: int | None = None
    raw: dict | None = None


class Bar(BaseModel):
    model_config = {"frozen": True}
    venue: Venue
    symbol: str
    timestamp_utc: datetime
    interval_seconds: Annotated[int, Field(ge=1, le=86400)]
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int | None = None
