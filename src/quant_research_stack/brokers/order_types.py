from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class OrderSide(StrEnum):
    buy = "buy"
    sell = "sell"


class TimeInForce(StrEnum):
    day = "day"
    gtc = "gtc"
    ioc = "ioc"
    fok = "fok"


class OrderType(StrEnum):
    market = "market"
    limit = "limit"
    stop = "stop"
    stop_limit = "stop_limit"
    oco = "oco"
    bracket = "bracket"


class OrderStatus(StrEnum):
    accepted = "accepted"
    partially_filled = "partially_filled"
    filled = "filled"
    canceled = "canceled"
    rejected = "rejected"
    expired = "expired"


class OrderIntent(BaseModel):
    model_config = {"frozen": True}
    client_order_id: Annotated[str, Field(min_length=8, max_length=64)]
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Annotated[float, Field(gt=0.0)]
    time_in_force: TimeInForce = TimeInForce.day
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    oco_limit_price: float | None = None
    oco_stop_price: float | None = None
    extended_hours: bool = False

    @model_validator(mode="after")
    def _required_prices_for_type(self) -> OrderIntent:
        t = self.type
        if t == OrderType.limit and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if t == OrderType.stop and self.stop_price is None:
            raise ValueError("stop order requires stop_price")
        if t == OrderType.stop_limit and (self.limit_price is None or self.stop_price is None):
            raise ValueError("stop_limit requires both limit_price and stop_price")
        if t == OrderType.bracket and (
            self.limit_price is None
            or self.take_profit_price is None
            or self.stop_loss_price is None
        ):
            raise ValueError("bracket requires entry limit_price, take_profit_price, stop_loss_price")
        if t == OrderType.oco and (self.oco_limit_price is None or self.oco_stop_price is None):
            raise ValueError("oco requires oco_limit_price and oco_stop_price")
        return self


class Order(BaseModel):
    model_config = {"frozen": True}
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    filled_quantity: float
    status: OrderStatus
    submitted_utc: datetime
    updated_utc: datetime


class Fill(BaseModel):
    model_config = {"frozen": True}
    client_order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    timestamp_utc: datetime
    commission: float = 0.0


class Position(BaseModel):
    model_config = {"frozen": True}
    symbol: str
    quantity: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


class Account(BaseModel):
    model_config = {"frozen": True}
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
