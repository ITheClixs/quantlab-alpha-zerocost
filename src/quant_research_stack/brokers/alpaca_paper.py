from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from quant_research_stack.brokers.capabilities import BrokerCapabilities, ensure_supported
from quant_research_stack.brokers.order_types import (
    Account,
    Fill,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

_CAPS = BrokerCapabilities(
    venue="alpaca_paper",
    supported_order_types=frozenset({
        OrderType.market,
        OrderType.limit,
        OrderType.stop,
        OrderType.stop_limit,
        OrderType.bracket,
        OrderType.oco,
    }),
    supported_time_in_force=frozenset({
        TimeInForce.day,
        TimeInForce.gtc,
        TimeInForce.ioc,
        TimeInForce.fok,
    }),
    supports_shorting=True,
    supports_fractional_shares=True,
    supports_extended_hours=True,
    max_orders_per_second=200,
    paper_only=True,
)


def build_order_payload(intent: OrderIntent) -> dict:
    payload: dict = {
        "symbol": intent.symbol,
        "side": intent.side.value,
        "type": intent.type.value,
        "qty": str(intent.quantity),
        "time_in_force": intent.time_in_force.value,
        "client_order_id": intent.client_order_id,
        "extended_hours": intent.extended_hours,
    }
    if intent.limit_price is not None:
        payload["limit_price"] = str(intent.limit_price)
    if intent.stop_price is not None:
        payload["stop_price"] = str(intent.stop_price)
    if intent.type == OrderType.bracket:
        payload["order_class"] = "bracket"
        payload["take_profit"] = {"limit_price": str(intent.take_profit_price)}
        payload["stop_loss"] = {"stop_price": str(intent.stop_loss_price)}
    if intent.type == OrderType.oco:
        payload["order_class"] = "oco"
        payload["take_profit"] = {"limit_price": str(intent.oco_limit_price)}
        payload["stop_loss"] = {"stop_price": str(intent.oco_stop_price)}
    return payload


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class AlpacaPaper:
    credentials_path: str = "~/.alpaca/paper_keys.json"
    base_url: str = "https://paper-api.alpaca.markets"
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        key, secret = _load_credentials(self.credentials_path)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent) -> Order:
        ensure_supported(self.capabilities, intent.type)
        payload = build_order_payload(intent)
        response = await self._client.post("/v2/orders", json=payload)
        response.raise_for_status()
        body = response.json()
        return _order_from_alpaca(intent, body)

    async def cancel_order(self, client_order_id: str) -> Order:
        order = await self.get_order(client_order_id)
        response = await self._client.delete(f"/v2/orders/{order.broker_order_id}")
        response.raise_for_status()
        return order.model_copy(update={"status": OrderStatus.canceled, "updated_utc": datetime.now(UTC)})

    async def get_order(self, client_order_id: str) -> Order:
        response = await self._client.get(
            f"/v2/orders:by_client_order_id?client_order_id={client_order_id}"
        )
        response.raise_for_status()
        body = response.json()
        return _order_from_alpaca_response(body)

    async def positions(self) -> list[Position]:
        response = await self._client.get("/v2/positions")
        response.raise_for_status()
        return [
            Position(
                symbol=row["symbol"],
                quantity=float(row["qty"]),
                avg_entry_price=float(row["avg_entry_price"]),
                market_value=float(row["market_value"]),
                unrealized_pnl=float(row["unrealized_pl"]),
            )
            for row in response.json()
        ]

    async def account(self) -> Account:
        response = await self._client.get("/v2/account")
        response.raise_for_status()
        body = response.json()
        return Account(
            equity=float(body["equity"]),
            cash=float(body["cash"]),
            buying_power=float(body["buying_power"]),
            currency=body.get("currency", "USD"),
        )

    async def stream_fills(self) -> AsyncIterator[Fill]:
        # Alpaca paper exposes fills via websocket; in S3 we poll /v2/account/activities
        # for simplicity. Live streaming via wss is reserved for S4 (live brokers).
        if False:
            yield  # type: ignore[unreachable]
        return

    async def close(self) -> None:
        await self._client.aclose()


def _order_from_alpaca(intent: OrderIntent, body: dict) -> Order:
    now = datetime.now(UTC)
    return Order(
        client_order_id=intent.client_order_id,
        broker_order_id=str(body.get("id", "")),
        symbol=intent.symbol,
        side=intent.side,
        type=intent.type,
        quantity=intent.quantity,
        filled_quantity=float(body.get("filled_qty", 0.0) or 0.0),
        status=OrderStatus(body.get("status", "accepted")),
        submitted_utc=now,
        updated_utc=now,
    )


def _order_from_alpaca_response(body: dict) -> Order:
    now = datetime.now(UTC)
    return Order(
        client_order_id=str(body["client_order_id"]),
        broker_order_id=str(body["id"]),
        symbol=str(body["symbol"]),
        side=OrderSide(body["side"]),
        type=OrderType(body["type"]),
        quantity=float(body["qty"]),
        filled_quantity=float(body.get("filled_qty", 0.0) or 0.0),
        status=OrderStatus(body["status"]),
        submitted_utc=now,
        updated_utc=now,
    )
