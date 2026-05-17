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
    venue="binance_testnet",
    supported_order_types=frozenset({
        OrderType.market, OrderType.limit, OrderType.stop_limit, OrderType.oco,
    }),
    supported_time_in_force=frozenset({TimeInForce.gtc, TimeInForce.ioc, TimeInForce.fok}),
    supports_shorting=False,
    supports_fractional_shares=True,
    supports_extended_hours=False,
    max_orders_per_second=10,
    paper_only=True,
)


def build_order_payload(intent: OrderIntent) -> dict:
    payload: dict = {
        "symbol": intent.symbol,
        "side": intent.side.value.upper(),
        "type": intent.type.value.upper(),
        "quantity": str(intent.quantity),
        "newClientOrderId": intent.client_order_id,
    }
    if intent.time_in_force is not None:
        payload["timeInForce"] = intent.time_in_force.value.upper()
    if intent.limit_price is not None:
        payload["price"] = str(intent.limit_price)
    if intent.stop_price is not None:
        payload["stopPrice"] = str(intent.stop_price)
    if intent.type == OrderType.oco:
        payload["type"] = "OCO"
        payload["price"] = str(intent.oco_limit_price)
        payload["stopPrice"] = str(intent.oco_stop_price)
    return payload


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class BinanceTestnet:
    credentials_path: str = "~/.binance/testnet_keys.json"
    rest_base_url: str = "https://testnet.binance.vision"
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        key, secret = _load_credentials(self.credentials_path)
        self._key = key
        self._secret = secret
        self._client = httpx.AsyncClient(
            base_url=self.rest_base_url,
            headers={"X-MBX-APIKEY": key},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent) -> Order:
        ensure_supported(self.capabilities, intent.type)
        payload = build_order_payload(intent)
        response = await self._client.post("/api/v3/order", data=payload)
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=intent.client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus(_translate_status(body.get("status", "NEW"))),
            submitted_utc=now,
            updated_utc=now,
        )

    async def cancel_order(self, client_order_id: str) -> Order:
        # The testnet REST API requires the broker order id for cancellation; tests stub this.
        response = await self._client.delete(f"/api/v3/order?origClientOrderId={client_order_id}")
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=str(body.get("symbol", "")),
            side=OrderSide(body.get("side", "buy").lower()),
            type=OrderType(body.get("type", "market").lower()),
            quantity=float(body.get("origQty", 0.0)),
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus.canceled,
            submitted_utc=now,
            updated_utc=now,
        )

    async def get_order(self, client_order_id: str) -> Order:
        response = await self._client.get(f"/api/v3/order?origClientOrderId={client_order_id}")
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=str(body.get("symbol", "")),
            side=OrderSide(body.get("side", "buy").lower()),
            type=OrderType(body.get("type", "market").lower()),
            quantity=float(body.get("origQty", 0.0)),
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus(_translate_status(body.get("status", "NEW"))),
            submitted_utc=now,
            updated_utc=now,
        )

    async def positions(self) -> list[Position]:
        response = await self._client.get("/api/v3/account")
        response.raise_for_status()
        body = response.json()
        out: list[Position] = []
        for bal in body.get("balances", []):
            free = float(bal.get("free", 0.0))
            if free == 0.0:
                continue
            out.append(Position(
                symbol=str(bal["asset"]),
                quantity=free,
                avg_entry_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
            ))
        return out

    async def account(self) -> Account:
        response = await self._client.get("/api/v3/account")
        response.raise_for_status()
        body = response.json()
        usdt = next((float(b["free"]) for b in body.get("balances", []) if b["asset"] == "USDT"), 0.0)
        return Account(equity=usdt, cash=usdt, buying_power=usdt, currency="USDT")

    async def stream_fills(self) -> AsyncIterator[Fill]:
        # Live user data stream is reserved for S4.
        if False:
            yield  # type: ignore[unreachable]
        return

    async def close(self) -> None:
        await self._client.aclose()


def _translate_status(s: str) -> str:
    return {
        "NEW": "accepted",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
    }.get(s.upper(), "accepted")
