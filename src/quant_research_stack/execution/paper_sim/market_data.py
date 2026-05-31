from __future__ import annotations

from dataclasses import dataclass

import httpx

_SPOT_URL = "https://api.binance.com/api/v3/ticker/price"
_PERP_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    ts_ms: int
    spot_price: float
    perp_mark: float
    funding_rate: float
    next_funding_ms: int

    @property
    def basis(self) -> float:
        return self.perp_mark / self.spot_price - 1.0


def parse_spot_price(payload: dict) -> float:
    return float(payload["price"])


def parse_premium_index(payload: dict) -> tuple[float, float, int]:
    return (
        float(payload["markPrice"]),
        float(payload["lastFundingRate"]),
        int(payload["nextFundingTime"]),
    )


class MarketDataPoller:
    """Polls free Binance public REST for spot price + perp mark/funding. No API key."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def snapshot(self, symbol: str, *, now_ms: int) -> MarketSnapshot:
        spot = await self._client.get(_SPOT_URL, params={"symbol": symbol})
        perp = await self._client.get(_PERP_URL, params={"symbol": symbol})
        spot.raise_for_status()
        perp.raise_for_status()
        spot_price = parse_spot_price(spot.json())
        mark, funding, next_ms = parse_premium_index(perp.json())
        return MarketSnapshot(symbol=symbol, ts_ms=now_ms, spot_price=spot_price,
                              perp_mark=mark, funding_rate=funding, next_funding_ms=next_ms)

    async def close(self) -> None:
        await self._client.aclose()
