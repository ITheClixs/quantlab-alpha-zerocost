from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass(frozen=True)
class FillModelConfig:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    half_spread_bps: float = 1.0
    fill_latency_ms: int = 50
    reject_if_notional_above_pct_adv: float | None = None
    partial_fill_max_pct_of_book: float = 0.10


def _mid(event: Tick | Bar) -> float:
    if isinstance(event, Bar):
        return (event.open + event.close) / 2.0
    return event.price


class FillModel:
    def __init__(self, cfg: FillModelConfig) -> None:
        self.cfg = cfg

    def synthesize(self, intent: OrderIntent, market_iter: Iterator[Tick | Bar]) -> list[Fill]:
        try:
            event = next(market_iter)
        except StopIteration:
            return []
        mid = _mid(event)
        direction = 1.0 if intent.side == OrderSide.buy else -1.0
        adverse_bps = self.cfg.half_spread_bps + self.cfg.slippage_bps
        fill_px = mid + direction * mid * adverse_bps * 1e-4
        notional = fill_px * intent.quantity
        commission = notional * self.cfg.commission_bps * 1e-4
        ts = event.timestamp_utc + timedelta(milliseconds=self.cfg.fill_latency_ms)
        return [Fill(
            client_order_id=intent.client_order_id,
            fill_id=f"{intent.client_order_id}-1",
            symbol=intent.symbol,
            side=intent.side,
            price=fill_px,
            quantity=intent.quantity,
            timestamp_utc=ts,
            commission=commission,
        )]
