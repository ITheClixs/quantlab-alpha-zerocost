from __future__ import annotations

from quant_research_stack.brokers.order_types import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot


def perp_symbol(spot_symbol: str) -> str:
    return f"{spot_symbol}PERP"


class FundingCarryStrategy:
    """Delta-neutral target-position rule: long spot + short perp at equal notional.

    1x unlevered (spec §0). Emits rebalancing market intents when a leg drifts beyond
    `rebalance_drift_bps` from its target notional.
    """

    def __init__(self, cfg: PaperSimConfig) -> None:
        self._cfg = cfg
        self._leg_notional = cfg.total_notional_usd * cfg.leverage / (2.0 * len(cfg.symbols))

    def rebalance_intents(self, snap: MarketSnapshot, *, positions: dict[str, float],
                          cycle: int) -> list[OrderIntent]:
        out: list[OrderIntent] = []
        drift = self._cfg.rebalance_drift_bps * 1e-4
        legs = (
            (snap.symbol, snap.spot_price, +self._leg_notional / snap.spot_price),       # long spot
            (perp_symbol(snap.symbol), snap.perp_mark, -self._leg_notional / snap.perp_mark),  # short perp
        )
        for leg_sym, price, target_qty in legs:
            cur = positions.get(leg_sym, 0.0)
            delta = target_qty - cur
            if abs(delta) * price < drift * self._leg_notional:
                continue
            out.append(OrderIntent(
                client_order_id=f"carry-{leg_sym}-{cycle:08d}",
                symbol=leg_sym,
                side=OrderSide.buy if delta > 0 else OrderSide.sell,
                type=OrderType.market,
                quantity=abs(delta),
                time_in_force=TimeInForce.ioc,
            ))
        return out
