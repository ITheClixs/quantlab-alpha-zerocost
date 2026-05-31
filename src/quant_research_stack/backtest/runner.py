from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_research_stack.backtest.metrics import (
    calmar_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
    turnover,
    value_at_risk,
)
from quant_research_stack.backtest.strategies.buy_and_hold import BuyAndHold
from quant_research_stack.backtest.strategies.moving_average_cross import MovingAverageCross
from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.feeds.market_types import Bar, Tick

_STRATEGIES = {
    "buy_and_hold": BuyAndHold,
    "moving_average_cross": MovingAverageCross,
}


@dataclass(frozen=True)
class BacktestConfig:
    events: Iterable[Tick | Bar]
    fill_model: FillModelConfig
    starting_cash: float
    strategy_name: str
    strategy_params: dict[str, Any]
    metrics_horizon_minutes: int = 5


@dataclass
class BacktestResult:
    fills: list[Fill]
    equity_curve: pl.DataFrame
    metrics: dict


def _build_strategy(name: str, params: dict):
    if name not in _STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")
    return _STRATEGIES[name](**params)


def _event_price(ev: Tick | Bar) -> float:
    return ev.close if isinstance(ev, Bar) else ev.price


class BacktestRunner:
    def __init__(self, cfg: BacktestConfig) -> None:
        self._cfg = cfg

    async def run(self) -> BacktestResult:
        fm = FillModel(self._cfg.fill_model)
        broker = NullBroker(fill_model=fm, starting_cash=self._cfg.starting_cash)
        strategy = _build_strategy(self._cfg.strategy_name, self._cfg.strategy_params)
        positions: dict[str, float] = {}
        avg_price: dict[str, float] = {}
        cash = float(self._cfg.starting_cash)
        equity_rows: list[dict] = []
        fills: list[Fill] = []
        for ev in self._cfg.events:
            broker.push_market_event(ev)
            intents = strategy.on_event(ev)
            for intent in intents:
                order = await broker.place_order(intent)
                for f in broker._fills.get(order.client_order_id, []):
                    fills.append(f)
                    strategy.on_fill(f)
                    sign = 1.0 if f.side == OrderSide.buy else -1.0
                    positions[f.symbol] = positions.get(f.symbol, 0.0) + sign * f.quantity
                    avg_price[f.symbol] = f.price
                    cash -= sign * f.price * f.quantity + f.commission
            mark_price = _event_price(ev)
            equity = cash + sum(
                qty * (mark_price if sym == ev.symbol else avg_price.get(sym, 0.0))
                for sym, qty in positions.items()
            )
            equity_rows.append({"timestamp_utc": ev.timestamp_utc, "equity": equity})
        equity_curve = (
            pl.DataFrame(equity_rows)
            if equity_rows
            else pl.DataFrame({"equity": [self._cfg.starting_cash]})
        )
        returns = (
            pl.Series("r", equity_curve["equity"].pct_change().drop_nulls().to_list())
            if equity_curve.height > 1
            else pl.Series("r", [])
        )
        metrics = {
            "total_return": total_return(equity_curve),
            "max_drawdown": max_drawdown(equity_curve),
            "sharpe_ratio": sharpe_ratio(returns, periods_per_year=252),
            "calmar_ratio": calmar_ratio(equity_curve),
            "hit_rate": hit_rate(fills),
            "turnover": turnover(fills, starting_cash=self._cfg.starting_cash),
            "value_at_risk_5pct": value_at_risk(returns, alpha=0.05),
            "n_fills": len(fills),
        }
        return BacktestResult(fills=fills, equity_curve=equity_curve, metrics=metrics)
