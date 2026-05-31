from __future__ import annotations

import math
from typing import Any, cast

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide


def total_return(equity_curve: pl.DataFrame) -> float:
    if equity_curve.height == 0:
        return 0.0
    start = float(cast(float, equity_curve["equity"][0]))
    end = float(cast(float, equity_curve["equity"][-1]))
    if start <= 0.0:
        return 0.0
    return (end - start) / start


def max_drawdown(equity_curve: pl.DataFrame) -> float:
    if equity_curve.height == 0:
        return 0.0
    peak = float("-inf")
    worst = 0.0
    for value in equity_curve["equity"].to_list():
        v = float(cast(float, value))
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < worst:
            worst = dd
    return worst


def sharpe_ratio(returns: pl.Series, periods_per_year: int) -> float:
    if returns.len() == 0:
        return 0.0
    mu_raw = returns.mean()
    sigma_raw = returns.std()
    if mu_raw is None or sigma_raw is None:
        return 0.0
    mu = float(cast(Any, mu_raw))
    sigma = float(cast(Any, sigma_raw))
    if sigma == 0.0 or math.isnan(sigma):
        return 0.0
    return (mu / sigma) * math.sqrt(periods_per_year)


def calmar_ratio(equity_curve: pl.DataFrame) -> float:
    dd = abs(max_drawdown(equity_curve))
    if dd == 0.0:
        return 0.0
    return total_return(equity_curve) / dd


def hit_rate(fills: list[Fill]) -> float:
    if len(fills) < 2:
        return 0.0
    sorted_fills = sorted(fills, key=lambda f: f.timestamp_utc)
    wins = 0
    pairs = 0
    open_price: float | None = None
    open_side: OrderSide | None = None
    for f in sorted_fills:
        if open_price is None:
            open_price = f.price
            open_side = f.side
            continue
        pairs += 1
        sign = 1.0 if open_side == OrderSide.buy else -1.0
        pnl = sign * (f.price - open_price)
        if pnl > 0:
            wins += 1
        open_price = None
        open_side = None
    if pairs == 0:
        return 0.0
    return wins / pairs


def turnover(fills: list[Fill], starting_cash: float) -> float:
    if starting_cash <= 0:
        return 0.0
    notional = sum(f.price * f.quantity for f in fills)
    return notional / starting_cash


def value_at_risk(returns: pl.Series, alpha: float = 0.05) -> float:
    if returns.len() == 0:
        return 0.0
    quantile = returns.quantile(alpha)
    return 0.0 if quantile is None else float(quantile)
