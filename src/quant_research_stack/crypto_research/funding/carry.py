"""Delta-neutral funding-carry backtest math (Strategy A).

Per unit gross notional, long spot + short perp, re-neutralized daily:

    spot_ret[t] = spot_close[t]/spot_close[t-1] - 1
    perp_ret[t] = perp_close[t]/perp_close[t-1] - 1
    price_pnl[t] = spot_ret[t] - perp_ret[t]      # long spot, short perp
    funding_pnl[t] = +funding_day[t]              # short receives funding when > 0
    gross[t] = price_pnl[t] + funding_pnl[t]
    net[t]   = gross[t] - cost[t]

Leak-safety: the short is established at the close of day t-1 (decision uses info <= t-1)
and earns day t's price move and day t's three funding settlements. Day 0 earns nothing
(entry day) — it only pays the entry cost. Returns are expressed per unit of one-side
notional (the carry yield), matching the annualized-funding convention. Crypto annualizes
at sqrt(365) / 365 (markets trade every day).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from numpy.typing import NDArray

F = NDArray[np.float64]
ANN_CRYPTO = 365.0


@dataclass(frozen=True)
class CarryResult:
    dates: list[date]
    net: F            # daily net return (per unit notional)
    gross: F
    funding: F        # funding component
    price: F          # price/basis component
    cost: F
    metrics: dict[str, float]


def metrics_365(net: F) -> dict[str, float]:
    f = net[np.isfinite(net)]
    if f.size < 2:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0, "total_return": 0.0}
    sd = float(np.std(f, ddof=1))
    sharpe = float(np.mean(f) / sd * np.sqrt(ANN_CRYPTO)) if sd > 0 else 0.0
    eq = np.cumprod(1.0 + f)
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    ann = float(eq[-1] ** (ANN_CRYPTO / f.size) - 1.0) if eq[-1] > 0 else -1.0
    return {"sharpe": sharpe, "ann_return": ann, "ann_vol": sd * np.sqrt(ANN_CRYPTO),
            "max_drawdown": dd, "calmar": float(ann / abs(dd)) if dd < 0 else 0.0,
            "total_return": float(eq[-1] - 1.0)}


def carry_returns(panel: pl.DataFrame, *, spot_taker_bps: float = 10.0,
                  perp_taker_bps: float = 5.0, rebalance: bool = True,
                  invert: bool = False, zero_funding: bool = False) -> CarryResult:
    """Delta-neutral carry daily returns.

    `invert` flips the book (long perp / short spot) — a placebo that must LOSE.
    `zero_funding` drops the funding leg — isolates the price/basis term (must be ~0),
    attributing the return to funding rather than to a price artifact.
    """
    dates = panel["date"].to_list()
    spot = panel["spot_close"].to_numpy().astype(np.float64)
    perp = panel["perp_close"].to_numpy().astype(np.float64)
    fund = panel["funding_day"].to_numpy().astype(np.float64)
    n = spot.size

    spot_ret = np.zeros(n)
    perp_ret = np.zeros(n)
    spot_ret[1:] = spot[1:] / spot[:-1] - 1.0
    perp_ret[1:] = perp[1:] / perp[:-1] - 1.0
    price = spot_ret - perp_ret                 # long spot, short perp
    funding = fund.copy()
    funding[0] = 0.0                            # entry day collects no funding
    if zero_funding:
        funding = np.zeros(n)
    if invert:
        price = -price
        funding = -funding
    gross = price + funding

    rt = (spot_taker_bps + perp_taker_bps) * 1e-4
    cost = np.zeros(n)
    cost[0] += rt                               # establish both legs
    cost[-1] += rt                              # unwind both legs
    if rebalance:
        cost[1:] += np.abs(price[1:]) * rt      # daily hedge-maintenance turnover
    net = gross - cost
    return CarryResult(dates=dates, net=net, gross=gross, funding=funding,
                       price=price, cost=cost, metrics=metrics_365(net))


def per_year(dates: list[date], net: F) -> dict[int, dict[str, float]]:
    years = np.array([d.year for d in dates])
    out: dict[int, dict[str, float]] = {}
    for y in sorted(set(years.tolist())):
        seg = net[years == y]
        m = metrics_365(seg)
        out[int(y)] = {"sharpe": round(m["sharpe"], 3),
                       "ann_return_pct": round(m["ann_return"] * 100, 2),
                       "total_pct": round(m["total_return"] * 100, 2),
                       "days": int(seg.size)}
    return out


def pooled_book(results: dict[str, CarryResult]) -> CarryResult:
    """Equal-weight daily-rebalanced book across assets sharing the same date grid."""
    names = list(results)
    ref = results[names[0]]
    stack = np.vstack([results[k].net for k in names])
    net = stack.mean(axis=0)
    g = np.vstack([results[k].gross for k in names]).mean(axis=0)
    fnd = np.vstack([results[k].funding for k in names]).mean(axis=0)
    prc = np.vstack([results[k].price for k in names]).mean(axis=0)
    cst = np.vstack([results[k].cost for k in names]).mean(axis=0)
    return CarryResult(dates=ref.dates, net=net, gross=g, funding=fnd, price=prc,
                       cost=cst, metrics=metrics_365(net))


def pnl_concentration(dates: list[date], net: F) -> dict[str, float]:
    """Share of total positive PnL from the single biggest year and single biggest day."""
    total = float(np.sum(net))
    if total <= 0:
        return {"top_year_share": 1.0, "top_day_share": 1.0, "total": total}
    years = np.array([d.year for d in dates])
    year_pnl = {int(y): float(np.sum(net[years == y])) for y in set(years.tolist())}
    top_year = max(year_pnl.values()) / total
    top_day = float(np.max(net)) / total
    return {"top_year_share": round(top_year, 3), "top_day_share": round(top_day, 4),
            "total": round(total, 4)}
