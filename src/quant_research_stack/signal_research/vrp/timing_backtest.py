"""Timing-mode backtest — single-instrument signal → daily net returns.

The alpha_eq M4 engine is built for cross-sectional baskets. VRP is a
single-index timing strategy: at each date, the rule emits a gross
exposure in [-1, +1] for one underlying (e.g. SPY). PnL accrues from
close(T) → close(T+1) times the gross_exposure observed at close(T).

Costs:
- per-trade turnover cost = commission + spread, applied each time
  position changes
- 1-bar delay variant shifts signal forward by N bars before PnL accrual
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimingCostConfig:
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 0.5

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.commission_bps_one_way + self.spread_bps_one_way)


@dataclass(frozen=True)
class TimingBacktestResult:
    daily_returns: pl.DataFrame  # date, gross_return, net_return, turnover
    sharpe_annual: float
    max_drawdown: float
    cumulative_return: float
    n_days: int
    turnover_total: float


def _safe_sharpe(rets: NDArray[np.float64]) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(rets)) / sd * float(np.sqrt(252.0))


def _max_dd(rets: NDArray[np.float64]) -> float:
    if rets.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def run_timing_backtest(
    *,
    signals: pl.DataFrame,
    underlying: pl.DataFrame,
    target_symbol: str,
    cost: TimingCostConfig,
    cost_stress_mult: float = 1.0,
    signal_delay_bars: int = 0,
) -> TimingBacktestResult:
    """Run a single-instrument timing backtest.

    `signals` must have columns (date, signal). `underlying` has bars for
    `target_symbol` with columns including (date, symbol, close).

    Position taken at close(T) earns return close(T+1)/close(T) - 1 (no fills
    on T+1 close, just the price move). Turnover cost charged on the day the
    position changes.
    """
    u = (
        underlying.filter(pl.col("symbol") == target_symbol)
        .sort("date")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("u_ret")
        )
    )
    s = signals.sort("date")
    if signal_delay_bars > 0:
        s = s.with_columns(pl.col("signal").shift(signal_delay_bars).alias("signal"))
    joined = (
        u.join(s, on="date", how="left")
        .with_columns(pl.col("signal").fill_null(0.0))
        .drop_nulls(subset=["u_ret"])
    )
    sig = joined["signal"].to_numpy().astype(np.float64)
    ret = joined["u_ret"].to_numpy().astype(np.float64)
    # position at close(T) applied to ret(T+1) — but our `u_ret` is already
    # the return from close(T-1) → close(T). So we need to shift signal one
    # day backward to make causality correct: position taken at close(T) earns
    # the next day's move close(T+1) - close(T).
    sig_lagged = np.concatenate([[0.0], sig[:-1]])
    gross_ret = sig_lagged * ret
    pos_change = np.abs(np.diff(sig_lagged, prepend=0.0))
    turnover_cost_per_bar = pos_change * cost.round_trip_bps * cost_stress_mult / 10_000.0
    net_ret = gross_ret - turnover_cost_per_bar

    daily = joined.select(["date"]).with_columns(
        pl.Series("gross_return", gross_ret),
        pl.Series("net_return", net_ret),
        pl.Series("turnover", pos_change),
    )
    return TimingBacktestResult(
        daily_returns=daily,
        sharpe_annual=_safe_sharpe(net_ret),
        max_drawdown=_max_dd(net_ret),
        cumulative_return=float(np.prod(1.0 + net_ret) - 1.0),
        n_days=int(net_ret.size),
        turnover_total=float(np.sum(pos_change)),
    )
