"""Single-asset directional backtest for the strategy-benchmark framework.

Cost model:
- Commission: `commission_bps_one_way` per turn
- Spread/slippage: half-spread approximated by `spread_bps`
- Total trip cost = (commission + spread) × |Δposition|

Fill convention:
- Signal generated at close of date t
- Position effective from close of date t (i.e. captured by today's strategy
  PnL is yesterday's signal × today's return — `signals.shift(1)`)
- Costs charged on `|position_t - position_{t-1}|`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class BacktestCostConfig:
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0  # half-spread approximation

    @property
    def total_one_way_bps(self) -> float:
        return self.commission_bps_one_way + self.spread_bps_one_way


@dataclass(frozen=True)
class BacktestResult:
    daily_net_return: NDArray[np.float64]
    daily_gross_return: NDArray[np.float64]
    daily_turnover: NDArray[np.float64]
    sharpe_annualised: float
    sortino_annualised: float
    total_return: float
    max_drawdown: float
    hit_rate: float
    n_trades: int
    avg_position: float
    annual_turnover: float


def _to_np(series: pl.Series) -> NDArray[np.float64]:
    return series.to_numpy().astype(np.float64)


DEFAULT_COST = BacktestCostConfig()


def run_single_asset_backtest(
    *,
    bars: pl.DataFrame,
    signals: pl.Series,
    cost: BacktestCostConfig | None = None,
) -> BacktestResult:
    if cost is None:
        cost = DEFAULT_COST
    """Run a strategy that holds `signals[t]` units (long if +, short if −,
    flat if 0) entered at the close of date t.

    Returns daily NET strategy returns plus summary statistics.
    """
    if bars.height != signals.len():
        raise ValueError(
            f"bars ({bars.height}) and signals ({signals.len()}) row count mismatch"
        )

    closes = _to_np(bars["close"])
    sig = _to_np(signals)
    # Replace NaN signals with 0 (flat) — produced when lookback isn't met yet.
    sig = np.where(np.isnan(sig), 0.0, sig)
    # Clip to [-1, +1] so a single strategy can't accidentally use 5x leverage
    sig = np.clip(sig, -1.0, 1.0)

    # Daily simple return of the underlying instrument
    underlying_ret = np.zeros_like(closes)
    underlying_ret[1:] = closes[1:] / closes[:-1] - 1.0

    # Position effective today = signal from yesterday's close
    pos_today = np.zeros_like(sig)
    pos_today[1:] = sig[:-1]

    gross_ret = pos_today * underlying_ret

    # Turnover = |position change between consecutive days|
    pos_prev = np.zeros_like(pos_today)
    pos_prev[1:] = pos_today[:-1]
    turnover = np.abs(pos_today - pos_prev)

    cost_drag = turnover * cost.total_one_way_bps / 10_000.0
    net_ret = gross_ret - cost_drag

    return BacktestResult(
        daily_net_return=net_ret,
        daily_gross_return=gross_ret,
        daily_turnover=turnover,
        sharpe_annualised=_annualised_sharpe(net_ret),
        sortino_annualised=_annualised_sortino(net_ret),
        total_return=float(np.prod(1.0 + net_ret) - 1.0),
        max_drawdown=_max_drawdown(net_ret),
        hit_rate=float(np.mean(net_ret[net_ret != 0.0] > 0)) if np.any(net_ret != 0.0) else 0.0,
        n_trades=int(np.sum(turnover > 1e-12)),
        avg_position=float(np.mean(np.abs(pos_today))),
        annual_turnover=float(np.sum(turnover) * 252 / max(1, len(turnover))),
    )


def _annualised_sharpe(r: NDArray[np.float64]) -> float:
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r)) / sd * np.sqrt(252.0)


def _annualised_sortino(r: NDArray[np.float64]) -> float:
    if r.size < 2:
        return 0.0
    downside = r[r < 0.0]
    if downside.size < 2:
        return 0.0
    sd = float(np.std(downside, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r)) / sd * np.sqrt(252.0)


def _max_drawdown(r: NDArray[np.float64]) -> float:
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())
