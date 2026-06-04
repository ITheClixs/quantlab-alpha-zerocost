"""Run the zoo: enumerate grid, backtest each on IS, assemble (T,N) returns, PBO,
purged+embargoed OOS tail. research_only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.backtest import BacktestCostConfig, run_single_asset_backtest
from quant_research_stack.strategy_benchmark.pbo import compute_pbo
from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES
from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig, ZooStrategySpec, enumerate_zoo
from quant_research_stack.strategy_benchmark.zoo.transforms import (
    apply_holding,
    apply_position_mode,
    apply_vol_target,
)
from quant_research_stack.strategy_benchmark.zoo.vol_estimators import rolling_vol

_TARGET_DAILY_VOL = 0.01


@dataclass(frozen=True)
class ZooResult:
    specs: list[ZooStrategySpec]
    metrics: pl.DataFrame
    is_returns: NDArray[np.float32]
    oos_returns: NDArray[np.float32]
    pbo: dict[str, Any]
    wall_clock_sec: float


def build_signal(bars: pl.DataFrame, spec: ZooStrategySpec) -> pl.Series:
    raw = SIGNAL_FAMILIES[spec.family](bars, lookback=spec.lookback, threshold=spec.threshold)
    pos = apply_position_mode(raw, mode=spec.position_mode)
    vol = rolling_vol(bars, window=spec.lookback, estimator=spec.vol_estimator)
    pos = apply_vol_target(pos, vol=vol, target_daily_vol=_TARGET_DAILY_VOL)
    return apply_holding(pos, holding=spec.holding)


def run_zoo(*, universes: dict[str, pl.DataFrame], grid: GridConfig,
            oos_fraction: float = 0.3, embargo_days: int = 10,
            cost: BacktestCostConfig | None = None, n_partitions: int = 16) -> ZooResult:
    t0 = time.perf_counter()
    cost = cost or BacktestCostConfig()
    specs = enumerate_zoo(universes=tuple(universes.keys()), grid=grid)
    all_dates = sorted({d for u in universes.values() for d in u["date"].to_list()})
    T = len(all_dates)
    idx = {d: i for i, d in enumerate(all_dates)}
    split = int(T * (1.0 - oos_fraction))
    is_rows = list(range(0, split))
    oos_rows = list(range(min(split + embargo_days, T), T))
    full = np.zeros((T, len(specs)), dtype=np.float32)
    is_sharpe = np.zeros(len(specs))
    oos_sharpe = np.zeros(len(specs))
    turn = np.zeros(len(specs))
    for j, spec in enumerate(specs):
        bars = universes[spec.universe]
        sig = build_signal(bars, spec)
        res = run_single_asset_backtest(bars=bars, signals=sig, cost=cost)
        for k, d in enumerate(bars["date"].to_list()):
            full[idx[d], j] = np.float32(res.daily_net_return[k])
        is_sharpe[j] = _ann_sharpe(full[is_rows, j].astype(np.float64))
        oos_sharpe[j] = _ann_sharpe(full[oos_rows, j].astype(np.float64))
        turn[j] = res.annual_turnover
    metrics = pl.DataFrame({
        "strategy_id": [s.strategy_id for s in specs],
        "universe": [s.universe for s in specs], "family": [s.family for s in specs],
        "lookback": [s.lookback for s in specs], "threshold": [s.threshold for s in specs],
        "vol_estimator": [s.vol_estimator for s in specs], "position_mode": [s.position_mode for s in specs],
        "holding": [s.holding for s in specs], "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "annual_turnover": turn,
    })
    is_mat = full[is_rows]
    oos_mat = full[oos_rows]
    pbo = _pbo_dict(compute_pbo(returns=is_mat.astype(np.float64), n_partitions=n_partitions))
    return ZooResult(specs, metrics, is_mat, oos_mat, pbo, time.perf_counter() - t0)


def _ann_sharpe(r: NDArray[np.float64]) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r, ddof=1) == 0.0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0))


def _pbo_dict(p: Any) -> dict[str, Any]:
    return {"pbo_probability": float(p.pbo_probability), "median_logit": float(p.median_logit),
            "n_strategies": int(p.n_strategies), "failure_rate": float(p.failure_rate)}
