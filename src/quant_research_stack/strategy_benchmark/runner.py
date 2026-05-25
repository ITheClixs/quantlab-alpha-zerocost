"""Strategy-benchmark runner: enumerate → backtest → PBO → DSR → metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.backtest import (
    BacktestCostConfig,
    run_single_asset_backtest,
)
from quant_research_stack.strategy_benchmark.dsr import compute_dsr
from quant_research_stack.strategy_benchmark.enumeration import (
    StrategySpec,
    enumerate_strategies,
)
from quant_research_stack.strategy_benchmark.pbo import PBOResult, compute_pbo
from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES


@dataclass(frozen=True)
class BenchmarkRun:
    strategies: list[StrategySpec]
    metrics: pl.DataFrame
    returns_matrix: NDArray[np.float64]
    pbo: PBOResult
    universe_bars: dict[str, pl.DataFrame]
    wall_clock_sec: float


def _aligned_universe_returns(
    *,
    universes: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Trim every universe to its own bars; PBO needs equal-length series per
    universe, but ACROSS universes we'll pad-align later in the matrix step.
    """
    return universes


_DEFAULT_COST = BacktestCostConfig()


def run_benchmark(
    *,
    universes: dict[str, pl.DataFrame],
    cost: BacktestCostConfig | None = None,
    n_partitions: int = 16,
) -> BenchmarkRun:
    if cost is None:
        cost = _DEFAULT_COST
    """Run all 1500 strategies across the provided universes.

    Parameters
    ----------
    universes : dict[universe_name, bars_df]
        bars_df must have columns: date, symbol, open, high, low, close, volume.
    """
    t0 = time.perf_counter()
    strategy_specs = enumerate_strategies(list(universes.keys()))

    # We pad every strategy's daily return series onto a common index =
    # union of all universe dates. Missing days → 0 (flat). This lets PBO
    # treat the 1500 strategies as a single (T, S) matrix.
    all_dates: list = sorted(
        {d for u in universes.values() for d in u["date"].to_list()}
    )
    T = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    S = len(strategy_specs)
    returns_matrix = np.zeros((T, S), dtype=np.float64)

    # Metrics columns
    sharpe_arr = np.zeros(S, dtype=np.float64)
    sortino_arr = np.zeros(S, dtype=np.float64)
    total_ret = np.zeros(S, dtype=np.float64)
    max_dd = np.zeros(S, dtype=np.float64)
    hit_rate = np.zeros(S, dtype=np.float64)
    annual_turn = np.zeros(S, dtype=np.float64)
    n_trades = np.zeros(S, dtype=np.int64)

    # For each universe, generate signals once per (family, lookback, threshold)
    # cluster to avoid redundant recomputation.
    for col_i, spec in enumerate(strategy_specs):
        bars = universes[spec.universe]
        sig_fn = SIGNAL_FAMILIES[spec.signal_family]
        signals = sig_fn(bars, lookback=spec.lookback, threshold=spec.threshold)
        res = run_single_asset_backtest(bars=bars, signals=signals, cost=cost)
        # Place per-day returns into the common matrix
        for k, d in enumerate(bars["date"].to_list()):
            returns_matrix[date_to_idx[d], col_i] = res.daily_net_return[k]
        sharpe_arr[col_i] = res.sharpe_annualised
        sortino_arr[col_i] = res.sortino_annualised
        total_ret[col_i] = res.total_return
        max_dd[col_i] = res.max_drawdown
        hit_rate[col_i] = res.hit_rate
        annual_turn[col_i] = res.annual_turnover
        n_trades[col_i] = res.n_trades

    metrics = pl.DataFrame(
        {
            "strategy_id": [s.strategy_id for s in strategy_specs],
            "universe": [s.universe for s in strategy_specs],
            "signal_family": [s.signal_family for s in strategy_specs],
            "lookback": [s.lookback for s in strategy_specs],
            "threshold": [s.threshold for s in strategy_specs],
            "sharpe": sharpe_arr,
            "sortino": sortino_arr,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "annual_turnover": annual_turn,
            "n_trades": n_trades,
        }
    )

    pbo_result = compute_pbo(returns=returns_matrix, n_partitions=n_partitions)

    return BenchmarkRun(
        strategies=strategy_specs,
        metrics=metrics,
        returns_matrix=returns_matrix,
        pbo=pbo_result,
        universe_bars=universes,
        wall_clock_sec=time.perf_counter() - t0,
    )


def deflate_top_strategies(
    *,
    run: BenchmarkRun,
    top_k: int = 25,
) -> pl.DataFrame:
    """Compute DSR on the top-K strategies by raw Sharpe.

    Adds dsr_psr_zero and dsr columns to the metrics frame for the top-K rows.
    """
    metrics = run.metrics
    sorted_metrics = metrics.sort("sharpe", descending=True).head(top_k)
    sr_estimates = metrics["sharpe"].to_numpy().astype(np.float64)

    dsr_vals = []
    psr_vals = []
    for spec_id in sorted_metrics["strategy_id"].to_list():
        idx = metrics.with_row_index().filter(pl.col("strategy_id") == spec_id)["index"][0]
        dsr_res = compute_dsr(
            returns=run.returns_matrix[:, idx],
            sharpe_estimates=sr_estimates,
            selected_idx=int(idx),
        )
        dsr_vals.append(dsr_res.dsr)
        psr_vals.append(dsr_res.psr_zero)

    return sorted_metrics.with_columns(
        pl.Series("psr_zero", psr_vals),
        pl.Series("dsr", dsr_vals),
    )


def write_returns_matrix(*, run: BenchmarkRun, path: Path) -> None:
    """Persist the full (T × S) returns matrix as a parquet for later analysis."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = {s.strategy_id: run.returns_matrix[:, i] for i, s in enumerate(run.strategies)}
    pl.DataFrame(cols).write_parquet(path)
