from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.crypto_research.perps.reports import write_perp_reports


def _finite_array(values: Sequence[float] | NDArray[np.float64] | pl.Series) -> NDArray[np.float64]:
    if isinstance(values, pl.Series):
        array = values.to_numpy().astype(np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _safe_sharpe(returns: NDArray[np.float64]) -> float:
    finite = returns[np.isfinite(returns)]
    if finite.size < 2:
        return 0.0
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(finite) / sd * math.sqrt(252.0))


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-9), 1.0 - 1e-9)
    return float(math.log(clipped / (1.0 - clipped)))


def _chronological_blocks(row_count: int, n_partitions: int) -> NDArray[np.int64]:
    indices = np.arange(row_count, dtype=np.int64)
    return np.floor(indices * n_partitions / row_count).astype(np.int64)


def estimate_registry_pbo(
    returns: pl.DataFrame,
    *,
    strategy_columns: Sequence[str],
    n_partitions: int = 8,
) -> dict[str, Any]:
    missing = [column for column in strategy_columns if column not in returns.columns]
    if missing:
        raise ValueError(f"missing strategy return columns: {missing}")
    if n_partitions < 2:
        raise ValueError("n_partitions must be at least 2")
    if len(strategy_columns) < 2:
        return {
            "status": "not_estimated",
            "reason": "fewer than two strategies",
            "pbo_probability": None,
            "strategy_count": len(strategy_columns),
            "block_count": 0,
            "split_count": 0,
            "oos_rank_percentiles": [],
            "logit_ranks": [],
            "selected_strategy_ids": [],
        }
    if returns.height < n_partitions:
        return {
            "status": "not_estimated",
            "reason": "fewer rows than chronological partitions",
            "pbo_probability": None,
            "strategy_count": len(strategy_columns),
            "block_count": 0,
            "split_count": 0,
            "oos_rank_percentiles": [],
            "logit_ranks": [],
            "selected_strategy_ids": [],
        }

    ordered = returns.sort("event_index") if "event_index" in returns.columns else returns
    matrix = ordered.select(list(strategy_columns)).fill_null(0.0).to_numpy().astype(np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    block_ids = _chronological_blocks(matrix.shape[0], n_partitions)
    blocks = sorted(int(value) for value in np.unique(block_ids).tolist())
    if len(blocks) < 2:
        return {
            "status": "not_estimated",
            "reason": "fewer than two non-empty chronological blocks",
            "pbo_probability": None,
            "strategy_count": len(strategy_columns),
            "block_count": len(blocks),
            "split_count": 0,
            "oos_rank_percentiles": [],
            "logit_ranks": [],
            "selected_strategy_ids": [],
        }

    block_scores = np.zeros((len(blocks), len(strategy_columns)), dtype=np.float64)
    for row, block in enumerate(blocks):
        block_matrix = matrix[block_ids == block]
        block_scores[row] = np.mean(block_matrix, axis=0)

    half = len(blocks) // 2
    rank_percentiles: list[float] = []
    logit_ranks: list[float] = []
    selected_ids: list[str] = []
    block_positions = list(range(len(blocks)))
    for train_positions_tuple in itertools.combinations(block_positions, half):
        train_positions = set(train_positions_tuple)
        test_positions = [position for position in block_positions if position not in train_positions]
        train_scores = np.mean(block_scores[list(train_positions)], axis=0)
        test_scores = np.mean(block_scores[test_positions], axis=0)
        selected_index = int(np.argmax(train_scores))
        ordered_oos = np.argsort(-test_scores)
        rank = int(np.where(ordered_oos == selected_index)[0][0]) + 1
        percentile = 1.0 - ((rank - 1) / max(len(strategy_columns) - 1, 1))
        rank_percentiles.append(float(percentile))
        logit_ranks.append(_logit(float(percentile)))
        selected_ids.append(str(strategy_columns[selected_index]))

    pbo_probability = float(np.mean(np.asarray(logit_ranks, dtype=np.float64) < 0.0)) if logit_ranks else 1.0
    return {
        "status": "computed",
        "pbo_probability": pbo_probability,
        "strategy_count": len(strategy_columns),
        "block_count": len(blocks),
        "split_count": len(logit_ranks),
        "oos_rank_percentiles": rank_percentiles,
        "logit_ranks": logit_ranks,
        "selected_strategy_ids": selected_ids,
    }


def bootstrap_sharpe_payload(
    returns: Sequence[float] | NDArray[np.float64] | pl.Series,
    *,
    resamples: int = 1000,
    seed: int = 17,
    block_length: int | None = None,
) -> dict[str, Any]:
    finite = _finite_array(returns)
    point = _safe_sharpe(finite)
    if finite.size < 5:
        return {
            "status": "not_enough_observations",
            "point_sharpe": point,
            "ci_lower_95": 0.0,
            "ci_upper_95": 0.0,
            "resamples": 0,
            "observations": int(finite.size),
        }
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    block = block_length if block_length is not None else max(1, int(round(finite.size ** (1.0 / 3.0))))
    sharpes = np.empty(resamples, dtype=np.float64)
    for sample_idx in range(resamples):
        sample = np.empty(finite.size, dtype=np.float64)
        out_idx = 0
        while out_idx < finite.size:
            start = int(rng.integers(0, finite.size))
            take = min(block, finite.size - out_idx)
            source_idx = (start + np.arange(take)) % finite.size
            sample[out_idx : out_idx + take] = finite[source_idx]
            out_idx += take
        sharpes[sample_idx] = _safe_sharpe(sample)
    lower, upper = np.percentile(sharpes, [2.5, 97.5])
    return {
        "status": "computed",
        "point_sharpe": point,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "resamples": int(resamples),
        "observations": int(finite.size),
        "block_length": int(block),
    }


def _sample_skew_kurtosis(returns: NDArray[np.float64]) -> tuple[float, float]:
    if returns.size < 3:
        return 0.0, 3.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0, 3.0
    centered = returns - float(np.mean(returns))
    z = centered / sd
    return float(np.mean(z**3)), float(np.mean(z**4))


def _expected_max_sharpe(*, observations: int, trials: int) -> float:
    if observations < 2 or trials <= 1:
        return 0.0
    normal = NormalDist()
    trial_count = max(2, trials)
    gamma = 0.5772156649015329
    q1 = min(max(1.0 - 1.0 / trial_count, 1e-6), 1.0 - 1e-6)
    q2 = min(max(1.0 - 1.0 / (trial_count * math.e), 1e-6), 1.0 - 1e-6)
    expected_max_z = (1.0 - gamma) * normal.inv_cdf(q1) + gamma * normal.inv_cdf(q2)
    return max(0.0, expected_max_z / math.sqrt(max(observations - 1, 1)))


def deflated_sharpe_payload(
    returns: Sequence[float] | NDArray[np.float64] | pl.Series,
    *,
    trials: int = 1,
) -> dict[str, Any]:
    finite = _finite_array(returns)
    if finite.size < 5:
        return {
            "status": "not_enough_observations",
            "probability": 0.0,
            "observations": int(finite.size),
            "trials": int(trials),
        }
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return {
            "status": "zero_variance",
            "probability": 0.0,
            "observations": int(finite.size),
            "trials": int(trials),
        }
    daily_sharpe = float(np.mean(finite)) / sd
    skew, kurtosis = _sample_skew_kurtosis(finite)
    benchmark = _expected_max_sharpe(observations=finite.size, trials=trials)
    denominator = math.sqrt(max(1e-12, 1.0 - skew * daily_sharpe + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2))
    z_score = (daily_sharpe - benchmark) * math.sqrt(finite.size - 1.0) / denominator
    probability = NormalDist().cdf(z_score)
    return {
        "status": "computed_approximation",
        "probability": float(probability),
        "z_score": float(z_score),
        "observations": int(finite.size),
        "trials": int(trials),
        "annual_sharpe": float(daily_sharpe * math.sqrt(252.0)),
        "benchmark_annual_sharpe": float(benchmark * math.sqrt(252.0)),
        "sample_skew": skew,
        "sample_kurtosis": kurtosis,
    }


def concentration_payload(
    trades: pl.DataFrame,
    *,
    return_column: str = "net_return",
    symbol_column: str = "symbol",
    event_time_column: str = "event_time",
    max_day_share: float = 0.25,
    max_trade_share: float = 0.20,
    max_symbol_share: float = 0.35,
) -> dict[str, Any]:
    required = {return_column, symbol_column, event_time_column}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing concentration columns: {sorted(missing)}")
    if trades.is_empty():
        return {
            "status": "empty",
            "best_day_positive_pnl_share": 0.0,
            "best_trade_positive_pnl_share": 0.0,
            "best_symbol_positive_pnl_share": 0.0,
            "best_day_net_return": 0.0,
            "best_trade_net_return": 0.0,
            "best_symbol": "",
            "concentration_blocker": False,
        }

    base = trades.select([symbol_column, event_time_column, return_column]).with_columns(
        pl.col(return_column).cast(pl.Float64).alias(return_column)
    )
    daily = base.group_by(event_time_column).agg(pl.col(return_column).sum().alias("bucket_net_return"))
    symbol = base.group_by(symbol_column).agg(pl.col(return_column).sum().alias("symbol_net_return"))
    daily_values = daily["bucket_net_return"].to_numpy().astype(np.float64)
    symbol_values = symbol["symbol_net_return"].to_numpy().astype(np.float64)
    trade_values = base[return_column].to_numpy().astype(np.float64)

    positive_day = np.maximum(daily_values, 0.0)
    positive_symbol = np.maximum(symbol_values, 0.0)
    positive_trade = np.maximum(trade_values, 0.0)
    day_sum = float(np.sum(positive_day))
    symbol_sum = float(np.sum(positive_symbol))
    trade_sum = float(np.sum(positive_trade))
    symbol_sorted = symbol.sort("symbol_net_return", descending=True)
    best_day_share = float(np.max(positive_day) / day_sum) if day_sum > 0.0 else 0.0
    best_trade_share = float(np.max(positive_trade) / trade_sum) if trade_sum > 0.0 else 0.0
    best_symbol_share = float(np.max(positive_symbol) / symbol_sum) if symbol_sum > 0.0 else 0.0
    blocker = best_day_share > max_day_share or best_trade_share > max_trade_share or best_symbol_share > max_symbol_share
    return {
        "status": "computed",
        "best_day_positive_pnl_share": best_day_share,
        "best_trade_positive_pnl_share": best_trade_share,
        "best_symbol_positive_pnl_share": best_symbol_share,
        "best_day_net_return": float(np.max(daily_values)) if daily_values.size else 0.0,
        "best_trade_net_return": float(np.max(trade_values)) if trade_values.size else 0.0,
        "best_symbol": str(symbol_sorted[symbol_column][0]) if symbol_sorted.height else "",
        "concentration_blocker": bool(blocker),
        "max_day_share": max_day_share,
        "max_trade_share": max_trade_share,
        "max_symbol_share": max_symbol_share,
    }


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def classify_perp_candidate(metrics: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    pbo = _metric_float(metrics, "pbo_probability")
    bootstrap_lower = _metric_float(metrics, "bootstrap_ci_lower_95")
    sharpe = _metric_float(metrics, "net_daily_sharpe")
    net_total = _metric_float(metrics, "net_total_return")
    cost_2x = _metric_float(metrics, "cost_2x_net_total_return")
    delay_1 = _metric_float(metrics, "delay_1_event_net_total_return")
    dsr_probability = _metric_float(metrics, "deflated_sharpe_probability")
    concentration_blocker = bool(metrics.get("concentration_blocker", False))

    if pbo is None or pbo > 0.25:
        blockers.append("missing_or_high_pbo")
    if bootstrap_lower is None or bootstrap_lower <= 0.0:
        blockers.append("bootstrap_ci_not_positive")
    if dsr_probability is not None and dsr_probability < 0.95:
        blockers.append("low_deflated_sharpe_probability")
    if sharpe is None or sharpe < 1.5:
        blockers.append("low_net_daily_sharpe")
    if net_total is None or net_total <= 0.0:
        blockers.append("non_positive_net_total_return")
    if cost_2x is None or cost_2x <= 0.0:
        blockers.append("fails_2x_cost_stress")
    if delay_1 is None or delay_1 <= 0.0:
        blockers.append("fails_1_event_delay_stress")
    if concentration_blocker:
        blockers.append("performance_concentration")
    blockers.append("free_data_research_only")

    hard_blockers = [blocker for blocker in blockers if blocker != "free_data_research_only"]
    return {
        "strategy_id": metrics.get("strategy_id", metrics.get("name", "")),
        "research_candidate": len(hard_blockers) == 0,
        "promotion_eligible": False,
        "production_candidate": False,
        "blockers": blockers,
    }


__all__ = [
    "bootstrap_sharpe_payload",
    "classify_perp_candidate",
    "concentration_payload",
    "deflated_sharpe_payload",
    "estimate_registry_pbo",
    "write_perp_reports",
]
