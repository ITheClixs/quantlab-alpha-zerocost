"""Strict diagnostics for signal_research walk-forward prediction artifacts.

The functions in this module evaluate already-produced predictions. They do
not tune the model and they intentionally keep promotion blocked when the
available artifacts are daily OHLCV proxies rather than point-in-time
institutional execution data.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)
from quant_research_stack.strategy_benchmark.pbo import compute_pbo

_DISCLAIMER = (
    "The project may be production-intended, but this artifact is research output only "
    "and is not automatically investment advice. External advisory or capital-management "
    "use requires legal, regulatory, licensing, and compliance review before deployment."
)


@dataclass(frozen=True)
class StrictBacktestDiagnosticsConfig:
    """Controls hedge-fund-style diagnostics for a fixed model artifact."""

    market_name: str
    cost_bps_one_way: float = 1.0
    cost_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    delay_bars: tuple[int, ...] = (0, 1)
    holding_horizon_days: int = 20
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42
    multiple_testing_trials: int = 1
    random_seed: int = 42


@dataclass(frozen=True)
class StrictBacktestDiagnosticsResult:
    """Materialized strict diagnostics for one market/profile."""

    market_name: str
    summary: dict[str, Any]
    variant_metrics: pl.DataFrame
    daily_returns: pl.DataFrame
    trade_audit: pl.DataFrame


def _safe_sharpe(returns: NDArray[np.float64]) -> float:
    finite = returns[np.isfinite(returns)]
    if finite.size < 2:
        return 0.0
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(finite) / sd * np.sqrt(252.0))


def _compound(returns: NDArray[np.float64]) -> float:
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return 0.0
    return float(np.prod(1.0 + finite) - 1.0)


def _profit_factor(returns: NDArray[np.float64]) -> float:
    wins = float(np.sum(returns[returns > 0.0]))
    losses = float(np.sum(returns[returns < 0.0]))
    if losses == 0.0:
        return math.inf if wins > 0.0 else 0.0
    return wins / abs(losses)


def _drawdown_stats(returns: NDArray[np.float64]) -> tuple[float, int]:
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return 0.0, 0
    equity = np.cumprod(1.0 + finite)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peaks, 1e-12) - 1.0
    duration = 0
    max_duration = 0
    for value in drawdown:
        if value < 0.0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return float(np.min(drawdown)), max_duration


def _month_returns(daily: pl.DataFrame) -> NDArray[np.float64]:
    if daily.is_empty():
        return np.array([], dtype=np.float64)
    monthly = (
        daily.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(((pl.col("daily_net_return") + 1.0).product() - 1.0).alias("monthly_net_return"))
        .sort("month")
    )
    return monthly["monthly_net_return"].to_numpy().astype(np.float64)


def _clean_predictions(predictions: pl.DataFrame) -> pl.DataFrame:
    required = {
        "date",
        "symbol",
        "future_return_horizon",
        "meta_position",
        "meta_probability",
        "primary_position",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction artifact missing required columns: {sorted(missing)}")
    frame = predictions.sort(["symbol", "date"])
    if "entry_close_proxy" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("entry_close_proxy"))
    if "fold" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("fold"))
    return frame


def _variant_frame(
    predictions: pl.DataFrame,
    *,
    variant: str,
    cost_bps_one_way: float,
    cost_multiplier: float,
    delay_bars: int = 0,
    invert: bool = False,
    random_seed: int | None = None,
) -> pl.DataFrame:
    frame = _clean_predictions(predictions)
    if random_seed is None:
        frame = frame.with_columns(
            pl.col("meta_position")
            .shift(delay_bars)
            .over("symbol")
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("diagnostic_position")
        )
        if invert:
            frame = frame.with_columns((-pl.col("diagnostic_position")).alias("diagnostic_position"))
    else:
        base = frame["meta_position"].to_numpy().astype(np.float64)
        rng = np.random.default_rng(random_seed)
        random_side = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=base.size)
        random_position = np.where(base != 0.0, random_side, 0.0)
        frame = frame.with_columns(pl.Series("diagnostic_position", random_position))

    position = frame["diagnostic_position"].to_numpy().astype(np.float64)
    future_return = frame["future_return_horizon"].to_numpy().astype(np.float64)
    gross_return = position * future_return
    round_trip_cost = np.where(
        position != 0.0,
        2.0 * cost_bps_one_way * 1e-4 * cost_multiplier,
        0.0,
    )
    net_return = gross_return - round_trip_cost
    direction_correct = np.where(position != 0.0, gross_return > 0.0, False)
    return frame.with_columns(
        [
            pl.lit(variant).alias("diagnostic_variant"),
            pl.Series("diagnostic_gross_return", gross_return),
            pl.Series("diagnostic_round_trip_cost", round_trip_cost),
            pl.Series("diagnostic_net_return", net_return),
            pl.Series("diagnostic_direction_correct", direction_correct),
        ]
    )


def _daily_returns(frame: pl.DataFrame, *, market_name: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            {
                "market": [],
                "diagnostic_variant": [],
                "date": [],
                "daily_gross_return": [],
                "daily_net_return": [],
                "active_positions": [],
            }
        )
    return (
        frame.group_by(["diagnostic_variant", "date"])
        .agg(
            [
                pl.col("diagnostic_gross_return").mean().alias("daily_gross_return"),
                pl.col("diagnostic_net_return").mean().alias("daily_net_return"),
                (pl.col("diagnostic_position") != 0.0).sum().alias("active_positions"),
            ]
        )
        .with_columns(pl.lit(market_name).alias("market"))
        .select(
            [
                "market",
                "diagnostic_variant",
                "date",
                "daily_gross_return",
                "daily_net_return",
                "active_positions",
            ]
        )
        .sort(["diagnostic_variant", "date"])
    )


def _metrics_for_variant(
    *,
    frame: pl.DataFrame,
    daily: pl.DataFrame,
    market_name: str,
    variant: str,
    cost_multiplier: float,
    delay_bars: int,
) -> dict[str, Any]:
    variant_daily = daily.filter(pl.col("diagnostic_variant") == variant)
    net_daily = variant_daily["daily_net_return"].to_numpy().astype(np.float64)
    gross_daily = variant_daily["daily_gross_return"].to_numpy().astype(np.float64)
    trades = frame.filter(pl.col("diagnostic_position") != 0.0)
    trade_net = trades["diagnostic_net_return"].to_numpy().astype(np.float64)
    trade_gross = trades["diagnostic_gross_return"].to_numpy().astype(np.float64)
    max_drawdown, drawdown_duration = _drawdown_stats(net_daily)
    monthly = _month_returns(variant_daily)
    long_trades = trades.filter(pl.col("diagnostic_position") > 0.0)
    short_trades = trades.filter(pl.col("diagnostic_position") < 0.0)
    net_total = _compound(net_daily)
    return {
        "market": market_name,
        "variant": variant,
        "cost_multiplier": cost_multiplier,
        "delay_bars": delay_bars,
        "daily_count": int(variant_daily.height),
        "prediction_rows": int(frame.height),
        "trade_count": int(trades.height),
        "turnover": float(trades.height / max(frame.height, 1)),
        "gross_total_return": _compound(gross_daily),
        "net_total_return": net_total,
        "avg_monthly_net_return": float(np.mean(monthly)) if monthly.size else 0.0,
        "net_daily_sharpe": _safe_sharpe(net_daily),
        "trade_sharpe": _safe_sharpe(trade_net),
        "max_drawdown": max_drawdown,
        "calmar": float(net_total / abs(max_drawdown)) if max_drawdown < 0.0 else 0.0,
        "drawdown_duration_days": int(drawdown_duration),
        "gross_hit_rate": float(np.mean(trade_gross > 0.0)) if trade_gross.size else 0.0,
        "net_hit_rate": float(np.mean(trade_net > 0.0)) if trade_net.size else 0.0,
        "directional_accuracy": float(np.mean(trade_gross > 0.0)) if trade_gross.size else 0.0,
        "avg_trade_gross_return": float(np.mean(trade_gross)) if trade_gross.size else 0.0,
        "avg_trade_net_return": float(np.mean(trade_net)) if trade_net.size else 0.0,
        "avg_win": float(np.mean(trade_net[trade_net > 0.0])) if np.any(trade_net > 0.0) else 0.0,
        "avg_loss": float(np.mean(trade_net[trade_net < 0.0])) if np.any(trade_net < 0.0) else 0.0,
        "profit_factor": _profit_factor(trade_net),
        "long_trade_count": int(long_trades.height),
        "short_trade_count": int(short_trades.height),
        "long_net_pnl_sum": float(long_trades["diagnostic_net_return"].sum()) if long_trades.height else 0.0,
        "short_net_pnl_sum": float(short_trades["diagnostic_net_return"].sum()) if short_trades.height else 0.0,
    }


def _concentration_payload(base_frame: pl.DataFrame, base_daily: pl.DataFrame) -> dict[str, Any]:
    if base_daily.is_empty():
        return {
            "best_day_positive_pnl_share": 0.0,
            "best_symbol_positive_pnl_share": 0.0,
            "remove_best_day_net_total_return": 0.0,
            "best_day_net_return": 0.0,
            "best_symbol": "",
        }
    daily_net = base_daily["daily_net_return"].to_numpy().astype(np.float64)
    best_day_idx = int(np.argmax(daily_net))
    without_best_day = np.delete(daily_net, best_day_idx)
    positive_day = np.maximum(daily_net, 0.0)
    positive_day_sum = float(np.sum(positive_day))
    symbol_pnl = (
        base_frame.group_by("symbol")
        .agg(pl.col("diagnostic_net_return").sum().alias("symbol_net_pnl"))
        .sort("symbol_net_pnl", descending=True)
    )
    symbol_net = symbol_pnl["symbol_net_pnl"].to_numpy().astype(np.float64)
    positive_symbol = np.maximum(symbol_net, 0.0)
    positive_symbol_sum = float(np.sum(positive_symbol))
    best_symbol = str(symbol_pnl["symbol"][0]) if symbol_pnl.height else ""
    return {
        "best_day_positive_pnl_share": (
            float(np.max(positive_day) / positive_day_sum) if positive_day_sum > 0.0 else 0.0
        ),
        "best_symbol_positive_pnl_share": (
            float(np.max(positive_symbol) / positive_symbol_sum) if positive_symbol_sum > 0.0 else 0.0
        ),
        "remove_best_day_net_total_return": _compound(without_best_day),
        "best_day_net_return": float(daily_net[best_day_idx]) if daily_net.size else 0.0,
        "best_symbol": best_symbol,
    }


def _bootstrap_payload(
    returns: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    finite = returns[np.isfinite(returns)]
    if finite.size < 5:
        return {
            "status": "not_enough_observations",
            "point_sharpe": _safe_sharpe(finite),
            "ci_lower_95": 0.0,
            "ci_upper_95": 0.0,
            "resamples": 0,
        }
    result = bootstrap_sharpe_ci(
        returns=finite,
        config=BootstrapConfig(n_resamples=resamples, seed=seed),
    )
    return {
        "status": "computed",
        "point_sharpe": result.point_sharpe,
        "ci_lower_95": result.ci_lower_95,
        "ci_upper_95": result.ci_upper_95,
        "resamples": resamples,
    }


def _sample_skew_kurtosis(returns: NDArray[np.float64]) -> tuple[float, float]:
    finite = returns[np.isfinite(returns)]
    if finite.size < 3:
        return 0.0, 3.0
    centered = finite - float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return 0.0, 3.0
    z = centered / sd
    return float(np.mean(z**3)), float(np.mean(z**4))


def _expected_max_daily_sharpe(*, observations: int, trials: int) -> float:
    if observations < 2 or trials <= 1:
        return 0.0
    normal = NormalDist()
    trial_count = max(2, trials)
    gamma = 0.5772156649015329
    q1 = min(max(1.0 - 1.0 / trial_count, 1e-6), 1.0 - 1e-6)
    q2 = min(max(1.0 - 1.0 / (trial_count * math.e), 1e-6), 1.0 - 1e-6)
    expected_max_z = (1.0 - gamma) * normal.inv_cdf(q1) + gamma * normal.inv_cdf(q2)
    return max(0.0, expected_max_z / math.sqrt(max(observations - 1, 1)))


def _deflated_sharpe_payload(
    returns: NDArray[np.float64],
    *,
    trials: int,
) -> dict[str, Any]:
    finite = returns[np.isfinite(returns)]
    if finite.size < 5:
        return {
            "status": "not_enough_observations",
            "probability": 0.0,
            "observations": int(finite.size),
            "trials": int(trials),
            "benchmark_annual_sharpe": 0.0,
        }
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return {
            "status": "zero_variance",
            "probability": 0.0,
            "observations": int(finite.size),
            "trials": int(trials),
            "benchmark_annual_sharpe": 0.0,
        }
    daily_sharpe = float(np.mean(finite)) / sd
    skew, kurtosis = _sample_skew_kurtosis(finite)
    benchmark_daily = _expected_max_daily_sharpe(observations=finite.size, trials=trials)
    denom = math.sqrt(max(1e-12, 1.0 - skew * daily_sharpe + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2))
    z_score = (daily_sharpe - benchmark_daily) * math.sqrt(finite.size - 1.0) / denom
    probability = NormalDist().cdf(z_score)
    return {
        "status": "computed_approximation",
        "probability": float(probability),
        "z_score": float(z_score),
        "observations": int(finite.size),
        "trials": int(trials),
        "benchmark_annual_sharpe": float(benchmark_daily * math.sqrt(252.0)),
        "sample_skew": skew,
        "sample_kurtosis": kurtosis,
        "note": "Approximate DSR/PSR-style diagnostic; promotion still requires full search-process DSR accounting.",
    }


def _pbo_payload(daily_returns: pl.DataFrame) -> dict[str, Any]:
    if daily_returns.is_empty():
        return {"status": "not_estimated", "reason": "empty daily returns"}
    wide = (
        daily_returns.pivot(
            index="date",
            on="diagnostic_variant",
            values="daily_net_return",
            aggregate_function="first",
        )
        .sort("date")
        .fill_null(0.0)
    )
    value_columns = [c for c in wide.columns if c != "date"]
    if len(value_columns) < 3:
        return {"status": "not_estimated", "reason": "fewer than three diagnostic variants"}
    returns = wide.select(value_columns).to_numpy().astype(np.float64)
    partitions = next((p for p in (8, 6, 4, 2) if returns.shape[0] // p >= 5), None)
    if partitions is None:
        return {"status": "not_estimated", "reason": "not enough daily observations for partitions"}
    try:
        result = compute_pbo(returns=returns, n_partitions=partitions)
    except ValueError as exc:
        return {"status": "not_estimated", "reason": str(exc)}
    return {
        "status": "diagnostic_variant_matrix_only",
        "pbo_probability": result.pbo_probability,
        "median_logit": result.median_logit,
        "n_partitions": result.n_partitions,
        "n_combinations": result.n_combinations,
        "n_strategies": result.n_strategies,
        "failure_rate": result.failure_rate,
        "note": "This is not a full search-process PBO and cannot satisfy the promotion gate.",
    }


def run_strict_backtest_diagnostics(
    predictions: pl.DataFrame,
    *,
    config: StrictBacktestDiagnosticsConfig,
) -> StrictBacktestDiagnosticsResult:
    """Run cost, delay, sanity-check, and audit diagnostics for one profile."""

    variant_frames: list[pl.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    for multiplier in config.cost_multipliers:
        variant = f"cost_{multiplier:g}x"
        frame = _variant_frame(
            predictions,
            variant=variant,
            cost_bps_one_way=config.cost_bps_one_way,
            cost_multiplier=multiplier,
        )
        daily = _daily_returns(frame, market_name=config.market_name)
        variant_frames.append(frame)
        variant_rows.append(
            _metrics_for_variant(
                frame=frame,
                daily=daily,
                market_name=config.market_name,
                variant=variant,
                cost_multiplier=multiplier,
                delay_bars=0,
            )
        )

    base_frame = _variant_frame(
        predictions,
        variant="base",
        cost_bps_one_way=config.cost_bps_one_way,
        cost_multiplier=1.0,
    )
    base_daily = _daily_returns(base_frame, market_name=config.market_name)

    for delay in config.delay_bars:
        if delay <= 0:
            continue
        variant = f"delay_{delay}_bar"
        frame = _variant_frame(
            predictions,
            variant=variant,
            cost_bps_one_way=config.cost_bps_one_way,
            cost_multiplier=1.0,
            delay_bars=delay,
        )
        daily = _daily_returns(frame, market_name=config.market_name)
        variant_frames.append(frame)
        variant_rows.append(
            _metrics_for_variant(
                frame=frame,
                daily=daily,
                market_name=config.market_name,
                variant=variant,
                cost_multiplier=1.0,
                delay_bars=delay,
            )
        )

    inverted_frame = _variant_frame(
        predictions,
        variant="inverted_signal",
        cost_bps_one_way=config.cost_bps_one_way,
        cost_multiplier=1.0,
        invert=True,
    )
    inverted_daily = _daily_returns(inverted_frame, market_name=config.market_name)
    variant_frames.append(inverted_frame)
    variant_rows.append(
        _metrics_for_variant(
            frame=inverted_frame,
            daily=inverted_daily,
            market_name=config.market_name,
            variant="inverted_signal",
            cost_multiplier=1.0,
            delay_bars=0,
        )
    )

    random_frame = _variant_frame(
        predictions,
        variant="random_same_trade_mask",
        cost_bps_one_way=config.cost_bps_one_way,
        cost_multiplier=1.0,
        random_seed=config.random_seed,
    )
    random_daily = _daily_returns(random_frame, market_name=config.market_name)
    variant_frames.append(random_frame)
    variant_rows.append(
        _metrics_for_variant(
            frame=random_frame,
            daily=random_daily,
            market_name=config.market_name,
            variant="random_same_trade_mask",
            cost_multiplier=1.0,
            delay_bars=0,
        )
    )

    all_variant_frames = pl.concat(variant_frames, how="vertical") if variant_frames else pl.DataFrame()
    all_daily = _daily_returns(all_variant_frames, market_name=config.market_name)
    variant_metrics = pl.DataFrame(variant_rows)
    base_metrics = variant_metrics.filter(pl.col("variant") == "cost_1x")
    if base_metrics.is_empty():
        base_metrics = variant_metrics.filter(pl.col("variant") == "base")
    base_daily_for_bootstrap = all_daily.filter(pl.col("diagnostic_variant") == "cost_1x")
    base_net = base_daily_for_bootstrap["daily_net_return"].to_numpy().astype(np.float64)
    concentration = _concentration_payload(base_frame, base_daily)
    bootstrap = _bootstrap_payload(
        base_net,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    dsr = _deflated_sharpe_payload(
        base_net,
        trials=max(config.multiple_testing_trials, len(variant_rows)),
    )
    pbo = _pbo_payload(all_daily)
    trade_audit = _trade_audit_frame(base_frame, config=config)

    base_row: dict[str, Any] = base_metrics.to_dicts()[0] if not base_metrics.is_empty() else {}
    promotion_blockers = [
        "research_validation_only: this benchmark uses daily OHLCV proxy artifacts, not point-in-time bid/ask/order-book data",
        "permanent holdout was not isolated for this Nasdaq/S&P profile benchmark",
        "PBO is diagnostic-only for robustness variants and does not account for the full historical search process",
        "DSR is an approximation unless wired to the complete strategy registry/trial ledger",
        "spread, slippage, borrow/funding, futures roll, queue position, and market impact are proxy-modeled, not venue-verified",
    ]
    summary = {
        "market": config.market_name,
        "status": "research_validation_only",
        "promotion_eligible": False,
        "paper_trade_candidate": False,
        "production_candidate": False,
        "promotion_blockers": promotion_blockers,
        "base_metrics": base_row,
        "concentration": concentration,
        "bootstrap_sharpe_ci": bootstrap,
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "config": asdict(config),
        "trade_audit_rows": int(trade_audit.height),
        "disclaimer": _DISCLAIMER,
    }
    return StrictBacktestDiagnosticsResult(
        market_name=config.market_name,
        summary=summary,
        variant_metrics=variant_metrics,
        daily_returns=all_daily,
        trade_audit=trade_audit,
    )


def _trade_audit_frame(
    base_frame: pl.DataFrame,
    *,
    config: StrictBacktestDiagnosticsConfig,
) -> pl.DataFrame:
    if base_frame.is_empty():
        return pl.DataFrame()
    audited = base_frame.with_columns(
        [
            pl.when(pl.col("diagnostic_position") > 0.0)
            .then(pl.lit("long"))
            .when(pl.col("diagnostic_position") < 0.0)
            .then(pl.lit("short"))
            .otherwise(pl.lit("flat"))
            .alias("side"),
            (
                pl.col("entry_close_proxy")
                * (1.0 + pl.col("future_return_horizon"))
            ).alias("exit_close_proxy"),
            pl.lit(config.holding_horizon_days).alias("holding_horizon_days"),
            pl.lit("close_to_close_daily_proxy").alias("execution_price_semantics"),
            pl.lit("not_available_in_daily_ohlcv_artifact").alias("bid_ask_source"),
            pl.col("diagnostic_round_trip_cost").alias("spread_slippage_cost_proxy"),
            pl.lit(0.0).alias("fee_cost_proxy"),
            pl.lit(0.0).alias("borrow_or_funding_cost_proxy"),
        ]
    )
    return audited.filter(pl.col("diagnostic_position") != 0.0).select(
        [
            "date",
            "fold",
            "symbol",
            "side",
            "meta_probability",
            "primary_position",
            "diagnostic_position",
            "entry_close_proxy",
            "exit_close_proxy",
            "future_return_horizon",
            "diagnostic_gross_return",
            "diagnostic_round_trip_cost",
            "spread_slippage_cost_proxy",
            "fee_cost_proxy",
            "borrow_or_funding_cost_proxy",
            "diagnostic_net_return",
            "holding_horizon_days",
            "diagnostic_direction_correct",
            "execution_price_semantics",
            "bid_ask_source",
        ]
    )


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.3%}"
    except (TypeError, ValueError):
        return str(value)


def _metrics_table(metrics: pl.DataFrame) -> list[str]:
    columns = [
        "variant",
        "trade_count",
        "gross_total_return",
        "net_total_return",
        "avg_monthly_net_return",
        "net_daily_sharpe",
        "max_drawdown",
        "net_hit_rate",
        "profit_factor",
    ]
    lines = [
        "| variant | trades | gross return | net return | avg monthly net | daily Sharpe | max DD | net hit | profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.select(columns).iter_rows(named=True):
        lines.append(
            "| "
            f"{row['variant']} | "
            f"{int(row['trade_count'])} | "
            f"{_fmt_pct(row['gross_total_return'])} | "
            f"{_fmt_pct(row['net_total_return'])} | "
            f"{_fmt_pct(row['avg_monthly_net_return'])} | "
            f"{_fmt_float(row['net_daily_sharpe'])} | "
            f"{_fmt_pct(row['max_drawdown'])} | "
            f"{_fmt_pct(row['net_hit_rate'])} | "
            f"{_fmt_float(row['profit_factor'])} |"
        )
    return lines


def render_strict_backtest_report(result: StrictBacktestDiagnosticsResult) -> str:
    """Render a market-level Markdown report."""

    summary = result.summary
    base = summary.get("base_metrics", {})
    concentration = summary.get("concentration", {})
    bootstrap = summary.get("bootstrap_sharpe_ci", {})
    dsr = summary.get("deflated_sharpe", {})
    pbo = summary.get("pbo", {})
    lines = [
        f"# {result.market_name} Strict Meta-Label Backtest",
        "",
        "## Status",
        f"- status: `{summary['status']}`",
        f"- promotion_eligible: `{summary['promotion_eligible']}`",
        f"- paper_trade_candidate: `{summary['paper_trade_candidate']}`",
        f"- production_candidate: `{summary['production_candidate']}`",
        f"- base net return: `{_fmt_pct(base.get('net_total_return', 0.0))}`",
        f"- base daily Sharpe: `{_fmt_float(base.get('net_daily_sharpe', 0.0))}`",
        f"- base average monthly net return: `{_fmt_pct(base.get('avg_monthly_net_return', 0.0))}`",
        f"- trade audit rows: `{summary['trade_audit_rows']}`",
        "",
        "## Return And Execution Semantics",
        "- returns are equal-weight averages of event-level forward-horizon returns grouped by signal date",
        "- event rows may overlap because the triple-barrier horizon is longer than one daily bar",
        "- execution is a close-to-close daily proxy; true bid/ask, next-open slippage, queue, roll, borrow, funding, and impact are unavailable in this artifact",
        "- daily Sharpe and monthly net are therefore research diagnostics, not production portfolio statistics",
        "",
        "## Cost, Delay, And Sanity Checks",
        *_metrics_table(result.variant_metrics),
        "",
        "## Bootstrap, DSR, And PBO",
        (
            f"- bootstrap Sharpe 95% CI: `{_fmt_float(bootstrap.get('ci_lower_95', 0.0))}` "
            f"to `{_fmt_float(bootstrap.get('ci_upper_95', 0.0))}` "
            f"(status `{bootstrap.get('status', '')}`)"
        ),
        (
            f"- deflated Sharpe probability: `{_fmt_float(dsr.get('probability', 0.0))}` "
            f"using `{dsr.get('trials', 0)}` trials "
            f"(status `{dsr.get('status', '')}`)"
        ),
        (
            f"- PBO status: `{pbo.get('status', '')}`; "
            f"PBO probability: `{_fmt_float(pbo.get('pbo_probability', 0.0))}`"
        ),
        f"- PBO note: {pbo.get('note', pbo.get('reason', ''))}",
        "",
        "## Concentration",
        f"- best-day positive PnL share: `{_fmt_pct(concentration.get('best_day_positive_pnl_share', 0.0))}`",
        f"- best-symbol positive PnL share: `{_fmt_pct(concentration.get('best_symbol_positive_pnl_share', 0.0))}`",
        f"- best symbol: `{concentration.get('best_symbol', '')}`",
        f"- net return after removing best day: `{_fmt_pct(concentration.get('remove_best_day_net_total_return', 0.0))}`",
        "",
        "## Promotion Blockers",
    ]
    lines.extend([f"- {reason}" for reason in summary["promotion_blockers"]])
    lines.extend(["", "## Disclaimer", _DISCLAIMER, ""])
    return "\n".join(lines)


def write_strict_backtest_artifacts(
    result: StrictBacktestDiagnosticsResult,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist strict diagnostics as JSON, Parquet, and Markdown artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "strict_diagnostics.json"
    metrics_path = output_dir / "strict_variant_metrics.parquet"
    daily_path = output_dir / "strict_daily_returns.parquet"
    audit_path = output_dir / "strict_trade_audit.parquet"
    report_path = output_dir / "strict_backtest_report.md"

    summary_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True, default=str) + "\n")
    result.variant_metrics.write_parquet(metrics_path)
    result.daily_returns.write_parquet(daily_path)
    result.trade_audit.write_parquet(audit_path)
    report_path.write_text(render_strict_backtest_report(result))
    return {
        "summary": summary_path,
        "variant_metrics": metrics_path,
        "daily_returns": daily_path,
        "trade_audit": audit_path,
        "report": report_path,
    }
