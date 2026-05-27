from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class PerpBacktestConfig:
    prediction_column: str = "prediction"
    symbol_column: str = "symbol"
    event_time_column: str = "event_time"
    horizon: int = 1
    min_signal_abs: float = 0.0
    min_edge_to_cost_ratio: float | None = None
    max_relative_spread: float | None = None
    min_top_of_book_depth: float | None = None
    fee_bps: float = 4.0
    slippage_bps: float = 1.0
    cost_multiplier: float = 1.0
    latency_events: int = 0
    invert_signal: bool = False


@dataclass(frozen=True)
class PerpBacktestResult:
    trades: pl.DataFrame
    metrics: dict[str, float | int]


def _safe_sharpe(returns: NDArray[np.float64]) -> float:
    finite = returns[np.isfinite(returns)]
    if finite.size < 2:
        return 0.0
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(finite) / sd * np.sqrt(float(finite.size)))


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


def _max_drawdown(returns: NDArray[np.float64]) -> float:
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + finite)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(peaks, 1e-12) - 1.0
    return float(np.min(drawdowns))


def _empty_result() -> PerpBacktestResult:
    return PerpBacktestResult(
        trades=pl.DataFrame(),
        metrics={
            "trade_count": 0,
            "gross_total_return": 0.0,
            "net_total_return": 0.0,
            "trade_sharpe": 0.0,
            "gross_hit_rate": 0.0,
            "net_hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "long_net_pnl_sum": 0.0,
            "short_net_pnl_sum": 0.0,
        },
    )


def _validate_columns(frame: pl.DataFrame, config: PerpBacktestConfig) -> None:
    required = {
        config.symbol_column,
        config.event_time_column,
        config.prediction_column,
        "best_bid",
        "best_ask",
        f"future_best_bid_{config.horizon}",
        f"future_best_ask_{config.horizon}",
        "relative_spread",
        "best_bid_size",
        "best_ask_size",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing backtest columns: {sorted(missing)}")


def _prepare_frame(frame: pl.DataFrame, config: PerpBacktestConfig) -> pl.DataFrame:
    prediction = pl.col(config.prediction_column).cast(pl.Float64)
    if config.latency_events > 0:
        prediction = prediction.shift(config.latency_events).over(config.symbol_column)
    if config.invert_signal:
        prediction = -prediction
    out = (
        frame.sort([config.symbol_column, config.event_time_column])
        .with_columns(
            [
                pl.col("best_bid").cast(pl.Float64),
                pl.col("best_ask").cast(pl.Float64),
                pl.col(f"future_best_bid_{config.horizon}").cast(pl.Float64),
                pl.col(f"future_best_ask_{config.horizon}").cast(pl.Float64),
                pl.col("best_bid_size").cast(pl.Float64),
                pl.col("best_ask_size").cast(pl.Float64),
                prediction.alias("diagnostic_prediction"),
            ]
        )
        .with_columns(pl.min_horizontal("best_bid_size", "best_ask_size").alias("top_of_book_depth"))
    )
    return out


def _apply_filters(frame: pl.DataFrame, config: PerpBacktestConfig, *, round_trip_cost: float) -> pl.DataFrame:
    out = frame.drop_nulls(
        [
            "diagnostic_prediction",
            "best_bid",
            "best_ask",
            f"future_best_bid_{config.horizon}",
            f"future_best_ask_{config.horizon}",
        ]
    )
    out = out.filter(pl.col("diagnostic_prediction").abs() > config.min_signal_abs)
    if config.max_relative_spread is not None:
        out = out.filter(pl.col("relative_spread") <= config.max_relative_spread)
    if config.min_top_of_book_depth is not None:
        out = out.filter(pl.col("top_of_book_depth") >= config.min_top_of_book_depth)
    if config.min_edge_to_cost_ratio is not None and round_trip_cost > 0.0:
        out = out.filter(pl.col("diagnostic_prediction").abs() >= config.min_edge_to_cost_ratio * round_trip_cost)
    return out


def _build_trades(frame: pl.DataFrame, config: PerpBacktestConfig, *, round_trip_cost: float) -> pl.DataFrame:
    future_bid = pl.col(f"future_best_bid_{config.horizon}")
    future_ask = pl.col(f"future_best_ask_{config.horizon}")
    side_expr = (
        pl.when(pl.col("diagnostic_prediction") > 0.0)
        .then(pl.lit("long"))
        .when(pl.col("diagnostic_prediction") < 0.0)
        .then(pl.lit("short"))
        .otherwise(pl.lit("flat"))
    )
    entry_price = pl.when(pl.col("diagnostic_prediction") > 0.0).then(pl.col("best_ask")).otherwise(pl.col("best_bid"))
    exit_price = pl.when(pl.col("diagnostic_prediction") > 0.0).then(future_bid).otherwise(future_ask)
    gross_return = (
        pl.when(pl.col("diagnostic_prediction") > 0.0)
        .then(future_bid / pl.col("best_ask") - 1.0)
        .otherwise(pl.col("best_bid") / future_ask - 1.0)
    )
    return frame.filter(pl.col("diagnostic_prediction") != 0.0).select(
        [
            pl.col(config.symbol_column).alias("symbol"),
            pl.col(config.event_time_column).alias("event_time"),
            side_expr.alias("side"),
            pl.col("diagnostic_prediction").alias("prediction"),
            entry_price.alias("entry_price"),
            exit_price.alias("exit_price"),
            pl.col("relative_spread"),
            pl.col("top_of_book_depth"),
            gross_return.alias("gross_return"),
            pl.lit(round_trip_cost).alias("cost_return"),
            (gross_return - round_trip_cost).alias("net_return"),
            pl.lit(config.horizon).alias("holding_horizon_events"),
            pl.lit(config.latency_events).alias("latency_events"),
        ]
    )


def _metrics(trades: pl.DataFrame) -> dict[str, float | int]:
    if trades.is_empty():
        return _empty_result().metrics
    gross = trades["gross_return"].to_numpy().astype(np.float64)
    net = trades["net_return"].to_numpy().astype(np.float64)
    long_trades = trades.filter(pl.col("side") == "long")
    short_trades = trades.filter(pl.col("side") == "short")
    return {
        "trade_count": int(trades.height),
        "gross_total_return": _compound(gross),
        "net_total_return": _compound(net),
        "trade_sharpe": _safe_sharpe(net),
        "gross_hit_rate": float(np.mean(gross > 0.0)) if gross.size else 0.0,
        "net_hit_rate": float(np.mean(net > 0.0)) if net.size else 0.0,
        "profit_factor": _profit_factor(net),
        "max_drawdown": _max_drawdown(net),
        "avg_trade_gross_return": float(np.mean(gross)) if gross.size else 0.0,
        "avg_trade_net_return": float(np.mean(net)) if net.size else 0.0,
        "long_trade_count": int(long_trades.height),
        "short_trade_count": int(short_trades.height),
        "long_net_pnl_sum": float(long_trades["net_return"].sum()) if long_trades.height else 0.0,
        "short_net_pnl_sum": float(short_trades["net_return"].sum()) if short_trades.height else 0.0,
    }


def run_event_backtest(frame: pl.DataFrame, *, config: PerpBacktestConfig) -> PerpBacktestResult:
    _validate_columns(frame, config)
    if frame.is_empty():
        return _empty_result()
    round_trip_cost = 2.0 * (config.fee_bps + config.slippage_bps) * 1e-4 * config.cost_multiplier
    prepared = _prepare_frame(frame, config)
    filtered = _apply_filters(prepared, config, round_trip_cost=round_trip_cost)
    if filtered.is_empty():
        return _empty_result()
    trades = _build_trades(filtered, config, round_trip_cost=round_trip_cost)
    return PerpBacktestResult(trades=trades, metrics=_metrics(trades))
