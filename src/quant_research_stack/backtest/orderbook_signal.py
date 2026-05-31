from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_research_stack.artifacts import safe_repo_id

ORDERBOOK_MODEL_NAMES: tuple[str, ...] = ("ridge", "hist_gradient", "ensemble_mean")

ORDERBOOK_IDENTIFIER_COLUMNS = {
    "dataset_id",
    "source_file",
    "symbol",
    "event_time",
    "transaction_time",
    "update_id",
}


@dataclass(frozen=True)
class OrderBookBacktestConfig:
    prediction_column: str = "prediction"
    target_column: str = "future_mid_return_1"
    symbol_column: str = "symbol"
    event_time_column: str = "event_time"
    relative_spread_column: str = "relative_spread"
    min_signal_abs: float = 0.0
    min_edge_over_cost: float = 0.0
    min_edge_to_cost_ratio: float | None = None
    max_relative_spread: float | None = None
    min_entry_depth: float | None = None
    spread_cost_multiplier: float = 1.0
    fee_bps: float = 1.0
    slippage_bps: float = 0.0
    starting_equity: float = 100_000.0
    max_trades: int | None = None
    invert_signal: bool = False


@dataclass(frozen=True)
class OrderBookSignalBacktestResult:
    trades: pl.DataFrame
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class OrderBookWalkForwardConfig:
    target_column: str = "future_mid_return_1"
    symbol_column: str = "symbol"
    event_time_column: str = "event_time"
    min_train_rows: int = 50_000
    test_rows: int = 10_000
    step_rows: int | None = None
    max_folds: int | None = 4
    max_train_rows_per_fold: int | None = 100_000
    ridge_alpha: float = 1.0
    hist_gradient_max_iter: int = 80
    min_signal_abs: float = 0.0
    min_edge_over_cost: float = 0.0
    max_relative_spread: float | None = None
    min_entry_depth: float | None = None
    fee_bps: float = 1.0
    starting_equity: float = 100_000.0


@dataclass(frozen=True)
class OrderBookFoldSpec:
    fold: int
    train_start_row: int
    train_end_row: int
    test_start_row: int
    test_end_row: int
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class OrderBookWalkForwardResult:
    feature_columns: list[str]
    predictions: pl.DataFrame
    fold_specs: list[OrderBookFoldSpec]
    fold_metrics: list[dict[str, float | int]]
    model_metrics: dict[str, dict[str, float | int]]
    backtest_metrics: dict[str, dict[str, float | int]]


def parse_levels(raw: Any) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, Iterable):
        return []
    levels: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            price = float(item[0])
            qty = float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(qty):
            levels.append((price, qty))
    return levels


def symbol_from_orderbook_path(path: Path) -> str:
    parent = path.parent.name
    if parent and parent.upper().endswith("USDT"):
        return parent.upper()
    for part in path.stem.split("_"):
        if part.upper().endswith("USDT"):
            return part.upper()
    return (parent or path.stem).upper()


def orderbook_features_from_frame(
    frame: pl.DataFrame,
    *,
    dataset_id: str,
    source_file: str,
    symbol: str,
    horizons: tuple[int, ...] = (1, 5, 15, 60),
    depth_levels: tuple[int, ...] = (1, 5, 10, 20),
) -> pl.DataFrame:
    if not {"bids", "asks"}.issubset(set(frame.columns)):
        raise ValueError("order-book frame must include bids and asks columns")

    bids_raw = frame["bids"].to_list()
    asks_raw = frame["asks"].to_list()
    bid_levels = [parse_levels(raw) for raw in bids_raw]
    ask_levels = [parse_levels(raw) for raw in asks_raw]

    rows: list[dict[str, Any]] = []
    for index, (bids, asks) in enumerate(zip(bid_levels, ask_levels, strict=False)):
        if not bids or not asks:
            continue
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        denom = best_bid_qty + best_ask_qty
        microprice = ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) / denom if denom else None
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "source_file": source_file,
            "symbol": symbol,
            "row_index": index,
            "event_time": frame["E"][index] if "E" in frame.columns else None,
            "transaction_time": frame["T"][index] if "T" in frame.columns else None,
            "update_id": frame["u"][index] if "u" in frame.columns else frame["lastUpdateId"][index] if "lastUpdateId" in frame.columns else None,
            "best_bid": best_bid,
            "best_bid_qty": best_bid_qty,
            "best_ask": best_ask,
            "best_ask_qty": best_ask_qty,
            "mid_price": mid,
            "spread": spread,
            "relative_spread": spread / mid if mid else None,
            "microprice_l1": microprice,
            "imbalance_l1": (best_bid_qty - best_ask_qty) / denom if denom else None,
        }
        for depth in depth_levels:
            bid_depth = sum(qty for _, qty in bids[:depth])
            ask_depth = sum(qty for _, qty in asks[:depth])
            depth_denom = bid_depth + ask_depth
            row[f"bid_depth_{depth}"] = bid_depth
            row[f"ask_depth_{depth}"] = ask_depth
            row[f"imbalance_depth_{depth}"] = (bid_depth - ask_depth) / depth_denom if depth_denom else None
        rows.append(row)

    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows).sort(["symbol", "row_index"])
    target_exprs: list[pl.Expr] = []
    for horizon in sorted(set(horizons)):
        if horizon <= 0:
            raise ValueError(f"horizons must be positive; got {horizon}")
        target_name = f"future_mid_return_{horizon}"
        target_exprs.append((pl.col("mid_price").shift(-horizon).over("symbol") / pl.col("mid_price") - 1.0).alias(target_name))
        target_exprs.append((pl.col("mid_price").shift(-horizon).over("symbol") > pl.col("mid_price")).cast(pl.Int8).alias(f"mid_direction_up_{horizon}"))
        target_exprs.append(pl.col("mid_price").shift(-horizon).over("symbol").alias(f"future_mid_price_{horizon}"))
        target_exprs.append(pl.col("best_bid").shift(-horizon).over("symbol").alias(f"future_best_bid_{horizon}"))
        target_exprs.append(pl.col("best_ask").shift(-horizon).over("symbol").alias(f"future_best_ask_{horizon}"))
    return out.with_columns(target_exprs)


def orderbook_features_from_file(
    path: Path,
    *,
    dataset_id: str,
    horizons: tuple[int, ...] = (1, 5, 15, 60),
    depth_levels: tuple[int, ...] = (1, 5, 10, 20),
    max_rows: int | None = None,
) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if max_rows is not None and max_rows > 0:
        frame = frame.head(max_rows)
    return orderbook_features_from_frame(
        frame,
        dataset_id=dataset_id,
        source_file=str(path),
        symbol=symbol_from_orderbook_path(path),
        horizons=horizons,
        depth_levels=depth_levels,
    )


def write_orderbook_feature_files(
    *,
    raw_root: Path,
    output_root: Path,
    dataset_id: str,
    symbols: set[str] | None = None,
    max_files_per_symbol: int | None = None,
    max_rows_per_file: int | None = None,
    horizons: tuple[int, ...] = (1, 5, 15, 60),
    depth_levels: tuple[int, ...] = (1, 5, 10, 20),
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for symbol_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        symbol = symbol_dir.name.upper()
        if symbols and symbol not in symbols:
            continue
        files = sorted(symbol_dir.glob("*.parquet"))
        if max_files_per_symbol is not None:
            files = files[:max_files_per_symbol]
        for path in files:
            features = orderbook_features_from_file(
                path,
                dataset_id=dataset_id,
                horizons=horizons,
                depth_levels=depth_levels,
                max_rows=max_rows_per_file,
            )
            if features.is_empty():
                continue
            relative = Path(symbol) / f"{safe_repo_id(path.stem)}.orderbook_features.parquet"
            out_path = output_root / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            features.write_parquet(out_path, compression="zstd")
            produced.append(out_path)
    return produced


def read_orderbook_feature_files(paths: list[Path], max_rows: int | None = None) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    remaining = max_rows
    for path in paths:
        frame = pl.read_parquet(path)
        if remaining is not None:
            frame = frame.head(remaining)
        if frame.is_empty():
            continue
        frames.append(frame)
        if remaining is not None:
            remaining -= frame.height
            if remaining <= 0:
                break
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _zero_mean_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    denom = float(np.sum(np.square(y_true)))
    if denom <= 0.0:
        return 0.0
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denom)


def _safe_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.size < 2 or b.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if math.isnan(value) else value


def _directional_accuracy(y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
    mask = (y_pred != 0.0) & (y_true != 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def _sharpe_ratio(returns: NDArray[np.float64], *, annualization: float) -> float:
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        if mean > 0.0:
            return 1_000_000.0
        if mean < 0.0:
            return -1_000_000.0
        return 0.0
    value = float(mean / std * math.sqrt(annualization))
    return value if math.isfinite(value) else 0.0


def _date_key(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        raw = float(value)
        magnitude = abs(raw)
        if magnitude > 1e17:
            seconds = raw / 1e9
        elif magnitude > 1e14:
            seconds = raw / 1e6
        elif magnitude > 1e11:
            seconds = raw / 1e3
        elif magnitude > 1e9:
            seconds = raw
        else:
            return None
        try:
            return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str) and len(value) >= 10:
        token = value[:10]
        try:
            return date.fromisoformat(token).isoformat()
        except ValueError:
            return None
    return None


def _daily_returns(
    trades: pl.DataFrame,
    *,
    event_time_column: str,
    net_return_column: str,
) -> NDArray[np.float64]:
    if event_time_column not in trades.columns or net_return_column not in trades.columns:
        return np.asarray([], dtype=np.float64)
    by_day: dict[str, list[float]] = defaultdict(list)
    for raw_time, raw_return in zip(
        trades[event_time_column].to_list(),
        trades[net_return_column].to_list(),
        strict=False,
    ):
        day = _date_key(raw_time)
        if day is None:
            continue
        try:
            ret = float(raw_return)
        except (TypeError, ValueError):
            continue
        if math.isfinite(ret):
            by_day[day].append(ret)
    if not by_day:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        [float(np.prod(np.asarray(day_returns, dtype=np.float64) + 1.0) - 1.0) for _, day_returns in sorted(by_day.items())],
        dtype=np.float64,
    )


def _target_horizon(target_column: str) -> int | None:
    match = re.fullmatch(r"future_mid_return_(\d+)", target_column)
    if match is None:
        return None
    return int(match.group(1))


def _optional_float_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64)


def _compound_total(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(returns + 1.0) - 1.0)


def _max_drawdown(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    equity = np.asarray(equity_values, dtype=np.float64)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    value = float(np.min(drawdowns))
    return value if math.isfinite(value) else 0.0


def _clean_prediction_frame(frame: pl.DataFrame, config: OrderBookBacktestConfig) -> pl.DataFrame:
    required = {config.prediction_column, config.target_column, config.relative_spread_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing order-book signal columns: {sorted(missing)}")
    sort_columns = [
        col
        for col in (config.symbol_column, config.event_time_column, "row_index")
        if col in frame.columns
    ]
    out = (
        frame.drop_nulls([config.prediction_column, config.target_column, config.relative_spread_column])
        .with_columns(
            [
                pl.col(config.prediction_column).cast(pl.Float64, strict=False).alias(config.prediction_column),
                pl.col(config.target_column).cast(pl.Float64, strict=False).alias(config.target_column),
                pl.col(config.relative_spread_column).cast(pl.Float64, strict=False).alias(config.relative_spread_column),
            ]
        )
        .filter(
            pl.col(config.prediction_column).is_finite()
            & pl.col(config.target_column).is_finite()
            & pl.col(config.relative_spread_column).is_finite()
            & (pl.col(config.relative_spread_column) >= 0.0)
        )
    )
    return out.sort(sort_columns) if sort_columns else out


def run_orderbook_signal_backtest(
    frame: pl.DataFrame,
    *,
    config: OrderBookBacktestConfig | None = None,
) -> OrderBookSignalBacktestResult:
    cfg = config or OrderBookBacktestConfig()
    if cfg.min_signal_abs < 0.0:
        raise ValueError("min_signal_abs must be non-negative")
    if cfg.min_edge_over_cost < 0.0:
        raise ValueError("min_edge_over_cost must be non-negative")
    if cfg.min_edge_to_cost_ratio is not None and cfg.min_edge_to_cost_ratio < 0.0:
        raise ValueError("min_edge_to_cost_ratio must be non-negative")
    if cfg.max_relative_spread is not None and cfg.max_relative_spread < 0.0:
        raise ValueError("max_relative_spread must be non-negative")
    if cfg.min_entry_depth is not None and cfg.min_entry_depth < 0.0:
        raise ValueError("min_entry_depth must be non-negative")
    if cfg.spread_cost_multiplier < 0.0:
        raise ValueError("spread_cost_multiplier must be non-negative")
    if cfg.fee_bps < 0.0:
        raise ValueError("fee_bps must be non-negative")
    if cfg.slippage_bps < 0.0:
        raise ValueError("slippage_bps must be non-negative")
    if cfg.min_entry_depth is not None and not {"bid_depth_1", "ask_depth_1"}.issubset(set(frame.columns)):
        raise ValueError("min_entry_depth requires bid_depth_1 and ask_depth_1 columns")

    clean = _clean_prediction_frame(frame, cfg)
    if clean.is_empty():
        return OrderBookSignalBacktestResult(
            trades=pl.DataFrame(),
            metrics={
                "rows": 0,
                "trade_count": 0,
                "trade_rate": 0.0,
                "directional_accuracy": 0.0,
                "zero_mean_r2": 0.0,
                "information_coefficient": 0.0,
                "gross_total_return": 0.0,
                "total_return": 0.0,
                "cost_drag_return": 0.0,
                "hit_rate": 0.0,
                "gross_hit_rate": 0.0,
                "net_hit_rate": 0.0,
                "avg_trade_gross_return": 0.0,
                "avg_trade_net_return": 0.0,
                "avg_trade_cost_return": 0.0,
                "avg_spread_cost_return": 0.0,
                "avg_fee_cost_return": 0.0,
                "avg_slippage_cost_return": 0.0,
                "avg_edge_to_cost_ratio": 0.0,
                "per_trade_sharpe_ratio": 0.0,
                "trade_sharpe_ratio": 0.0,
                "daily_sharpe_ratio": 0.0,
                "daily_return_count": 0,
                "max_drawdown": 0.0,
                "long_trade_count": 0,
                "short_trade_count": 0,
                "long_total_return": 0.0,
                "short_total_return": 0.0,
                "long_avg_net_return": 0.0,
                "short_avg_net_return": 0.0,
                "candidate_count": 0,
                "filtered_count": 0,
                "ending_equity": cfg.starting_equity,
            },
        )

    y_pred = clean[cfg.prediction_column].to_numpy().astype(np.float64)
    y_true = clean[cfg.target_column].to_numpy().astype(np.float64)
    direction_acc = _directional_accuracy(y_pred, y_true)
    fee_return = 2.0 * cfg.fee_bps * 1e-4
    slippage_return = 2.0 * cfg.slippage_bps * 1e-4
    horizon = _target_horizon(cfg.target_column)
    future_mid_column = f"future_mid_price_{horizon}" if horizon is not None else ""
    future_bid_column = f"future_best_bid_{horizon}" if horizon is not None else ""
    future_ask_column = f"future_best_ask_{horizon}" if horizon is not None else ""
    signal_expr = -pl.col(cfg.prediction_column) if cfg.invert_signal else pl.col(cfg.prediction_column)
    entry_mid_expr = _optional_float_expr(clean, "mid_price")
    entry_bid_expr = (
        _optional_float_expr(clean, "best_bid")
        if "best_bid" in clean.columns
        else entry_mid_expr * (1.0 - (pl.col(cfg.relative_spread_column) / 2.0))
    )
    entry_ask_expr = (
        _optional_float_expr(clean, "best_ask")
        if "best_ask" in clean.columns
        else entry_mid_expr * (1.0 + (pl.col(cfg.relative_spread_column) / 2.0))
    )
    exit_mid_expr = (
        _optional_float_expr(clean, future_mid_column)
        if future_mid_column in clean.columns
        else entry_mid_expr * (1.0 + pl.col(cfg.target_column))
    )
    exit_bid_expr = (
        _optional_float_expr(clean, future_bid_column)
        if future_bid_column in clean.columns
        else exit_mid_expr * (1.0 - (pl.col(cfg.relative_spread_column) / 2.0))
    )
    exit_ask_expr = (
        _optional_float_expr(clean, future_ask_column)
        if future_ask_column in clean.columns
        else exit_mid_expr * (1.0 + (pl.col(cfg.relative_spread_column) / 2.0))
    )
    candidates = (
        clean.with_columns(signal_expr.alias("__decision_signal"))
        .filter((pl.col("__decision_signal").abs() >= cfg.min_signal_abs) & (pl.col("__decision_signal") != 0.0))
        .with_columns(
            [
                pl.when(pl.col("__decision_signal") > 0.0).then(1.0).otherwise(-1.0).alias("position_side"),
                pl.when(pl.col("__decision_signal") > 0.0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side"),
                pl.col(cfg.prediction_column).alias("predicted_return"),
                pl.col("__decision_signal").alias("decision_signal_return"),
                pl.col("__decision_signal").abs().alias("signal_abs"),
                entry_mid_expr.alias("entry_mid"),
                entry_bid_expr.alias("entry_bid"),
                entry_ask_expr.alias("entry_ask"),
                exit_mid_expr.alias("exit_mid"),
                exit_bid_expr.alias("exit_bid"),
                exit_ask_expr.alias("exit_ask"),
                pl.col(cfg.target_column).alias("realized_mid_return"),
                pl.lit(fee_return).alias("fee_cost_return"),
                pl.lit(slippage_return).alias("slippage_cost_return"),
                pl.lit(horizon).alias("holding_horizon"),
            ]
        )
        .with_columns(
            pl.when(pl.col("position_side") > 0.0)
            .then(((pl.col("entry_ask") - pl.col("entry_mid")) + (pl.col("exit_mid") - pl.col("exit_bid"))) / pl.col("entry_mid"))
            .otherwise(((pl.col("entry_mid") - pl.col("entry_bid")) + (pl.col("exit_ask") - pl.col("exit_mid"))) / pl.col("entry_mid"))
            .alias("__price_spread_cost_return")
        )
        .with_columns(
            (
                pl.when(
                    pl.col("__price_spread_cost_return").is_finite()
                    & pl.col("__price_spread_cost_return").is_not_null()
                    & (pl.col("__price_spread_cost_return") >= 0.0)
                )
                .then(pl.col("__price_spread_cost_return"))
                .otherwise(pl.col(cfg.relative_spread_column))
                * cfg.spread_cost_multiplier
            ).alias("spread_cost_return")
        )
        .with_columns(
            (
                pl.col("spread_cost_return")
                + pl.col("fee_cost_return")
                + pl.col("slippage_cost_return")
            ).alias("estimated_round_trip_cost")
        )
        .with_columns(
            [
                pl.col("estimated_round_trip_cost").alias("cost_return"),
                (pl.col("signal_abs") - pl.col("estimated_round_trip_cost")).alias("predicted_edge_over_cost"),
                pl.when(pl.col("estimated_round_trip_cost") > 0.0)
                .then(pl.col("signal_abs") / pl.col("estimated_round_trip_cost"))
                .otherwise(float("inf"))
                .alias("edge_to_cost_ratio"),
            ]
        )
    )
    candidate_count = candidates.height
    trade_frame = candidates.filter(pl.col("predicted_edge_over_cost") >= cfg.min_edge_over_cost)
    if cfg.min_edge_to_cost_ratio is not None:
        trade_frame = trade_frame.filter(pl.col("edge_to_cost_ratio") > cfg.min_edge_to_cost_ratio)
    if cfg.max_relative_spread is not None:
        trade_frame = trade_frame.filter(pl.col(cfg.relative_spread_column) <= cfg.max_relative_spread)
    if cfg.min_entry_depth is not None:
        trade_frame = trade_frame.filter(
            pl.when(pl.col("position_side") > 0.0)
            .then(pl.col("ask_depth_1") >= cfg.min_entry_depth)
            .otherwise(pl.col("bid_depth_1") >= cfg.min_entry_depth)
        )
    if cfg.max_trades is not None:
        trade_frame = trade_frame.head(cfg.max_trades)
    filtered_count = candidate_count - trade_frame.height

    if trade_frame.is_empty():
        return OrderBookSignalBacktestResult(
            trades=trade_frame,
            metrics={
                "rows": clean.height,
                "trade_count": 0,
                "trade_rate": 0.0,
                "directional_accuracy": direction_acc,
                "zero_mean_r2": _zero_mean_r2(y_true, y_pred),
                "information_coefficient": _safe_corr(y_pred, y_true),
                "gross_total_return": 0.0,
                "total_return": 0.0,
                "cost_drag_return": 0.0,
                "hit_rate": 0.0,
                "gross_hit_rate": 0.0,
                "net_hit_rate": 0.0,
                "avg_trade_gross_return": 0.0,
                "avg_trade_net_return": 0.0,
                "avg_trade_cost_return": 0.0,
                "avg_spread_cost_return": 0.0,
                "avg_fee_cost_return": 0.0,
                "avg_slippage_cost_return": 0.0,
                "avg_edge_to_cost_ratio": 0.0,
                "per_trade_sharpe_ratio": 0.0,
                "trade_sharpe_ratio": 0.0,
                "daily_sharpe_ratio": 0.0,
                "daily_return_count": 0,
                "max_drawdown": 0.0,
                "long_trade_count": 0,
                "short_trade_count": 0,
                "long_total_return": 0.0,
                "short_total_return": 0.0,
                "long_avg_net_return": 0.0,
                "short_avg_net_return": 0.0,
                "candidate_count": candidate_count,
                "filtered_count": filtered_count,
                "ending_equity": cfg.starting_equity,
            },
        )

    enriched = trade_frame.with_columns(
        [
            (pl.col("position_side") * pl.col(cfg.target_column)).alias("gross_return"),
        ]
    ).with_columns(
        [
            (pl.col("gross_return") - pl.col("cost_return")).alias("net_return"),
            (pl.col("gross_return") > 0.0).alias("prediction_direction_correct"),
            (pl.col("gross_return") > 0.0).alias("gross_hit"),
            ((pl.col("gross_return") - pl.col("cost_return")) > 0.0).alias("net_hit"),
            pl.col(cfg.event_time_column).alias("timestamp") if cfg.event_time_column in trade_frame.columns else pl.lit(None).alias("timestamp"),
        ]
    )

    equity = cfg.starting_equity
    gross_equity = cfg.starting_equity
    equity_values: list[float] = []
    gross_equity_values: list[float] = []
    for gross_ret, net_ret in zip(enriched["gross_return"].to_list(), enriched["net_return"].to_list(), strict=False):
        gross_equity *= 1.0 + float(gross_ret)
        equity *= 1.0 + float(net_ret)
        gross_equity_values.append(gross_equity)
        equity_values.append(equity)
    trades = enriched.with_columns(
        [
            pl.Series("gross_equity", gross_equity_values),
            pl.Series("equity", equity_values),
        ]
    )
    gross_returns = trades["gross_return"].to_numpy().astype(np.float64)
    net_returns = trades["net_return"].to_numpy().astype(np.float64)
    cost_returns = trades["cost_return"].to_numpy().astype(np.float64)
    spread_cost_returns = trades["spread_cost_return"].to_numpy().astype(np.float64)
    fee_cost_returns = trades["fee_cost_return"].to_numpy().astype(np.float64)
    slippage_cost_returns = trades["slippage_cost_return"].to_numpy().astype(np.float64)
    edge_to_cost_ratios = trades["edge_to_cost_ratio"].to_numpy().astype(np.float64)
    long_returns = trades.filter(pl.col("position_side") > 0.0)["net_return"].to_numpy().astype(np.float64)
    short_returns = trades.filter(pl.col("position_side") < 0.0)["net_return"].to_numpy().astype(np.float64)
    daily_returns = _daily_returns(
        trades,
        event_time_column=cfg.event_time_column,
        net_return_column="net_return",
    )
    total = float(equity / cfg.starting_equity - 1.0)
    gross_total = float(gross_equity / cfg.starting_equity - 1.0)
    metrics = {
        "rows": clean.height,
        "trade_count": trades.height,
        "trade_rate": float(trades.height / clean.height) if clean.height else 0.0,
        "directional_accuracy": direction_acc,
        "zero_mean_r2": _zero_mean_r2(y_true, y_pred),
        "information_coefficient": _safe_corr(y_pred, y_true),
        "gross_total_return": gross_total,
        "total_return": total,
        "cost_drag_return": gross_total - total,
        "hit_rate": float(np.mean(net_returns > 0.0)),
        "gross_hit_rate": float(np.mean(gross_returns > 0.0)),
        "net_hit_rate": float(np.mean(net_returns > 0.0)),
        "avg_trade_gross_return": float(np.mean(gross_returns)),
        "avg_trade_net_return": float(np.mean(net_returns)),
        "avg_trade_cost_return": float(np.mean(cost_returns)),
        "avg_spread_cost_return": float(np.mean(spread_cost_returns)),
        "avg_fee_cost_return": float(np.mean(fee_cost_returns)),
        "avg_slippage_cost_return": float(np.mean(slippage_cost_returns)),
        "avg_edge_to_cost_ratio": float(np.mean(edge_to_cost_ratios[np.isfinite(edge_to_cost_ratios)]))
        if np.any(np.isfinite(edge_to_cost_ratios))
        else 0.0,
        "per_trade_sharpe_ratio": _sharpe_ratio(net_returns, annualization=1.0),
        "trade_sharpe_ratio": _sharpe_ratio(net_returns, annualization=float(net_returns.size)),
        "daily_sharpe_ratio": _sharpe_ratio(daily_returns, annualization=252.0),
        "daily_return_count": int(daily_returns.size),
        "max_drawdown": _max_drawdown(equity_values),
        "long_trade_count": int(long_returns.size),
        "short_trade_count": int(short_returns.size),
        "long_total_return": _compound_total(long_returns),
        "short_total_return": _compound_total(short_returns),
        "long_avg_net_return": float(np.mean(long_returns)) if long_returns.size else 0.0,
        "short_avg_net_return": float(np.mean(short_returns)) if short_returns.size else 0.0,
        "candidate_count": candidate_count,
        "filtered_count": filtered_count,
        "avg_signal_abs": float(cast(float, trades["signal_abs"].mean())),
        "ending_equity": equity,
    }
    return OrderBookSignalBacktestResult(trades=trades, metrics=metrics)


def _is_label_like(column: str, target_column: str) -> bool:
    if column == target_column:
        return True
    return column.startswith(("future_", "direction_", "mid_direction_"))


def orderbook_feature_columns(
    frame: pl.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> list[str]:
    if feature_columns is not None:
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing configured feature columns: {sorted(missing)}")
        return list(feature_columns)
    cols: list[str] = []
    for name, dtype in frame.schema.items():
        if name in ORDERBOOK_IDENTIFIER_COLUMNS or name.startswith("__") or _is_label_like(name, target_column):
            continue
        if dtype.is_numeric():
            cols.append(name)
    if not cols:
        raise ValueError("no numeric order-book feature columns are available")
    return sorted(cols)


def _clean_supervised_frame(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    symbol_column: str,
    event_time_column: str,
) -> pl.DataFrame:
    required = set(feature_columns) | {target_column, symbol_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing order-book supervised columns: {sorted(missing)}")
    sort_columns = [col for col in (symbol_column, event_time_column, "row_index") if col in frame.columns]
    return (
        frame.drop_nulls([target_column])
        .with_columns(
            [
                *[pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in feature_columns],
                pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
            ]
        )
        .filter(pl.col(target_column).is_finite())
        .sort(sort_columns)
        .with_row_index("__wf_row")
    )


def _row_splits(
    frame: pl.DataFrame,
    *,
    min_train_rows: int,
    test_rows: int,
    step_rows: int | None,
    max_folds: int | None,
) -> list[OrderBookFoldSpec]:
    if min_train_rows < 1:
        raise ValueError("min_train_rows must be positive")
    if test_rows < 1:
        raise ValueError("test_rows must be positive")
    step = step_rows or test_rows
    if step < 1:
        raise ValueError("step_rows must be positive")
    if frame.height <= min_train_rows:
        raise ValueError(f"not enough rows for walk-forward split: rows={frame.height} min_train_rows={min_train_rows}")

    specs: list[OrderBookFoldSpec] = []
    train_end = min_train_rows
    fold = 0
    while train_end < frame.height:
        test_end = min(train_end + test_rows, frame.height)
        if test_end <= train_end:
            break
        specs.append(
            OrderBookFoldSpec(
                fold=fold,
                train_start_row=0,
                train_end_row=train_end - 1,
                test_start_row=train_end,
                test_end_row=test_end - 1,
                train_rows=train_end,
                test_rows=test_end - train_end,
            )
        )
        fold += 1
        if test_end >= frame.height:
            break
        train_end += step

    if max_folds is not None and max_folds > 0 and len(specs) > max_folds:
        selected = specs[-max_folds:]
        return [
            OrderBookFoldSpec(
                fold=i,
                train_start_row=spec.train_start_row,
                train_end_row=spec.train_end_row,
                test_start_row=spec.test_start_row,
                test_end_row=spec.test_end_row,
                train_rows=spec.train_rows,
                test_rows=spec.test_rows,
            )
            for i, spec in enumerate(selected)
        ]
    return specs


def _tail_train(frame: pl.DataFrame, max_rows: int | None) -> pl.DataFrame:
    if max_rows is None or max_rows <= 0 or frame.height <= max_rows:
        return frame
    return frame.tail(max_rows)


def _to_numpy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    x = frame.select(feature_columns).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
    y = frame[target_column].to_numpy().astype(np.float64)
    return (
        np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _make_models(config: OrderBookWalkForwardConfig) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha)),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=config.hist_gradient_max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }


def _evaluate_prediction_metrics(
    frame: pl.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
) -> dict[str, float | int]:
    clean = frame.drop_nulls([prediction_column, target_column]).with_columns(
        [
            pl.col(prediction_column).cast(pl.Float64, strict=False).alias(prediction_column),
            pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
        ]
    )
    if clean.is_empty():
        return {
            "rows": 0,
            "directional_accuracy": 0.0,
            "zero_mean_r2": 0.0,
            "information_coefficient": 0.0,
        }
    y_pred = clean[prediction_column].to_numpy().astype(np.float64)
    y_true = clean[target_column].to_numpy().astype(np.float64)
    return {
        "rows": clean.height,
        "directional_accuracy": _directional_accuracy(y_pred, y_true),
        "zero_mean_r2": _zero_mean_r2(y_true, y_pred),
        "information_coefficient": _safe_corr(y_pred, y_true),
    }


def run_orderbook_walk_forward(
    frame: pl.DataFrame,
    *,
    config: OrderBookWalkForwardConfig | None = None,
    feature_columns: list[str] | None = None,
) -> OrderBookWalkForwardResult:
    cfg = config or OrderBookWalkForwardConfig()
    features = orderbook_feature_columns(frame, target_column=cfg.target_column, feature_columns=feature_columns)
    clean = _clean_supervised_frame(
        frame,
        feature_columns=features,
        target_column=cfg.target_column,
        symbol_column=cfg.symbol_column,
        event_time_column=cfg.event_time_column,
    )
    specs = _row_splits(
        clean,
        min_train_rows=cfg.min_train_rows,
        test_rows=cfg.test_rows,
        step_rows=cfg.step_rows,
        max_folds=cfg.max_folds,
    )
    if not specs:
        raise ValueError("walk-forward split produced no folds")

    prediction_frames: list[pl.DataFrame] = []
    fold_metrics: list[dict[str, float | int]] = []
    horizon = _target_horizon(cfg.target_column)
    future_keep_columns = (
        [f"future_mid_price_{horizon}", f"future_best_bid_{horizon}", f"future_best_ask_{horizon}"]
        if horizon is not None
        else []
    )
    keep_columns = [
        col
        for col in (
            cfg.symbol_column,
            cfg.event_time_column,
            "row_index",
            "source_file",
            "mid_price",
            "best_bid",
            "best_ask",
            "spread",
            "relative_spread",
            "bid_depth_1",
            "ask_depth_1",
            cfg.target_column,
            *future_keep_columns,
        )
        if col in clean.columns
    ]
    for spec in specs:
        train = clean.filter(pl.col("__wf_row") <= spec.train_end_row)
        test = clean.filter((pl.col("__wf_row") >= spec.test_start_row) & (pl.col("__wf_row") <= spec.test_end_row))
        train = _tail_train(train, cfg.max_train_rows_per_fold)
        x_train, y_train = _to_numpy(train, feature_columns=features, target_column=cfg.target_column)
        x_test, _ = _to_numpy(test, feature_columns=features, target_column=cfg.target_column)
        models = _make_models(cfg)
        preds: dict[str, NDArray[np.float64]] = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            preds[model_name] = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
        preds["ensemble_mean"] = np.mean(np.column_stack([preds["ridge"], preds["hist_gradient"]]), axis=1)
        fold_pred = test.select(keep_columns).with_columns(
            [
                pl.lit(spec.fold).alias("fold"),
                *[pl.Series(f"pred_{name}", values) for name, values in preds.items()],
            ]
        )
        prediction_frames.append(fold_pred)
        row: dict[str, float | int] = {
            "fold": spec.fold,
            "train_rows": train.height,
            "test_rows": test.height,
        }
        for name in ORDERBOOK_MODEL_NAMES:
            metrics = _evaluate_prediction_metrics(
                fold_pred,
                prediction_column=f"pred_{name}",
                target_column=cfg.target_column,
            )
            row[f"{name}_directional_accuracy"] = metrics["directional_accuracy"]
            row[f"{name}_zero_mean_r2"] = metrics["zero_mean_r2"]
        fold_metrics.append(row)

    predictions = pl.concat(prediction_frames, how="vertical")
    model_metrics = {
        name: _evaluate_prediction_metrics(
            predictions,
            prediction_column=f"pred_{name}",
            target_column=cfg.target_column,
        )
        for name in ORDERBOOK_MODEL_NAMES
    }
    backtest_metrics = {
        name: run_orderbook_signal_backtest(
            predictions,
            config=OrderBookBacktestConfig(
                prediction_column=f"pred_{name}",
                target_column=cfg.target_column,
                symbol_column=cfg.symbol_column,
                event_time_column=cfg.event_time_column,
                min_signal_abs=cfg.min_signal_abs,
                min_edge_over_cost=cfg.min_edge_over_cost,
                max_relative_spread=cfg.max_relative_spread,
                min_entry_depth=cfg.min_entry_depth,
                fee_bps=cfg.fee_bps,
                starting_equity=cfg.starting_equity,
            ),
        ).metrics
        for name in ORDERBOOK_MODEL_NAMES
    }
    return OrderBookWalkForwardResult(
        feature_columns=features,
        predictions=predictions,
        fold_specs=specs,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
        backtest_metrics=backtest_metrics,
    )


def train_final_orderbook_models(
    frame: pl.DataFrame,
    *,
    config: OrderBookWalkForwardConfig | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    cfg = config or OrderBookWalkForwardConfig()
    features = orderbook_feature_columns(frame, target_column=cfg.target_column, feature_columns=feature_columns)
    clean = _clean_supervised_frame(
        frame,
        feature_columns=features,
        target_column=cfg.target_column,
        symbol_column=cfg.symbol_column,
        event_time_column=cfg.event_time_column,
    )
    x, y = _to_numpy(clean, feature_columns=features, target_column=cfg.target_column)
    models = _make_models(cfg)
    for model in models.values():
        model.fit(x, y)
    return models


def save_orderbook_model_artifacts(
    *,
    models: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, model in models.items():
        path = output_dir / f"{safe_repo_id(name)}.joblib"
        joblib.dump(
            {
                "model": model,
                "features": feature_columns,
                "target": target_column,
                "metadata": metadata or {},
            },
            path,
        )
        paths[name] = str(path)
    return paths
