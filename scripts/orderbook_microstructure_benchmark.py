from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.backtest.orderbook_signal import (
    ORDERBOOK_MODEL_NAMES,
    OrderBookBacktestConfig,
    OrderBookWalkForwardConfig,
    read_orderbook_feature_files,
    run_orderbook_signal_backtest,
    run_orderbook_walk_forward,
    save_orderbook_model_artifacts,
    train_final_orderbook_models,
    write_orderbook_feature_files,
)

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.3f}%"
    except Exception:
        return str(value)


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _symbols_arg(values: list[str]) -> set[str]:
    symbols: set[str] = set()
    for value in values:
        for part in value.split(","):
            symbol = part.strip().upper()
            if symbol:
                symbols.add(symbol)
    return symbols


def _date_span_from_paths(paths: list[Path]) -> dict[str, Any]:
    dates = sorted({match.group(1) for path in paths if (match := re.search(r"(\d{4}-\d{2}-\d{2})", str(path)))})
    return {
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "date_count": len(dates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Binance futures order-book microstructure benchmark.")
    parser.add_argument("--raw-root", default="data/raw/huggingface/predict-quant__binance-future-orderbook")
    parser.add_argument("--output-root", default="experiments/orderbook_microstructure")
    parser.add_argument("--report", default=None)
    parser.add_argument("--symbols", action="append", default=["BTCUSDT"], help="Comma-separated symbols or repeated flag.")
    parser.add_argument("--max-files-per-symbol", type=int, default=1)
    parser.add_argument("--max-rows-per-file", type=int, default=120_000)
    parser.add_argument("--max-feature-rows", type=int, default=250_000)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=(1, 5, 15, 60))
    parser.add_argument("--depth-levels", type=_parse_int_tuple, default=(1, 5, 10, 20))
    parser.add_argument("--target-column", default="future_mid_return_5")
    parser.add_argument("--min-train-rows", type=int, default=60_000)
    parser.add_argument("--test-rows", type=int, default=15_000)
    parser.add_argument("--step-rows", type=int, default=15_000)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=90_000)
    parser.add_argument("--hist-gradient-max-iter", type=int, default=40)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--min-signal-abs-sweep", type=_parse_float_tuple, default=(0.0, 0.00002, 0.00005, 0.0001, 0.0002))
    parser.add_argument("--min-edge-over-cost-sweep", type=_parse_float_tuple, default=(0.0, 0.00005, 0.0001, 0.0002))
    parser.add_argument("--edge-to-cost-k-sweep", type=_parse_float_tuple, default=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0))
    parser.add_argument("--target-horizon-sweep", type=_parse_int_tuple, default=())
    parser.add_argument("--max-relative-spread", type=float, default=None)
    parser.add_argument("--min-entry-depth", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--save-final-artifacts", action="store_true")
    return parser.parse_args()


def _best_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    daily_sharpe = float(row.get("daily_sharpe_ratio", 0.0))
    trade_sharpe = float(row.get("trade_sharpe_ratio", 0.0))
    return (
        1.0 if float(row.get("trade_count", 0.0)) > 0.0 else 0.0,
        daily_sharpe if daily_sharpe != 0.0 else trade_sharpe,
        trade_sharpe,
        float(row.get("total_return", 0.0)),
        float(row.get("hit_rate", 0.0)),
    )


def _best_backtest(backtests: list[dict[str, Any]]) -> dict[str, Any]:
    traded = [row for row in backtests if float(row.get("trade_count", 0.0)) > 0.0]
    if not traded:
        return {}
    return max(traded, key=_best_score)


def _prediction_summary(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"rows": 0, "symbols": 0, "min_event_time": None, "max_event_time": None}
    summary = frame.select(
        [
            pl.len().alias("rows"),
            pl.n_unique("symbol").alias("symbols"),
            pl.min("event_time").alias("min_event_time"),
            pl.max("event_time").alias("max_event_time"),
        ]
    ).row(0, named=True)
    return dict(summary)


def _zero_mean_r2(values: np.ndarray, predictions: np.ndarray) -> float:
    denom = float(np.sum(np.square(values)))
    if denom <= 0.0:
        return 0.0
    return float(1.0 - np.sum(np.square(values - predictions)) / denom)


def _safe_corr(values: np.ndarray, predictions: np.ndarray) -> float:
    if values.size < 2 or predictions.size < 2 or float(np.std(values)) == 0.0 or float(np.std(predictions)) == 0.0:
        return 0.0
    out = float(np.corrcoef(values, predictions)[0, 1])
    return out if isfinite(out) else 0.0


def _prediction_metrics(frame: pl.DataFrame, *, prediction_column: str, target_column: str) -> dict[str, Any]:
    if frame.is_empty() or prediction_column not in frame.columns or target_column not in frame.columns:
        return {"rows": 0, "ic": 0.0, "zero_mean_r2": 0.0, "directional_accuracy": 0.0}
    clean = frame.drop_nulls([prediction_column, target_column]).with_columns(
        [
            pl.col(prediction_column).cast(pl.Float64, strict=False).alias(prediction_column),
            pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
        ]
    )
    if clean.is_empty():
        return {"rows": 0, "ic": 0.0, "zero_mean_r2": 0.0, "directional_accuracy": 0.0}
    preds = clean[prediction_column].to_numpy().astype(np.float64)
    target = clean[target_column].to_numpy().astype(np.float64)
    return {
        "rows": clean.height,
        "ic": _safe_corr(preds, target),
        "zero_mean_r2": _zero_mean_r2(target, preds),
        "directional_accuracy": float(np.mean((preds > 0.0) == (target > 0.0))),
    }


def _base_backtest_config(
    *,
    prediction_column: str,
    target_column: str,
    args: argparse.Namespace,
    min_signal_abs: float = 0.0,
    min_edge_over_cost: float = 0.0,
    min_edge_to_cost_ratio: float | None = None,
    spread_cost_multiplier: float = 1.0,
    fee_bps: float | None = None,
    invert_signal: bool = False,
    max_trades: int | None = None,
) -> OrderBookBacktestConfig:
    return OrderBookBacktestConfig(
        prediction_column=prediction_column,
        target_column=target_column,
        min_signal_abs=min_signal_abs,
        min_edge_over_cost=min_edge_over_cost,
        min_edge_to_cost_ratio=min_edge_to_cost_ratio,
        max_relative_spread=args.max_relative_spread,
        min_entry_depth=args.min_entry_depth,
        spread_cost_multiplier=spread_cost_multiplier,
        fee_bps=args.fee_bps if fee_bps is None else fee_bps,
        slippage_bps=args.slippage_bps,
        starting_equity=args.starting_equity,
        max_trades=max_trades,
        invert_signal=invert_signal,
    )


def _run_cost_aware_sweep(
    predictions: pl.DataFrame,
    *,
    target_column: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in ORDERBOOK_MODEL_NAMES:
        for k in args.edge_to_cost_k_sweep:
            result = run_orderbook_signal_backtest(
                predictions,
                config=_base_backtest_config(
                    prediction_column=f"pred_{model_name}",
                    target_column=target_column,
                    args=args,
                    min_edge_to_cost_ratio=float(k),
                ),
            )
            rows.append(
                {
                    "model": model_name,
                    "target_column": target_column,
                    "edge_to_cost_k": float(k),
                    **result.metrics,
                }
            )
    return rows


def _best_debug_row(backtests: list[dict[str, Any]], cost_aware: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [*backtests, *cost_aware]
    return _best_backtest(candidates)


def _config_from_row(row: dict[str, Any], *, target_column: str, args: argparse.Namespace) -> OrderBookBacktestConfig:
    return _base_backtest_config(
        prediction_column=f"pred_{row.get('model', 'ensemble_mean')}",
        target_column=target_column,
        args=args,
        min_signal_abs=float(row.get("min_signal_abs", 0.0)),
        min_edge_over_cost=float(row.get("min_edge_over_cost", 0.0)),
        min_edge_to_cost_ratio=float(row["edge_to_cost_k"]) if "edge_to_cost_k" in row else None,
    )


def _run_cost_regimes(
    predictions: pl.DataFrame,
    *,
    model_name: str,
    target_column: str,
    args: argparse.Namespace,
    max_trades: int | None = None,
) -> list[dict[str, Any]]:
    regimes = [
        ("no_cost", 0.0, 0.0),
        ("spread_only", 1.0, 0.0),
        ("fee_only", 0.0, args.fee_bps),
        ("spread_plus_fee", 1.0, args.fee_bps),
    ]
    rows: list[dict[str, Any]] = []
    for name, spread_multiplier, fee_bps in regimes:
        result = run_orderbook_signal_backtest(
            predictions,
            config=_base_backtest_config(
                prediction_column=f"pred_{model_name}",
                target_column=target_column,
                args=args,
                spread_cost_multiplier=spread_multiplier,
                fee_bps=fee_bps,
                max_trades=max_trades,
            ),
        )
        rows.append({"regime": name, "model": model_name, **result.metrics})
    return rows


def _selected_trade_cost_decomposition(trades: pl.DataFrame) -> list[dict[str, Any]]:
    if trades.is_empty():
        return []
    gross = trades["gross_return"].to_numpy().astype(np.float64)
    spread = trades["spread_cost_return"].to_numpy().astype(np.float64)
    fee = trades["fee_cost_return"].to_numpy().astype(np.float64)
    slippage = trades["slippage_cost_return"].to_numpy().astype(np.float64)
    regimes = {
        "selected_no_cost": gross,
        "selected_spread_only": gross - spread,
        "selected_fee_only": gross - fee,
        "selected_full_cost": gross - spread - fee - slippage,
    }
    rows: list[dict[str, Any]] = []
    for name, returns in regimes.items():
        rows.append(
            {
                "regime": name,
                "trade_count": int(returns.size),
                "gross_hit_rate": float(np.mean(gross > 0.0)),
                "net_hit_rate": float(np.mean(returns > 0.0)),
                "avg_trade_gross_return": float(np.mean(gross)),
                "avg_trade_net_return": float(np.mean(returns)),
                "total_return": float(np.prod(returns + 1.0) - 1.0),
            }
        )
    return rows


def _run_inverted_signal(
    predictions: pl.DataFrame,
    *,
    model_name: str,
    target_column: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = run_orderbook_signal_backtest(
        predictions,
        config=_base_backtest_config(
            prediction_column=f"pred_{model_name}",
            target_column=target_column,
            args=args,
            invert_signal=True,
        ),
    )
    return {"model": model_name, "inverted": True, **result.metrics}


def _run_null_baselines(
    predictions: pl.DataFrame,
    *,
    model_name: str,
    target_column: str,
    args: argparse.Namespace,
    trade_count: int,
) -> list[dict[str, Any]]:
    if predictions.is_empty():
        return []
    rng = np.random.default_rng(42)
    n_rows = predictions.height
    selected_count = min(max(trade_count, 1), n_rows)
    selected = set(int(idx) for idx in rng.choice(n_rows, size=selected_count, replace=False))
    random_values = np.zeros(n_rows, dtype=np.float64)
    random_signs = rng.choice(np.asarray([-1.0, 1.0]), size=selected_count)
    for offset, row_index in enumerate(sorted(selected)):
        random_values[row_index] = random_signs[offset] * 1.0
    source = predictions[f"pred_{model_name}"].to_numpy().astype(np.float64)
    shuffled = np.array(source, copy=True)
    rng.shuffle(shuffled)
    frames = {
        "random_same_trade_count": predictions.with_columns(pl.Series("__baseline_prediction", random_values)),
        "shuffled_predictions": predictions.with_columns(pl.Series("__baseline_prediction", shuffled)),
        "always_long": predictions.with_columns(pl.lit(1.0).alias("__baseline_prediction")),
        "always_short": predictions.with_columns(pl.lit(-1.0).alias("__baseline_prediction")),
    }
    rows: list[dict[str, Any]] = []
    for baseline_name, frame in frames.items():
        result = run_orderbook_signal_backtest(
            frame,
            config=_base_backtest_config(
                prediction_column="__baseline_prediction",
                target_column=target_column,
                args=args,
                max_trades=trade_count if baseline_name in {"always_long", "always_short"} and trade_count > 0 else None,
            ),
        )
        rows.append({"baseline": baseline_name, **result.metrics})
    return rows


def _bucket_expr(column: str, *, edges: tuple[float, ...], labels: tuple[str, ...]) -> pl.Expr:
    expr = pl.when(pl.col(column) < edges[0]).then(pl.lit(labels[0]))
    for index, edge in enumerate(edges[1:], start=1):
        expr = expr.when(pl.col(column) < edge).then(pl.lit(labels[index]))
    return expr.otherwise(pl.lit(labels[-1]))


def _summarize_grouped_trades(trades: pl.DataFrame, *, group_column: str) -> list[dict[str, Any]]:
    if trades.is_empty() or group_column not in trades.columns:
        return []
    return [
        dict(row)
        for row in trades.group_by(group_column)
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.mean("gross_return").alias("avg_gross_return"),
                pl.mean("net_return").alias("avg_net_return"),
                pl.mean("cost_return").alias("avg_cost_return"),
                pl.mean("net_hit").alias("net_hit_rate"),
                ((pl.col("net_return") + 1.0).product() - 1.0).alias("total_return"),
            ]
        )
        .sort(group_column)
        .iter_rows(named=True)
    ]


def _trade_diagnostics(trades: pl.DataFrame) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "edge_to_cost_buckets": [],
            "spread_regime": [],
            "volatility_regime": [],
            "liquidity_regime": [],
        }
    out = trades
    out = out.with_columns(
        _bucket_expr(
            "edge_to_cost_ratio",
            edges=(1.0, 1.5, 2.0, 3.0, 4.0),
            labels=("<1x", "1-1.5x", "1.5-2x", "2-3x", "3-4x", ">=4x"),
        ).alias("edge_to_cost_bucket")
    )
    if "relative_spread" in out.columns:
        spread_q = out.select(
            [
                pl.quantile("relative_spread", 0.33).alias("low"),
                pl.quantile("relative_spread", 0.66).alias("high"),
            ]
        ).row(0, named=True)
        out = out.with_columns(
            pl.when(pl.col("relative_spread") <= float(spread_q["low"]))
            .then(pl.lit("tight"))
            .when(pl.col("relative_spread") <= float(spread_q["high"]))
            .then(pl.lit("medium"))
            .otherwise(pl.lit("wide"))
            .alias("spread_regime")
        )
    out = out.with_columns(pl.col("realized_mid_return").abs().alias("__abs_realized_mid_return"))
    vol_q = out.select(
        [
            pl.quantile("__abs_realized_mid_return", 0.33).alias("low"),
            pl.quantile("__abs_realized_mid_return", 0.66).alias("high"),
        ]
    ).row(0, named=True)
    out = out.with_columns(
        pl.when(pl.col("__abs_realized_mid_return") <= float(vol_q["low"]))
        .then(pl.lit("low_vol"))
        .when(pl.col("__abs_realized_mid_return") <= float(vol_q["high"]))
        .then(pl.lit("mid_vol"))
        .otherwise(pl.lit("high_vol"))
        .alias("volatility_regime")
    )
    if {"bid_depth_1", "ask_depth_1", "position_side"}.issubset(set(out.columns)):
        out = out.with_columns(
            pl.when(pl.col("position_side") > 0.0)
            .then(pl.col("ask_depth_1"))
            .otherwise(pl.col("bid_depth_1"))
            .alias("__entry_depth")
        )
        depth_q = out.select(
            [
                pl.quantile("__entry_depth", 0.33).alias("low"),
                pl.quantile("__entry_depth", 0.66).alias("high"),
            ]
        ).row(0, named=True)
        out = out.with_columns(
            pl.when(pl.col("__entry_depth") <= float(depth_q["low"]))
            .then(pl.lit("thin"))
            .when(pl.col("__entry_depth") <= float(depth_q["high"]))
            .then(pl.lit("normal"))
            .otherwise(pl.lit("deep"))
            .alias("liquidity_regime")
        )
    return {
        "edge_to_cost_buckets": _summarize_grouped_trades(out, group_column="edge_to_cost_bucket"),
        "spread_regime": _summarize_grouped_trades(out, group_column="spread_regime"),
        "volatility_regime": _summarize_grouped_trades(out, group_column="volatility_regime"),
        "liquidity_regime": _summarize_grouped_trades(out, group_column="liquidity_regime"),
    }


def _audit_columns(trades: pl.DataFrame) -> list[str]:
    preferred = [
        "timestamp",
        "side",
        "predicted_return",
        "decision_signal_return",
        "entry_mid",
        "entry_bid",
        "entry_ask",
        "exit_mid",
        "exit_bid",
        "exit_ask",
        "realized_mid_return",
        "gross_return",
        "spread_cost_return",
        "fee_cost_return",
        "slippage_cost_return",
        "estimated_round_trip_cost",
        "edge_to_cost_ratio",
        "net_return",
        "holding_horizon",
        "prediction_direction_correct",
        "gross_hit",
        "net_hit",
    ]
    return [column for column in preferred if column in trades.columns]


def _classify_failure(
    *,
    best: dict[str, Any],
    debug_diagnostics: dict[str, Any],
    cost_aware_backtests: list[dict[str, Any]],
    horizon_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    inverted = debug_diagnostics.get("inverted_signal", {})
    regimes = {row["regime"]: row for row in debug_diagnostics.get("cost_regimes", [])}
    no_cost = regimes.get("no_cost", {})
    spread_only = regimes.get("spread_only", {})
    fee_only = regimes.get("fee_only", {})
    active_k = [row for row in cost_aware_backtests if int(row.get("trade_count", 0)) > 0]
    higher_horizon = max(horizon_sweep, key=lambda row: float(row.get("gross_total_return", 0.0)), default={})
    labels: list[str] = []
    evidence: list[str] = []
    if float(inverted.get("gross_total_return", 0.0)) > float(best.get("gross_total_return", 0.0)):
        labels.append("signal sign bug possible")
    else:
        evidence.append("inverted signal did not improve gross or net performance")
    if float(best.get("gross_hit_rate", 0.0)) <= 0.1:
        labels.append("threshold selects poor traded subset")
        evidence.append("selected-trade gross hit rate is very low")
    if len(active_k) <= 1:
        labels.append("threshold too low / edge barely clears cost")
        evidence.append("only the lowest cost-aware k generated trades")
    if float(no_cost.get("total_return", 0.0)) > 0.0 and float(fee_only.get("total_return", 0.0)) < 0.0:
        labels.append("costs too high for taker execution")
        evidence.append("no-cost gross strategy is positive while fee-only selected trades are negative")
    if float(spread_only.get("total_return", 0.0)) > 0.0 and float(fee_only.get("total_return", 0.0)) < 0.0:
        evidence.append("spread alone is not the dominant blocker in this run; round-trip fees dominate selected trades")
    if higher_horizon and float(higher_horizon.get("gross_total_return", 0.0)) > float(best.get("gross_total_return", 0.0)):
        labels.append("horizon may be too short")
        evidence.append(f"horizon {higher_horizon.get('horizon')} had the strongest gross PnL among tested horizons")
    if not labels:
        labels.append("genuinely predictive but untradable signal")
    return {
        "labels": sorted(set(labels)),
        "evidence": evidence,
    }


def _write_report(
    path: Path,
    *,
    run_id: str,
    git_sha: str,
    args: argparse.Namespace,
    feature_paths: list[Path],
    feature_rows: int,
    feature_symbols: int,
    feature_columns: list[str],
    feature_date_span: dict[str, Any],
    prediction_summary: dict[str, Any],
    model_metrics: dict[str, dict[str, float | int]],
    backtests: list[dict[str, Any]],
    cost_aware_backtests: list[dict[str, Any]],
    best: dict[str, Any],
    debug_diagnostics: dict[str, Any],
    horizon_sweep: list[dict[str, Any]],
    failure_classification: dict[str, Any],
) -> None:
    lines = [
        f"# Order-Book Microstructure Benchmark `{run_id}`",
        "",
        "## Scope",
        "",
        "This run trains order-book signal heads on local Binance futures L2 depth snapshots/updates and backtests future-midprice signals with explicit spread and fee costs.",
        "",
        "## Configuration",
        "",
        f"- Git SHA: `{git_sha}`",
        f"- Raw root: `{args.raw_root}`",
        f"- Symbols: `{', '.join(sorted(_symbols_arg(args.symbols)))}`",
        f"- Max files per symbol: `{args.max_files_per_symbol}`",
        f"- Max rows per file: `{args.max_rows_per_file}`",
        f"- Max feature rows: `{args.max_feature_rows}`",
        f"- Horizons: `{args.horizons}`",
        f"- Depth levels: `{args.depth_levels}`",
        f"- Target column: `{args.target_column}`",
        f"- Min train rows: `{args.min_train_rows}`",
        f"- Test rows: `{args.test_rows}`",
        f"- Max folds: `{args.max_folds}`",
        f"- Fee: `{args.fee_bps}` bps per side",
        f"- Slippage: `{args.slippage_bps}` bps per side",
        f"- Min signal abs sweep: `{args.min_signal_abs_sweep}`",
        f"- Min edge over cost sweep: `{args.min_edge_over_cost_sweep}`",
        f"- Edge-to-cost k sweep: `{args.edge_to_cost_k_sweep}`",
        f"- Target horizon sweep: `{args.target_horizon_sweep}`",
        f"- Max relative spread: `{args.max_relative_spread}`",
        f"- Min entry depth: `{args.min_entry_depth}`",
        "",
        "## Data",
        "",
        f"- Feature files produced: `{len(feature_paths)}`",
        f"- Feature source date coverage: `{feature_date_span.get('first_date')}` to `{feature_date_span.get('last_date')}` across `{feature_date_span.get('date_count')}` file dates",
        f"- Feature rows loaded: `{feature_rows:,}`",
        f"- Symbols loaded: `{feature_symbols:,}`",
        f"- Prediction rows: `{_fmt(prediction_summary.get('rows', 0))}`",
        f"- Prediction symbols: `{_fmt(prediction_summary.get('symbols', 0))}`",
        f"- Event time range: `{prediction_summary.get('min_event_time')}` to `{prediction_summary.get('max_event_time')}`",
        "",
        "## Feature Columns",
        "",
        ", ".join(f"`{col}`" for col in feature_columns),
        "",
        "## Model Accuracy",
        "",
        "| model | rows | directional acc. | zero-mean R2 | IC |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in ORDERBOOK_MODEL_NAMES:
        metrics = model_metrics.get(model_name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_name}`",
                    _fmt(metrics.get("rows", 0)),
                    _pct(metrics.get("directional_accuracy", 0.0)),
                    _fmt(metrics.get("zero_mean_r2", 0.0)),
                    _fmt(metrics.get("information_coefficient", 0.0)),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Costed Backtest Sweep",
        "",
        "Selection prioritizes annualized daily Sharpe when at least two daily returns are available, then event-trade Sharpe, then net return.",
        "",
        "| model | min signal abs | min edge > cost | candidates | filtered | trades | daily Sharpe | trade Sharpe | trade rate | hit rate | avg gross/trade | avg cost/trade | avg net/trade | total net | gross total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in backtests:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['model']}`",
                    _fmt(row["min_signal_abs"]),
                    _fmt(row["min_edge_over_cost"]),
                    _fmt(row.get("candidate_count", 0)),
                    _fmt(row.get("filtered_count", 0)),
                    _fmt(row.get("trade_count", 0)),
                    _fmt(row.get("daily_sharpe_ratio", 0.0)),
                    _fmt(row.get("trade_sharpe_ratio", 0.0)),
                    _pct(row.get("trade_rate", 0.0)),
                    _pct(row.get("hit_rate", 0.0)),
                    _pct(row.get("avg_trade_gross_return", 0.0)),
                    _pct(row.get("avg_trade_cost_return", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                    _pct(row.get("gross_total_return", 0.0)),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Cost-Aware Threshold Sweep",
        "",
        "This sweep trades only when `abs(prediction) > k * estimated_round_trip_cost`.",
        "",
        "| model | k | candidates | filtered | trades | gross hit | net hit | daily Sharpe | max drawdown | avg cost/trade | avg net/trade | total net | gross total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cost_aware_backtests:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['model']}`",
                    _fmt(row["edge_to_cost_k"]),
                    _fmt(row.get("candidate_count", 0)),
                    _fmt(row.get("filtered_count", 0)),
                    _fmt(row.get("trade_count", 0)),
                    _pct(row.get("gross_hit_rate", 0.0)),
                    _pct(row.get("net_hit_rate", 0.0)),
                    _fmt(row.get("daily_sharpe_ratio", 0.0)),
                    _pct(row.get("max_drawdown", 0.0)),
                    _pct(row.get("avg_trade_cost_return", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                    _pct(row.get("gross_total_return", 0.0)),
                ]
            )
            + " |"
        )
    lines += ["", "## Best Candidate", ""]
    if best:
        lines += [
            f"- Model: `{best['model']}`",
            f"- Min signal abs: `{best['min_signal_abs']}`",
            f"- Min edge over cost: `{best.get('min_edge_over_cost')}`",
            f"- Edge-to-cost k: `{best.get('edge_to_cost_k')}`",
            f"- Trades: `{_fmt(best.get('trade_count', 0))}`",
            f"- Candidates: `{_fmt(best.get('candidate_count', 0))}`",
            f"- Filtered: `{_fmt(best.get('filtered_count', 0))}`",
            f"- Total net return: `{_pct(best.get('total_return', 0.0))}`",
            f"- Gross total return: `{_pct(best.get('gross_total_return', 0.0))}`",
            f"- Gross hit rate: `{_pct(best.get('gross_hit_rate', 0.0))}`",
            f"- Net hit rate: `{_pct(best.get('net_hit_rate', 0.0))}`",
            f"- Daily Sharpe: `{_fmt(best.get('daily_sharpe_ratio', 0.0))}`",
            f"- Event-trade Sharpe: `{_fmt(best.get('trade_sharpe_ratio', 0.0))}`",
            f"- Max drawdown: `{_pct(best.get('max_drawdown', 0.0))}`",
            f"- Daily return count: `{_fmt(best.get('daily_return_count', 0))}`",
            f"- Hit rate: `{_pct(best.get('hit_rate', 0.0))}`",
            f"- Avg net/trade: `{_pct(best.get('avg_trade_net_return', 0.0))}`",
            f"- Avg spread cost/trade: `{_pct(best.get('avg_spread_cost_return', 0.0))}`",
            f"- Avg fee cost/trade: `{_pct(best.get('avg_fee_cost_return', 0.0))}`",
            f"- Long net return: `{_pct(best.get('long_total_return', 0.0))}` on `{_fmt(best.get('long_trade_count', 0))}` trades",
            f"- Short net return: `{_pct(best.get('short_total_return', 0.0))}` on `{_fmt(best.get('short_trade_count', 0))}` trades",
        ]
    else:
        lines.append("No costed backtest candidate was produced.")
    lines += [
        "",
        "## Failure Classification",
        "",
        f"- Labels: `{', '.join(failure_classification.get('labels', []))}`",
    ]
    for item in failure_classification.get("evidence", []):
        lines.append(f"- Evidence: {item}")
    lines += [
        "",
        "## Trading Conversion Debug",
        "",
        f"- Best-trade audit parquet: `{debug_diagnostics.get('audit_parquet')}`",
        f"- Best-trade audit CSV: `{debug_diagnostics.get('audit_csv')}`",
        f"- All-row prediction metrics: `{debug_diagnostics.get('all_prediction_metrics')}`",
        f"- Traded-row prediction metrics: `{debug_diagnostics.get('traded_prediction_metrics')}`",
        "",
        "### Cost Regimes",
        "",
        "The first table holds the selected trade set fixed and recomputes PnL under each cost component.",
        "",
        "| regime | trades | gross hit | net hit | avg gross/trade | avg net/trade | total net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in debug_diagnostics.get("selected_trade_cost_decomposition", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['regime']}`",
                    _fmt(row.get("trade_count", 0)),
                    _pct(row.get("gross_hit_rate", 0.0)),
                    _pct(row.get("net_hit_rate", 0.0)),
                    _pct(row.get("avg_trade_gross_return", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "The second table reruns the backtest under each cost regime, so the trade set may change as gates change.",
        "",
        "| regime | trades | gross hit | net hit | avg gross/trade | avg spread cost | avg fee cost | avg net/trade | total net | gross total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in debug_diagnostics.get("cost_regimes", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['regime']}`",
                    _fmt(row.get("trade_count", 0)),
                    _pct(row.get("gross_hit_rate", 0.0)),
                    _pct(row.get("net_hit_rate", 0.0)),
                    _pct(row.get("avg_trade_gross_return", 0.0)),
                    _pct(row.get("avg_spread_cost_return", 0.0)),
                    _pct(row.get("avg_fee_cost_return", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                    _pct(row.get("gross_total_return", 0.0)),
                ]
            )
            + " |"
        )
    inverted = debug_diagnostics.get("inverted_signal", {})
    lines += [
        "",
        "### Inverted Signal",
        "",
        f"- Trades: `{_fmt(inverted.get('trade_count', 0))}`",
        f"- Gross hit rate: `{_pct(inverted.get('gross_hit_rate', 0.0))}`",
        f"- Net hit rate: `{_pct(inverted.get('net_hit_rate', 0.0))}`",
        f"- Total net return: `{_pct(inverted.get('total_return', 0.0))}`",
        f"- Gross total return: `{_pct(inverted.get('gross_total_return', 0.0))}`",
        "",
        "### Null Baselines",
        "",
        "| baseline | trades | gross hit | net hit | avg net/trade | total net | gross total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in debug_diagnostics.get("null_baselines", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['baseline']}`",
                    _fmt(row.get("trade_count", 0)),
                    _pct(row.get("gross_hit_rate", 0.0)),
                    _pct(row.get("net_hit_rate", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                    _pct(row.get("gross_total_return", 0.0)),
                ]
            )
            + " |"
        )
    for title, key in (
        ("Edge-To-Cost Buckets", "edge_to_cost_buckets"),
        ("Spread Regime", "spread_regime"),
        ("Volatility Regime", "volatility_regime"),
        ("Liquidity Regime", "liquidity_regime"),
    ):
        lines += [
            "",
            f"### {title}",
            "",
            "| bucket | trades | net hit | avg gross | avg cost | avg net | total net |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in debug_diagnostics.get("trade_diagnostics", {}).get(key, []):
            group_value = row.get(key[:-1] if key.endswith("s") else key, row.get(key, row.get("edge_to_cost_bucket", "")))
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{group_value}`",
                        _fmt(row.get("trade_count", 0)),
                        _pct(row.get("net_hit_rate", 0.0)),
                        _pct(row.get("avg_gross_return", 0.0)),
                        _pct(row.get("avg_cost_return", 0.0)),
                        _pct(row.get("avg_net_return", 0.0)),
                        _pct(row.get("total_return", 0.0)),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Horizon Sweep",
        "",
        "| horizon | best model | k | IC | zero-mean R2 | directional acc. | trades | gross hit | net hit | avg gross/trade | avg net/trade | turnover | net PnL | gross PnL |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in horizon_sweep:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("horizon", 0)),
                    f"`{row.get('model', '')}`",
                    _fmt(row.get("edge_to_cost_k", 0.0)),
                    _fmt(row.get("information_coefficient", 0.0)),
                    _fmt(row.get("zero_mean_r2", 0.0)),
                    _pct(row.get("directional_accuracy", 0.0)),
                    _fmt(row.get("trade_count", 0)),
                    _pct(row.get("gross_hit_rate", 0.0)),
                    _pct(row.get("net_hit_rate", 0.0)),
                    _pct(row.get("avg_trade_gross_return", 0.0)),
                    _pct(row.get("avg_trade_net_return", 0.0)),
                    _pct(row.get("trade_rate", 0.0)),
                    _pct(row.get("total_return", 0.0)),
                    _pct(row.get("gross_total_return", 0.0)),
                ]
            )
            + " |"
        )
    traded_variants = sum(1 for row in backtests if float(row.get("trade_count", 0.0)) > 0.0)
    candidate_variants = sum(1 for row in backtests if float(row.get("candidate_count", 0.0)) > 0.0)
    max_candidates = max((int(row.get("candidate_count", 0)) for row in backtests), default=0)
    max_trades = max((int(row.get("trade_count", 0)) for row in backtests), default=0)
    lines += [
        "",
        "## Gate Diagnostic",
        "",
        f"- Backtest variants with raw candidates: `{candidate_variants}` / `{len(backtests)}`",
        f"- Backtest variants with executed trades: `{traded_variants}` / `{len(backtests)}`",
        f"- Max raw candidates in one variant: `{max_candidates:,}`",
        f"- Max executed trades in one variant: `{max_trades:,}`",
    ]
    if traded_variants == 0 and candidate_variants > 0:
        lines.append(
            "- All raw candidates were filtered after applying predicted-edge-over-cost, spread, and depth gates. "
            "This indicates directional predictability without enough predicted markout magnitude to pay aggressive execution costs."
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- This is a research benchmark, not a live HFT simulator.",
        "- PnL is based on future midprice markout minus spread crossing and fee assumptions; it does not model queue position, partial fills, latency, exchange rebates, funding, or market impact.",
        "- Daily Sharpe is annualized from grouped UTC event dates when at least two daily returns are available; event-trade Sharpe is not a calendar annualized live Sharpe.",
        "- Raw Binance depth updates are parsed as observed top-of-book snapshots; this benchmark does not rebuild a full exchange book from deltas.",
        "- `not_investment_advice: true`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root) / run_id
    feature_dir = output_root / "features"
    raw_root = Path(args.raw_root)
    report_path = Path(args.report) if args.report else Path("reports") / f"orderbook_microstructure_benchmark_{run_id}.md"
    if not raw_root.exists():
        console.print(f"[red]Raw order-book root does not exist:[/red] {raw_root}")
        return 1

    symbols = _symbols_arg(args.symbols)
    console.print(f"[bold]Preparing order-book features[/bold] {sorted(symbols)} -> {feature_dir}")
    feature_paths = write_orderbook_feature_files(
        raw_root=raw_root,
        output_root=feature_dir,
        dataset_id=raw_root.name,
        symbols=symbols,
        max_files_per_symbol=args.max_files_per_symbol,
        max_rows_per_file=args.max_rows_per_file,
        horizons=args.horizons,
        depth_levels=args.depth_levels,
    )
    if not feature_paths:
        console.print("[red]No feature files produced.[/red]")
        return 1

    features = read_orderbook_feature_files(feature_paths, max_rows=args.max_feature_rows)
    if features.is_empty():
        console.print("[red]No feature rows loaded.[/red]")
        return 1
    feature_symbols = int(features["symbol"].n_unique()) if "symbol" in features.columns else 0
    wf_config = OrderBookWalkForwardConfig(
        target_column=args.target_column,
        min_train_rows=args.min_train_rows,
        test_rows=args.test_rows,
        step_rows=args.step_rows,
        max_folds=args.max_folds,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
        hist_gradient_max_iter=args.hist_gradient_max_iter,
        fee_bps=args.fee_bps,
        starting_equity=args.starting_equity,
    )
    console.print("[bold]Running walk-forward training[/bold]")
    wf = run_orderbook_walk_forward(features, config=wf_config)
    predictions_path = output_root / "walk_forward_predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    wf.predictions.write_parquet(predictions_path, compression="zstd")

    backtests: list[dict[str, Any]] = []
    for model_name in ORDERBOOK_MODEL_NAMES:
        for min_signal_abs in args.min_signal_abs_sweep:
            for min_edge_over_cost in args.min_edge_over_cost_sweep:
                result = run_orderbook_signal_backtest(
                    wf.predictions,
                    config=OrderBookBacktestConfig(
                        prediction_column=f"pred_{model_name}",
                        target_column=args.target_column,
                        min_signal_abs=float(min_signal_abs),
                        min_edge_over_cost=float(min_edge_over_cost),
                        max_relative_spread=args.max_relative_spread,
                        min_entry_depth=args.min_entry_depth,
                        slippage_bps=args.slippage_bps,
                        fee_bps=args.fee_bps,
                        starting_equity=args.starting_equity,
                    ),
                )
                row = {
                    "model": model_name,
                    "min_signal_abs": float(min_signal_abs),
                    "min_edge_over_cost": float(min_edge_over_cost),
                    "max_relative_spread": args.max_relative_spread,
                    "min_entry_depth": args.min_entry_depth,
                    **result.metrics,
                }
                backtests.append(row)
                signal_slug = str(min_signal_abs).replace(".", "p")
                edge_slug = str(min_edge_over_cost).replace(".", "p")
                trades_path = output_root / f"{model_name}_signal_{signal_slug}_edge_{edge_slug}_trades.parquet"
                result.trades.write_parquet(trades_path, compression="zstd")

    cost_aware_backtests = _run_cost_aware_sweep(wf.predictions, target_column=args.target_column, args=args)
    best = _best_debug_row(backtests, cost_aware_backtests)
    best_model = str(best.get("model", "ensemble_mean"))
    best_result = run_orderbook_signal_backtest(
        wf.predictions,
        config=_config_from_row(best, target_column=args.target_column, args=args),
    )
    audit_frame = best_result.trades.select(_audit_columns(best_result.trades)) if not best_result.trades.is_empty() else pl.DataFrame()
    audit_parquet = output_root / "best_trade_pnl_audit.parquet"
    audit_csv = output_root / "best_trade_pnl_audit.csv"
    audit_frame.write_parquet(audit_parquet, compression="zstd")
    audit_frame.write_csv(audit_csv)
    debug_diagnostics = {
        "audit_parquet": str(audit_parquet),
        "audit_csv": str(audit_csv),
        "all_prediction_metrics": _prediction_metrics(
            wf.predictions,
            prediction_column=f"pred_{best_model}",
            target_column=args.target_column,
        ),
        "traded_prediction_metrics": _prediction_metrics(
            best_result.trades,
            prediction_column="predicted_return",
            target_column=args.target_column,
        ),
        "cost_regimes": _run_cost_regimes(
            wf.predictions,
            model_name=best_model,
            target_column=args.target_column,
            args=args,
        ),
        "selected_trade_cost_decomposition": _selected_trade_cost_decomposition(best_result.trades),
        "inverted_signal": _run_inverted_signal(
            wf.predictions,
            model_name=best_model,
            target_column=args.target_column,
            args=args,
        ),
        "null_baselines": _run_null_baselines(
            wf.predictions,
            model_name=best_model,
            target_column=args.target_column,
            args=args,
            trade_count=int(best.get("trade_count", 0)),
        ),
        "trade_diagnostics": _trade_diagnostics(best_result.trades),
    }

    horizon_sweep: list[dict[str, Any]] = []
    for horizon in args.target_horizon_sweep:
        target_column = f"future_mid_return_{horizon}"
        if target_column not in features.columns:
            horizon_sweep.append({"horizon": horizon, "error": f"missing {target_column}"})
            continue
        horizon_config = OrderBookWalkForwardConfig(
            target_column=target_column,
            min_train_rows=args.min_train_rows,
            test_rows=args.test_rows,
            step_rows=args.step_rows,
            max_folds=args.max_folds,
            max_train_rows_per_fold=args.max_train_rows_per_fold,
            hist_gradient_max_iter=args.hist_gradient_max_iter,
            fee_bps=args.fee_bps,
            starting_equity=args.starting_equity,
        )
        console.print(f"[bold]Running horizon sweep[/bold] h={horizon}")
        wf_horizon = run_orderbook_walk_forward(features, config=horizon_config)
        horizon_predictions_path = output_root / f"walk_forward_predictions_h{horizon}.parquet"
        wf_horizon.predictions.write_parquet(horizon_predictions_path, compression="zstd")
        horizon_cost_aware = _run_cost_aware_sweep(wf_horizon.predictions, target_column=target_column, args=args)
        horizon_best = _best_backtest(horizon_cost_aware)
        model_name = str(horizon_best.get("model", ""))
        metrics = wf_horizon.model_metrics.get(model_name, {}) if model_name else {}
        horizon_sweep.append(
            {
                "horizon": horizon,
                "target_column": target_column,
                "prediction_path": str(horizon_predictions_path),
                "model": model_name,
                "edge_to_cost_k": horizon_best.get("edge_to_cost_k", 0.0),
                "information_coefficient": metrics.get("information_coefficient", 0.0),
                "zero_mean_r2": metrics.get("zero_mean_r2", 0.0),
                "directional_accuracy": metrics.get("directional_accuracy", 0.0),
                **horizon_best,
            }
        )
    failure_classification = _classify_failure(
        best=best,
        debug_diagnostics=debug_diagnostics,
        cost_aware_backtests=cost_aware_backtests,
        horizon_sweep=horizon_sweep,
    )

    artifact_paths: dict[str, str] = {}
    if args.save_final_artifacts:
        final_models = train_final_orderbook_models(features, config=wf_config, feature_columns=wf.feature_columns)
        artifact_paths = save_orderbook_model_artifacts(
            models=final_models,
            feature_columns=wf.feature_columns,
            target_column=args.target_column,
            output_dir=output_root / "models",
            metadata={
                "raw_root": str(raw_root),
                "symbols": sorted(symbols),
                "trained_on": "all loaded feature rows after benchmark",
            },
        )

    payload = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": vars(args),
        "feature_paths": [str(path) for path in feature_paths],
        "feature_rows": features.height,
        "feature_symbols": feature_symbols,
        "feature_columns": wf.feature_columns,
        "feature_date_span": _date_span_from_paths(feature_paths),
        "fold_specs": [asdict(spec) for spec in wf.fold_specs],
        "fold_metrics": wf.fold_metrics,
        "prediction_summary": _prediction_summary(wf.predictions),
        "model_metrics": wf.model_metrics,
        "backtests": backtests,
        "cost_aware_backtests": cost_aware_backtests,
        "best": best,
        "debug_diagnostics": debug_diagnostics,
        "horizon_sweep": horizon_sweep,
        "failure_classification": failure_classification,
        "artifact_paths": artifact_paths,
        "prediction_path": str(predictions_path),
        "report_path": str(report_path),
    }
    _write_json(output_root / "summary.json", payload)
    _write_report(
        report_path,
        run_id=run_id,
        git_sha=payload["git_sha"],
        args=args,
        feature_paths=feature_paths,
        feature_rows=features.height,
        feature_symbols=feature_symbols,
        feature_columns=wf.feature_columns,
        feature_date_span=payload["feature_date_span"],
        prediction_summary=payload["prediction_summary"],
        model_metrics=wf.model_metrics,
        backtests=backtests,
        cost_aware_backtests=cost_aware_backtests,
        best=best,
        debug_diagnostics=debug_diagnostics,
        horizon_sweep=horizon_sweep,
        failure_classification=failure_classification,
    )
    console.print(f"[bold]Best[/bold] {best}")
    console.print(f"Wrote {report_path}")
    console.print(f"Wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
