from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--max-relative-spread", type=float, default=None)
    parser.add_argument("--min-entry-depth", type=float, default=None)
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
    best: dict[str, Any],
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
        f"- Min signal abs sweep: `{args.min_signal_abs_sweep}`",
        f"- Min edge over cost sweep: `{args.min_edge_over_cost_sweep}`",
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
    lines += ["", "## Best Candidate", ""]
    if best:
        lines += [
            f"- Model: `{best['model']}`",
            f"- Min signal abs: `{best['min_signal_abs']}`",
            f"- Min edge over cost: `{best['min_edge_over_cost']}`",
            f"- Trades: `{_fmt(best.get('trade_count', 0))}`",
            f"- Candidates: `{_fmt(best.get('candidate_count', 0))}`",
            f"- Filtered: `{_fmt(best.get('filtered_count', 0))}`",
            f"- Total net return: `{_pct(best.get('total_return', 0.0))}`",
            f"- Gross total return: `{_pct(best.get('gross_total_return', 0.0))}`",
            f"- Daily Sharpe: `{_fmt(best.get('daily_sharpe_ratio', 0.0))}`",
            f"- Event-trade Sharpe: `{_fmt(best.get('trade_sharpe_ratio', 0.0))}`",
            f"- Daily return count: `{_fmt(best.get('daily_return_count', 0))}`",
            f"- Hit rate: `{_pct(best.get('hit_rate', 0.0))}`",
            f"- Avg net/trade: `{_pct(best.get('avg_trade_net_return', 0.0))}`",
        ]
    else:
        lines.append("No costed backtest candidate was produced.")
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

    best = _best_backtest(backtests)
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
        "best": best,
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
        best=best,
    )
    console.print(f"[bold]Best[/bold] {best}")
    console.print(f"Wrote {report_path}")
    console.print(f"Wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
