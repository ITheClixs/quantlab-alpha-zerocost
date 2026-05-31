from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from equity_signal_backtest import (  # noqa: E402
    DATASETS,
    EquityDatasetSpec,
    _first_existing_path,
    _forward_horizons_for_target,
    _read_table,
)
from rich.console import Console

from quant_research_stack.backtest.equity_signal import normalize_equity_ohlcv
from quant_research_stack.backtest.equity_walk_forward import (
    MODEL_NAMES,
    EquityWalkForwardConfig,
    run_equity_walk_forward,
    save_signal_artifacts,
    train_final_equity_models,
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


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.3f}%"
    except Exception:
        return str(value)


def _model_row(name: str, result: dict[str, Any]) -> str:
    accuracy = result["model_metrics"][name]
    backtest = result["backtest_metrics"][name]
    return (
        "| "
        + " | ".join(
            [
                f"`{name}`",
                _fmt(accuracy["rows"]),
                _pct(accuracy["directional_accuracy"]),
                _fmt(accuracy["rank_ic_mean"]),
                _pct(accuracy["top_bottom_spread_return"]),
                _pct(backtest["total_return"]),
                _pct(backtest["gross_total_return"]),
                _pct(backtest["cost_drag_return"]),
                _fmt(backtest["sharpe_ratio"]),
                _pct(backtest["max_drawdown"]),
            ]
        )
        + " |"
    )


def _write_report(
    *,
    path: Path,
    run_id: str,
    git_sha: str,
    results: list[dict[str, Any]],
    config: EquityWalkForwardConfig,
    max_rows_per_dataset: int | None,
    tail_dates_per_dataset: int | None,
    save_predictions: bool,
) -> None:
    lines = [
        f"# Equity Walk-Forward Retrain Report `{run_id}`",
        "",
        "## Scope",
        "",
        "This benchmark trains dedicated OHLCV signal heads separately for each selected US equity universe.",
        "Every fold trains only on dates before the test window, then evaluates out-of-sample predictions with the same cost-aware daily dollar-neutral long/short backtest used for the generic OHLCV benchmark.",
        "",
        "## Configuration",
        "",
        f"- Git SHA: `{git_sha}`",
        f"- Target: `{config.target_column}`",
        "- Features: past/current OHLCV-derived features only",
        f"- Models: `{', '.join(MODEL_NAMES)}`",
        f"- Minimum training dates: `{config.min_train_dates}`",
        f"- Test window dates: `{config.test_window_dates}`",
        f"- Step dates: `{config.step_dates}`",
        f"- Max folds: `{config.max_folds}`",
        f"- Max train rows per fold: `{config.max_train_rows_per_fold}`",
        f"- Selection fraction: `{config.selection_fraction:.2%}`",
        f"- Cost model: `{config.cost_bps}` bps one-way, two gross turns per daily rebalance",
        f"- Starting equity: `{config.starting_equity}`",
        f"- Max rows per dataset: `{max_rows_per_dataset}`",
        f"- Tail dates per dataset: `{tail_dates_per_dataset}`",
        f"- Save predictions: `{str(save_predictions).lower()}`",
        "",
        "## Results",
        "",
    ]
    for result in results:
        lines += [
            f"### `{result['dataset']}` - {result['label']}",
            "",
            f"- Source: `{result['source_path']}`",
            f"- Raw rows: `{result['raw_rows']:,}`",
            f"- Normalized rows: `{result['normalized_rows']:,}`",
            f"- Symbols: `{result['symbols']:,}`",
            f"- Dates: `{result['dates']:,}`",
            f"- Walk-forward folds: `{len(result['folds'])}`",
            f"- Feature count: `{len(result['feature_columns'])}`",
            f"- Artifacts: `{result['artifact_dir']}`",
            "",
            "| model | rows | directional acc. | rank IC | top-bottom spread | net return | gross return | cost drag | Sharpe | max drawdown |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in MODEL_NAMES:
            lines.append(_model_row(name, result))
        lines += [
            "",
            "Fold windows:",
            "",
            "| fold | train dates | train window | test dates | test window | train rows | test rows |",
            "|---:|---:|---|---:|---|---:|---:|",
        ]
        for fold in result["fold_metrics"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(fold["fold"]),
                        _fmt(fold["train_dates"]),
                        f"`{fold['train_start']}` to `{fold['train_end']}`",
                        _fmt(fold["test_dates"]),
                        f"`{fold['test_start']}` to `{fold['test_end']}`",
                        _fmt(fold["train_rows"]),
                        _fmt(fold["test_rows"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Interpretation Notes",
        "",
        "- This is still daily OHLCV, not futures tick/order-book data, so it cannot prove HFT viability.",
        "- The benchmark is leakage-aware at the date level but does not model short borrow, exchange-specific fees, queue position, or intraday fills.",
        "- `ensemble_mean` is a fixed average of ridge and histogram-gradient predictions inside each fold; it is cooperative but intentionally simple.",
        "- `not_investment_advice: true`",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark universe-specific OHLCV signal heads.")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), default=[])
    parser.add_argument("--output-root", default="experiments/equity_walk_forward")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--target-column", default="future_return_1")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--tail-dates-per-dataset", type=int, default=None)
    parser.add_argument("--min-train-dates", type=int, default=756)
    parser.add_argument("--test-window-dates", type=int, default=126)
    parser.add_argument("--step-dates", type=int, default=126)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=500_000)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--hist-gradient-max-iter", type=int, default=80)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--max-symbols-per-side", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--skip-final-artifacts", action="store_true")
    return parser.parse_args()


def _tail_dates(frame: pl.DataFrame, *, date_column: str, n_dates: int | None) -> pl.DataFrame:
    if n_dates is None or n_dates <= 0:
        return frame
    dates = frame.select(date_column).unique().sort(date_column).tail(n_dates)[date_column].to_list()
    return frame.filter(pl.col(date_column).is_in(dates))


def _normalize_dataset(
    spec: EquityDatasetSpec,
    *,
    target_column: str,
    max_rows: int | None,
    tail_dates: int | None,
) -> tuple[Path, pl.DataFrame, pl.DataFrame]:
    source_path = _first_existing_path(spec)
    raw = _read_table(source_path, max_rows)
    normalized = normalize_equity_ohlcv(
        raw,
        dataset_id=spec.name,
        date_column=spec.date_column,
        symbol_column=spec.symbol_column,
        open_column=spec.open_column,
        high_column=spec.high_column,
        low_column=spec.low_column,
        close_column=spec.close_column,
        volume_column=spec.volume_column,
        forward_horizons=_forward_horizons_for_target(target_column),
    )
    normalized = _tail_dates(normalized, date_column="date", n_dates=tail_dates)
    return source_path, raw, normalized


def _run_dataset(
    *,
    spec: EquityDatasetSpec,
    run_dir: Path,
    config: EquityWalkForwardConfig,
    max_rows: int | None,
    tail_dates: int | None,
    save_predictions: bool,
    save_final_artifacts: bool,
) -> dict[str, Any]:
    console.print(f"[bold]Walk-forward retrain[/bold] {spec.name}")
    source_path, raw, normalized = _normalize_dataset(
        spec,
        target_column=config.target_column,
        max_rows=max_rows,
        tail_dates=tail_dates,
    )
    result = run_equity_walk_forward(normalized, config)
    artifact_dir = run_dir / spec.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        result.predictions.write_parquet(artifact_dir / "walk_forward_predictions.parquet", compression="zstd")
    if save_final_artifacts:
        final_models = train_final_equity_models(normalized, config, feature_columns=result.feature_columns)
        save_signal_artifacts(
            models=final_models,
            feature_columns=result.feature_columns,
            target_column=config.target_column,
            output_dir=artifact_dir / "models",
            metadata={
                "dataset": spec.name,
                "source_path": str(source_path),
                "trained_on": "all rows with non-null target after walk-forward benchmark",
            },
        )
    payload: dict[str, Any] = {
        "dataset": spec.name,
        "label": spec.label,
        "source_path": str(source_path),
        "artifact_dir": str(artifact_dir),
        "raw_rows": raw.height,
        "normalized_rows": normalized.height,
        "symbols": int(normalized["symbol"].n_unique()),
        "dates": int(normalized["date"].n_unique()),
        "start_date": str(normalized["date"].min()),
        "end_date": str(normalized["date"].max()),
        "tail_dates_per_dataset": tail_dates,
        "feature_columns": result.feature_columns,
        "folds": [fold.__dict__ for fold in result.fold_specs],
        "fold_metrics": result.fold_metrics,
        "model_metrics": result.model_metrics,
        "backtest_metrics": result.backtest_metrics,
        "saved_predictions": save_predictions,
        "saved_final_artifacts": save_final_artifacts,
    }
    _write_json(artifact_dir / "summary.json", payload)
    best = max(MODEL_NAMES, key=lambda name: float(result.backtest_metrics[name]["total_return"]))
    console.print(
        f"  best={best} net_return={float(result.backtest_metrics[best]['total_return']):.3%} "
        f"rank_ic={float(result.model_metrics[best]['rank_ic_mean']):.4f} folds={len(result.fold_specs)}"
    )
    return payload


def main() -> int:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    config = EquityWalkForwardConfig(
        target_column=args.target_column,
        min_train_dates=args.min_train_dates,
        test_window_dates=args.test_window_dates,
        step_dates=args.step_dates,
        max_folds=args.max_folds,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
        ridge_alpha=args.ridge_alpha,
        hist_gradient_max_iter=args.hist_gradient_max_iter,
        selection_fraction=args.selection_fraction,
        cost_bps=args.cost_bps,
        starting_equity=args.starting_equity,
        max_symbols_per_side=args.max_symbols_per_side,
    )
    selected = args.dataset or ["sp500", "nasdaq", "nyse"]
    results = [
        _run_dataset(
            spec=DATASETS[name],
            run_dir=run_dir,
            config=config,
            max_rows=args.max_rows_per_dataset,
            tail_dates=args.tail_dates_per_dataset,
            save_predictions=not args.no_save_predictions,
            save_final_artifacts=not args.skip_final_artifacts,
        )
        for name in selected
    ]
    summary = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config.__dict__,
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "tail_dates_per_dataset": args.tail_dates_per_dataset,
        "datasets": results,
    }
    _write_json(run_dir / "summary.json", summary)
    report_path = report_root / f"equity_walk_forward_retrain_{run_id}.md"
    _write_report(
        path=report_path,
        run_id=run_id,
        git_sha=summary["git_sha"],
        results=results,
        config=config,
        max_rows_per_dataset=args.max_rows_per_dataset,
        tail_dates_per_dataset=args.tail_dates_per_dataset,
        save_predictions=not args.no_save_predictions,
    )
    console.print(f"[bold green]Wrote[/bold green] {report_path}")
    console.print(f"[bold green]Artifacts[/bold green] {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
