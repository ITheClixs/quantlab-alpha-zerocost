from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_research_stack.backtest.equity_signal import (
    evaluate_signal_accuracy,
    load_signal_model,
    normalize_equity_ohlcv,
    predict_signal_frame,
    run_long_short_signal_backtest,
)

console = Console()


@dataclass(frozen=True)
class EquityDatasetSpec:
    name: str
    label: str
    paths: tuple[Path, ...]
    date_column: str
    symbol_column: str
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"


DATASETS: dict[str, EquityDatasetSpec] = {
    "sp500": EquityDatasetSpec(
        name="sp500",
        label="S&P 500 daily equities",
        paths=(
            Path("data/raw/huggingface/jwigginton__timeseries-daily-sp500/data/train-00000-of-00001.parquet"),
        ),
        date_column="date",
        symbol_column="symbol",
    ),
    "nasdaq": EquityDatasetSpec(
        name="nasdaq",
        label="NASDAQ daily equities",
        paths=(Path("data/raw/huggingface/benstaf__nasdaq_2013_2023/trade_data_2019_2023.csv"),),
        date_column="date",
        symbol_column="tic",
    ),
    "nyse": EquityDatasetSpec(
        name="nyse",
        label="NYSE daily equities",
        paths=(Path("data/raw/kaggle/datasets/dgawlik__nyse/prices-split-adjusted.csv"),),
        date_column="date",
        symbol_column="symbol",
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _first_existing_path(spec: EquityDatasetSpec) -> Path:
    for path in spec.paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"none of the configured paths exist for {spec.name}: {spec.paths}")


def _read_table(path: Path, max_rows: int | None) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
        if max_rows is not None:
            lf = lf.head(max_rows)
        return lf.collect()
    if suffix == ".csv":
        lf = pl.scan_csv(path, ignore_errors=True, infer_schema_length=10000)
        if max_rows is not None:
            lf = lf.head(max_rows)
        return lf.collect()
    raise ValueError(f"unsupported dataset file type: {path}")


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.3f}%"
    except Exception:
        return str(value)


def _number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _metric_subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key, 0.0) for key in keys}


def _human_join(values: list[str]) -> str:
    if not values:
        return "no datasets"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _write_markdown_report(
    *,
    path: Path,
    run_id: str,
    model_artifact: str,
    model_features: list[str],
    target_column: str,
    dataset_results: list[dict[str, Any]],
    cost_bps: float,
    selection_fraction: float,
    git_sha: str,
    starting_equity: float,
    max_rows_per_dataset: int | None,
    max_symbols_per_side: int | None,
    save_signals: bool,
) -> None:
    scope = _human_join([str(result["label"]) for result in dataset_results])
    lines = [
        f"# Equity Signal Backtest Report `{run_id}`",
        "",
        "## Scope",
        "",
        f"This report evaluates the locally persisted market signal head on historical {scope} data.",
        "The Jane Street S1 stack is not applied here because its inference contract is opaque `feature_00..feature_78`; mapping listed-equity OHLCV into those features would be out-of-domain.",
        "",
        "## Configuration",
        "",
        f"- Git SHA: `{git_sha}`",
        f"- Signal model artifact: `{model_artifact}`",
        f"- Model feature count: `{len(model_features)}`",
        f"- Model features: `{', '.join(model_features)}`",
        f"- Strategy: daily dollar-neutral long/short, top/bottom `{selection_fraction:.2%}` by predicted next-day return",
        f"- Cost model: `{cost_bps}` bps per one-way notional, two gross notional turns per daily rebalance",
        f"- Starting equity: `{starting_equity}`",
        f"- Max rows per dataset: `{max_rows_per_dataset}`",
        f"- Max symbols per side: `{max_symbols_per_side}`",
        f"- Save signal parquet: `{str(save_signals).lower()}`",
        f"- Target: `{target_column}`",
        "",
        "## Headline Results",
        "",
        "| dataset | rows | symbols | dates | directional acc. | rank IC | top-bottom spread | net return | Sharpe | max drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in dataset_results:
        accuracy = result["accuracy_metrics"]
        backtest = result["backtest_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result['dataset']}`",
                    _number(accuracy["rows"]),
                    _number(result["symbols"]),
                    _number(result["dates"]),
                    _percent(accuracy["directional_accuracy"]),
                    _number(accuracy["rank_ic_mean"]),
                    _percent(accuracy["top_bottom_spread_return"]),
                    _percent(backtest["total_return"]),
                    _number(backtest["sharpe_ratio"]),
                    _percent(backtest["max_drawdown"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Dataset Details",
        "",
    ]
    for result in dataset_results:
        accuracy = result["accuracy_metrics"]
        backtest = result["backtest_metrics"]
        lines += [
            f"### `{result['dataset']}` - {result['label']}",
            "",
            f"- Source: `{result['source_path']}`",
            f"- Date range: `{result['start_date']}` to `{result['end_date']}`",
            f"- Normalized rows: `{result['normalized_rows']:,}`",
            f"- Evaluated signal rows: `{accuracy['rows']:,}`",
            f"- Symbols: `{result['symbols']:,}`",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
        for key, value in _metric_subset(
            accuracy,
            (
                "directional_accuracy",
                "positive_precision",
                "negative_precision",
                "zero_mean_r2",
                "information_coefficient",
                "rank_ic_mean",
                "rank_ic_std",
                "top_mean_forward_return",
                "bottom_mean_forward_return",
                "top_bottom_spread_return",
                "positive_signal_share",
            ),
        ).items():
            rendered = _percent(value) if "return" in key or "accuracy" in key or "precision" in key or key.endswith("share") else _number(value)
            lines.append(f"| `{key}` | {rendered} |")
        lines += [
            "",
            "| backtest metric | value |",
            "|---|---:|",
        ]
        for key, value in _metric_subset(
            backtest,
            (
                "n_days",
                "total_return",
                "gross_total_return",
                "cost_drag_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "hit_rate",
                "avg_daily_turnover",
                "avg_daily_net_return",
                "avg_daily_gross_return",
            ),
        ).items():
            rendered = _percent(value) if "return" in key or key == "max_drawdown" or key == "hit_rate" else _number(value)
            lines.append(f"| `{key}` | {rendered} |")
        lines += [
            "",
            f"Artifacts: `{result['artifact_dir']}`",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "- Treat these as out-of-sample transfer diagnostics for the locally trained generic OHLCV signal head, not as production trading results.",
        "- A useful signal should show positive rank IC, positive top-bottom forward-return spread, and survive the explicit cost model.",
        "- Negative zero-mean R2 can coexist with useful rank spread; for trading, ranking quality and costed portfolio returns matter more than raw point-forecast MSE.",
        "- The current model artifact was not trained specifically on these US equity universes. A dedicated walk-forward retrain per universe should be the next upgrade before promotion decisions.",
        "",
        "## Limitations",
        "",
        "- Daily close-to-close simulation only; no intraday fill timing, opening auction, borrow fees, locate constraints, dividends, delistings, or point-in-time constituent membership are modeled.",
        "- The long/short book uses equal weights and a simple daily full-rebalance cost approximation.",
        "- The S&P and NYSE historical datasets may contain survivorship and vendor-adjustment bias. Use point-in-time universe files before treating results as capacity evidence.",
        "- `not_investment_advice: true`",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest local market signal heads on S&P/Nasdaq/NYSE daily equities.")
    parser.add_argument("--model-artifact", default="experiments/local_signal_training/market/ridge.joblib")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), default=[])
    parser.add_argument("--output-root", default="experiments/equity_signal_backtests")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--max-symbols-per-side", type=int, default=None)
    parser.add_argument("--no-save-signals", action="store_true")
    return parser.parse_args()


def _forward_horizons_for_target(target_column: str) -> tuple[int, ...]:
    match = re.fullmatch(r"future_return_(\d+)", target_column)
    if match is None:
        return (1,)
    horizon = int(match.group(1))
    return tuple(sorted({1, horizon}))


def _run_dataset(
    *,
    spec: EquityDatasetSpec,
    model_artifact: Any,
    run_dir: Path,
    max_rows: int | None,
    selection_fraction: float,
    cost_bps: float,
    starting_equity: float,
    max_symbols_per_side: int | None,
    save_signals: bool,
) -> dict[str, Any]:
    source_path = _first_existing_path(spec)
    console.print(f"[bold]Loading[/bold] {spec.name}: {source_path}")
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
        forward_horizons=_forward_horizons_for_target(model_artifact.target_column),
    )
    signals = predict_signal_frame(normalized, model_artifact)
    accuracy = evaluate_signal_accuracy(
        signals,
        target_column=model_artifact.target_column,
        selection_quantile=selection_fraction,
    )
    backtest = run_long_short_signal_backtest(
        signals,
        target_column=model_artifact.target_column,
        starting_equity=starting_equity,
        selection_fraction=selection_fraction,
        cost_bps=cost_bps,
        max_symbols_per_side=max_symbols_per_side,
    )

    artifact_dir = run_dir / spec.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    signal_cols = [
        "dataset_id",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1",
        "realized_vol_5",
        "realized_vol_20",
        "realized_vol_60",
        model_artifact.target_column,
        "prediction",
        "prediction_abs",
        "signal_side",
    ]
    if save_signals:
        signals.select([col for col in signal_cols if col in signals.columns]).write_parquet(
            artifact_dir / "signals.parquet",
            compression="zstd",
        )
    backtest.daily_curve.write_parquet(artifact_dir / "daily_curve.parquet", compression="zstd")

    result: dict[str, Any] = {
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
        "accuracy_metrics": accuracy,
        "backtest_metrics": backtest.metrics,
    }
    _write_json(artifact_dir / "metrics.json", result)
    console.print(
        f"  rows={accuracy['rows']:,} directional={float(accuracy['directional_accuracy']):.3f} "
        f"rank_ic={float(accuracy['rank_ic_mean']):.4f} net_return={float(backtest.metrics['total_return']):.3%}"
    )
    return result


def main() -> int:
    args = parse_args()
    selected = args.dataset or ["sp500", "nasdaq", "nyse"]
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    model_artifact = load_signal_model(args.model_artifact)
    dataset_results = []
    for name in selected:
        dataset_results.append(
            _run_dataset(
                spec=DATASETS[name],
                model_artifact=model_artifact,
                run_dir=run_dir,
                max_rows=args.max_rows_per_dataset,
                selection_fraction=args.selection_fraction,
                cost_bps=args.cost_bps,
                starting_equity=args.starting_equity,
                max_symbols_per_side=args.max_symbols_per_side,
                save_signals=not args.no_save_signals,
            )
        )

    summary = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "model_artifact": str(model_artifact.artifact_path),
        "model_features": model_artifact.feature_columns,
        "target_column": model_artifact.target_column,
        "selection_fraction": args.selection_fraction,
        "cost_bps": args.cost_bps,
        "starting_equity": args.starting_equity,
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "max_symbols_per_side": args.max_symbols_per_side,
        "save_signals": not args.no_save_signals,
        "datasets": dataset_results,
    }
    _write_json(run_dir / "summary.json", summary)
    report_path = report_root / f"equity_signal_backtest_{run_id}.md"
    _write_markdown_report(
        path=report_path,
        run_id=run_id,
        model_artifact=str(model_artifact.artifact_path),
        model_features=model_artifact.feature_columns,
        target_column=model_artifact.target_column,
        dataset_results=dataset_results,
        cost_bps=args.cost_bps,
        selection_fraction=args.selection_fraction,
        git_sha=summary["git_sha"],
        starting_equity=args.starting_equity,
        max_rows_per_dataset=args.max_rows_per_dataset,
        max_symbols_per_side=args.max_symbols_per_side,
        save_signals=not args.no_save_signals,
    )
    console.print(f"[bold green]Wrote[/bold green] {report_path}")
    console.print(f"[bold green]Artifacts[/bold green] {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
