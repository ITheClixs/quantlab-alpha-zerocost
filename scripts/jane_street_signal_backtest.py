from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
from rich.console import Console

from quant_research_stack.alpha.io import LoadConfig, scan_jane_street
from quant_research_stack.backtest.jane_street_signal import (
    evaluate_prediction_column,
    run_grouped_long_short_backtest,
)

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _selected_tail_groups(lf: pl.LazyFrame, group_column: str, max_rows: int) -> list[int]:
    counts = (
        lf.group_by(group_column)
        .agg(pl.len().alias("__n"))
        .sort(group_column, descending=True)
        .collect()
    )
    if counts.is_empty():
        raise ValueError("no Jane Street rows available")
    cum = counts.with_columns(pl.col("__n").cum_sum().alias("__cum"))
    under = cum.filter(pl.col("__cum") <= max_rows)
    if under.height == 0:
        selected = cum.head(1)
    elif under.height < cum.height:
        selected = cum.head(under.height + 1)
    else:
        selected = under
    return [int(x) for x in selected[group_column].sort().to_list()]


def reconstruct_holdout_index(run_dir: Path) -> pl.DataFrame:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    data_cfg = metadata["hyperparams"]["data"]
    target_column = str(data_cfg["target_column"])
    weight_column = str(data_cfg["weight_column"])
    group_column = str(data_cfg["group_column"])
    holdout_fraction = float(data_cfg["permanent_holdout_fraction"])
    row_budget = int(metadata["hyperparams"].get("max_rows_streaming") or data_cfg["max_rows"])
    root = data_cfg["jane_street_root"]
    lf = scan_jane_street(
        root,
        LoadConfig(
            target_column=target_column,
            weight_column=weight_column,
            group_column=group_column,
            holdout_fraction=holdout_fraction,
        ),
    )
    selected_groups = _selected_tail_groups(lf, group_column, row_budget)
    group_count = len(selected_groups)
    holdout_n = max(1, int(group_count * holdout_fraction))
    holdout_groups = selected_groups[-holdout_n:]
    columns = [group_column, "time_id", "symbol_id", weight_column, target_column]
    return (
        lf.filter(pl.col(group_column).is_in(holdout_groups))
        .select(columns)
        .collect()
        .sort(group_column)
    )


def _load_aligned_holdout(run_dir: Path) -> pl.DataFrame:
    preds = pl.read_parquet(run_dir / "predictions.parquet").filter(pl.col("split") == "holdout")
    index = reconstruct_holdout_index(run_dir)
    if preds.height != index.height:
        raise ValueError(f"holdout row mismatch: predictions={preds.height} reconstructed={index.height}")
    aligned = pl.concat([index, preds.drop(["split", "weight"])], how="horizontal")
    target_diff = (
        (aligned["responder_6"].cast(pl.Float64) - aligned["target_actual"].cast(pl.Float64))
        .abs()
        .max()
    )
    weight_diff = (
        (index["weight"].cast(pl.Float64) - preds["weight"].cast(pl.Float64))
        .abs()
        .max()
    )
    if target_diff is None or weight_diff is None:
        raise ValueError("alignment check failed on empty holdout")
    target_diff_f = float(cast(Any, target_diff))
    weight_diff_f = float(cast(Any, weight_diff))
    if target_diff_f > 1e-5 or weight_diff_f > 1e-5:
        raise ValueError(f"holdout alignment mismatch: target_diff={target_diff_f} weight_diff={weight_diff_f}")
    return aligned


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _write_report(
    *,
    path: Path,
    run_id: str,
    source_run_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    holdout: pl.DataFrame,
    model_metrics: dict[str, dict[str, float | int]],
    backtest_metrics: dict[str, dict[str, float | int]],
    git_sha: str,
) -> None:
    date_min = str(cast(Any, holdout["date_id"].min()))
    date_max = str(cast(Any, holdout["date_id"].max()))
    lines = [
        f"# Jane Street S1 Signal Backtest `{run_id}`",
        "",
        "## Scope",
        "",
        "This benchmark tests the serious S1 stack on its native Jane Street competition-style market.",
        "The data is anonymized: rows have `date_id`, `time_id`, `symbol_id`, `weight`, `feature_00..feature_78`, and `responder_6`; there are no exchange prices or OHLCV bars, so PnL is reported in weighted responder units rather than dollars.",
        "",
        "## Configuration",
        "",
        f"- Git SHA: `{git_sha}`",
        f"- Source run: `{source_run_dir}`",
        f"- Training source: `{metadata['hyperparams']['data']['jane_street_root']}`",
        f"- Training row cap: `{metadata['hyperparams'].get('max_rows_streaming')}`",
        f"- Target: `{metadata['hyperparams']['data']['target_column']}`",
        f"- Weight: `{metadata['hyperparams']['data']['weight_column']}`",
        f"- Holdout date range: `{date_min}` to `{date_max}`",
        f"- Holdout rows: `{holdout.height:,}`",
        f"- Holdout dates: `{holdout['date_id'].n_unique():,}`",
        f"- Holdout symbols: `{holdout['symbol_id'].n_unique():,}`",
        f"- Holdout time buckets: `{holdout['time_id'].n_unique():,}`",
        "- Long/short pseudo-backtest: top/bottom `10%` predictions per `date_id`, weighted by Jane Street `weight`.",
        "",
        "## Model Metrics",
        "",
        "| model | rows | weighted R2 | weighted directional acc. | positive precision | negative precision | weighted sign capture | weighted IC | positive signal share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in model_metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    _fmt(metrics["rows"]),
                    _fmt(metrics["weighted_zero_mean_r2"]),
                    _fmt(metrics["weighted_directional_accuracy"]),
                    _fmt(metrics["positive_precision"]),
                    _fmt(metrics["negative_precision"]),
                    _fmt(metrics["weighted_sign_capture"]),
                    _fmt(metrics["weighted_information_coefficient"]),
                    _fmt(metrics["positive_signal_share"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Long/Short Pseudo-Backtest",
        "",
        "| model | dates | total pnl units | mean pnl units | mean long-short spread | sharpe-like | hit rate | max drawdown units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in backtest_metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    _fmt(metrics["n_groups"]),
                    _fmt(metrics["total_pnl_units"]),
                    _fmt(metrics["mean_pnl_units"]),
                    _fmt(metrics["mean_long_short_spread"]),
                    _fmt(metrics["sharpe_like"]),
                    _fmt(metrics["hit_rate"]),
                    _fmt(metrics["max_drawdown_units"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Output dir: `{output_dir}`",
        "- `summary.json`: all metrics and run metadata",
        "- `stacked_daily_curve.parquet`: date-level pseudo-PnL curve for the stacked model",
        "",
        "## Interpretation Notes",
        "",
        "- `weighted_zero_mean_r2` is the native Jane Street competition-style score.",
        "- Pseudo-PnL is not dollar PnL because the dataset does not expose tradable prices, spreads, or fills.",
        "- This is the correct native benchmark for the S1 stack; testing this model on S&P/Nasdaq/NYSE OHLCV would require an invalid feature mapping from OHLCV to opaque `feature_00..feature_78`.",
        "- `not_investment_advice: true`",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest an S1 Jane Street run on native holdout predictions.")
    parser.add_argument("--run-dir", default="experiments/alpha_s1/20260523-160541")
    parser.add_argument("--output-root", default="experiments/jane_street_signal_backtests")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = Path(args.run_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source_run_dir / "metadata.json").read_text())

    console.print(f"[bold]Reconstructing holdout[/bold] for {source_run_dir}")
    holdout = _load_aligned_holdout(source_run_dir)
    prediction_columns = ["stacked", "ridge", "lgb", "xgb", "cat", "mlp", "seq"]
    model_metrics = {
        column: evaluate_prediction_column(holdout, column, target_column="target_actual", weight_column="weight")
        for column in prediction_columns
    }
    backtest_results = {
        column: run_grouped_long_short_backtest(
            holdout,
            column,
            target_column="target_actual",
            weight_column="weight",
            group_column="date_id",
            selection_fraction=args.selection_fraction,
        )
        for column in prediction_columns
    }
    backtest_metrics = {column: result.metrics for column, result in backtest_results.items()}
    backtest_results["stacked"].daily_curve.write_parquet(output_dir / "stacked_daily_curve.parquet")
    summary = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "source_run_dir": str(source_run_dir),
        "selection_fraction": args.selection_fraction,
        "holdout_rows": holdout.height,
        "holdout_dates": holdout["date_id"].n_unique(),
        "holdout_symbols": holdout["symbol_id"].n_unique(),
        "holdout_time_buckets": holdout["time_id"].n_unique(),
        "model_metrics": model_metrics,
        "backtest_metrics": backtest_metrics,
    }
    _write_json(output_dir / "summary.json", summary)
    report_path = report_root / f"jane_street_signal_backtest_{run_id}.md"
    _write_report(
        path=report_path,
        run_id=run_id,
        source_run_dir=source_run_dir,
        output_dir=output_dir,
        metadata=metadata,
        holdout=holdout,
        model_metrics=model_metrics,
        backtest_metrics=backtest_metrics,
        git_sha=str(summary["git_sha"]),
    )
    console.print(f"[bold green]Wrote[/bold green] {report_path}")
    console.print(f"[bold green]Artifacts[/bold green] {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
