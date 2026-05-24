from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from rich.console import Console

from quant_research_stack.artifacts import safe_repo_id
from quant_research_stack.backtest.equity_dataset_loop import (
    EquityDatasetCandidate,
    EquityDatasetLoopConfig,
    best_overall_result,
    probe_candidate,
    run_dataset_candidate,
)

console = Console()


def _hf_dir(repo_id: str) -> Path:
    return Path("data/raw/huggingface") / safe_repo_id(repo_id)


def candidate_entries() -> list[tuple[EquityDatasetCandidate, tuple[str, ...]]]:
    sp500_daily = _hf_dir("jwigginton/timeseries-daily-sp500")
    nasdaq_daily = _hf_dir("benstaf/nasdaq_2013_2023")
    sp500_2025 = _hf_dir("mospira/sp500-daily-candles-2025")
    hexquant = _hf_dir("HexQuant/Stocks-Daily-Price")
    nasdaq_shifted = _hf_dir("zexianli/nasdaq_shifted")
    nasdaq_close = _hf_dir("Q-bert/NASDAQ-Daily-Close-Random-100")
    return [
        (
            EquityDatasetCandidate(
                name="hf_sp500_daily",
                label="HF S&P 500 daily OHLCV",
                repo_id="jwigginton/timeseries-daily-sp500",
                paths=(sp500_daily / "data",),
                date_column="date",
                symbol_column="symbol",
                universe="sp500",
                priority=10,
            ),
            ("*.parquet", "README*"),
        ),
        (
            EquityDatasetCandidate(
                name="hf_nasdaq_2013_2023",
                label="HF NASDAQ 2013-2023 daily OHLCV/features",
                repo_id="benstaf/nasdaq_2013_2023",
                paths=(nasdaq_daily / "train_data_2013_2018.csv", nasdaq_daily / "trade_data_2019_2023.csv"),
                date_column="date",
                symbol_column="tic",
                universe="nasdaq",
                priority=20,
            ),
            ("train_data_2013_2018.csv", "trade_data_2019_2023.csv", "README*"),
        ),
        (
            EquityDatasetCandidate(
                name="hf_hexquant_stocks_daily",
                label="HF HexQuant broad stocks daily OHLCV",
                repo_id="HexQuant/Stocks-Daily-Price",
                paths=(hexquant / "data",),
                date_column="date",
                symbol_column="symbol",
                universe="broad_us_equity",
                priority=30,
            ),
            ("*.parquet", "README*"),
        ),
        (
            EquityDatasetCandidate(
                name="hf_sp500_2025",
                label="HF S&P 500 2025 daily candles",
                repo_id="mospira/sp500-daily-candles-2025",
                paths=(sp500_2025 / "sp500-daily-tickers-2025.csv",),
                date_column="date",
                symbol_column="ticker",
                universe="sp500",
                priority=40,
            ),
            ("*.csv", "README*"),
        ),
        (
            EquityDatasetCandidate(
                name="hf_nasdaq_shifted_reject_probe",
                label="HF NASDAQ shifted/news probe",
                repo_id="zexianli/nasdaq_shifted",
                paths=(nasdaq_shifted,),
                date_column="Date",
                symbol_column="symbol",
                universe="nasdaq",
                priority=90,
            ),
            ("*.csv", "README*"),
        ),
        (
            EquityDatasetCandidate(
                name="hf_nasdaq_close_sequence_reject_probe",
                label="HF NASDAQ close sequence probe",
                repo_id="Q-bert/NASDAQ-Daily-Close-Random-100",
                paths=(nasdaq_close / "data",),
                date_column="date",
                symbol_column="symbol",
                universe="nasdaq",
                priority=100,
            ),
            ("*.parquet", "README*"),
        ),
    ]


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


def _best_metrics(result: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if result.best_model is None:
        return {}, {}, {}
    return (
        result.monthly_metrics.get(result.best_model, {}),
        result.backtest_metrics.get(result.best_model, {}),
        result.model_metrics.get(result.best_model, {}),
    )


def _result_payload(result: Any) -> dict[str, Any]:
    candidate = result.candidate
    return {
        "candidate": {**candidate.__dict__, "paths": [str(path) for path in candidate.paths]},
        "status": result.status,
        "reason": result.reason,
        "best_model": result.best_model,
        "model_metrics": result.model_metrics,
        "backtest_metrics": result.backtest_metrics,
        "monthly_metrics": result.monthly_metrics,
        "feature_columns": result.feature_columns,
        "artifact_paths": result.artifact_paths,
        "raw_rows": result.raw_rows,
        "normalized_rows": result.normalized_rows,
        "prediction_rows": result.prediction_rows,
        "symbols": result.symbols,
        "dates": result.dates,
    }


def _download_if_needed(candidate: EquityDatasetCandidate, allow_patterns: tuple[str, ...], fetch_missing: bool) -> str | None:
    if probe_candidate(candidate).files or not fetch_missing:
        return None
    local_dir = _hf_dir(candidate.repo_id)
    console.print(f"[bold]Fetching HF dataset[/bold] {candidate.repo_id} -> {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=candidate.repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=list(allow_patterns),
        max_workers=4,
    )
    return candidate.repo_id


def _write_report(
    path: Path,
    *,
    run_id: str,
    git_sha: str,
    results: list[Any],
    best: Any | None,
    downloaded: list[str],
    config: EquityDatasetLoopConfig,
    target_avg_monthly_net_return: float,
    target_total_return: float,
) -> None:
    lines = [
        f"# HF Equity Dataset Loop `{run_id}`",
        "",
        "## Scope",
        "",
        "This run probes Hugging Face equity datasets, fetches missing configured datasets when enabled, rejects non-OHLCV schemas, trains dedicated OHLCV signal heads, and backtests out-of-sample predictions by month.",
        "",
        "## Configuration",
        "",
        f"- Git SHA: `{git_sha}`",
        f"- Target column: `{config.target_column}`",
        f"- Min train dates: `{config.min_train_dates}`",
        f"- Test window dates: `{config.test_window_dates}`",
        f"- Max folds: `{config.max_folds}`",
        f"- Max rows per dataset: `{config.max_rows_per_dataset}`",
        f"- Tail dates per dataset: `{config.tail_dates_per_dataset}`",
        f"- Min close: `{config.min_close}`",
        f"- Min dollar volume: `{config.min_dollar_volume}`",
        f"- Max abs future return: `{config.max_abs_future_return}`",
        f"- Cost: `{config.cost_bps}` bps one-way",
        f"- Selection fraction: `{config.selection_fraction:.2%}`",
        f"- Target avg monthly net return: `{target_avg_monthly_net_return}`",
        f"- Target total return: `{target_total_return}`",
        f"- Downloaded this run: `{', '.join(downloaded) if downloaded else 'none'}`",
        "",
        "## Leaderboard",
        "",
        "| dataset | status | best model | months | avg monthly net | positive months | total net | gross total | rank IC | dir. acc. | reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        monthly, backtest, model = _best_metrics(result)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.candidate.name}`",
                    f"`{result.status}`",
                    f"`{result.best_model or ''}`",
                    _fmt(monthly.get("months", 0)),
                    _pct(monthly.get("avg_monthly_net_return", 0.0)),
                    _pct(monthly.get("positive_month_share", 0.0)),
                    _pct(backtest.get("total_return", 0.0)),
                    _pct(backtest.get("gross_total_return", 0.0)),
                    _fmt(model.get("rank_ic_mean", 0.0)),
                    _pct(model.get("directional_accuracy", 0.0)),
                    str(result.reason or ""),
                ]
            )
            + " |"
        )
    lines += ["", "## Best Candidate", ""]
    if best is None or best.best_model is None:
        lines.append("No usable candidate produced a model.")
    else:
        monthly, backtest, model = _best_metrics(best)
        passed = (
            float(monthly.get("avg_monthly_net_return", 0.0)) >= target_avg_monthly_net_return
            and float(backtest.get("total_return", 0.0)) >= target_total_return
        )
        lines += [
            f"- Dataset: `{best.candidate.name}`",
            f"- Repo: `{best.candidate.repo_id}`",
            f"- Model: `{best.best_model}`",
            f"- Avg monthly net return: `{float(monthly.get('avg_monthly_net_return', 0.0)):.6g}`",
            f"- Avg monthly net income: `{float(monthly.get('avg_monthly_net_income', 0.0)):.6g}`",
            f"- Total net return: `{float(backtest.get('total_return', 0.0)):.6g}`",
            f"- Gross total return: `{float(backtest.get('gross_total_return', 0.0)):.6g}`",
            f"- Rank IC: `{float(model.get('rank_ic_mean', 0.0)):.6g}`",
            f"- Target passed: `{str(passed).lower()}`",
        ]
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- The controller searches a finite configured dataset list; it does not run unbounded retries until profitability appears.",
        "- Poor net return with positive gross return usually means the signal is too high-turnover for the cost model.",
        "- Rejected datasets remain in the report so adapters can be added deliberately.",
        "- `not_investment_advice: true`",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterate over HF equity OHLCV datasets, train, and monthly-backtest models.")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--output-root", default="experiments/equity_hf_dataset_loop")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--stop-on-target", action="store_true")
    parser.add_argument("--target-avg-monthly-net-return", type=float, default=0.0)
    parser.add_argument("--target-total-return", type=float, default=0.0)
    parser.add_argument("--tail-dates-per-dataset", type=int, default=1000)
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--min-close", type=float, default=0.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-abs-future-return", type=float, default=None)
    parser.add_argument("--min-train-dates", type=int, default=504)
    parser.add_argument("--test-window-dates", type=int, default=63)
    parser.add_argument("--step-dates", type=int, default=63)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=150_000)
    parser.add_argument("--hist-gradient-max-iter", type=int, default=40)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--max-symbols-per-side", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--skip-final-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = sorted(candidate_entries(), key=lambda item: item[0].priority)
    if args.candidate:
        selected = set(args.candidate)
        entries = [item for item in entries if item[0].name in selected or item[0].repo_id in selected]
    if args.max_candidates is not None:
        entries = entries[: args.max_candidates]

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = EquityDatasetLoopConfig(
        min_train_dates=args.min_train_dates,
        test_window_dates=args.test_window_dates,
        step_dates=args.step_dates,
        max_folds=args.max_folds,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
        hist_gradient_max_iter=args.hist_gradient_max_iter,
        selection_fraction=args.selection_fraction,
        cost_bps=args.cost_bps,
        starting_equity=args.starting_equity,
        max_symbols_per_side=args.max_symbols_per_side,
        max_rows_per_dataset=args.max_rows_per_dataset,
        tail_dates_per_dataset=args.tail_dates_per_dataset,
        min_close=args.min_close,
        min_dollar_volume=args.min_dollar_volume,
        max_abs_future_return=args.max_abs_future_return,
        save_predictions=not args.no_save_predictions,
        save_final_artifacts=not args.skip_final_artifacts,
    )

    downloaded: list[str] = []
    results = []
    for candidate, allow_patterns in entries:
        downloaded_repo = _download_if_needed(candidate, allow_patterns, args.fetch_missing)
        if downloaded_repo is not None:
            downloaded.append(downloaded_repo)
        console.print(f"[bold]Evaluating[/bold] {candidate.name}")
        result = run_dataset_candidate(candidate, config, output_dir=run_dir / candidate.name)
        results.append(result)
        if result.status == "ok" and result.best_model is not None:
            monthly, backtest, _ = _best_metrics(result)
            console.print(
                f"  best={result.best_model} avg_monthly={float(monthly.get('avg_monthly_net_return', 0.0)):.3%} "
                f"total={float(backtest.get('total_return', 0.0)):.3%}"
            )
            if (
                args.stop_on_target
                and float(monthly.get("avg_monthly_net_return", 0.0)) >= args.target_avg_monthly_net_return
                and float(backtest.get("total_return", 0.0)) >= args.target_total_return
            ):
                break
        else:
            console.print(f"  {result.status}: {result.reason}")

    best = best_overall_result(results)
    payload = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config.__dict__,
        "downloaded": downloaded,
        "target_avg_monthly_net_return": args.target_avg_monthly_net_return,
        "target_total_return": args.target_total_return,
        "best": None if best is None else {"candidate": best.candidate.name, "model": best.best_model},
        "results": [_result_payload(result) for result in results],
    }
    _write_json(run_dir / "summary.json", payload)
    report_path = Path(args.report_root) / f"equity_hf_dataset_loop_{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report_path,
        run_id=run_id,
        git_sha=payload["git_sha"],
        results=results,
        best=best,
        downloaded=downloaded,
        config=config,
        target_avg_monthly_net_return=args.target_avg_monthly_net_return,
        target_total_return=args.target_total_return,
    )
    console.print(f"[bold green]Wrote[/bold green] {report_path}")
    console.print(f"[bold green]Artifacts[/bold green] {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
