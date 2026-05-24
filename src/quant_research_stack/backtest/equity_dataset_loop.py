from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import polars as pl

from quant_research_stack.backtest.equity_signal import (
    normalize_equity_ohlcv,
    run_long_short_signal_backtest,
)
from quant_research_stack.backtest.equity_walk_forward import (
    MODEL_NAMES,
    EquityWalkForwardConfig,
    run_equity_walk_forward,
    save_signal_artifacts,
    train_final_equity_models,
)


@dataclass(frozen=True)
class EquityDatasetCandidate:
    name: str
    label: str
    repo_id: str
    paths: tuple[Path, ...]
    date_column: str
    symbol_column: str
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"
    universe: str = "unknown"
    priority: int = 100


@dataclass(frozen=True)
class EquityDatasetLoopConfig:
    target_column: str = "future_return_1"
    min_train_dates: int = 504
    test_window_dates: int = 63
    step_dates: int = 63
    max_folds: int | None = 4
    max_train_rows_per_fold: int | None = 150_000
    ridge_alpha: float = 10.0
    hist_gradient_max_iter: int = 40
    selection_fraction: float = 0.10
    cost_bps: float = 5.0
    rebalance_every_n_days: int = 1
    starting_equity: float = 100_000.0
    max_symbols_per_side: int | None = None
    max_rows_per_dataset: int | None = None
    tail_dates_per_dataset: int | None = 1_000
    min_close: float = 0.0
    min_dollar_volume: float = 0.0
    max_abs_future_return: float | None = None
    save_predictions: bool = True
    save_final_artifacts: bool = True


@dataclass(frozen=True)
class DatasetProbeResult:
    candidate: str
    usable: bool
    reason: str
    files: tuple[str, ...]
    schema: tuple[str, ...]


@dataclass
class DatasetCandidateResult:
    candidate: EquityDatasetCandidate
    status: Literal["ok", "rejected", "error"]
    reason: str | None
    best_model: str | None
    model_metrics: dict[str, dict[str, float | int]]
    backtest_metrics: dict[str, dict[str, float | int]]
    monthly_metrics: dict[str, dict[str, float | int]]
    feature_columns: list[str]
    artifact_paths: dict[str, str]
    raw_rows: int
    normalized_rows: int
    prediction_rows: int
    symbols: int
    dates: int


def _existing_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".parquet"}))
    return sorted(dict.fromkeys(files))


def _collect_schema(path: Path) -> tuple[str, ...]:
    if path.suffix.lower() == ".parquet":
        return tuple(pl.scan_parquet(path).collect_schema().names())
    if path.suffix.lower() == ".csv":
        return tuple(pl.scan_csv(path, ignore_errors=True, infer_schema_length=10_000).collect_schema().names())
    return ()


def probe_candidate(candidate: EquityDatasetCandidate) -> DatasetProbeResult:
    files = _existing_files(candidate.paths)
    if not files:
        return DatasetProbeResult(
            candidate=candidate.name,
            usable=False,
            reason="no local CSV/parquet files found",
            files=(),
            schema=(),
        )
    schema = set(_collect_schema(files[0]))
    required = {
        candidate.date_column,
        candidate.symbol_column,
        candidate.open_column,
        candidate.high_column,
        candidate.low_column,
        candidate.close_column,
        candidate.volume_column,
    }
    missing = sorted(required - schema)
    if missing:
        return DatasetProbeResult(
            candidate=candidate.name,
            usable=False,
            reason=f"missing OHLCV columns: {missing}",
            files=tuple(str(path) for path in files),
            schema=tuple(sorted(schema)),
        )
    return DatasetProbeResult(
        candidate=candidate.name,
        usable=True,
        reason="usable OHLCV schema",
        files=tuple(str(path) for path in files),
        schema=tuple(sorted(schema)),
    )


def _read_one(path: Path, max_rows: int | None) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif suffix == ".csv":
        lf = pl.scan_csv(path, ignore_errors=True, infer_schema_length=10_000)
    else:
        raise ValueError(f"unsupported dataset file: {path}")
    if max_rows is not None:
        lf = lf.head(max_rows)
    return lf.collect()


def read_candidate_frame(candidate: EquityDatasetCandidate, max_rows: int | None = None) -> pl.DataFrame:
    files = _existing_files(candidate.paths)
    if not files:
        raise FileNotFoundError(f"no files for candidate {candidate.name}")
    frames: list[pl.DataFrame] = []
    remaining = max_rows
    for path in files:
        frame = _read_one(path, remaining)
        if frame.is_empty():
            continue
        frames.append(frame)
        if remaining is not None:
            remaining -= frame.height
            if remaining <= 0:
                break
    if not frames:
        raise ValueError(f"all files were empty for candidate {candidate.name}")
    return pl.concat(frames, how="diagonal_relaxed")


def _tail_dates(frame: pl.DataFrame, *, date_column: str, n_dates: int | None) -> pl.DataFrame:
    if n_dates is None or n_dates <= 0:
        return frame
    if date_column not in frame.columns:
        raise ValueError(f"missing date column: {date_column}")
    dates = frame.select(date_column).unique().sort(date_column).tail(n_dates)[date_column].to_list()
    return frame.filter(pl.col(date_column).is_in(dates))


def filter_normalized_equity_frame(
    frame: pl.DataFrame,
    *,
    target_column: str,
    min_close: float = 0.0,
    min_dollar_volume: float = 0.0,
    max_abs_future_return: float | None = None,
) -> pl.DataFrame:
    out = frame
    if min_close > 0.0 and "close" in out.columns:
        out = out.filter(pl.col("close") >= min_close)
    if min_dollar_volume > 0.0 and "dollar_volume" in out.columns:
        out = out.filter(pl.col("dollar_volume") >= min_dollar_volume)
    if max_abs_future_return is not None and max_abs_future_return > 0.0 and target_column in out.columns:
        out = out.filter(pl.col(target_column).abs() <= max_abs_future_return)
    return out


def build_monthly_return_table(daily_curve: pl.DataFrame) -> pl.DataFrame:
    if daily_curve.is_empty():
        return pl.DataFrame(
            {
                "month": [],
                "n_days": [],
                "net_return": [],
                "gross_return": [],
                "net_income": [],
                "gross_income": [],
                "cost_return_sum": [],
            }
        )
    required = {"date", "net_return", "gross_return", "cost_return", "equity", "gross_equity"}
    missing = required - set(daily_curve.columns)
    if missing:
        raise ValueError(f"daily curve missing monthly columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    curve = daily_curve.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 7).alias("__month"))
    for group in curve.partition_by("__month", maintain_order=True):
        month = str(group["__month"][0])
        first_net = float(group["net_return"][0])
        first_gross = float(group["gross_return"][0])
        end_equity = float(group["equity"][-1])
        end_gross_equity = float(group["gross_equity"][-1])
        start_equity = end_equity
        if 1.0 + first_net > 0.0:
            start_equity = float(group["equity"][0]) / (1.0 + first_net)
        start_gross_equity = end_gross_equity
        if 1.0 + first_gross > 0.0:
            start_gross_equity = float(group["gross_equity"][0]) / (1.0 + first_gross)
        net_income = end_equity - start_equity
        gross_income = end_gross_equity - start_gross_equity
        rows.append(
            {
                "month": month,
                "n_days": int(group.height),
                "net_return": (end_equity / start_equity - 1.0) if start_equity > 0.0 else 0.0,
                "gross_return": (end_gross_equity / start_gross_equity - 1.0) if start_gross_equity > 0.0 else 0.0,
                "net_income": net_income,
                "gross_income": gross_income,
                "cost_return_sum": float(group["cost_return"].sum()),
            }
        )
    return pl.DataFrame(rows)


def summarize_monthly_returns(monthly: pl.DataFrame) -> dict[str, float | int]:
    if monthly.is_empty():
        return {
            "months": 0,
            "avg_monthly_net_return": 0.0,
            "median_monthly_net_return": 0.0,
            "positive_month_share": 0.0,
            "best_month_return": 0.0,
            "worst_month_return": 0.0,
            "total_net_income": 0.0,
            "avg_monthly_net_income": 0.0,
        }
    net = monthly["net_return"]
    income = monthly["net_income"]
    return {
        "months": int(monthly.height),
        "avg_monthly_net_return": float(cast(float, net.mean())),
        "median_monthly_net_return": float(cast(float, net.median())),
        "positive_month_share": float(cast(float, (net > 0.0).mean())),
        "best_month_return": float(cast(float, net.max())),
        "worst_month_return": float(cast(float, net.min())),
        "total_net_income": float(cast(float, income.sum())),
        "avg_monthly_net_income": float(cast(float, income.mean())),
    }


def _walk_forward_config(config: EquityDatasetLoopConfig) -> EquityWalkForwardConfig:
    return EquityWalkForwardConfig(
        target_column=config.target_column,
        min_train_dates=config.min_train_dates,
        test_window_dates=config.test_window_dates,
        step_dates=config.step_dates,
        max_folds=config.max_folds,
        max_train_rows_per_fold=config.max_train_rows_per_fold,
        ridge_alpha=config.ridge_alpha,
        hist_gradient_max_iter=config.hist_gradient_max_iter,
        selection_fraction=config.selection_fraction,
        cost_bps=config.cost_bps,
        starting_equity=config.starting_equity,
        max_symbols_per_side=config.max_symbols_per_side,
    )


def _score_candidate_model(result: DatasetCandidateResult, model_name: str) -> tuple[float, float, float]:
    monthly = result.monthly_metrics.get(model_name, {})
    backtest = result.backtest_metrics.get(model_name, {})
    model = result.model_metrics.get(model_name, {})
    return (
        float(monthly.get("avg_monthly_net_return", 0.0)),
        float(backtest.get("total_return", 0.0)),
        float(model.get("rank_ic_mean", 0.0)),
    )


def _best_model(result: DatasetCandidateResult) -> str | None:
    if not result.monthly_metrics:
        return None
    return max(result.monthly_metrics, key=lambda name: _score_candidate_model(result, name))


def run_dataset_candidate(
    candidate: EquityDatasetCandidate,
    config: EquityDatasetLoopConfig,
    *,
    output_dir: Path,
) -> DatasetCandidateResult:
    probe = probe_candidate(candidate)
    empty = DatasetCandidateResult(
        candidate=candidate,
        status="rejected",
        reason=probe.reason,
        best_model=None,
        model_metrics={},
        backtest_metrics={},
        monthly_metrics={},
        feature_columns=[],
        artifact_paths={},
        raw_rows=0,
        normalized_rows=0,
        prediction_rows=0,
        symbols=0,
        dates=0,
    )
    if not probe.usable:
        return empty

    try:
        raw = read_candidate_frame(candidate, config.max_rows_per_dataset)
        normalized = normalize_equity_ohlcv(
            raw,
            dataset_id=candidate.name,
            date_column=candidate.date_column,
            symbol_column=candidate.symbol_column,
            open_column=candidate.open_column,
            high_column=candidate.high_column,
            low_column=candidate.low_column,
            close_column=candidate.close_column,
            volume_column=candidate.volume_column,
        )
        normalized = _tail_dates(normalized, date_column="date", n_dates=config.tail_dates_per_dataset)
        normalized = filter_normalized_equity_frame(
            normalized,
            target_column=config.target_column,
            min_close=config.min_close,
            min_dollar_volume=config.min_dollar_volume,
            max_abs_future_return=config.max_abs_future_return,
        )
        wf_config = _walk_forward_config(config)
        wf = run_equity_walk_forward(normalized, wf_config)
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.save_predictions:
            wf.predictions.write_parquet(output_dir / "walk_forward_predictions.parquet", compression="zstd")

        backtest_metrics: dict[str, dict[str, float | int]] = {}
        monthly_metrics: dict[str, dict[str, float | int]] = {}
        for model_name in MODEL_NAMES:
            bt = run_long_short_signal_backtest(
                wf.predictions,
                prediction_column=f"pred_{model_name}",
                target_column=config.target_column,
                date_column="date",
                starting_equity=config.starting_equity,
                selection_fraction=config.selection_fraction,
                cost_bps=config.cost_bps,
                rebalance_every_n_days=config.rebalance_every_n_days,
                max_symbols_per_side=config.max_symbols_per_side,
            )
            backtest_metrics[model_name] = bt.metrics
            monthly = build_monthly_return_table(bt.daily_curve)
            monthly.write_parquet(output_dir / f"{model_name}_monthly_returns.parquet", compression="zstd")
            monthly_metrics[model_name] = summarize_monthly_returns(monthly)

        artifact_paths: dict[str, str] = {}
        if config.save_final_artifacts:
            final_models = train_final_equity_models(normalized, wf_config, feature_columns=wf.feature_columns)
            artifact_paths = save_signal_artifacts(
                models=final_models,
                feature_columns=wf.feature_columns,
                target_column=config.target_column,
                output_dir=output_dir / "models",
                metadata={
                    "dataset": candidate.name,
                    "repo_id": candidate.repo_id,
                    "trained_on": "all normalized rows with non-null target after loop benchmark",
                },
            )

        result = DatasetCandidateResult(
            candidate=candidate,
            status="ok",
            reason=None,
            best_model=None,
            model_metrics=wf.model_metrics,
            backtest_metrics=backtest_metrics,
            monthly_metrics=monthly_metrics,
            feature_columns=wf.feature_columns,
            artifact_paths=artifact_paths,
            raw_rows=raw.height,
            normalized_rows=normalized.height,
            prediction_rows=wf.predictions.height,
            symbols=int(normalized["symbol"].n_unique()),
            dates=int(normalized["date"].n_unique()),
        )
        result.best_model = _best_model(result)
        return result
    except Exception as exc:
        empty.status = "error"
        empty.reason = f"{type(exc).__name__}: {exc}"
        return empty


def best_overall_result(results: list[DatasetCandidateResult]) -> DatasetCandidateResult | None:
    ok = [result for result in results if result.status == "ok" and result.best_model is not None]
    if not ok:
        return None
    return max(ok, key=lambda result: _score_candidate_model(result, result.best_model or ""))
