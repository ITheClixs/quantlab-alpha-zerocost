"""Walk-forward supervised training for the triple-barrier meta-labeler.

This module is intentionally research-validation only. It trains a secondary
RandomForest meta-labeler on chronological folds and writes auditable artifacts,
but it never marks a candidate as promotion-eligible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.signal_research.methodology.meta_labeling import (
    MetaLabelingEligibility,
)
from quant_research_stack.signal_research.papers.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierWrapper,
    label_triple_barrier,
)

_FEATURE_COLUMNS = (
    "primary_position",
    "log_return_1",
    "log_return_5",
    "log_return_lookback",
    "realized_vol_20",
    "volume_z_20",
)

_DISCLAIMER = (
    "The project may be production-intended, but this artifact is research output only "
    "and is not automatically investment advice. External advisory or capital-management "
    "use requires legal, regulatory, licensing, and compliance review before deployment."
)


@dataclass(frozen=True)
class MetaLabelWalkForwardConfig:
    lookback_days: int = 20
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 63
    purge_days: int = 20
    min_train_events: int = 200
    random_forest_estimators: int = 200
    probability_threshold: float = 0.55
    cost_bps_one_way: float = 1.0
    seed: int = 42
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)


@dataclass(frozen=True)
class MetaLabelWalkForwardResult:
    predictions: pl.DataFrame
    fold_metrics: pl.DataFrame
    summary: dict[str, Any]


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


def _daily_returns(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.is_empty():
        return pl.DataFrame({"date": [], "daily_gross_return": [], "daily_net_return": []})
    return predictions.group_by("date").agg(
        [
            pl.col("gross_return").mean().alias("daily_gross_return"),
            pl.col("net_return").mean().alias("daily_net_return"),
        ]
    ).sort("date")


def _feature_frame(panel: pl.DataFrame, config: MetaLabelWalkForwardConfig) -> pl.DataFrame:
    required = {"date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df = panel.sort(["symbol", "date"]).with_columns(
        [
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("log_return_1"),
            (pl.col("close").log() - pl.col("close").shift(5).over("symbol").log()).alias("log_return_5"),
            (
                pl.col("close").log()
                - pl.col("close").shift(config.lookback_days).over("symbol").log()
            ).alias("log_return_lookback"),
            (
                pl.col("close").shift(-config.triple_barrier.vertical_barrier_days).over("symbol")
                / pl.col("close")
                - 1.0
            ).alias("future_return_horizon"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("log_return_1").rolling_std(window_size=20, min_samples=20).over("symbol").alias("realized_vol_20"),
            (
                (pl.col("volume") - pl.col("volume").rolling_mean(window_size=20, min_samples=20).over("symbol"))
                / (pl.col("volume").rolling_std(window_size=20, min_samples=20).over("symbol") + 1e-12)
            ).alias("volume_z_20"),
            pl.when(pl.col("log_return_lookback") > 0.0)
            .then(1.0)
            .when(pl.col("log_return_lookback") < 0.0)
            .then(-1.0)
            .otherwise(0.0)
            .alias("primary_position"),
        ]
    )
    frames: list[pl.DataFrame] = []
    for _, group in df.group_by("symbol", maintain_order=True):
        labels = label_triple_barrier(
            close=group["close"].to_numpy().astype(np.float64),
            positions=group["primary_position"].to_numpy().astype(np.float64),
            cfg=config.triple_barrier,
        )
        frames.append(group.with_columns(pl.Series("triple_barrier_label", labels)))
    labeled = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    finite_cols = [*_FEATURE_COLUMNS, "triple_barrier_label", "future_return_horizon"]
    return labeled.filter(pl.all_horizontal([pl.col(c).is_finite() for c in finite_cols]))


def _xy(frame: pl.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        frame.select(_FEATURE_COLUMNS).to_numpy().astype(np.float64),
        frame["triple_barrier_label"].to_numpy().astype(np.float64),
    )


def _predict_fold(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    fold: int,
    config: MetaLabelWalkForwardConfig,
) -> pl.DataFrame:
    x_train, y_train = _xy(train)
    x_test, _ = _xy(test)
    wrapper = TripleBarrierWrapper(
        TripleBarrierConfig(
            vertical_barrier_days=config.triple_barrier.vertical_barrier_days,
            profit_take_multiplier=config.triple_barrier.profit_take_multiplier,
            stop_loss_multiplier=config.triple_barrier.stop_loss_multiplier,
            vol_estimator_window=config.triple_barrier.vol_estimator_window,
            seed=config.seed + fold,
        ),
        MetaLabelingEligibility(True, ""),
    )
    wrapper.fit_labeled_events(
        features_at_event=x_train,
        labels=y_train,
        n_estimators=config.random_forest_estimators,
    )
    probabilities = wrapper.predict_trade_probability(x_test)
    primary = test["primary_position"].to_numpy().astype(np.float64)
    meta_position = wrapper.filter_positions(
        primary_positions=primary,
        features_at_event=x_test,
        probability_threshold=config.probability_threshold,
    )
    future_return = test["future_return_horizon"].to_numpy().astype(np.float64)
    gross_return = meta_position * future_return
    round_trip_cost = 2.0 * config.cost_bps_one_way * 1e-4
    net_return = gross_return - np.where(meta_position != 0.0, round_trip_cost, 0.0)
    return test.select(["date", "symbol", "primary_position", "future_return_horizon", "close"]).with_columns(
        [
            pl.col("close").alias("entry_close_proxy"),
            pl.lit(fold).alias("fold"),
            pl.Series("meta_probability", probabilities),
            pl.Series("meta_position", meta_position),
            pl.Series("gross_return", gross_return),
            pl.Series("net_return", net_return),
        ]
    ).drop("close")


def _fold_metrics(preds: pl.DataFrame, *, fold: int, train: pl.DataFrame, test: pl.DataFrame) -> dict[str, Any]:
    daily = _daily_returns(preds)
    net = daily["daily_net_return"].to_numpy().astype(np.float64)
    gross = daily["daily_gross_return"].to_numpy().astype(np.float64)
    trades = preds.filter(pl.col("meta_position") != 0.0)
    traded_net = trades["net_return"].to_numpy().astype(np.float64) if trades.height else np.array([], dtype=np.float64)
    return {
        "fold": fold,
        "train_start": str(train["date"].min()),
        "train_end": str(train["date"].max()),
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
        "train_events": train.height,
        "test_events": test.height,
        "trade_count": trades.height,
        "gross_total_return": _compound(gross),
        "net_total_return": _compound(net),
        "net_sharpe": _safe_sharpe(net),
        "net_hit_rate": float(np.mean(traded_net > 0.0)) if traded_net.size else 0.0,
    }


def _summary(predictions: pl.DataFrame, fold_metrics: pl.DataFrame, config: MetaLabelWalkForwardConfig) -> dict[str, Any]:
    daily = _daily_returns(predictions)
    net = daily["daily_net_return"].to_numpy().astype(np.float64) if daily.height else np.array([], dtype=np.float64)
    gross = daily["daily_gross_return"].to_numpy().astype(np.float64) if daily.height else np.array([], dtype=np.float64)
    trades = predictions.filter(pl.col("meta_position") != 0.0) if predictions.height else pl.DataFrame()
    return {
        "status": "research_validation_only",
        "promotion_eligible": False,
        "paper_trade_candidate": False,
        "production_candidate": False,
        "reason": "walk-forward supervised training artifact; not a promotion gate pass",
        "fold_count": fold_metrics.height,
        "prediction_rows": predictions.height,
        "trade_count": trades.height,
        "return_semantics": (
            "equal-weight daily compound return over fold predictions; event rows may use overlapping "
            "triple-barrier horizons and are research-validation only"
        ),
        "gross_total_return": _compound(gross),
        "net_total_return": _compound(net),
        "net_sharpe": _safe_sharpe(net),
        "symbols": sorted(predictions["symbol"].unique().to_list()) if predictions.height else [],
        "date_start": str(predictions["date"].min()) if predictions.height else "",
        "date_end": str(predictions["date"].max()) if predictions.height else "",
        "config": asdict(config),
        "disclaimer": _DISCLAIMER,
    }


def train_meta_label_walk_forward(
    *,
    panel: pl.DataFrame,
    config: MetaLabelWalkForwardConfig,
) -> MetaLabelWalkForwardResult:
    events = _feature_frame(panel, config)
    dates = sorted(events["date"].unique().to_list())
    preds: list[pl.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    fold = 0
    start = config.train_window_days
    while start < len(dates):
        train_start_idx = max(0, start - config.train_window_days)
        train_end_idx = start - config.purge_days - 1
        test_start_idx = start
        test_end_idx = min(len(dates) - 1, start + config.test_window_days - 1)
        if train_end_idx < train_start_idx or test_start_idx > test_end_idx:
            start += config.step_days
            continue
        train = events.filter(
            (pl.col("date") >= dates[train_start_idx]) & (pl.col("date") <= dates[train_end_idx])
        )
        test = events.filter(
            (pl.col("date") >= dates[test_start_idx]) & (pl.col("date") <= dates[test_end_idx])
        )
        labels = train["triple_barrier_label"].to_numpy().astype(np.float64)
        if train.height < config.min_train_events or len(set(labels.astype(int).tolist())) < 2 or test.is_empty():
            start += config.step_days
            continue
        fold_pred = _predict_fold(train=train, test=test, fold=fold, config=config)
        preds.append(fold_pred)
        metrics.append(_fold_metrics(fold_pred, fold=fold, train=train, test=test))
        fold += 1
        start += config.step_days
    predictions = pl.concat(preds, how="vertical") if preds else pl.DataFrame()
    fold_metrics = pl.DataFrame(metrics) if metrics else pl.DataFrame()
    return MetaLabelWalkForwardResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        summary=_summary(predictions, fold_metrics, config),
    )


def write_meta_label_walk_forward_artifacts(
    result: MetaLabelWalkForwardResult,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    fold_metrics_path = output_dir / "fold_metrics.parquet"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    result.predictions.write_parquet(predictions_path)
    result.fold_metrics.write_parquet(fold_metrics_path)
    summary_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True, default=str) + "\n")
    report_path.write_text(
        "\n".join(
            [
                "# Triple-Barrier Meta-Label Walk-Forward Training",
                "",
                f"- status: `{result.summary['status']}`",
                f"- folds: `{result.summary['fold_count']}`",
                f"- prediction rows: `{result.summary['prediction_rows']}`",
                f"- trades: `{result.summary['trade_count']}`",
                f"- return semantics: {result.summary['return_semantics']}",
                f"- net total return: `{result.summary['net_total_return']:.6g}`",
                f"- net Sharpe: `{result.summary['net_sharpe']:.6g}`",
                "",
                "## Disclaimer",
                _DISCLAIMER,
                "",
            ]
        )
    )
    return {
        "predictions": predictions_path,
        "fold_metrics": fold_metrics_path,
        "summary": summary_path,
        "report": report_path,
    }
