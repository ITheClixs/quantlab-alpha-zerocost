from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXCLUDED_FEATURE_PREFIXES = ("future_", "pred_")
_EXCLUDED_FEATURE_COLUMNS = {
    "dataset_id",
    "source",
    "event_type",
    "symbol",
    "event_time",
    "received_utc",
    "exchange_event_time",
    "aggressor_side",
    "side",
}


@dataclass(frozen=True)
class PerpWalkForwardConfig:
    target_column: str = "future_mid_return_1"
    symbol_column: str = "symbol"
    event_time_column: str = "event_time"
    min_train_rows: int = 50_000
    test_rows: int = 10_000
    step_rows: int | None = None
    embargo_rows: int = 0
    max_folds: int | None = 4
    max_train_rows_per_fold: int | None = 100_000
    ridge_alpha: float = 1.0
    hist_gradient_max_iter: int = 80


@dataclass(frozen=True)
class PerpWalkForwardResult:
    feature_columns: list[str]
    predictions: pl.DataFrame
    fold_specs: list[dict[str, Any]]
    fold_metrics: list[dict[str, float | int]]
    model_metrics: dict[str, dict[str, float | int]]


def _feature_columns(frame: pl.DataFrame, target_column: str) -> list[str]:
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        if name == target_column or name in _EXCLUDED_FEATURE_COLUMNS:
            continue
        if name.startswith(_EXCLUDED_FEATURE_PREFIXES):
            continue
        if dtype.is_numeric():
            columns.append(name)
    if not columns:
        raise ValueError("no numeric feature columns available for perp walk-forward training")
    return columns


def _xy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = frame.select(feature_columns).to_numpy().astype(np.float64)
    y = frame[target_column].to_numpy().astype(np.float64)
    return x, y


def _safe_corr(values: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    if values.size < 2 or predictions.size < 2:
        return 0.0
    if float(np.std(values)) == 0.0 or float(np.std(predictions)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(values, predictions)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _zero_mean_r2(values: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    denom = float(np.sum(np.square(values)))
    if denom <= 0.0:
        return 0.0
    return float(1.0 - np.sum(np.square(values - predictions)) / denom)


def _directional_accuracy(values: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    mask = (values != 0.0) & (predictions != 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(values[mask]) == np.sign(predictions[mask])))


def _metrics(values: NDArray[np.float64], predictions: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "rows": int(values.size),
        "ic": _safe_corr(values, predictions),
        "zero_mean_r2": _zero_mean_r2(values, predictions),
        "directional_accuracy": _directional_accuracy(values, predictions),
    }


def _fit_predict(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    config: PerpWalkForwardConfig,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    x_train, y_train = _xy(train, feature_columns=feature_columns, target_column=target_column)
    x_test, y_test = _xy(test, feature_columns=feature_columns, target_column=target_column)
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.ridge_alpha),
    )
    hist = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=config.hist_gradient_max_iter,
            random_state=17,
        ),
    )
    ridge.fit(x_train, y_train)
    hist.fit(x_train, y_train)
    pred_ridge = ridge.predict(x_test).astype(np.float64)
    pred_hist = hist.predict(x_test).astype(np.float64)
    return {
        "pred_ridge": pred_ridge,
        "pred_hist_gradient": pred_hist,
        "pred_ensemble_mean": ((pred_ridge + pred_hist) / 2.0).astype(np.float64),
    }, y_test


def train_perp_walk_forward(
    frame: pl.DataFrame,
    *,
    config: PerpWalkForwardConfig,
) -> PerpWalkForwardResult:
    if config.target_column not in frame.columns:
        raise ValueError(f"missing target column: {config.target_column}")
    if config.event_time_column not in frame.columns:
        raise ValueError(f"missing event time column: {config.event_time_column}")
    if config.min_train_rows <= 0 or config.test_rows <= 0:
        raise ValueError("min_train_rows and test_rows must be positive")
    if config.embargo_rows < 0:
        raise ValueError("embargo_rows must be non-negative")

    features = _feature_columns(frame, config.target_column)
    required = [*features, config.target_column, config.symbol_column, config.event_time_column]
    ordered = frame.sort([config.event_time_column, config.symbol_column]).drop_nulls(required)
    step_rows = config.step_rows or config.test_rows
    predictions: list[pl.DataFrame] = []
    fold_specs: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float | int]] = []
    model_metric_rows: dict[str, list[dict[str, float | int]]] = {
        "ridge": [],
        "hist_gradient": [],
        "ensemble_mean": [],
    }

    fold = 0
    start = config.min_train_rows + config.embargo_rows
    while start < ordered.height:
        if config.max_folds is not None and fold >= config.max_folds:
            break
        train_end = start - config.embargo_rows
        train_start = max(0, train_end - (config.max_train_rows_per_fold or train_end))
        test_end = min(ordered.height, start + config.test_rows)
        test = ordered.slice(start, test_end - start)
        if test.is_empty():
            break
        test_start_time = test[config.event_time_column][0]
        train = ordered.slice(train_start, train_end - train_start).filter(pl.col(config.event_time_column) < test_start_time)
        if train.height < config.min_train_rows or test.is_empty():
            break

        pred_map, y_test = _fit_predict(
            train,
            test,
            feature_columns=features,
            target_column=config.target_column,
            config=config,
        )
        prediction_series = [pl.Series(name, values) for name, values in pred_map.items()]
        fold_pred = test.select([config.symbol_column, config.event_time_column, config.target_column]).with_columns(
            [pl.lit(fold).alias("fold"), *prediction_series]
        )
        predictions.append(fold_pred)
        fold_specs.append(
            {
                "fold": fold,
                "train_start_row": train_start,
                "train_end_row": train_end,
                "test_start_row": start,
                "test_end_row": test_end,
                "train_rows": train.height,
                "test_rows": test.height,
                "train_start_time": train[config.event_time_column][0],
                "train_end_time": train[config.event_time_column][-1],
                "test_start_time": test[config.event_time_column][0],
                "test_end_time": test[config.event_time_column][-1],
            }
        )
        for column, model_name in (
            ("pred_ridge", "ridge"),
            ("pred_hist_gradient", "hist_gradient"),
            ("pred_ensemble_mean", "ensemble_mean"),
        ):
            row = _metrics(y_test, pred_map[column])
            row["fold"] = fold
            model_metric_rows[model_name].append(row)
        ensemble_metrics = _metrics(y_test, pred_map["pred_ensemble_mean"])
        ensemble_metrics["fold"] = fold
        fold_metrics.append(ensemble_metrics)
        fold += 1
        start += step_rows

    prediction_frame = pl.concat(predictions, how="vertical") if predictions else pl.DataFrame()
    model_metrics = {
        name: {
            "folds": len(rows),
            "rows": int(sum(int(row["rows"]) for row in rows)),
            "mean_ic": float(np.mean([float(row["ic"]) for row in rows])) if rows else 0.0,
            "mean_zero_mean_r2": float(np.mean([float(row["zero_mean_r2"]) for row in rows])) if rows else 0.0,
            "mean_directional_accuracy": float(np.mean([float(row["directional_accuracy"]) for row in rows]))
            if rows
            else 0.0,
        }
        for name, rows in model_metric_rows.items()
    }
    return PerpWalkForwardResult(
        feature_columns=features,
        predictions=prediction_frame,
        fold_specs=fold_specs,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
    )
