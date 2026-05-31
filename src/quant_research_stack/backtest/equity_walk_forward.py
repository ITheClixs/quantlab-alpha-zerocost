from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_research_stack.backtest.equity_signal import (
    evaluate_signal_accuracy,
    run_long_short_signal_backtest,
)

DEFAULT_EQUITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1",
    "gap_return_1",
    "open_close_return",
    "log_return_1",
    "high_low_range",
    "dollar_volume",
    "volume_change_1",
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "return_5_mean",
    "return_20_mean",
    "return_60_mean",
)

MODEL_NAMES: tuple[str, ...] = ("ridge", "hist_gradient", "ensemble_mean")


@dataclass(frozen=True)
class EquityWalkForwardConfig:
    target_column: str = "future_return_1"
    date_column: str = "date"
    symbol_column: str = "symbol"
    min_train_dates: int = 756
    test_window_dates: int = 126
    step_dates: int = 126
    max_folds: int | None = 6
    max_train_rows_per_fold: int | None = 500_000
    ridge_alpha: float = 10.0
    hist_gradient_max_iter: int = 80
    selection_fraction: float = 0.10
    cost_bps: float = 5.0
    starting_equity: float = 100_000.0
    max_symbols_per_side: int | None = None


@dataclass(frozen=True)
class EquityFoldSpec:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_dates: int
    test_dates: int


@dataclass(frozen=True)
class EquityWalkForwardResult:
    feature_columns: list[str]
    predictions: pl.DataFrame
    fold_metrics: list[dict[str, Any]]
    model_metrics: dict[str, dict[str, float | int]]
    backtest_metrics: dict[str, dict[str, float | int]]
    fold_specs: list[EquityFoldSpec]


@dataclass
class AverageEnsembleRegressor:
    models: list[Any]

    def predict(self, x: NDArray[np.float32] | NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.models:
            raise ValueError("AverageEnsembleRegressor requires at least one model")
        preds = [np.asarray(model.predict(x), dtype=np.float64).reshape(-1) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)


def equity_feature_columns(
    frame: pl.DataFrame,
    *,
    target_column: str = "future_return_1",
    feature_columns: list[str] | None = None,
) -> list[str]:
    if feature_columns is not None:
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing configured feature columns: {sorted(missing)}")
        return list(feature_columns)
    cols = [col for col in DEFAULT_EQUITY_FEATURE_COLUMNS if col in frame.columns and col != target_column]
    if not cols:
        raise ValueError("no default equity OHLCV feature columns are available")
    return cols


def walk_forward_date_splits(
    frame: pl.DataFrame,
    *,
    date_column: str = "date",
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
    max_folds: int | None = None,
) -> list[EquityFoldSpec]:
    if min_train_dates < 1:
        raise ValueError("min_train_dates must be positive")
    if test_window_dates < 1:
        raise ValueError("test_window_dates must be positive")
    if step_dates < 1:
        raise ValueError("step_dates must be positive")
    if date_column not in frame.columns:
        raise ValueError(f"missing date column: {date_column}")

    dates = [str(value) for value in frame.select(date_column).unique().sort(date_column)[date_column].to_list()]
    if len(dates) <= min_train_dates:
        raise ValueError(
            f"not enough dates for walk-forward split: dates={len(dates)} min_train_dates={min_train_dates}"
        )

    specs: list[EquityFoldSpec] = []
    train_end_idx = min_train_dates
    fold = 0
    while train_end_idx < len(dates):
        test_end_idx = min(train_end_idx + test_window_dates, len(dates))
        if test_end_idx <= train_end_idx:
            break
        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        specs.append(
            EquityFoldSpec(
                fold=fold,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_dates=len(train_dates),
                test_dates=len(test_dates),
            )
        )
        fold += 1
        if test_end_idx >= len(dates):
            break
        train_end_idx += step_dates

    if max_folds is not None and max_folds > 0 and len(specs) > max_folds:
        selected = specs[-max_folds:]
        return [
            EquityFoldSpec(
                fold=i,
                train_start=spec.train_start,
                train_end=spec.train_end,
                test_start=spec.test_start,
                test_end=spec.test_end,
                train_dates=spec.train_dates,
                test_dates=spec.test_dates,
            )
            for i, spec in enumerate(selected)
        ]
    return specs


def _date_bound_filter(column: str, start: str, end: str) -> pl.Expr:
    return (pl.col(column).cast(pl.Utf8) >= start) & (pl.col(column).cast(pl.Utf8) <= end)


def _clean_supervised_frame(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    date_column: str,
    symbol_column: str,
) -> pl.DataFrame:
    required = set(feature_columns) | {target_column, date_column, symbol_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing walk-forward columns: {sorted(missing)}")
    return (
        frame.drop_nulls([target_column])
        .with_columns(
            [
                *[pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in feature_columns],
                pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
            ]
        )
        .filter(pl.col(target_column).is_finite())
        .sort([date_column, symbol_column])
    )


def _tail_training_rows(frame: pl.DataFrame, max_rows: int | None) -> pl.DataFrame:
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


def _make_base_models(config: EquityWalkForwardConfig) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha)),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=config.hist_gradient_max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }


def _fold_frames(
    frame: pl.DataFrame,
    spec: EquityFoldSpec,
    *,
    date_column: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = frame.filter(_date_bound_filter(date_column, spec.train_start, spec.train_end))
    test = frame.filter(_date_bound_filter(date_column, spec.test_start, spec.test_end))
    if train.is_empty() or test.is_empty():
        raise ValueError(f"empty fold partition for fold={spec.fold}")
    return train, test


def run_equity_walk_forward(
    frame: pl.DataFrame,
    config: EquityWalkForwardConfig,
    *,
    feature_columns: list[str] | None = None,
) -> EquityWalkForwardResult:
    features = equity_feature_columns(frame, target_column=config.target_column, feature_columns=feature_columns)
    clean = _clean_supervised_frame(
        frame,
        feature_columns=features,
        target_column=config.target_column,
        date_column=config.date_column,
        symbol_column=config.symbol_column,
    )
    fold_specs = walk_forward_date_splits(
        clean,
        date_column=config.date_column,
        min_train_dates=config.min_train_dates,
        test_window_dates=config.test_window_dates,
        step_dates=config.step_dates,
        max_folds=config.max_folds,
    )
    if not fold_specs:
        raise ValueError("walk-forward split produced no folds")

    prediction_frames: list[pl.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for spec in fold_specs:
        train, test = _fold_frames(clean, spec, date_column=config.date_column)
        train = _tail_training_rows(train, config.max_train_rows_per_fold)
        x_train, y_train = _to_numpy(train, feature_columns=features, target_column=config.target_column)
        x_test, _ = _to_numpy(test, feature_columns=features, target_column=config.target_column)
        models = _make_base_models(config)
        preds: dict[str, NDArray[np.float64]] = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            preds[name] = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
        preds["ensemble_mean"] = np.mean(np.column_stack([preds["ridge"], preds["hist_gradient"]]), axis=1)

        fold_pred = test.select([config.date_column, config.symbol_column, config.target_column]).with_columns(
            [
                pl.lit(spec.fold).alias("fold"),
                *[pl.Series(f"pred_{name}", values) for name, values in preds.items()],
            ]
        )
        prediction_frames.append(fold_pred)
        row: dict[str, Any] = {
            "fold": spec.fold,
            "train_start": spec.train_start,
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "train_rows": train.height,
            "test_rows": test.height,
            "train_dates": spec.train_dates,
            "test_dates": spec.test_dates,
        }
        for name in MODEL_NAMES:
            metrics = evaluate_signal_accuracy(
                fold_pred,
                prediction_column=f"pred_{name}",
                target_column=config.target_column,
                date_column=config.date_column,
                selection_quantile=config.selection_fraction,
            )
            row[f"{name}_rank_ic_mean"] = metrics["rank_ic_mean"]
            row[f"{name}_directional_accuracy"] = metrics["directional_accuracy"]
            row[f"{name}_top_bottom_spread_return"] = metrics["top_bottom_spread_return"]
        fold_metrics.append(row)

    predictions = pl.concat(prediction_frames, how="vertical")
    model_metrics = {
        name: evaluate_signal_accuracy(
            predictions,
            prediction_column=f"pred_{name}",
            target_column=config.target_column,
            date_column=config.date_column,
            selection_quantile=config.selection_fraction,
        )
        for name in MODEL_NAMES
    }
    backtest_metrics = {
        name: run_long_short_signal_backtest(
            predictions,
            prediction_column=f"pred_{name}",
            target_column=config.target_column,
            date_column=config.date_column,
            starting_equity=config.starting_equity,
            selection_fraction=config.selection_fraction,
            cost_bps=config.cost_bps,
            max_symbols_per_side=config.max_symbols_per_side,
        ).metrics
        for name in MODEL_NAMES
    }
    return EquityWalkForwardResult(
        feature_columns=features,
        predictions=predictions,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
        backtest_metrics=backtest_metrics,
        fold_specs=fold_specs,
    )


def train_final_equity_models(
    frame: pl.DataFrame,
    config: EquityWalkForwardConfig,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    features = equity_feature_columns(frame, target_column=config.target_column, feature_columns=feature_columns)
    clean = _clean_supervised_frame(
        frame,
        feature_columns=features,
        target_column=config.target_column,
        date_column=config.date_column,
        symbol_column=config.symbol_column,
    )
    x, y = _to_numpy(clean, feature_columns=features, target_column=config.target_column)
    models = _make_base_models(config)
    for model in models.values():
        model.fit(x, y)
    models["ensemble_mean"] = AverageEnsembleRegressor(models=[models["ridge"], models["hist_gradient"]])
    return models


def save_signal_artifacts(
    *,
    models: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
    output_dir: str | Path,
    metadata: dict[str, Any],
    compression: int = 3,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        path = output_path / f"{name}.joblib"
        joblib.dump(
            {
                "model": models[name],
                "features": list(feature_columns),
                "target": target_column,
                "metadata": {**metadata, "model_name": name},
            },
            path,
            compress=compression,
        )
        paths[name] = str(path)
    return paths
