from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_research_stack.artifacts import safe_repo_id, write_json

IDENTIFIER_COLUMNS = {
    "dataset_id",
    "source_file",
    "symbol",
    "timestamp",
    "event_time",
    "transaction_time",
    "update_id",
}


@dataclass(frozen=True)
class SignalTrainingTask:
    name: str
    input_root: Path
    target_column: str
    rows_per_file: int
    max_files: int | None
    validation_fraction: float
    ridge_alpha: float
    hist_gradient_max_iter: int
    output_root: Path


def zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.sum(np.square(y_true)))
    if denominator <= 0:
        return 0.0
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denominator)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def is_label_like(column: str, target_column: str) -> bool:
    if column == target_column:
        return True
    prefixes = ("future_", "direction_", "mid_direction_")
    return column.startswith(prefixes)


def candidate_feature_columns(frame: pl.DataFrame, target_column: str) -> list[str]:
    cols = []
    for name, dtype in frame.schema.items():
        if name in IDENTIFIER_COLUMNS or name.startswith("__") or is_label_like(name, target_column):
            continue
        if dtype.is_numeric():
            cols.append(name)
    return sorted(cols)


def read_supervised_file(path: Path, target_column: str, rows_per_file: int) -> pl.DataFrame | None:
    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception:
        return None
    if target_column not in schema.names():
        return None
    try:
        frame = (
            pl.scan_parquet(path)
            .with_row_index("__row_index")
            .head(rows_per_file)
            .with_columns(pl.lit(str(path)).alias("__source_file"))
            .collect()
            .drop_nulls([target_column])
            .filter(pl.col(target_column).is_finite())
        )
    except Exception:
        return None
    return frame if not frame.is_empty() else None


def load_training_frame(input_root: Path, target_column: str, rows_per_file: int, max_files: int | None) -> tuple[pl.DataFrame, list[str]]:
    files = sorted(input_root.rglob("*.parquet"))
    frames: list[pl.DataFrame] = []
    used_files: list[str] = []
    for file_path in files:
        if max_files is not None and len(frames) >= max_files:
            break
        frame = read_supervised_file(file_path, target_column, rows_per_file)
        if frame is None:
            continue
        frames.append(frame)
        used_files.append(str(file_path))
    if not frames:
        raise FileNotFoundError(f"No supervised parquet files with target {target_column!r} under {input_root}")
    combined = pl.concat(frames, how="diagonal_relaxed").sort(["__source_file", "__row_index"])
    return combined, used_files


def time_ordered_split(frame: pl.DataFrame, validation_fraction: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    split_at = max(1, int(math.floor(frame.height * (1.0 - validation_fraction))))
    if split_at >= frame.height:
        split_at = frame.height - 1
    train = frame.slice(0, split_at)
    validation = frame.slice(split_at)
    if train.is_empty() or validation.is_empty():
        raise ValueError("training split produced an empty partition")
    return train, validation


def to_numpy(frame: pl.DataFrame, features: list[str], target_column: str) -> tuple[np.ndarray, np.ndarray]:
    x = frame.select(features).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
    y = frame[target_column].to_numpy().astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


def train_task(task: SignalTrainingTask) -> dict[str, Any]:
    frame, used_files = load_training_frame(task.input_root, task.target_column, task.rows_per_file, task.max_files)
    features = candidate_feature_columns(frame, task.target_column)
    if not features:
        raise ValueError(f"No numeric feature columns found for {task.name}")

    train, validation = time_ordered_split(frame, task.validation_fraction)
    x_train, y_train = to_numpy(train, features, task.target_column)
    x_valid, y_valid = to_numpy(validation, features, task.target_column)
    zero_pred = np.zeros_like(y_valid)

    task_output = task.output_root / safe_repo_id(task.name)
    task_output.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=task.ridge_alpha)),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=task.hist_gradient_max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }
    model_results = {
        "zero": {
            "zero_mean_r2": zero_mean_r2(y_valid, zero_pred),
            "directional_accuracy": directional_accuracy(y_valid, zero_pred),
        }
    }
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        model_path = task_output / f"{model_name}.joblib"
        joblib.dump({"model": model, "features": features, "target": task.target_column}, model_path)
        model_results[model_name] = {
            "zero_mean_r2": zero_mean_r2(y_valid, pred),
            "directional_accuracy": directional_accuracy(y_valid, pred),
            "artifact": str(model_path),
        }

    best_model = max(model_results, key=lambda key: model_results[key]["zero_mean_r2"])
    report = {
        "task": task.name,
        "input_root": str(task.input_root),
        "target_column": task.target_column,
        "rows": frame.height,
        "train_rows": train.height,
        "validation_rows": validation.height,
        "source_file_count": len(used_files),
        "source_files_sample": used_files[:20],
        "feature_count": len(features),
        "features": features,
        "metrics": model_results,
        "best_model": best_model,
        "limitations": [
            "Local sampled benchmark; not a live trading result.",
            "Validation is time-ordered over sampled files, not a Kaggle public leaderboard score.",
            "Large GGUF LLM is not fine-tuned; these are locally trained signal heads for the wrapped quant system.",
        ],
    }
    write_json(task_output / "training_report.json", report)
    return report


def write_training_summary(reports: list[dict[str, Any]], output_path: str | Path) -> None:
    write_json(output_path, {"tasks": reports})
