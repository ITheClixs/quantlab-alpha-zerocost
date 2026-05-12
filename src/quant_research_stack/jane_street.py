from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_research_stack.artifacts import write_json

TARGET_COLUMN = "responder_6"
WEIGHT_COLUMN = "weight"
DATE_COLUMN = "date_id"


@dataclass(frozen=True)
class BenchmarkResult:
    dataset_path: str
    rows: int
    train_rows: int
    validation_rows: int
    feature_count: int
    target: str
    metric: str
    zero_baseline_r2: float
    ridge_r2: float | None
    ridge_artifact: str | None
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "rows": self.rows,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_count": self.feature_count,
            "target": self.target,
            "metric": self.metric,
            "zero_baseline_r2": self.zero_baseline_r2,
            "ridge_r2": self.ridge_r2,
            "ridge_artifact": self.ridge_artifact,
            "limitations": list(self.limitations),
        }


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape != weights.shape:
        raise ValueError("y_true, y_pred, and weights must have matching shapes")
    numerator = np.sum(weights * np.square(y_true - y_pred))
    denominator = np.sum(weights * np.square(y_true))
    if denominator <= 0:
        return 0.0
    return float(1.0 - numerator / denominator)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("feature_"))


def validate_jane_street_frame(frame: pl.DataFrame) -> None:
    required = {DATE_COLUMN, WEIGHT_COLUMN, TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Jane Street frame is missing required columns: {missing}")
    if not feature_columns(frame):
        raise ValueError("Jane Street frame has no feature_* columns")


def time_ordered_train_validation_split(frame: pl.DataFrame, validation_fraction: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    dates = frame.select(DATE_COLUMN).unique().sort(DATE_COLUMN)[DATE_COLUMN].to_list()
    if len(dates) < 2:
        raise ValueError("Need at least two distinct date_id values for time-ordered validation")
    validation_date_count = max(1, int(round(len(dates) * validation_fraction)))
    validation_start = dates[-validation_date_count]
    train = frame.filter(pl.col(DATE_COLUMN) < validation_start)
    validation = frame.filter(pl.col(DATE_COLUMN) >= validation_start)
    if train.is_empty() or validation.is_empty():
        raise ValueError("Time split produced an empty train or validation frame")
    return train, validation


def find_train_parquet_paths(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Jane Street input root does not exist: {root}")
    train_dir = root / "train.parquet"
    if train_dir.is_dir():
        parts = sorted(path for path in train_dir.rglob("*.parquet") if path.is_file() and not path.name.startswith("."))
        if parts:
            return parts
    preferred = sorted(path for path in root.rglob("*.parquet") if path.is_file() and "train" in path.as_posix().lower())
    if preferred:
        return preferred
    parquet_files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return [parquet_files[0]]


def load_train_frame(input_root: str | Path, sample_rows: int | None = None) -> tuple[Path, pl.DataFrame]:
    train_paths = find_train_parquet_paths(input_root)
    scan = pl.scan_parquet([str(path) for path in train_paths])
    if sample_rows is not None:
        scan = scan.head(sample_rows)
    frame = scan.collect()
    validate_jane_street_frame(frame)
    return train_paths[0], frame


def _frame_to_numpy(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return frame.select(cols).fill_null(0.0).to_numpy().astype(np.float32)


def run_local_baseline(
    input_root: str | Path,
    *,
    sample_rows: int | None = None,
    validation_fraction: float = 0.2,
    output_root: str | Path | None = None,
) -> BenchmarkResult:
    train_path, frame = load_train_frame(input_root, sample_rows=sample_rows)
    cols = feature_columns(frame)
    train, validation = time_ordered_train_validation_split(frame, validation_fraction=validation_fraction)

    y_valid = validation[TARGET_COLUMN].to_numpy()
    weights = validation[WEIGHT_COLUMN].to_numpy()
    zero_pred = np.zeros_like(y_valid, dtype=np.float64)
    zero_score = weighted_zero_mean_r2(y_valid, zero_pred, weights)

    ridge_score: float | None
    ridge_artifact: str | None = None
    try:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
        model.fit(_frame_to_numpy(train, cols), train[TARGET_COLUMN].to_numpy())
        ridge_pred = model.predict(_frame_to_numpy(validation, cols))
        ridge_score = weighted_zero_mean_r2(y_valid, ridge_pred, weights)
        if output_root is not None:
            model_dir = Path(output_root)
            model_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = model_dir / "jane_street_ridge.joblib"
            joblib.dump({"model": model, "features": cols, "target": TARGET_COLUMN}, artifact_path)
            ridge_artifact = str(artifact_path)
    except Exception:
        ridge_score = None

    return BenchmarkResult(
        dataset_path=str(train_path),
        rows=frame.height,
        train_rows=train.height,
        validation_rows=validation.height,
        feature_count=len(cols),
        target=TARGET_COLUMN,
        metric="weighted_zero_mean_r2",
        zero_baseline_r2=zero_score,
        ridge_r2=ridge_score,
        ridge_artifact=ridge_artifact,
        limitations=(
            "Local validation only; not a Kaggle leaderboard score.",
            "Ridge baseline is a sanity check, not a tuned competition model.",
        ),
    )


def write_benchmark_report(result: BenchmarkResult, report_path: str | Path) -> None:
    write_json(report_path, result.as_dict())
