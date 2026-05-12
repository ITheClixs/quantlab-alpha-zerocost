from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.local_training import (
    SignalTrainingTask,
    candidate_feature_columns,
    directional_accuracy,
    load_training_frame,
    time_ordered_split,
    train_task,
    zero_mean_r2,
)


def fixture_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset_id": ["x"] * 8,
            "symbol": ["AAA"] * 8,
            "timestamp": list(range(8)),
            "close": [10.0, 10.1, 10.3, 10.2, 10.5, 10.7, 10.8, 11.0],
            "return_1": [0.0, 0.01, 0.02, -0.01, 0.03, 0.02, 0.01, 0.02],
            "realized_vol_5": [0.1] * 8,
            "future_return_1": [0.01, 0.02, -0.01, 0.03, 0.02, 0.01, 0.02, 0.0],
            "direction_up_1": [1, 1, 0, 1, 1, 1, 1, 0],
        }
    )


def test_zero_mean_r2_and_directional_accuracy() -> None:
    assert zero_mean_r2([1.0, -1.0], [1.0, 0.0]) == pytest.approx(0.5)
    assert directional_accuracy([1.0, -1.0], [0.1, 0.2]) == pytest.approx(0.5)


def test_candidate_feature_columns_excludes_labels_and_identifiers() -> None:
    features = candidate_feature_columns(fixture_frame(), "future_return_1")
    assert "close" in features
    assert "return_1" in features
    assert "future_return_1" not in features
    assert "direction_up_1" not in features
    assert "symbol" not in features


def test_load_training_frame_reads_supervised_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    fixture_frame().write_parquet(path)
    frame, files = load_training_frame(tmp_path, "future_return_1", rows_per_file=4, max_files=10)
    assert frame.height == 4
    assert files == [str(path)]


def test_time_ordered_split_keeps_later_rows_for_validation() -> None:
    train, validation = time_ordered_split(fixture_frame(), 0.25)
    assert train.height == 6
    assert validation.height == 2
    assert train["timestamp"].max() < validation["timestamp"].min()


def test_train_task_writes_model_artifacts(tmp_path: Path) -> None:
    input_root = tmp_path / "processed" / "market"
    input_root.mkdir(parents=True)
    fixture_frame().write_parquet(input_root / "sample.parquet")
    task = SignalTrainingTask(
        name="market",
        input_root=input_root,
        target_column="future_return_1",
        rows_per_file=8,
        max_files=10,
        validation_fraction=0.25,
        ridge_alpha=1.0,
        hist_gradient_max_iter=5,
        output_root=tmp_path / "experiments",
    )
    report = train_task(task)
    assert report["rows"] == 8
    assert (tmp_path / "experiments" / "market" / "ridge.joblib").exists()
    assert (tmp_path / "experiments" / "market" / "hist_gradient.joblib").exists()
