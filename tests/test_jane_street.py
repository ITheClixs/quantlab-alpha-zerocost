from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_research_stack.jane_street import (
    feature_columns,
    find_train_parquet_paths,
    run_local_baseline,
    time_ordered_train_validation_split,
    validate_jane_street_frame,
    weighted_zero_mean_r2,
)


def fixture_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1, 2, 2],
            "time_id": [0, 1, 0, 1, 0, 1],
            "weight": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
            "feature_00": [0.1, 0.2, 0.2, 0.4, 0.5, 0.7],
            "feature_01": [1.0, 0.9, 0.7, 0.5, 0.3, 0.2],
            "responder_6": [0.01, -0.02, 0.03, -0.01, 0.04, 0.02],
        }
    )


def test_weighted_zero_mean_r2_matches_hand_calculation() -> None:
    y = np.array([1.0, 2.0, -1.0])
    pred = np.array([0.5, 1.0, 0.0])
    weights = np.array([1.0, 2.0, 1.0])
    expected = 1.0 - (1.0 * 0.25 + 2.0 * 1.0 + 1.0 * 1.0) / (1.0 * 1.0 + 2.0 * 4.0 + 1.0 * 1.0)
    assert weighted_zero_mean_r2(y, pred, weights) == pytest.approx(expected)


def test_feature_columns_are_sorted() -> None:
    assert feature_columns(fixture_frame()) == ["feature_00", "feature_01"]


def test_validate_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="responder_6"):
        validate_jane_street_frame(fixture_frame().drop("responder_6"))


def test_time_split_uses_later_dates_for_validation() -> None:
    train, validation = time_ordered_train_validation_split(fixture_frame(), validation_fraction=0.34)
    assert train["date_id"].max() < validation["date_id"].min()
    assert set(validation["date_id"].to_list()) == {2}


def test_run_local_baseline_writes_fixture_score(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    fixture_frame().write_parquet(path)
    result = run_local_baseline(path, output_root=tmp_path / "model")
    assert result.rows == 6
    assert result.feature_count == 2
    assert result.metric == "weighted_zero_mean_r2"
    assert result.ridge_r2 is not None
    assert result.ridge_artifact is not None
    assert Path(result.ridge_artifact).exists()


def test_find_train_parquet_paths_ignores_hidden_files_in_partitioned_dir(tmp_path: Path) -> None:
    train_dir = tmp_path / "train.parquet" / "partition_id=0"
    train_dir.mkdir(parents=True)
    fixture_frame().write_parquet(train_dir / "part-0.parquet")
    (tmp_path / "train.parquet" / ".DS_Store").write_text("not parquet", encoding="utf-8")
    paths = find_train_parquet_paths(tmp_path)
    assert paths == [train_dir / "part-0.parquet"]
