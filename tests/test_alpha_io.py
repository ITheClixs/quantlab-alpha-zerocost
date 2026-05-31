from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha.io import (
    LoadConfig,
    load_jane_street,
    permanent_holdout_split,
    scan_jane_street,
    select_tail_by_row_budget,
)


@pytest.fixture
def fake_js(tmp_path: Path) -> Path:
    df = pl.DataFrame({
        "date_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "symbol_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "feature_00": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "responder_6": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.01, 0.04, 0.02, -0.03],
        "weight": [1.0] * 10,
    })
    path = tmp_path / "fake.parquet"
    df.write_parquet(path)
    return path


def test_load_jane_street_reads_parquet(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    df = load_jane_street(fake_js, cfg)
    assert df.height == 10
    assert "responder_6" in df.columns
    assert "weight" in df.columns


def test_load_jane_street_root_prefers_train_parquet(tmp_path: Path) -> None:
    root = tmp_path / "js"
    train_dir = root / "train.parquet"
    test_dir = root / "test.parquet"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    pl.DataFrame({
        "date_id": [0, 1],
        "symbol_id": [1, 1],
        "feature_09": [1, 2],
        "responder_6": [0.1, -0.2],
        "weight": [1.0, 1.0],
    }).write_parquet(train_dir / "part-0.parquet")
    pl.DataFrame({
        "date_id": [0],
        "feature_09": [1.5],
        "weight": [1.0],
    }).write_parquet(test_dir / "part-0.parquet")

    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    df = load_jane_street(root, cfg)

    assert df.height == 2
    assert "responder_6" in df.columns


def test_permanent_holdout_split_by_date_id(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id", holdout_fraction=0.4)
    df = load_jane_street(fake_js, cfg)
    train, holdout = permanent_holdout_split(df, cfg)
    train_dates = set(train["date_id"].to_list())
    holdout_dates = set(holdout["date_id"].to_list())
    assert train_dates.isdisjoint(holdout_dates), "holdout dates leaked into train"
    # 5 unique dates; 40% holdout -> last 2 dates (3, 4)
    assert holdout_dates == {3, 4}


def test_permanent_holdout_split_chronological(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id", holdout_fraction=0.4)
    df = load_jane_street(fake_js, cfg)
    train, holdout = permanent_holdout_split(df, cfg)
    assert max(train["date_id"]) < min(holdout["date_id"]), "holdout must come AFTER train"


def test_scan_jane_street_returns_lazyframe(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(fake_js, cfg)
    assert isinstance(lf, pl.LazyFrame)
    schema_names = set(lf.collect_schema().names())
    assert {"responder_6", "weight", "date_id"} <= schema_names


def test_scan_jane_street_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"date_id": [0, 1], "feature_00": [1.0, 2.0]}).write_parquet(path)
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    with pytest.raises(ValueError, match="missing columns"):
        scan_jane_street(path, cfg)


def test_scan_jane_street_missing_path_raises(tmp_path: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    with pytest.raises(FileNotFoundError):
        scan_jane_street(tmp_path / "does_not_exist.parquet", cfg)


def test_select_tail_by_row_budget_picks_recent_groups(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(fake_js, cfg)
    df = select_tail_by_row_budget(lf, "date_id", max_rows=5)
    # 5 dates × 2 rows = 10 total. Budget 5 ⇒ 2 full dates (4 rows) + 1 overshoot (6 rows total)
    # Selection MUST be the most-recent contiguous dates.
    dates = sorted(set(df["date_id"].to_list()))
    assert dates == [2, 3, 4], dates
    assert df.height == 6


def test_select_tail_by_row_budget_includes_single_oversized_group(tmp_path: Path) -> None:
    path = tmp_path / "uneven.parquet"
    pl.DataFrame({
        "date_id": [0] * 100 + [1] * 100 + [2] * 100,
        "symbol_id": list(range(300)),
        "feature_00": [0.1] * 300,
        "responder_6": [0.0] * 300,
        "weight": [1.0] * 300,
    }).write_parquet(path)
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(path, cfg)
    df = select_tail_by_row_budget(lf, "date_id", max_rows=50)
    assert sorted(set(df["date_id"].to_list())) == [2]
    assert df.height == 100  # single most-recent group, even though it exceeds budget


def test_select_tail_by_row_budget_returns_all_when_budget_exceeds_total(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(fake_js, cfg)
    df = select_tail_by_row_budget(lf, "date_id", max_rows=10_000)
    assert df.height == 10
    assert sorted(set(df["date_id"].to_list())) == [0, 1, 2, 3, 4]


def test_select_tail_by_row_budget_zero_raises(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(fake_js, cfg)
    with pytest.raises(ValueError, match="max_rows must be positive"):
        select_tail_by_row_budget(lf, "date_id", max_rows=0)


def test_select_tail_by_row_budget_returns_sorted_ascending(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    lf = scan_jane_street(fake_js, cfg)
    df = select_tail_by_row_budget(lf, "date_id", max_rows=5)
    dates = df["date_id"].to_list()
    assert dates == sorted(dates), "select_tail_by_row_budget must return chronologically-sorted rows"
