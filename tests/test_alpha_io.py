from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha.io import (
    LoadConfig,
    load_jane_street,
    permanent_holdout_split,
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
