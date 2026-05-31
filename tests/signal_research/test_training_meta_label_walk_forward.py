"""Walk-forward supervised training tests for the triple-barrier meta-labeler."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
    write_meta_label_walk_forward_artifacts,
)


def test_meta_label_walk_forward_trains_without_train_test_overlap(
    synthetic_daily_bars: pl.DataFrame,
) -> None:
    result = train_meta_label_walk_forward(
        panel=synthetic_daily_bars,
        config=MetaLabelWalkForwardConfig(
            lookback_days=10,
            train_window_days=90,
            test_window_days=30,
            step_days=30,
            purge_days=5,
            min_train_events=30,
            random_forest_estimators=25,
            probability_threshold=0.55,
            cost_bps_one_way=1.0,
            seed=11,
        ),
    )

    assert result.fold_metrics.height > 0
    assert result.predictions.height > 0
    assert {
        "fold",
        "date",
        "symbol",
        "primary_position",
        "meta_probability",
        "meta_position",
        "gross_return",
        "net_return",
    }.issubset(set(result.predictions.columns))
    assert result.summary["status"] == "research_validation_only"
    assert result.summary["promotion_eligible"] is False

    for row in result.fold_metrics.iter_rows(named=True):
        assert row["train_end"] < row["test_start"]


def test_meta_label_walk_forward_writes_research_only_artifacts(
    tmp_path: Path,
    synthetic_daily_bars: pl.DataFrame,
) -> None:
    result = train_meta_label_walk_forward(
        panel=synthetic_daily_bars,
        config=MetaLabelWalkForwardConfig(
            lookback_days=10,
            train_window_days=90,
            test_window_days=30,
            step_days=30,
            purge_days=5,
            min_train_events=30,
            random_forest_estimators=10,
            probability_threshold=0.55,
            cost_bps_one_way=1.0,
            seed=13,
        ),
    )

    written = write_meta_label_walk_forward_artifacts(
        result,
        output_dir=tmp_path / "meta_label_run",
    )

    assert written["predictions"].exists()
    assert written["fold_metrics"].exists()
    assert written["summary"].exists()
    report = written["report"].read_text()
    assert "research_validation_only" in report
    assert "not automatically investment advice" in report
