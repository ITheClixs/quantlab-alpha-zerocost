from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.alpha.features import (
    FeatureConfig,
    add_cross_sectional_ranks,
    add_lag_features,
    add_noise_feature,
    add_rolling_features,
    build_feature_frame,
    no_future_leakage,
)


@pytest.fixture
def panel() -> pl.DataFrame:
    return pl.DataFrame({
        "date_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "symbol_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "feature_00": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "responder_6": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.01, 0.04, 0.02, -0.03],
    })


def test_add_lag_features_produces_expected_columns(panel: pl.DataFrame) -> None:
    out = add_lag_features(panel, ["feature_00"], lags=[1, 2], group_col="symbol_id", time_col="date_id")
    assert "feature_00_lag1" in out.columns
    assert "feature_00_lag2" in out.columns


def test_add_lag_features_does_not_use_future_values(panel: pl.DataFrame) -> None:
    out = add_lag_features(panel, ["feature_00"], lags=[1], group_col="symbol_id", time_col="date_id").sort(["symbol_id", "date_id"])
    sym1 = out.filter(pl.col("symbol_id") == 1)
    # for symbol 1, sorted by date: feature_00 = [0.1, 0.3, 0.5, 0.7, 0.9]
    # lag1 must be [None, 0.1, 0.3, 0.5, 0.7]
    assert sym1["feature_00_lag1"].to_list() == [None, 0.1, 0.3, 0.5, 0.7]


def test_add_rolling_features_emits_mean_and_std(panel: pl.DataFrame) -> None:
    out = add_rolling_features(panel, ["feature_00"], windows=[2], group_col="symbol_id", time_col="date_id")
    assert "feature_00_roll2_mean" in out.columns
    assert "feature_00_roll2_std" in out.columns


def test_add_cross_sectional_ranks_per_date(panel: pl.DataFrame) -> None:
    out = add_cross_sectional_ranks(panel, ["feature_00"], date_col="date_id")
    # for date 0, feature_00 = [0.1, 0.2]; ranks [0, 1]
    d0 = out.filter(pl.col("date_id") == 0).sort("symbol_id")
    assert d0["feature_00_rank_xs"].to_list() == [0.0, 1.0]


def test_add_noise_feature_deterministic(panel: pl.DataFrame) -> None:
    out_a = add_noise_feature(panel, seed=123)
    out_b = add_noise_feature(panel, seed=123)
    assert out_a["noise_seed123"].to_list() == out_b["noise_seed123"].to_list()


def test_no_future_leakage_detects_leak() -> None:
    bad = pl.DataFrame({
        "date_id": [0, 1, 2, 3],
        "symbol_id": [1, 1, 1, 1],
        "leaky_feat": [10.0, 20.0, 30.0, 40.0],
        "responder_6": [10.0, 20.0, 30.0, 40.0],
    })
    leaks = no_future_leakage(bad, target_col="responder_6", group_col="symbol_id", time_col="date_id")
    assert "leaky_feat" in leaks


def test_no_future_leakage_passes_clean() -> None:
    clean = pl.DataFrame({
        "date_id": [0, 1, 2, 3],
        "symbol_id": [1, 1, 1, 1],
        "lagged_feat": [None, 10.0, 20.0, 30.0],
        "responder_6": [10.0, 20.0, 30.0, 40.0],
    })
    leaks = no_future_leakage(clean, target_col="responder_6", group_col="symbol_id", time_col="date_id")
    assert leaks == []


def test_build_feature_frame_end_to_end(panel: pl.DataFrame) -> None:
    cfg = FeatureConfig(
        lag_windows=[1],
        rolling_windows=[2],
        include_noise_feature=True,
        cross_sectional_ranks=True,
        noise_seed=42,
    )
    out = build_feature_frame(panel, cfg, base_features=["feature_00"], date_col="date_id", symbol_col="symbol_id")
    expected = {"feature_00_lag1", "feature_00_roll2_mean", "feature_00_roll2_std", "feature_00_rank_xs", "noise_seed42"}
    assert expected.issubset(set(out.columns))
