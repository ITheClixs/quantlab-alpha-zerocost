from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.backtest.equity_dataset_loop import (
    EquityDatasetCandidate,
    EquityDatasetLoopConfig,
    build_monthly_return_table,
    filter_normalized_equity_frame,
    probe_candidate,
    run_dataset_candidate,
)


def _daily_curve() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-01"],
            "net_return": [0.10, -0.05, 0.02],
            "gross_return": [0.11, -0.04, 0.03],
            "cost_return": [0.01, 0.01, 0.01],
            "equity": [110_000.0, 104_500.0, 106_590.0],
            "gross_equity": [111_000.0, 106_560.0, 109_756.8],
        }
    )


def _raw_ohlcv(days: int = 12) -> pl.DataFrame:
    rows = []
    for symbol, base, drift in [("AAA", 10.0, 0.10), ("BBB", 20.0, -0.07), ("CCC", 30.0, 0.05)]:
        for idx in range(days):
            close = base + drift * idx + 0.02 * (idx % 2)
            rows.append(
                {
                    "date": f"2024-01-{idx + 2:02d}",
                    "symbol": symbol,
                    "open": close - 0.03,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1000.0 + idx,
                }
            )
    return pl.DataFrame(rows)


def test_build_monthly_return_table_reports_compounded_net_income() -> None:
    monthly = build_monthly_return_table(_daily_curve())

    assert monthly["month"].to_list() == ["2024-01", "2024-02"]
    jan = monthly.row(0, named=True)
    assert jan["n_days"] == 2
    assert jan["net_return"] == pytest.approx(0.045)
    assert jan["net_income"] == pytest.approx(4_500.0)
    feb = monthly.row(1, named=True)
    assert feb["net_income"] == pytest.approx(2_090.0)


def test_probe_candidate_rejects_missing_ohlcv_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pl.DataFrame({"date": ["2024-01-02"], "symbol": ["AAA"], "open": [1.0], "close": [1.1]}).write_csv(path)
    candidate = EquityDatasetCandidate(
        name="bad",
        label="Bad",
        repo_id="local/bad",
        paths=(path,),
        date_column="date",
        symbol_column="symbol",
    )

    probe = probe_candidate(candidate)

    assert probe.usable is False
    assert "high" in probe.reason
    assert "volume" in probe.reason


def test_filter_normalized_equity_frame_removes_low_quality_rows() -> None:
    frame = pl.DataFrame(
        {
            "close": [0.5, 10.0, 12.0, 20.0],
            "dollar_volume": [10_000_000.0, 50_000.0, 5_000_000.0, 6_000_000.0],
            "future_return_1": [0.01, 0.02, 0.50, -0.03],
        }
    )

    filtered = filter_normalized_equity_frame(
        frame,
        target_column="future_return_1",
        min_close=5.0,
        min_dollar_volume=1_000_000.0,
        max_abs_future_return=0.25,
    )

    assert filtered.height == 1
    assert filtered["close"].to_list() == [20.0]


def test_run_dataset_candidate_selects_best_model_and_monthly_metrics(tmp_path: Path) -> None:
    raw_path = tmp_path / "ohlcv.csv"
    _raw_ohlcv().write_csv(raw_path)
    candidate = EquityDatasetCandidate(
        name="unit",
        label="Unit",
        repo_id="local/unit",
        paths=(raw_path,),
        date_column="date",
        symbol_column="symbol",
    )
    config = EquityDatasetLoopConfig(
        min_train_dates=4,
        test_window_dates=2,
        step_dates=2,
        max_folds=2,
        max_train_rows_per_fold=None,
        hist_gradient_max_iter=5,
        selection_fraction=0.5,
        cost_bps=0.0,
    )

    result = run_dataset_candidate(candidate, config, output_dir=tmp_path / "out")

    assert result.status == "ok"
    assert result.best_model in {"ridge", "hist_gradient", "ensemble_mean"}
    assert result.monthly_metrics[result.best_model]["months"] >= 1
    assert result.backtest_metrics[result.best_model]["rebalance_every_n_days"] == 1
    assert (tmp_path / "out" / "models" / f"{result.best_model}.joblib").exists()
