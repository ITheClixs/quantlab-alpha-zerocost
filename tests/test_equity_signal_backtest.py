from __future__ import annotations

import sys
from pathlib import Path

import joblib
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from quant_research_stack.backtest.equity_signal import (
    SignalModelArtifact,
    evaluate_signal_accuracy,
    load_signal_model,
    normalize_equity_ohlcv,
    predict_signal_frame,
    run_long_short_signal_backtest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from equity_signal_backtest import (  # noqa: E402
    EquityDatasetSpec,
    _run_dataset,
    _write_markdown_report,
)


def _fixture_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "open": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "high": [10.5, 11.5, 12.5, 20.5, 19.5, 18.5],
            "low": [9.5, 10.5, 11.5, 19.5, 18.5, 17.5],
            "close": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "volume": [1000, 1100, 1200, 2000, 1900, 1800],
        }
    )


def test_normalize_equity_ohlcv_builds_features_and_future_returns_without_lookahead() -> None:
    frame = normalize_equity_ohlcv(
        _fixture_ohlcv(),
        dataset_id="unit",
        date_column="date",
        symbol_column="ticker",
    )

    aaa = frame.filter(pl.col("symbol") == "AAA").sort("timestamp_utc")
    assert aaa["return_1"].to_list()[0] is None
    assert aaa["future_return_1"].to_list()[0] == pytest.approx(0.10)
    assert aaa["future_return_1"].to_list()[-1] is None
    assert "high_low_range" in frame.columns
    assert "realized_vol_5" in frame.columns


def test_evaluate_signal_accuracy_reports_directional_and_rank_quality() -> None:
    frame = pl.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "prediction": [0.40, 0.20, -0.10, -0.30],
            "future_return_1": [0.05, 0.02, -0.01, -0.04],
        }
    )

    metrics = evaluate_signal_accuracy(frame, prediction_column="prediction")

    assert metrics["rows"] == 4
    assert metrics["directional_accuracy"] == pytest.approx(1.0)
    assert metrics["positive_precision"] == pytest.approx(1.0)
    assert metrics["negative_precision"] == pytest.approx(1.0)
    assert metrics["top_bottom_spread_return"] > 0.0
    assert metrics["rank_ic_mean"] > 0.9


def test_long_short_backtest_is_cost_aware() -> None:
    frame = pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "B"],
            "prediction": [0.5, -0.5, 0.4, -0.4],
            "future_return_1": [0.10, -0.08, 0.04, -0.02],
        }
    )

    free = run_long_short_signal_backtest(frame, cost_bps=0.0, selection_fraction=0.5)
    costly = run_long_short_signal_backtest(frame, cost_bps=25.0, selection_fraction=0.5)

    assert free.daily_curve.height == 2
    assert free.metrics["total_return"] > 0.0
    assert costly.metrics["total_return"] < free.metrics["total_return"]
    assert costly.metrics["cost_drag_return"] > 0.0


def test_long_short_backtest_can_rebalance_less_often_to_reduce_turnover() -> None:
    frame = pl.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "symbol": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "prediction": [0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2],
            "future_return_1": [0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.04, -0.02],
        }
    )

    daily = run_long_short_signal_backtest(frame, cost_bps=10.0, selection_fraction=0.5)
    slower = run_long_short_signal_backtest(
        frame,
        cost_bps=10.0,
        selection_fraction=0.5,
        rebalance_every_n_days=2,
    )

    assert daily.metrics["n_rebalances"] == 4
    assert slower.metrics["n_rebalances"] == 2
    assert slower.metrics["avg_daily_turnover"] < daily.metrics["avg_daily_turnover"]
    assert slower.metrics["cost_drag_return"] < daily.metrics["cost_drag_return"]


def test_predict_signal_frame_uses_persisted_feature_order(tmp_path: Path) -> None:
    x = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    y = [0.1, 0.2, 0.3]
    model = LinearRegression().fit(x, y)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "features": ["feature_b", "feature_a"], "target": "future_return_1"}, path)

    artifact = load_signal_model(path)
    frame = pl.DataFrame({"feature_a": [10.0], "feature_b": [1.0]})
    predicted = predict_signal_frame(frame, artifact)

    assert predicted["prediction"].to_list() == pytest.approx([0.1])


def test_cli_dataset_run_uses_model_artifact_target_column(tmp_path: Path) -> None:
    rows = []
    for symbol, base in [("AAA", 10.0), ("BBB", 20.0)]:
        for offset in range(6):
            close = base + offset
            rows.append(
                {
                    "date": f"2024-01-{offset + 2:02d}",
                    "symbol": symbol,
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + offset,
                }
            )
    csv_path = tmp_path / "equities.csv"
    pl.DataFrame(rows).write_csv(csv_path)
    model = LinearRegression().fit([[10.0], [20.0]], [0.1, -0.1])
    artifact = SignalModelArtifact(
        model=model,
        feature_columns=["close"],
        target_column="future_return_5",
        artifact_path=tmp_path / "model.joblib",
    )

    result = _run_dataset(
        spec=EquityDatasetSpec(
            name="unit",
            label="Unit equities",
            paths=(csv_path,),
            date_column="date",
            symbol_column="symbol",
        ),
        model_artifact=artifact,
        run_dir=tmp_path / "run",
        max_rows=None,
        selection_fraction=0.5,
        cost_bps=0.0,
        starting_equity=100_000.0,
        max_symbols_per_side=None,
        save_signals=False,
    )

    assert result["accuracy_metrics"]["rows"] == 2
    assert result["backtest_metrics"]["n_days"] == 1


def test_markdown_report_scope_and_config_follow_selected_datasets(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    _write_markdown_report(
        path=report_path,
        run_id="unit",
        model_artifact="model.joblib",
        model_features=["close"],
        target_column="future_return_1",
        dataset_results=[
            {
                "dataset": "sp500",
                "label": "S&P 500 daily equities",
                "source_path": "sp500.parquet",
                "artifact_dir": "experiments/unit/sp500",
                "normalized_rows": 10,
                "symbols": 2,
                "dates": 5,
                "start_date": "2024-01-02",
                "end_date": "2024-01-08",
                "accuracy_metrics": {
                    "rows": 8,
                    "directional_accuracy": 0.5,
                    "positive_precision": 0.5,
                    "negative_precision": 0.5,
                    "zero_mean_r2": 0.0,
                    "information_coefficient": 0.0,
                    "rank_ic_mean": 0.0,
                    "rank_ic_std": 0.0,
                    "top_mean_forward_return": 0.01,
                    "bottom_mean_forward_return": -0.01,
                    "top_bottom_spread_return": 0.02,
                    "positive_signal_share": 0.5,
                },
                "backtest_metrics": {
                    "n_days": 4,
                    "total_return": 0.01,
                    "gross_total_return": 0.02,
                    "cost_drag_return": 0.01,
                    "annualized_return": 0.5,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.01,
                    "hit_rate": 0.5,
                    "avg_daily_turnover": 2.0,
                    "avg_daily_net_return": 0.002,
                    "avg_daily_gross_return": 0.003,
                },
            }
        ],
        cost_bps=5.0,
        selection_fraction=0.1,
        git_sha="abc123",
        starting_equity=100_000.0,
        max_rows_per_dataset=1000,
        max_symbols_per_side=10,
        save_signals=False,
    )

    text = report_path.read_text()
    assert "S&P 500 daily equities data" in text
    assert "NASDAQ, and NYSE" not in text
    assert "Max rows per dataset: `1000`" in text
    assert "Max symbols per side: `10`" in text
    assert "Save signal parquet: `false`" in text
