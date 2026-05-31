from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.backtest.report import BacktestReport
from quant_research_stack.backtest.runner import BacktestResult
from quant_research_stack.brokers.order_types import Fill, OrderSide


def _result() -> BacktestResult:
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="BTCUSDT", side=OrderSide.buy,
             price=100.0, quantity=1.0, timestamp_utc=datetime(2026, 5, 17, tzinfo=UTC)),
    ]
    eq = pl.DataFrame({
        "timestamp_utc": [datetime(2026, 5, 17, 10, i, tzinfo=UTC) for i in range(5)],
        "equity": [100_000.0, 100_010.0, 100_020.0, 99_990.0, 100_050.0],
    })
    return BacktestResult(
        fills=fills, equity_curve=eq,
        metrics={
            "total_return": 0.0005, "max_drawdown": -0.0003, "sharpe_ratio": 1.2,
            "calmar_ratio": 0.5, "hit_rate": 0.5, "turnover": 0.001,
            "value_at_risk_5pct": -0.0002, "n_fills": 1,
        },
    )


def test_writes_all_required_artifacts(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    files = {p.name for p in tmp_path.glob("*")}
    assert "metrics.json" in files
    assert "fills.parquet" in files
    assert "equity_curve.parquet" in files
    assert "report.md" in files
    assert "equity_curve.png" in files
    assert "drawdown.png" in files


def test_metrics_json_round_trips(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["sharpe_ratio"] == 1.2
    assert payload["run_id"] == "20260517-120000"
    assert payload["strategy_name"] == "buy_and_hold"


def test_markdown_includes_metric_table(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    md = (tmp_path / "report.md").read_text()
    assert "sharpe_ratio" in md
    assert "buy_and_hold" in md
    assert "![equity_curve](equity_curve.png)" in md
