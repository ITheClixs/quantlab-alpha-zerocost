from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.validation.daily_report import PerSignalRow
from quant_research_stack.validation.pnl import (
    compute_daily_pnl_metrics,
    load_historical_daily_pnl_pct,
)


def _row(
    *,
    signal_id: str,
    predicted_direction: int,
    fill_price: float,
    weight: float,
    realized_return: float,
    fee: float = 0.0,
) -> PerSignalRow:
    return PerSignalRow(
        signal_id=signal_id,
        symbol="AAPL",
        predicted_score=0.01 * predicted_direction,
        confidence=0.75,
        predicted_direction=predicted_direction,
        s2_decision="pass",
        fill_price=fill_price,
        horizon_minutes=5,
        realized_return=realized_return,
        realized_direction=1 if realized_return > 0 else -1,
        hit=predicted_direction == (1 if realized_return > 0 else -1),
        weight=weight,
        fill_ts_utc=datetime(2026, 5, 20, 13, 35, tzinfo=UTC),
        fee=fee,
    )


def test_compute_daily_pnl_metrics_uses_direction_notional_and_fee() -> None:
    rows = [
        _row(signal_id="long", predicted_direction=1, fill_price=100.0, weight=2.0, realized_return=0.01, fee=1.5),
        _row(signal_id="short", predicted_direction=-1, fill_price=50.0, weight=4.0, realized_return=-0.02, fee=0.5),
    ]

    metrics = compute_daily_pnl_metrics(rows, starting_equity=1000.0, historical_daily_pnl_pct=[0.001, 0.002])

    assert metrics.daily_pnl == pytest.approx(4.0)
    assert metrics.daily_pnl_pct == pytest.approx(0.004)
    assert metrics.daily_dd_pct == 0.0
    assert metrics.sharpe_rolling > 0.0


def test_compute_daily_pnl_metrics_reports_drawdown_for_negative_day() -> None:
    rows = [
        _row(signal_id="loss", predicted_direction=1, fill_price=100.0, weight=10.0, realized_return=-0.02, fee=5.0),
    ]

    metrics = compute_daily_pnl_metrics(rows, starting_equity=1000.0)

    assert metrics.daily_pnl == pytest.approx(-25.0)
    assert metrics.daily_pnl_pct == pytest.approx(-0.025)
    assert metrics.daily_dd_pct == pytest.approx(0.025)
    assert metrics.sharpe_rolling == 0.0


def test_load_historical_daily_pnl_pct_reads_recent_validation_parquets(tmp_path: Path) -> None:
    recent = pl.DataFrame({
        "signal_id": ["sig-recent"],
        "predicted_dir": [1],
        "fill_price": [100.0],
        "realized_return": [0.01],
        "weight": [2.0],
        "fee": [1.0],
    })
    old = pl.DataFrame({
        "signal_id": ["sig-old"],
        "predicted_dir": [1],
        "fill_price": [100.0],
        "realized_return": [0.50],
        "weight": [100.0],
        "fee": [0.0],
    })
    recent.write_parquet(tmp_path / "2026-05-19.parquet")
    old.write_parquet(tmp_path / "2026-05-01.parquet")

    history = load_historical_daily_pnl_pct(
        tmp_path,
        before_date="2026-05-20",
        window_days=7,
        starting_equity=1000.0,
    )

    assert history == pytest.approx([0.001])
