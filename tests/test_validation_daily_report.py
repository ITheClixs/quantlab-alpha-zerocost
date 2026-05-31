from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from quant_research_stack.validation.daily_report import (
    DailyReportInputs,
    PerSignalRow,
    build_per_signal_table,
    render_markdown,
)
from quant_research_stack.validation.hit_rate import HitRateResult
from quant_research_stack.validation.reconcile import ReconcileSummary


def _row(
    signal_id: str = "sig-1",
    symbol: str = "AAPL",
    predicted_score: float = 0.05,
    confidence: float = 0.7,
    predicted_dir: int = 1,
    s2_decision: str = "pass",
    fill_price: float | None = 100.0,
    horizon_minutes: int = 5,
    realized_return: float = 0.005,
    realized_dir: int = 1,
    hit: bool | None = True,
    weight: float = 1.0,
    fill_ts: datetime | None = None,
    fee: float = 0.0,
) -> PerSignalRow:
    return PerSignalRow(
        signal_id=signal_id, symbol=symbol, predicted_score=predicted_score,
        confidence=confidence, predicted_direction=predicted_dir, s2_decision=s2_decision,
        fill_price=fill_price, horizon_minutes=horizon_minutes,
        realized_return=realized_return, realized_direction=realized_dir, hit=hit,
        weight=weight, fill_ts_utc=fill_ts or datetime(2026, 5, 20, 13, 35, tzinfo=UTC),
        fee=fee,
    )


def _inputs() -> DailyReportInputs:
    rows = [
        _row(signal_id="sig-1", predicted_dir=1, realized_dir=1, hit=True),
        _row(signal_id="sig-2", predicted_dir=1, realized_dir=-1, hit=False, realized_return=-0.01),
        _row(signal_id="sig-3", predicted_dir=0, s2_decision="veto", fill_price=None,
             hit=None, realized_return=math.nan, realized_dir=0),
    ]
    return DailyReportInputs(
        date_str="2026-05-20",
        stage="paper",
        broker_name="alpaca_paper",
        rows=rows,
        hit_rate=HitRateResult(hit_rate=0.5, n_signals=2, n_hits=1, governor_block_rate=1 / 3),
        reconcile=ReconcileSummary(
            book_equity=Decimal("100000"), broker_equity=Decimal("100000"),
            diff_bps=0.0, flagged=False,
        ),
        daily_pnl_pct=0.42,
        daily_dd_pct=0.31,
        sharpe_rolling=1.18,
        days_in_paper=18,
        min_trading_days=30,
        thresholds={
            "hit_rate_min": 0.53,
            "sharpe_min": 1.0,
            "max_daily_dd_pct": 0.05,
            "governor_block_rate_max": 0.50,
        },
    )


def test_render_markdown_contains_required_sections() -> None:
    md = render_markdown(_inputs())
    assert "QuantLab paper validation — 2026-05-20" in md
    assert "## Headline" in md
    assert "## Per-signal table" in md
    assert "## Position-book reconciliation" in md
    assert "## TV chart cross-check (operator-filled)" in md
    assert "## Promotion gate status (informational)" in md
    assert "n_signals: 3" in md


def test_render_markdown_marks_failed_gate_red() -> None:
    inp = _inputs()
    md = render_markdown(inp)
    assert "hit_rate_min (0.53):" in md
    line = [line for line in md.splitlines() if line.startswith("- hit_rate_min")][0]
    assert "❌" in line


def test_render_markdown_marks_passing_gate_green() -> None:
    inp = _inputs()
    inp_passed = DailyReportInputs(
        **{**inp.__dict__,
           "hit_rate": HitRateResult(hit_rate=0.6, n_signals=2, n_hits=1, governor_block_rate=0.0)},
    )
    md = render_markdown(inp_passed)
    line = [line for line in md.splitlines() if line.startswith("- hit_rate_min")][0]
    assert "✅" in line


def test_build_per_signal_table_returns_polars_dataframe_with_expected_schema() -> None:
    df = build_per_signal_table(_inputs().rows)
    assert isinstance(df, pl.DataFrame)
    expected = {
        "signal_id", "symbol", "predicted_score", "confidence", "predicted_dir",
        "s2_decision", "fill_price", "horizon_minutes", "realized_return",
        "realized_dir", "hit", "weight", "fill_ts_utc", "fee",
    }
    assert set(df.columns) == expected
    assert df.height == 3


def test_build_per_signal_table_preserves_null_for_vetoed_signal() -> None:
    df = build_per_signal_table(_inputs().rows)
    veto_row = df.filter(pl.col("signal_id") == "sig-3").row(0, named=True)
    assert veto_row["fill_price"] is None
    assert veto_row["hit"] is None
