"""Backtest report writer — required sections + prototype banner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.backtest.report import (
    ReportInputs,
    write_report,
)


def _inputs(label: str) -> ReportInputs:
    return ReportInputs(
        run_id="20260524T120000Z",
        git_sha="deadbeef",
        data_manifest_sha256="a" * 64,
        data_quality_label=label,
        cohort="full_universe",
        daily_returns=pl.DataFrame({
            "date": [date(2020, 1, 3)], "net_return": [0.001], "gross_return": [0.0015],
            "commission_drag": [0.0001], "spread_drag": [0.0002],
            "borrow_drag": [0.0001], "financing_drag": [0.0],
        }),
        decomposition_bps={
            "gross_alpha": 12.0, "cash_dividend": 1.0, "commission": 1.0,
            "spread": 2.0, "borrow": 1.0, "financing": 0.0, "net_alpha": 8.0,
        },
        sensitivity_rows=[],
    )


def test_report_emits_banner_when_prototype_only(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_report(out, _inputs("survivorship_prototype_only"))
    text = out.read_text()
    assert "prototype-only" in text.lower() or "survivorship_prototype_only" in text
    assert "not_investment_advice: true" in text
    assert "Configuration" in text


def test_report_no_prototype_banner_when_pit_safe(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_report(out, _inputs("pit_safe"))
    text = out.read_text()
    assert "prototype-only" not in text.lower()
