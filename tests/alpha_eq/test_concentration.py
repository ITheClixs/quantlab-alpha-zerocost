"""Concentration check (spec §6.4-11)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.diagnostics.concentration import (
    ConcentrationReport,
    check_concentration,
)


def test_single_stock_above_25pct_flagged() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, k) for k in (2, 3, 6, 7)],
            "symbol": ["A", "A", "A", "B"],
            "sector": ["tech", "tech", "tech", "tech"],
            "net_pnl": [100.0, 100.0, 100.0, 50.0],
        }
    )
    rep = check_concentration(pnl=pnl, max_stock_frac=0.25, max_month_frac=0.35, max_sector_frac=0.50)
    assert isinstance(rep, ConcentrationReport)
    assert rep.stock_violation is True


def test_no_violations_when_balanced() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 2, 3), date(2020, 3, 6), date(2020, 4, 7)],
            "symbol": ["A", "B", "C", "D"],
            "sector": ["tech", "finance", "energy", "health"],
            "net_pnl": [25.0, 25.0, 25.0, 25.0],
        }
    )
    rep = check_concentration(pnl=pnl, max_stock_frac=0.25, max_month_frac=0.35, max_sector_frac=0.50)
    assert not (rep.stock_violation or rep.month_violation or rep.sector_violation)
