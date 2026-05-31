"""Delisting-return audit (spec §2.9)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.delisting_audit import (
    DelistingAuditResult,
    audit_delistings,
    classify_exit,
)


def test_classify_exit_basic() -> None:
    assert classify_exit(reason="bankruptcy", terminal_return_known=True) == "delisted_captured"
    assert classify_exit(reason="bankruptcy", terminal_return_known=False) == "delisted_missing"
    assert classify_exit(reason="acquired", terminal_return_known=True) == "merger_captured"
    assert classify_exit(reason="acquired", terminal_return_known=False) == "merger_missing"
    assert classify_exit(reason="ticker_change", terminal_return_known=False) == "ticker_changed"
    assert classify_exit(reason="unknown", terminal_return_known=False) == "unknown_exit"


def test_audit_delistings_counts_and_threshold() -> None:
    panel = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 2), date(2020, 1, 3),
                date(2020, 1, 2),
                date(2020, 1, 2), date(2020, 1, 3),
            ],
            "symbol": ["AAA", "AAA", "BBB", "CCC", "CCC"],
            "close": [100.0, 101.0, 50.0, 200.0, 199.0],
        }
    )
    exits = pl.DataFrame(
        {
            "symbol": ["BBB"],
            "exit_date": [date(2020, 1, 3)],
            "exit_reason": ["acquired"],
            "terminal_return_known": [True],
            "terminal_return_value": [-0.10],
        }
    )
    result = audit_delistings(panel=panel, exits=exits)
    assert isinstance(result, DelistingAuditResult)
    assert result.counters["merger_captured"] == 1
    assert result.counters["unknown_exit"] == 0


def test_audit_delistings_flags_unknown_exit() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 2)],
            "symbol": ["AAA", "BBB"],
            "close": [100.0, 50.0],
        }
    )
    panel_next = pl.DataFrame(
        {"date": [date(2020, 1, 3)], "symbol": ["AAA"], "close": [101.0]}
    )
    full = pl.concat([panel, panel_next])
    exits = pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "exit_date": pl.Date,
            "exit_reason": pl.Utf8,
            "terminal_return_known": pl.Boolean,
            "terminal_return_value": pl.Float64,
        }
    )
    result = audit_delistings(panel=full, exits=exits)
    assert result.counters["unknown_exit"] >= 1
