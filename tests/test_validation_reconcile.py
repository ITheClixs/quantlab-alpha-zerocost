from __future__ import annotations

from decimal import Decimal

import pytest

from quant_research_stack.validation.reconcile import (
    ReconcileSummary,
    summarize_reconciliation,
)


def test_zero_diff_when_book_matches_broker() -> None:
    summary = summarize_reconciliation(
        book_equity=Decimal("100000"), broker_equity=Decimal("100000"), max_diff_bps=1.0,
    )
    assert isinstance(summary, ReconcileSummary)
    assert summary.diff_bps == pytest.approx(0.0, abs=1e-9)
    assert summary.flagged is False


def test_1_bp_diff_not_flagged_at_threshold() -> None:
    summary = summarize_reconciliation(
        book_equity=Decimal("100010"), broker_equity=Decimal("100000"), max_diff_bps=1.0,
    )
    assert summary.diff_bps == pytest.approx(1.0, abs=1e-3)
    assert summary.flagged is False


def test_2_bps_diff_flagged() -> None:
    summary = summarize_reconciliation(
        book_equity=Decimal("100020"), broker_equity=Decimal("100000"), max_diff_bps=1.0,
    )
    assert summary.diff_bps > 1.0
    assert summary.flagged is True


def test_zero_broker_equity_flagged_as_divergence() -> None:
    summary = summarize_reconciliation(
        book_equity=Decimal("100"), broker_equity=Decimal("0"), max_diff_bps=1.0,
    )
    assert summary.flagged is True
