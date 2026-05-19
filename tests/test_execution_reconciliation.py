from __future__ import annotations

from decimal import Decimal

import pytest

from quant_research_stack.execution.reconciliation import (
    ReconciliationResult,
    diff_book_vs_broker,
)


def test_zero_diff_when_book_matches_broker() -> None:
    broker_equity = Decimal("100000")
    result = diff_book_vs_broker(book_equity=Decimal("100000"), broker_equity=broker_equity)
    assert isinstance(result, ReconciliationResult)
    assert result.diff_bps == pytest.approx(0.0, abs=1e-9)


def test_one_bp_diff() -> None:
    book = Decimal("100010")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.diff_bps == pytest.approx(1.0, abs=1e-4)


def test_exceeds_threshold_at_2_bps() -> None:
    book = Decimal("100020")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.diff_bps > 1.0
    assert result.exceeds_threshold(max_diff_bps=1.0) is True


def test_within_threshold_at_half_bp() -> None:
    book = Decimal("100005")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.exceeds_threshold(max_diff_bps=1.0) is False


def test_zero_broker_equity_treated_as_divergence() -> None:
    result = diff_book_vs_broker(book_equity=Decimal("100"), broker_equity=Decimal("0"))
    assert result.exceeds_threshold(max_diff_bps=1.0) is True
