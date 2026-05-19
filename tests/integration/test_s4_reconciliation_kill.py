from __future__ import annotations

from decimal import Decimal

import pytest

pytestmark = pytest.mark.s4_integration


def test_5_bps_diff_exceeds_threshold() -> None:
    from quant_research_stack.execution.reconciliation import diff_book_vs_broker

    book = Decimal("100050")
    broker = Decimal("100000")
    res = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert res.diff_bps == pytest.approx(5.0, abs=1e-3)
    assert res.exceeds_threshold(max_diff_bps=1.0) is True
