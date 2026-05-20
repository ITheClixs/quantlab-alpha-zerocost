from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_research_stack.execution.reconciliation import diff_book_vs_broker


@dataclass(frozen=True)
class ReconcileSummary:
    book_equity: Decimal
    broker_equity: Decimal
    diff_bps: float
    flagged: bool


def summarize_reconciliation(
    book_equity: Decimal,
    broker_equity: Decimal,
    max_diff_bps: float,
) -> ReconcileSummary:
    diff = diff_book_vs_broker(book_equity=book_equity, broker_equity=broker_equity)
    return ReconcileSummary(
        book_equity=book_equity,
        broker_equity=broker_equity,
        diff_bps=float(diff.diff_bps),
        flagged=diff.exceeds_threshold(max_diff_bps=max_diff_bps),
    )
