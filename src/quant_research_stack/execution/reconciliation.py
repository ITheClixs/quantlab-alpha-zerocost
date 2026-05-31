from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ReconciliationResult:
    book_equity: Decimal
    broker_equity: Decimal
    diff_bps: float

    def exceeds_threshold(self, max_diff_bps: float) -> bool:
        return self.diff_bps > max_diff_bps


def diff_book_vs_broker(book_equity: Decimal, broker_equity: Decimal) -> ReconciliationResult:
    if broker_equity <= 0:
        return ReconciliationResult(book_equity=book_equity, broker_equity=broker_equity, diff_bps=float("inf"))
    diff = abs(book_equity - broker_equity)
    bps = float(diff / broker_equity * Decimal("10000"))
    return ReconciliationResult(book_equity=book_equity, broker_equity=broker_equity, diff_bps=bps)
