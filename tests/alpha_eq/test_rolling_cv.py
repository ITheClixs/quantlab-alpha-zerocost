"""Rolling-window 10y/2y CV diagnostic (spec §4.2)."""

from __future__ import annotations

from datetime import date, timedelta

from quant_research_stack.alpha_eq.diagnostics.rolling_cv import (
    RollingWindow,
    build_rolling_windows,
)


def test_rolling_windows_chronological_and_non_overlapping_validation() -> None:
    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(10 * 365)]
    wins = build_rolling_windows(dates, train_years=5, valid_years=1, step_years=1)
    assert all(isinstance(w, RollingWindow) for w in wins)
    assert len(wins) > 0
    for w in wins:
        assert max(w.train_dates) < min(w.validation_dates)
    for prev, nxt in zip(wins[:-1], wins[1:], strict=True):
        prev_end = max(prev.validation_dates)
        nxt_start = min(nxt.validation_dates)
        assert nxt_start >= prev_end - timedelta(days=400)
