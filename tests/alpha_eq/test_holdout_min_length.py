"""Minimum holdout length ≥ 3 years (spec §3.6, §6.4-2)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutTooShortError,
    assert_min_holdout_length,
)


def test_holdout_min_length_passes_with_756_days() -> None:
    holdout = [date(2020, 1, 1) + timedelta(days=i) for i in range(756)]
    assert_min_holdout_length(holdout, min_trading_days=756)  # no raise


def test_holdout_min_length_raises_when_too_short() -> None:
    holdout = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    with pytest.raises(HoldoutTooShortError):
        assert_min_holdout_length(holdout, min_trading_days=756)
