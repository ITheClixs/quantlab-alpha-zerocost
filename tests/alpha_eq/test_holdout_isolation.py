"""Permanent holdout isolation (spec §3.6) — load guard."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutAccessError,
    HoldoutGate,
    compute_holdout_dates,
)


def test_compute_holdout_uses_last_20_percent() -> None:
    dates = [date(2020, 1, d) for d in range(2, 12)]  # 10 dates
    dev, hold = compute_holdout_dates(dates, fraction=0.2)
    assert len(hold) == 2
    assert hold == sorted(hold)
    assert dev[-1] < hold[0]


def test_holdout_gate_blocks_training_caller() -> None:
    holdout = [date(2020, 1, 9), date(2020, 1, 10)]
    gate = HoldoutGate(holdout_dates=holdout)
    panel = pl.DataFrame(
        {"date": [date(2020, 1, 9), date(2020, 1, 8)], "symbol": ["A", "A"], "x": [1.0, 2.0]}
    )
    with pytest.raises(HoldoutAccessError):
        gate.filter_for_caller(panel, caller="training")


def test_holdout_gate_allows_inference_evaluate_holdout() -> None:
    holdout = [date(2020, 1, 9)]
    gate = HoldoutGate(holdout_dates=holdout)
    panel = pl.DataFrame({"date": [date(2020, 1, 9)], "symbol": ["A"], "x": [1.0]})
    out = gate.filter_for_caller(panel, caller="inference.evaluate_holdout")
    assert out.height == 1
