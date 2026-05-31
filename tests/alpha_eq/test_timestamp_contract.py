"""feature_as_of_date < execution_date hard invariant (spec §3.1, §3.5)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.alpha_eq.features.timestamps import (
    TimestampContractError,
    assert_feature_before_execution,
    attach_execution_date,
    attach_feature_as_of_date,
)


def test_after_close_convention_attaches_same_day_as_of() -> None:
    df = pl.DataFrame(
        {"date": [date(2020, 1, 2), date(2020, 1, 3)], "symbol": ["A", "A"]}
    )
    out = attach_feature_as_of_date(df, convention="after_close_t")
    assert out["feature_as_of_date"].to_list() == [date(2020, 1, 2), date(2020, 1, 3)]


def test_execution_date_is_next_trading_day() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "symbol": ["A", "A", "A"],
        }
    )
    out = attach_execution_date(df, convention="next_trading_day")
    rows = out.to_dicts()
    # 2020-01-03 (Fri) → 2020-01-06 (Mon, weekend skip)
    assert rows[1]["execution_date"] == date(2020, 1, 6)


def test_assert_feature_before_execution_passes() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 2)],
            "execution_date": [date(2020, 1, 3)],
        }
    )
    assert_feature_before_execution(df)


def test_assert_feature_before_execution_raises_on_violation() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 3)],
            "execution_date": [date(2020, 1, 3)],
        }
    )
    with pytest.raises(TimestampContractError):
        assert_feature_before_execution(df)
