"""Backtest row contract (spec §5.2)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.alpha_eq.backtest.contracts import (
    BacktestContractError,
    assert_backtest_row_contract,
)


def test_assert_backtest_row_contract_passes_on_well_formed_row() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 2)],
            "execution_date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_target_notional": [1000.0],
            "fill_price": [101.0],
        }
    )
    assert_backtest_row_contract(df)


def test_assert_backtest_row_contract_fails_when_feature_not_before_execution() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 3)],
            "execution_date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_target_notional": [1000.0],
            "fill_price": [101.0],
        }
    )
    with pytest.raises(BacktestContractError):
        assert_backtest_row_contract(df)


def test_assert_backtest_row_contract_required_columns() -> None:
    df = pl.DataFrame({"symbol": ["AAPL"]})
    with pytest.raises(BacktestContractError):
        assert_backtest_row_contract(df)
