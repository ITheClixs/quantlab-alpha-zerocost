"""Backtest row temporal contract (spec §5.2)."""

from __future__ import annotations

import polars as pl

_REQUIRED_ROW_COLS: tuple[str, ...] = (
    "feature_as_of_date",
    "execution_date",
    "symbol",
    "signed_target_notional",
    "fill_price",
)


class BacktestContractError(RuntimeError):
    pass


def assert_backtest_row_contract(df: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_ROW_COLS if c not in df.columns]
    if missing:
        raise BacktestContractError(f"missing required columns: {missing}")
    bad = df.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise BacktestContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )
