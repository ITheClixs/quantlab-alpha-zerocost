"""Three-price-series builder (spec §2.3)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.corporate_actions import (
    PriceSeriesBundle,
    build_three_series,
    de_total_return_to_tradable,
)


def _toy_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
            "symbol": ["A", "A", "A", "A"],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000_000, 1_100_000, 1_050_000, 1_200_000],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))


def _toy_dividends() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ex_date": ["2020-01-06"],
            "symbol": ["A"],
            "dividend_per_share": [0.50],
        }
    ).with_columns(pl.col("ex_date").str.strptime(pl.Date, "%Y-%m-%d"))


def test_build_three_series_when_source_is_split_adjusted() -> None:
    bundle = build_three_series(
        panel=_toy_panel(),
        dividends=_toy_dividends(),
        source_is_total_return=False,
    )
    assert isinstance(bundle, PriceSeriesBundle)
    # tradable_* == split-adjusted in v1
    assert bundle.tradable["close"].to_list() == [100.5, 101.5, 102.5, 103.5]
    # total-return reflects the 0.50 dividend reinvested on ex-date 2020-01-06
    tr = bundle.total_return["close_tr"].to_list()
    assert tr[0] == 100.5
    assert tr[1] == 101.5
    assert tr[2] > 102.5  # bumped by dividend reinvestment
    assert tr[3] > 103.5


def test_de_total_return_inverse_of_total_return_build() -> None:
    panel = _toy_panel()
    divs = _toy_dividends()
    bundle = build_three_series(panel=panel, dividends=divs, source_is_total_return=False)
    rebuilt = de_total_return_to_tradable(bundle.total_return, divs)
    # rebuilt should match the split-adjusted tradable close within float tolerance
    assert (
        (rebuilt["close"] - bundle.tradable["close"]).abs().max() < 1e-9
    )
