"""Cross-sectional ranks within the date-t tradable PIT universe (spec §3.3-5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.cross_sectional_ranks import (
    build_cross_sectional_ranks,
)


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 2)] * 4 + [date(2020, 1, 3)] * 4,
            "symbol": ["A", "B", "C", "D"] * 2,
            "in_universe": [True, True, True, False, True, True, False, True],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 99.0, 0.5],
        }
    )


def test_rank_is_only_within_in_universe() -> None:
    df = build_cross_sectional_ranks(
        _panel(),
        columns=("feature_value",),
        universe_col="in_universe",
    )
    row_d = df.filter((pl.col("symbol") == "D") & (pl.col("date") == date(2020, 1, 2)))
    assert row_d["rank_feature_value"][0] is None
    row_a = df.filter((pl.col("symbol") == "A") & (pl.col("date") == date(2020, 1, 2)))
    # min rank=1, n=3 → (1-1)/(3-1)=0 → -0.5
    assert abs(row_a["rank_feature_value"][0] - (-0.5)) < 1e-12


def test_rank_invariant_to_out_of_universe_rows() -> None:
    base = _panel()
    extra_oou = pl.DataFrame(
        {
            "date": [date(2020, 1, 2)],
            "symbol": ["X"],
            "in_universe": [False],
            "feature_value": [-99999.0],
        }
    )
    joined = pl.concat([base, extra_oou])
    a = build_cross_sectional_ranks(
        base, columns=("feature_value",), universe_col="in_universe"
    ).filter(pl.col("date") == date(2020, 1, 2))
    b = build_cross_sectional_ranks(
        joined, columns=("feature_value",), universe_col="in_universe"
    ).filter((pl.col("date") == date(2020, 1, 2)) & (pl.col("symbol") != "X"))
    assert a.sort("symbol")["rank_feature_value"].to_list() == b.sort("symbol")["rank_feature_value"].to_list()
