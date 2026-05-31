"""Label builders y_raw / y_vn / y_xs (spec §3.2)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.labels import build_labels


def _toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7)] * 3,
            "symbol": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "close_tr": [100.0, 101.0, 102.0, 103.0,
                          50.0, 50.5, 51.0, 51.5,
                          200.0, 199.0, 201.0, 199.0],
            "realized_vol_20": [0.02] * 12,
            "in_universe": [True] * 12,
        }
    )


def test_labels_present() -> None:
    out = build_labels(_toy(), close_tr="close_tr", vol_col="realized_vol_20", universe_col="in_universe")
    for col in ("y_raw", "y_vn", "y_xs"):
        assert col in out.columns


def test_y_xs_zero_mean_per_date_among_universe() -> None:
    out = build_labels(_toy(), close_tr="close_tr", vol_col="realized_vol_20", universe_col="in_universe")
    by_date = out.filter(pl.col("y_xs").is_not_null()).group_by("date").agg(
        pl.col("y_xs").mean().alias("mu")
    )
    for mu in by_date["mu"].to_list():
        assert abs(mu) < 1e-9
