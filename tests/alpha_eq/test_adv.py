"""Dollar ADV (spec §2.5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.adv import build_adv_20d_dollar


def test_adv_is_lagged_by_one_day() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7, 8, 9, 10)],
            "symbol": ["A"] * 7,
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_400, 1_500, 1_600],
        }
    )
    adv = build_adv_20d_dollar(panel, window=3)
    nulls = adv["adv_20d_dollar_lag1"].is_null().to_list()
    assert nulls[0] is True
    assert nulls[1] is True
    # Row index 3 (date 2020-01-07): window=3 lagged by 1 uses rows 0..2.
    expected = float(
        sorted([10.0 * 1_000, 11.0 * 1_100, 12.0 * 1_200])[1]
    )
    assert abs(adv["adv_20d_dollar_lag1"][3] - expected) < 1e-9


def test_adv_uses_dollar_not_share_volume() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7)],
            "symbol": ["A"] * 4,
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        }
    )
    adv = build_adv_20d_dollar(panel, window=2)
    vals = [v for v in adv["adv_20d_dollar_lag1"].to_list() if v is not None]
    assert all(abs(v - 100.0) < 1e-9 for v in vals)
