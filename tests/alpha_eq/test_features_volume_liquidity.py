"""Volume / liquidity features (spec §3.3-4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.volume_liquidity import (
    build_volume_liquidity,
)


def _toy() -> pl.DataFrame:
    dates_full = pl.date_range(
        date(2020, 1, 2), date(2020, 5, 31), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(60).to_list()
    return pl.DataFrame(
        {
            "date": dates,
            "symbol": ["A"] * 60,
            "close": [100.0 + i * 0.1 for i in range(60)],
            "volume": [1_000_000 + i * 1_000 for i in range(60)],
        }
    )


def test_volume_liquidity_columns_present_and_no_turnover_proxy() -> None:
    df = build_volume_liquidity(_toy(), window=20)
    assert "dollar_volume" in df.columns
    assert "log_dollar_volume_20d" in df.columns
    assert "volume_zscore_20d" in df.columns
    # spec §3.3-4: turnover_proxy_20 is DROPPED unless real shares-outstanding source
    assert "turnover_proxy_20" not in df.columns
