"""Market regime features + VIX fallback rule (spec §3.3-6)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.market_regime import (
    build_market_regime,
)


def _panel(n: int = 80) -> pl.DataFrame:
    dates_full = pl.date_range(
        date(1988, 1, 4), date(1988, 12, 31), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(n).to_list()
    rows = []
    for d in dates:
        for s in ["A", "B", "C"]:
            rows.append({"date": d, "symbol": s, "close": 100.0 + (hash(s) % 5)})
    return pl.DataFrame(rows)


def test_vix_fallback_when_no_vix_provided() -> None:
    df = build_market_regime(panel=_panel(), vix=None, spy_close=None)
    assert "vix_close" in df.columns
    assert "vix_is_proxy" in df.columns
    assert all(df["vix_is_proxy"].to_list())


def test_no_truncation_when_vix_missing_early_dates() -> None:
    """Missing VIX must NOT silently drop early rows."""
    panel = _panel()
    n_before = panel.height
    df = build_market_regime(panel=panel, vix=None, spy_close=None)
    assert df.height == n_before
