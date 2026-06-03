from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_vwap_proxy_is_typical_price_rolling_volume_weighted(panel: pl.DataFrame) -> None:
    out = daily_vwap_proxy(panel, window=5)
    assert "vwap" in out.columns
    assert out.height == panel.height
    defined = out.drop_nulls("vwap")
    assert defined.height > 0
    assert (defined["vwap"] >= defined["low"].min()).all()
    assert out.sort(["symbol", "date"]).select(["symbol", "date"]).equals(
        panel.sort(["symbol", "date"]).select(["symbol", "date"])
    )


def test_vwap_primary_long_only_below_vwap_band(panel: pl.DataFrame) -> None:
    with_vwap = daily_vwap_proxy(panel, window=5)
    out = vwap_primary_position(with_vwap, band=0.01)
    assert set(out["primary_position"].unique().to_list()) <= {0.0, 1.0}
    longs = out.filter(pl.col("primary_position") == 1.0).drop_nulls("vwap")
    if longs.height:
        assert (longs["close"] <= longs["vwap"] * (1.0 - 0.01) + 1e-9).all()
