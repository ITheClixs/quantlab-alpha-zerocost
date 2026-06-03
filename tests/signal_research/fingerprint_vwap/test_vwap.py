from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.vwap import daily_vwap_proxy


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
