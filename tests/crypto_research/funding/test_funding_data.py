from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_research_stack.crypto_research.funding.data import (
    NORMALIZED_COLUMNS,
    annualized_funding,
    funding_day_url,
    normalize_funding,
)

_RAW = pl.DataFrame({
    "calc_time": [1609459200002, 1609488000006, 1609516800003],
    "funding_interval_hours": [8, 8, 8],
    "last_funding_rate": [0.00022753, 0.00026336, 0.00034457],
})


def test_normalize_maps_and_timestamps() -> None:
    out = normalize_funding(_RAW, symbol="btcusdt")
    assert out.columns == NORMALIZED_COLUMNS
    assert out["symbol"][0] == "BTCUSDT"
    assert out["interval_hours"][0] == 8
    assert out["funding_time"][0] == datetime.fromtimestamp(1609459200002 / 1000.0, tz=UTC)
    assert out["funding_rate"][0] == pytest.approx(0.00022753)


def test_annualized_funding() -> None:
    out = normalize_funding(_RAW, symbol="BTCUSDT")
    ann = annualized_funding(out)
    assert ann == pytest.approx(float(_RAW["last_funding_rate"].mean()) * 3 * 365)
    assert ann > 0  # positive funding regime


def test_normalize_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        normalize_funding(_RAW.drop("last_funding_rate"), symbol="BTCUSDT")


def test_funding_url_shape() -> None:
    assert funding_day_url("BTCUSDT", "2021-01").endswith(
        "/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2021-01.zip")
