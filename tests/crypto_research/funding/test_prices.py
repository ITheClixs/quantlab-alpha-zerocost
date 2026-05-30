"""Offline unit tests for funding-carry price loaders + alignment."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.crypto_research.funding import prices


def test_klines_url_spot_and_perp() -> None:
    assert prices.klines_url("btcusdt", "spot", "2024-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )
    assert prices.klines_url("ethusdt", "perp", "2024-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-2024-01.zip"
    )


def test_base_rejects_unknown_market() -> None:
    with pytest.raises(ValueError, match="market must be one of"):
        prices.klines_url("btcusdt", "options", "2024-01")


def test_normalize_klines_ms_epoch() -> None:
    raw = pl.DataFrame({
        "open_time": [1_704_067_200_000, 1_704_153_600_000],  # 2024-01-01, 2024-01-02 (ms)
        "open": ["42000.0", "42500.0"],
        "close": ["42500.0", "44000.0"],
    })
    out = prices.normalize_klines(raw, symbol="btcusdt")
    assert out.columns == prices.NORMALIZED_COLUMNS
    assert out["symbol"].to_list() == ["BTCUSDT", "BTCUSDT"]
    assert out["ts"].dt.date().to_list() == [date(2024, 1, 1), date(2024, 1, 2)]
    assert out["close"].to_list() == [42500.0, 44000.0]


def test_normalize_klines_us_epoch_detected() -> None:
    raw = pl.DataFrame({
        "open_time": [1_704_067_200_000_000],  # microseconds
        "open": ["1.0"], "close": ["2.0"],
    })
    out = prices.normalize_klines(raw, symbol="ethusdt")
    assert out["ts"].dt.date().to_list() == [date(2024, 1, 1)]


def test_normalize_klines_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        prices.normalize_klines(pl.DataFrame({"open_time": [1], "open": [1.0]}), symbol="x")


def test_daily_funding_sums_settlements() -> None:
    funding = pl.DataFrame({
        "symbol": ["BTCUSDT"] * 4,
        "funding_time": [
            "2024-01-01T00:00:00", "2024-01-01T08:00:00",
            "2024-01-01T16:00:00", "2024-01-02T00:00:00",
        ],
        "funding_rate": [0.0001, 0.0002, 0.0001, 0.0003],
        "interval_hours": [8] * 4,
    }).with_columns(pl.col("funding_time").str.to_datetime(time_zone="UTC"))
    out = prices.daily_funding(funding)
    row0 = out.filter(pl.col("date") == date(2024, 1, 1))
    assert row0["settlements"].item() == 3
    assert row0["funding_day"].item() == pytest.approx(0.0004)


def test_align_carry_inner_join_and_basis() -> None:
    spot = pl.DataFrame({
        "symbol": ["BTCUSDT"] * 2,
        "ts": pl.Series(["2024-01-01", "2024-01-02"]).str.to_datetime(time_zone="UTC"),
        "open": [100.0, 110.0], "close": [100.0, 110.0],
    })
    perp = pl.DataFrame({
        "symbol": ["BTCUSDT"] * 2,
        "ts": pl.Series(["2024-01-01", "2024-01-02"]).str.to_datetime(time_zone="UTC"),
        "open": [101.0, 110.0], "close": [101.0, 110.0],  # perp 1% rich day 1, flat day 2
    })
    funding = pl.DataFrame({
        "symbol": ["BTCUSDT"] * 2,
        "funding_time": pl.Series(["2024-01-01T00:00:00", "2024-01-02T00:00:00"]).str.to_datetime(time_zone="UTC"),
        "funding_rate": [0.0001, 0.0002], "interval_hours": [8, 8],
    })
    out = prices.align_carry(funding, spot, perp)
    assert out["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]
    assert out["basis"].to_list() == pytest.approx([0.01, 0.0])
    assert out["funding_day"].to_list() == pytest.approx([0.0001, 0.0002])
