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
        "high": ["43000.0", "44500.0"],
        "low": ["41500.0", "42100.0"],
        "close": ["42500.0", "44000.0"],
    })
    out = prices.normalize_klines(raw, symbol="btcusdt")
    assert out.columns == prices.NORMALIZED_COLUMNS
    assert out["symbol"].to_list() == ["BTCUSDT", "BTCUSDT"]
    assert out["ts"].dt.date().to_list() == [date(2024, 1, 1), date(2024, 1, 2)]
    assert out["close"].to_list() == [42500.0, 44000.0]
    assert out["high"].to_list() == [43000.0, 44500.0]


def test_normalize_klines_us_epoch_detected() -> None:
    raw = pl.DataFrame({
        "open_time": [1_704_067_200_000_000],  # microseconds
        "open": ["1.0"], "high": ["1.5"], "low": ["0.9"], "close": ["2.0"],
    })
    out = prices.normalize_klines(raw, symbol="ethusdt")
    assert out["ts"].dt.date().to_list() == [date(2024, 1, 1)]


def test_normalize_klines_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        prices.normalize_klines(pl.DataFrame({"open_time": [1], "open": [1.0]}), symbol="x")


def test_klines_url_8h_interval() -> None:
    assert prices.klines_url("btcusdt", "perp", "2024-01", "8h") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-2024-01.zip"
    )


def test_align_carry_8h_joins_on_settlement_ts() -> None:
    ts = pl.Series(["2024-01-01T00:00:00", "2024-01-01T08:00:00"]).str.to_datetime(time_zone="UTC")
    spot = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "ts": ts, "open": [100.0, 100.0],
                         "high": [101.0, 102.0], "low": [99.0, 99.5], "close": [100.0, 101.0]})
    perp = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "ts": ts, "open": [100.5, 101.0],
                         "high": [102.0, 103.0], "low": [99.5, 100.0], "close": [101.0, 101.5]})
    funding = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "funding_time": ts,
                            "funding_rate": [0.0001, 0.0002], "interval_hours": [8, 8]})
    out = prices.align_carry_8h(funding, spot, perp)
    assert out.height == 2
    assert {"spot_high", "spot_low", "perp_high", "perp_low", "funding_rate", "basis"} <= set(out.columns)
    assert out["basis"][0] == pytest.approx(101.0 / 100.0 - 1.0)


def test_align_carry_8h_matches_jittered_funding_times() -> None:
    """Regression: Binance funding calc_time has ms jitter; the 8h join must still
    match every settlement (exact-timestamp join previously dropped ~45%)."""
    ts = pl.Series(["2024-01-01T00:00:00", "2024-01-01T08:00:00"]).str.to_datetime(time_zone="UTC")
    # funding times jittered a few ms past the boundary, as Binance emits them
    fts = pl.Series(["2024-01-01T00:00:00.002", "2024-01-01T08:00:00.001"]).str.to_datetime(time_zone="UTC")
    spot = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "ts": ts, "open": [100.0, 100.0],
                         "high": [101.0, 102.0], "low": [99.0, 99.5], "close": [100.0, 101.0]})
    perp = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "ts": ts, "open": [100.5, 101.0],
                         "high": [102.0, 103.0], "low": [99.5, 100.0], "close": [101.0, 101.5]})
    funding = pl.DataFrame({"symbol": ["BTCUSDT"] * 2, "funding_time": fts,
                            "funding_rate": [0.0001, 0.0002], "interval_hours": [8, 8]})
    out = prices.align_carry_8h(funding, spot, perp)
    assert out.height == 2  # both jittered settlements matched
    assert out["funding_rate"].to_list() == pytest.approx([0.0001, 0.0002])


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
