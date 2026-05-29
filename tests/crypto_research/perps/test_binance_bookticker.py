from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import polars as pl
import pytest

from quant_research_stack.crypto_research.perps.binance_bookticker import (
    BOOKTICKER_RAW_COLUMNS,
    bookticker_day_url,
    normalize_bookticker,
    read_bookticker_zip_bytes,
    read_bookticker_zip_stream,
)

_RAW = pl.DataFrame(
    {
        "update_id": [1, 2, 3],
        "best_bid_price": [27077.3, 27077.3, 27077.4],
        "best_bid_qty": [19.35, 19.34, 0.0],
        "best_ask_price": [27077.4, 27077.4, 27077.5],
        "best_ask_qty": [8.6, 8.6, 5.0],
        "transaction_time": [1684237787209, 1684237787218, 1684237787224],
        "event_time": [1684237787214, 1684237787222, 1684237787228],
    }
)


def test_normalize_maps_columns_and_times() -> None:
    out = normalize_bookticker(_RAW, symbol="btcusdt")
    assert out.columns[:6] == ["symbol", "event_time", "best_bid", "best_ask", "best_bid_size", "best_ask_size"]
    assert out["symbol"][0] == "BTCUSDT"
    assert out["best_bid"][0] == 27077.3
    assert out["best_ask"][0] == 27077.4
    assert out["best_bid_size"][0] == 19.35
    assert out["best_ask_size"][0] == 8.6
    assert out["event_time"][0] == datetime.fromtimestamp(1684237787214 / 1000.0, tz=UTC)


def test_normalize_is_sorted_and_drops_crossed_or_nonpositive() -> None:
    raw = _RAW.with_columns(
        pl.Series("best_ask_price", [27077.4, 27077.2, 27077.5]),  # row1 crossed: bid>=ask
        pl.Series("best_bid_price", [27077.3, 27077.3, 27077.4]),
    )
    out = normalize_bookticker(raw, symbol="BTCUSDT")
    # row index 1 has best_bid(27077.3) >= best_ask(27077.2) -> dropped
    assert out.height == 2


def test_normalize_max_rows_caps_output() -> None:
    out = normalize_bookticker(_RAW, symbol="BTCUSDT", max_rows=2)
    assert out.height == 2


def test_normalize_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        normalize_bookticker(_RAW.drop("best_bid_price"), symbol="BTCUSDT")


def test_day_url_shape() -> None:
    url = bookticker_day_url("BTCUSDT", "2024-04-01")
    assert url.endswith("/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-04-01.zip")
    assert url.startswith("https://data.binance.vision/")


def test_read_zip_bytes_with_header_and_cap() -> None:
    csv = (
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        "1,27077.3,19.35,27077.4,8.6,1684237787209,1684237787214\n"
        "2,27077.3,19.34,27077.4,8.6,1684237787218,1684237787222\n"
        "3,27077.4,1.0,27077.5,5.0,1684237787224,1684237787228\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-bookTicker-2024-04-01.csv", csv)
    df = read_bookticker_zip_bytes(buf.getvalue(), max_rows=2)
    assert df.columns == BOOKTICKER_RAW_COLUMNS
    assert df.height == 2


def _zip_bytes(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-bookTicker-2024-03-29.csv", csv)
    return buf.getvalue()


def test_stream_reader_caps_rows_and_parses() -> None:
    csv = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
    csv += "".join(
        f"{i},27000.{i},1.0,27001.{i},2.0,168400000000{i % 10},168400000001{i % 10}\n" for i in range(500)
    )
    df = read_bookticker_zip_stream(io.BytesIO(_zip_bytes(csv)), max_rows=100)
    assert df.columns == BOOKTICKER_RAW_COLUMNS
    assert df.height == 100
    norm = normalize_bookticker(df, symbol="BTCUSDT")
    assert norm["best_bid"][0] == 27000.0


def test_stream_reader_reads_all_when_uncapped() -> None:
    csv = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
    csv += "".join(f"{i},27000.0,1.0,27001.0,2.0,1684000000000,1684000000001\n" for i in range(37))
    df = read_bookticker_zip_stream(io.BytesIO(_zip_bytes(csv)))
    assert df.height == 37


def test_read_zip_bytes_without_header() -> None:
    csv = (
        "1,27077.3,19.35,27077.4,8.6,1684237787209,1684237787214\n"
        "2,27077.3,19.34,27077.4,8.6,1684237787218,1684237787222\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-bookTicker-2024-04-01.csv", csv)
    df = read_bookticker_zip_bytes(buf.getvalue())
    assert df.columns == BOOKTICKER_RAW_COLUMNS
    assert df.height == 2
    assert df["best_bid_price"][0] == 27077.3
