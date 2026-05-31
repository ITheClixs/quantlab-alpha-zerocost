from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.perps.normalize import (
    normalize_agg_trade_frame,
    normalize_book_ticker_frame,
    normalize_depth_frame,
    write_normalized_events,
)


def test_normalize_book_ticker_frame_writes_required_columns() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "u": [2, 1],
            "s": ["ETHUSDT", "BTCUSDT"],
            "b": ["3500.0", "70000.0"],
            "B": ["10", "5"],
            "a": ["3500.5", "70001.0"],
            "A": ["11", "6"],
        }
    )

    out = normalize_book_ticker_frame(frame, received_utc=received_utc, dataset_id="unit-book")

    assert {"dataset_id", "symbol", "event_time", "best_bid", "best_ask", "relative_spread"}.issubset(out.columns)
    assert out["dataset_id"].to_list() == ["unit-book", "unit-book"]
    assert out["symbol"].to_list() == ["BTCUSDT", "ETHUSDT"]
    assert out["relative_spread"][0] > 0.0


def test_normalize_agg_trade_frame_sorts_and_preserves_side() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "E": [1710000000200, 1710000000100],
            "s": ["BTCUSDT", "BTCUSDT"],
            "a": [102, 101],
            "p": ["70001.0", "70000.5"],
            "q": ["0.10", "0.25"],
            "T": [1710000000200, 1710000000100],
            "m": [False, True],
        }
    )

    out = normalize_agg_trade_frame(frame, received_utc=received_utc, dataset_id="unit-trades")

    assert out["trade_id"].to_list() == [101, 102]
    assert out["aggressor_side"].to_list() == ["sell", "buy"]
    assert out["dataset_id"].to_list() == ["unit-trades", "unit-trades"]


def test_normalize_depth_frame_converts_levels_and_sorts() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "E": [1710000000200, 1710000000100],
            "s": ["ETHUSDT", "BTCUSDT"],
            "U": [200, 100],
            "u": [205, 105],
            "b": [
                [["3500.0", "1.5"]],
                [["70000.0", "1.0"], ["69999.5", "0.5"]],
            ],
            "a": [
                [["3500.5", "1.0"]],
                [["70000.5", "0.75"]],
            ],
        }
    )

    out = normalize_depth_frame(frame, received_utc=received_utc, dataset_id="unit-depth")

    assert out["symbol"].to_list() == ["BTCUSDT", "ETHUSDT"]
    assert out["bids"].to_list()[0] == [[70000.0, 1.0], [69999.5, 0.5]]
    assert out["asks"].to_list()[1] == [[3500.5, 1.0]]
    assert out["dataset_id"].to_list() == ["unit-depth", "unit-depth"]


def test_normalize_depth_frame_accepts_hf_bids_asks_json_strings() -> None:
    received_utc = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "E": [1710000000100],
            "s": ["BTCUSDT"],
            "U": [100],
            "u": [105],
            "bids": ['[["70000.0","1.0"],["69999.5","0.5"]]'],
            "asks": ['[["70000.5","0.75"]]'],
        }
    )

    out = normalize_depth_frame(frame, received_utc=received_utc)

    assert out["bids"].to_list()[0] == [[70000.0, 1.0], [69999.5, 0.5]]
    assert out["asks"].to_list()[0] == [[70000.5, 0.75]]


def test_write_normalized_events_writes_parquet_and_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "events.parquet"
    manifest_path = tmp_path / "manifest.json"
    frame = pl.DataFrame({"symbol": ["BTCUSDT"], "event_time": [datetime(2026, 5, 26, tzinfo=UTC)]})
    manifest = {"dataset_id": "unit", "row_count": 1}

    written = write_normalized_events(
        frame,
        output_path=output_path,
        manifest_path=manifest_path,
        manifest_payload=manifest,
    )

    assert written["events"] == output_path
    assert written["manifest"] == manifest_path
    assert pl.read_parquet(output_path).height == 1
    assert json.loads(manifest_path.read_text())["dataset_id"] == "unit"
