from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.perps.manifest import build_dataset_manifest


def test_manifest_records_hash_schema_and_timestamp_semantics(tmp_path: Path) -> None:
    data_path = tmp_path / "book.parquet"
    pl.DataFrame({"symbol": ["BTCUSDT"], "best_bid": [70000.0], "best_ask": [70000.5]}).write_parquet(data_path)

    manifest = build_dataset_manifest(
        dataset_id="unit-book",
        source="unit",
        paths=[data_path],
        symbols=["BTCUSDT"],
        timestamp_semantics="event_time from exchange milliseconds",
        quality_label="unit_test",
    )

    assert manifest["dataset_id"] == "unit-book"
    assert manifest["source"] == "unit"
    assert manifest["symbols"] == ["BTCUSDT"]
    assert manifest["timestamp_semantics"] == "event_time from exchange milliseconds"
    assert manifest["quality_label"] == "unit_test"
    assert manifest["row_count"] == 1
    assert manifest["files"][0]["sha256"]
    assert manifest["files"][0]["row_count"] == 1
    assert "best_bid" in manifest["files"][0]["schema"]


def test_manifest_records_unreadable_schema_without_failing(tmp_path: Path) -> None:
    data_path = tmp_path / "events.jsonl"
    data_path.write_text('{"symbol":"BTCUSDT","price":1.0}\n')

    manifest = build_dataset_manifest(
        dataset_id="unit-jsonl",
        source="unit",
        paths=[data_path],
        symbols=["BTCUSDT"],
        timestamp_semantics="jsonl fixture timestamp",
        quality_label="unit_test",
    )

    assert manifest["row_count"] == 1
    assert manifest["files"][0]["schema"]["price"] == "Float64"
