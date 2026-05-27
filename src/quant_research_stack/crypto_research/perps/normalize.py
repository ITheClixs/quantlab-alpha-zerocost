from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_research_stack.crypto_research.perps.events import (
    normalize_agg_trade,
    normalize_book_ticker,
    normalize_depth_update,
)


def _rows_to_frame(rows: list[dict[str, Any]], *, dataset_id: str | None) -> pl.DataFrame:
    if dataset_id is not None:
        for row in rows:
            row["dataset_id"] = dataset_id
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["symbol", "event_time"])


def normalize_book_ticker_frame(
    frame: pl.DataFrame,
    *,
    received_utc: datetime,
    dataset_id: str | None = None,
) -> pl.DataFrame:
    rows = [normalize_book_ticker(row, received_utc=received_utc) for row in frame.to_dicts()]
    out = _rows_to_frame(rows, dataset_id=dataset_id)
    if out.is_empty():
        return out
    return out.with_columns(
        [
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            (
                (pl.col("best_ask") - pl.col("best_bid"))
                / ((pl.col("best_ask") + pl.col("best_bid")) / 2.0)
            ).alias("relative_spread"),
        ]
    )


def normalize_agg_trade_frame(
    frame: pl.DataFrame,
    *,
    received_utc: datetime,
    dataset_id: str | None = None,
) -> pl.DataFrame:
    rows = [normalize_agg_trade(row, received_utc=received_utc) for row in frame.to_dicts()]
    return _rows_to_frame(rows, dataset_id=dataset_id)


def _depth_payload(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "b" not in out and "bids" in out:
        out["b"] = json.loads(out["bids"]) if isinstance(out["bids"], str) else out["bids"]
    if "a" not in out and "asks" in out:
        out["a"] = json.loads(out["asks"]) if isinstance(out["asks"], str) else out["asks"]
    return out


def normalize_depth_frame(
    frame: pl.DataFrame,
    *,
    received_utc: datetime,
    dataset_id: str | None = None,
) -> pl.DataFrame:
    rows = [normalize_depth_update(_depth_payload(row), received_utc=received_utc) for row in frame.to_dicts()]
    return _rows_to_frame(rows, dataset_id=dataset_id)


def write_normalized_events(
    frame: pl.DataFrame,
    *,
    output_path: Path,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
) -> dict[str, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True, default=str) + "\n")
    return {"events": output_path, "manifest": manifest_path}
