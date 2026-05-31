from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source_path: str
    row_count: int
    symbols: list[str]
    start: str
    end: str
    schema: dict[str, str]
    sha256: str | None
    timestamp_column: str
    timestamp_semantics: str
    known_limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Period:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ChronologicalPeriods:
    development: Period
    validation: Period
    holdout: Period

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {
            "development": {"start": self.development.start.isoformat(), "end": self.development.end.isoformat()},
            "validation": {"start": self.validation.start.isoformat(), "end": self.validation.end.isoformat()},
            "holdout": {"start": self.holdout.start.isoformat(), "end": self.holdout.end.isoformat()},
        }


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_from_frame(
    frame: pl.DataFrame,
    *,
    dataset_id: str,
    source_path: Path,
    timestamp_column: str,
    timestamp_semantics: str = "bar close timestamp; signal uses only current and past bars",
    known_limitations: list[str] | None = None,
) -> DatasetManifest:
    if frame.is_empty():
        raise ValueError("cannot build a dataset manifest from an empty frame")
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    symbols = sorted(str(value) for value in frame.get_column("symbol").unique().to_list()) if "symbol" in frame.columns else []
    summary = frame.select(
        [
            pl.len().alias("rows"),
            pl.min(timestamp_column).alias("start"),
            pl.max(timestamp_column).alias("end"),
        ]
    ).row(0, named=True)
    return DatasetManifest(
        dataset_id=dataset_id,
        source_path=str(source_path),
        row_count=int(summary["rows"]),
        symbols=symbols,
        start=summary["start"].isoformat() if hasattr(summary["start"], "isoformat") else str(summary["start"]),
        end=summary["end"].isoformat() if hasattr(summary["end"], "isoformat") else str(summary["end"]),
        schema={name: str(dtype) for name, dtype in frame.schema.items()},
        sha256=file_sha256(source_path) if source_path.exists() and source_path.is_file() else None,
        timestamp_column=timestamp_column,
        timestamp_semantics=timestamp_semantics,
        known_limitations=known_limitations or [],
    )


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def load_btcusdt_1m_panel(path: Path, *, months: int = 18, max_rows: int | None = None) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if "datetime" not in frame.columns:
        raise ValueError("expected BTCUSDT 1m parquet with a datetime column")
    frame = (
        frame.rename({"datetime": "timestamp"})
        .with_columns(pl.lit("BTCUSDT").alias("symbol"))
        .sort("timestamp")
    )
    start_ts = frame.select(pl.col("timestamp").max().dt.offset_by(f"-{months}mo")).item()
    out = frame.filter(pl.col("timestamp") >= start_ts)
    if max_rows is not None and max_rows > 0 and out.height > max_rows:
        out = out.tail(max_rows)
    return out


def build_chronological_periods(
    frame: pl.DataFrame,
    *,
    timestamp_column: str,
    holdout_fraction: float = 0.15,
    validation_fraction: float = 0.25,
) -> ChronologicalPeriods:
    if frame.height < 10:
        raise ValueError("not enough rows to build chronological research periods")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    ordered = frame.sort(timestamp_column)
    n_rows = ordered.height
    dev_end_idx = int(n_rows * (1.0 - holdout_fraction - validation_fraction))
    val_end_idx = int(n_rows * (1.0 - holdout_fraction))
    if dev_end_idx <= 0 or val_end_idx <= dev_end_idx or val_end_idx >= n_rows:
        raise ValueError("period fractions produce an empty development, validation, or holdout period")
    ts = ordered.get_column(timestamp_column)
    return ChronologicalPeriods(
        development=Period(name="development", start=ts[0], end=ts[dev_end_idx - 1]),
        validation=Period(name="validation", start=ts[dev_end_idx], end=ts[val_end_idx - 1]),
        holdout=Period(name="holdout", start=ts[val_end_idx], end=ts[n_rows - 1]),
    )


def slice_period(frame: pl.DataFrame, *, timestamp_column: str, period: Period) -> pl.DataFrame:
    return frame.filter((pl.col(timestamp_column) >= period.start) & (pl.col(timestamp_column) <= period.end))


def chronological_blocks(
    frame: pl.DataFrame,
    *,
    timestamp_column: str,
    block_count: int,
) -> list[pl.DataFrame]:
    if block_count < 2:
        raise ValueError("block_count must be at least 2")
    ordered = frame.sort(timestamp_column)
    if ordered.height < block_count:
        raise ValueError("not enough rows to build chronological blocks")
    block_size = ordered.height // block_count
    blocks: list[pl.DataFrame] = []
    for index in range(block_count):
        start = index * block_size
        end = ordered.height if index == block_count - 1 else (index + 1) * block_size
        blocks.append(ordered.slice(start, end - start))
    return blocks
