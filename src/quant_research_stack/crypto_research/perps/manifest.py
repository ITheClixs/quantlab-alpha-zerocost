from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_supported_frame(path: Path) -> pl.DataFrame | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path)
        if suffix in {".json", ".jsonl", ".ndjson"}:
            return pl.read_ndjson(path)
    except Exception:
        return None
    return None


def _file_payload(path: Path) -> dict[str, Any]:
    frame = _read_supported_frame(path)
    schema = {name: str(dtype) for name, dtype in frame.schema.items()} if frame is not None else {}
    row_count = int(frame.height) if frame is not None else None
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "row_count": row_count,
        "schema": schema,
    }


def build_dataset_manifest(
    *,
    dataset_id: str,
    source: str,
    paths: list[Path],
    symbols: list[str],
    timestamp_semantics: str,
    quality_label: str,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("paths must contain at least one file")
    files = [_file_payload(Path(path)) for path in paths]
    row_count = sum(int(file["row_count"] or 0) for file in files)
    return {
        "dataset_id": dataset_id,
        "source": source,
        "symbols": sorted({symbol.upper() for symbol in symbols}),
        "timestamp_semantics": timestamp_semantics,
        "quality_label": quality_label,
        "row_count": row_count,
        "files": files,
    }
