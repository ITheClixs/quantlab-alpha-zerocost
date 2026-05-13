from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_research_stack.artifacts import read_yaml

console = Console()


def iter_jsonl_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text", "")
    return {
        "id": str(record["id"]),
        "source_type": str(record.get("source_type", "")),
        "source_path": str(record.get("source_path", "")),
        "chunk_index": int(record.get("chunk_index", 0)),
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_words": len(text.split()),
    }


def _row_bytes(row: dict[str, Any]) -> int:
    return sum(len(str(value).encode("utf-8")) for value in row.values())


def write_parquet_shards(rows: Iterable[dict[str, Any]], out_dir: str | Path, shard_target_mb: int) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_bytes = shard_target_mb * 1024 * 1024
    buffer: list[dict[str, Any]] = []
    buffer_bytes = 0
    shard_paths: list[Path] = []
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes, shard_index
        if not buffer:
            return
        df = pl.DataFrame(buffer)
        path = out / f"shard_{shard_index:05d}.parquet"
        df.write_parquet(path, compression="zstd")
        shard_paths.append(path)
        shard_index += 1
        buffer = []
        buffer_bytes = 0

    for row in rows:
        buffer.append(row)
        buffer_bytes += _row_bytes(row)
        if buffer_bytes >= target_bytes:
            flush()
    flush()
    return shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert chunked research JSONL to Parquet shards.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shard-target-mb", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    corpus_cfg = config.get("research_corpus", {})
    input_path = Path(args.input or Path(paths["processed_research_root"]) / "research_corpus.jsonl")
    output_dir = Path(args.output_dir or Path(paths["processed_research_root"]) / "parquet")
    shard_target_mb = int(args.shard_target_mb or corpus_cfg.get("parquet_shard_target_mb", 256))

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}. Run scripts/prepare_research_corpus.py first.[/red]")
        return 2

    rows = (record_to_row(rec) for rec in iter_jsonl_records(input_path))
    shards = write_parquet_shards(rows, output_dir, shard_target_mb)
    console.print(f"Wrote {len(shards)} parquet shards to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
