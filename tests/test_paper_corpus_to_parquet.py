from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from paper_corpus_to_parquet import (  # noqa: E402
    iter_jsonl_records,
    record_to_row,
    write_parquet_shards,
)


def test_iter_jsonl_records_yields_dicts(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    records = list(iter_jsonl_records(jsonl))
    assert records == [{"a": 1}, {"a": 2}]


def test_record_to_row_computes_sha256_and_word_count() -> None:
    record = {
        "id": "paper_pdf:foo.pdf:0",
        "source_type": "paper_pdf",
        "source_path": "foo.pdf",
        "chunk_index": 0,
        "text": "hello world test",
    }
    row = record_to_row(record)
    assert row["id"] == "paper_pdf:foo.pdf:0"
    assert row["source_type"] == "paper_pdf"
    assert row["n_words"] == 3
    assert row["sha256"] == hashlib.sha256(b"hello world test").hexdigest()


def test_write_parquet_shards_writes_at_least_one_shard(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"id-{i}",
            "source_type": "paper_pdf",
            "source_path": "x.pdf",
            "chunk_index": i,
            "text": "x " * 200,
            "sha256": hashlib.sha256((str(i) * 16).encode()).hexdigest(),
            "n_words": 200,
        }
        for i in range(50)
    ]
    out_dir = tmp_path / "out"
    written = write_parquet_shards(rows, out_dir, shard_target_mb=1)
    assert len(written) >= 1
    df = pl.concat([pl.read_parquet(path) for path in written])
    assert df.height == 50
    assert {"id", "source_type", "source_path", "chunk_index", "text", "sha256", "n_words"} <= set(df.columns)


def test_write_parquet_shards_empty_input_writes_nothing(tmp_path: Path) -> None:
    written = write_parquet_shards([], tmp_path / "out", shard_target_mb=256)
    assert written == []
