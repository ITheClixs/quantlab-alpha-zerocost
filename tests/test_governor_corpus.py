from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_research_stack.governor.corpus import Chunk, load_corpus


def _write_fixture(tmp_path: Path) -> Path:
    df = pl.DataFrame({
        "id": ["paper_pdf:a:0", "paper_pdf:a:1", "paper_pdf:b:0"],
        "source_type": ["paper_pdf"] * 3,
        "source_path": ["a.pdf", "a.pdf", "b.pdf"],
        "chunk_index": [0, 1, 0],
        "text": ["alpha text one", "alpha text two", "beta text"],
        "sha256": ["aa", "ab", "ba"],
        "n_words": [3, 3, 2],
    })
    out = tmp_path / "shard_00000.parquet"
    df.write_parquet(out)
    return tmp_path


def test_load_corpus_reads_parquet_shards(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    assert len(corpus) == 3


def test_corpus_id_lookup_returns_chunk(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    chunk = corpus["paper_pdf:a:1"]
    assert isinstance(chunk, Chunk)
    assert chunk.text == "alpha text two"
    assert chunk.source_path == "a.pdf"


def test_corpus_membership(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    assert "paper_pdf:a:0" in corpus
    assert "missing-id" not in corpus


def test_corpus_iter_yields_all_chunks(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    ids = sorted(c.id for c in corpus)
    assert ids == ["paper_pdf:a:0", "paper_pdf:a:1", "paper_pdf:b:0"]


def test_corpus_sha_is_stable(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    sha_a = load_corpus(parquet_dir).sha256
    sha_b = load_corpus(parquet_dir).sha256
    assert sha_a == sha_b
    assert len(sha_a) == 64
