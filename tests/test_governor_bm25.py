from __future__ import annotations

from pathlib import Path

from quant_research_stack.governor.bm25_index import BM25Index, build_bm25_index, load_bm25_index, save_bm25_index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex


def _corpus() -> CorpusIndex:
    chunks = [
        Chunk(id="a", source_type="t", source_path="p", chunk_index=0,
              text="order flow imbalance equity prediction", sha256="aa", n_words=5),
        Chunk(id="b", source_type="t", source_path="p", chunk_index=1,
              text="mean reversion crypto microstructure tick", sha256="bb", n_words=5),
        Chunk(id="c", source_type="t", source_path="p", chunk_index=2,
              text="momentum trending equities stocks", sha256="cc", n_words=4),
    ]
    return CorpusIndex(chunks)


def test_build_bm25_index_returns_top_n_for_lexical_match() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("order flow imbalance", n=2)
    assert hits[0] == "a"
    assert len(hits) == 2


def test_top_k_returns_chunk_ids_only() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("microstructure tick", n=3)
    assert all(isinstance(h, str) for h in hits)
    assert hits[0] == "b"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = build_bm25_index(_corpus())
    save_bm25_index(idx, tmp_path / "bm25.pkl")
    loaded = load_bm25_index(tmp_path / "bm25.pkl")
    assert isinstance(loaded, BM25Index)
    assert loaded.top_k("momentum", n=1) == ["c"]


def test_top_k_n_larger_than_corpus_returns_all() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("anything", n=10)
    assert len(hits) == 3
