from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.governor.dense_index import (
    DenseIndex,
    build_dense_index_from_vectors,
    load_dense_index,
    save_dense_index,
)


def _vectors(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_build_dense_index_returns_top_k() -> None:
    chunk_ids = ("a", "b", "c", "d", "e")
    vectors = _vectors()
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    query = vectors[2]
    hits = idx.top_k(query, n=3)
    assert hits[0] == "c"
    assert len(hits) == 3


def test_dense_index_returns_unique_ids() -> None:
    idx = build_dense_index_from_vectors(("x", "y", "z"), _vectors()[:3])
    hits = idx.top_k(_vectors()[0], n=10)
    assert len(set(hits)) == len(hits)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    chunk_ids = ("a", "b", "c", "d", "e")
    vectors = _vectors()
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    npy_path = tmp_path / "dense.npy"
    faiss_path = tmp_path / "dense.faiss"
    save_dense_index(idx, npy_path, faiss_path)
    loaded = load_dense_index(npy_path, faiss_path, chunk_ids=chunk_ids)
    assert isinstance(loaded, DenseIndex)
    assert loaded.top_k(vectors[1], n=1) == ["b"]


def test_query_unit_norm_required() -> None:
    chunk_ids = ("a",)
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    # FlatIP requires unit-norm queries for cosine equivalence; test we accept any vector
    hits = idx.top_k(np.array([2.0, 0.0], dtype=np.float32), n=1)
    assert hits == ["a"]
