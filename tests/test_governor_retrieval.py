from __future__ import annotations

import numpy as np

from quant_research_stack.governor.bm25_index import build_bm25_index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.dense_index import build_dense_index_from_vectors
from quant_research_stack.governor.reranker import StubReranker
from quant_research_stack.governor.retrieval import HybridRetriever


def _setup() -> tuple[HybridRetriever, CorpusIndex]:
    chunks = [
        Chunk(id="a", source_type="t", source_path="p", chunk_index=0, text="order flow imbalance equity", sha256="x", n_words=4),
        Chunk(id="b", source_type="t", source_path="p", chunk_index=1, text="mean reversion crypto micro", sha256="y", n_words=4),
        Chunk(id="c", source_type="t", source_path="p", chunk_index=2, text="momentum trending stocks", sha256="z", n_words=3),
    ]
    corpus = CorpusIndex(chunks)
    bm25 = build_bm25_index(corpus)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dense = build_dense_index_from_vectors(("a", "b", "c"), vecs)
    retriever = HybridRetriever(corpus=corpus, bm25=bm25, dense=dense, reranker=StubReranker())
    return retriever, corpus


def test_retrieve_returns_at_most_k() -> None:
    retriever, _ = _setup()
    out = retriever.retrieve("order flow", bm25_n=3, dense_n=3, k=2, query_vector=np.zeros(8, dtype=np.float32))
    assert len(out) <= 2


def test_retrieve_returns_unique_chunks() -> None:
    retriever, _ = _setup()
    out = retriever.retrieve("order flow imbalance", bm25_n=3, dense_n=3, k=3, query_vector=np.zeros(8, dtype=np.float32))
    assert len({c.id for c in out}) == len(out)


def test_retrieve_returns_chunk_dataclass() -> None:
    retriever, corpus = _setup()
    out = retriever.retrieve("momentum trending", bm25_n=3, dense_n=3, k=1, query_vector=np.zeros(8, dtype=np.float32))
    assert out
    assert out[0].id in corpus
