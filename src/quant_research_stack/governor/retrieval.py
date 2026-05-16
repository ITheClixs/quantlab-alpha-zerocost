from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.governor.bm25_index import BM25Index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.dense_index import DenseIndex
from quant_research_stack.governor.reranker import RerankCandidate, Reranker


@dataclass(frozen=True)
class HybridRetriever:
    corpus: CorpusIndex
    bm25: BM25Index
    dense: DenseIndex
    reranker: Reranker

    def retrieve(self, query: str, *, bm25_n: int, dense_n: int, k: int, query_vector: NDArray[np.float32]) -> list[Chunk]:
        bm25_hits = self.bm25.top_k(query, n=bm25_n)
        dense_hits = self.dense.top_k(query_vector, n=dense_n)
        seen: set[str] = set()
        union_ids: list[str] = []
        for cid in bm25_hits + dense_hits:
            if cid in self.corpus and cid not in seen:
                union_ids.append(cid)
                seen.add(cid)
        candidates = [
            RerankCandidate(id=cid, text=self.corpus[cid].text) for cid in union_ids
        ]
        reranked = self.reranker.rerank(query, candidates)
        return [self.corpus[cand.id] for cand in reranked[:k]]
