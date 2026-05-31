from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class RerankCandidate:
    id: str
    text: str


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]: ...


class StubReranker:
    """Deterministic reranker for tests. Scores by token overlap (no model)."""

    def rerank(self, query: str, candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
        q_tokens = set(query.lower().split())
        scored = [
            (sum(1 for t in cand.text.lower().split() if t in q_tokens), cand)
            for cand in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cand for _, cand in scored]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder.

    Loaded lazily so unit tests can use StubReranker without downloading a model.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self._model_dir = Path(model_dir)
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(str(self._model_dir))

    def rerank(self, query: str, candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
        self._load()
        assert self._model is not None
        cand_list = list(candidates)
        if not cand_list:
            return []
        pairs = [(query, cand.text) for cand in cand_list]
        scores = self._model.predict(pairs)
        scored = list(zip(scores, cand_list, strict=True))
        scored.sort(key=lambda pair: float(pair[0]), reverse=True)
        return [cand for _, cand in scored]
