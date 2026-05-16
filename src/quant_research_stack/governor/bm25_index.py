from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from quant_research_stack.governor.corpus import CorpusIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Index:
    chunk_ids: tuple[str, ...]
    bm25: BM25Okapi

    def top_k(self, query: str, n: int) -> list[str]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if not scores.size:
            return []
        order = scores.argsort()[::-1][:n]
        return [self.chunk_ids[int(i)] for i in order]


def build_bm25_index(corpus: CorpusIndex) -> BM25Index:
    chunks = list(corpus)
    tokenized = [_tokenize(c.text) for c in chunks]
    return BM25Index(
        chunk_ids=tuple(c.id for c in chunks),
        bm25=BM25Okapi(tokenized),
    )


def save_bm25_index(index: BM25Index, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"chunk_ids": index.chunk_ids, "bm25": index.bm25}, handle)


def load_bm25_index(path: str | Path) -> BM25Index:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return BM25Index(chunk_ids=payload["chunk_ids"], bm25=payload["bm25"])
