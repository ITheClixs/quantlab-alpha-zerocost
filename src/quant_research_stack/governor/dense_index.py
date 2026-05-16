from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import faiss


@dataclass(frozen=True)
class DenseIndex:
    chunk_ids: tuple[str, ...]
    index: faiss.IndexFlatIP
    vectors: NDArray[np.float32]

    def top_k(self, query: NDArray[np.float32], n: int) -> list[str]:
        q = np.atleast_2d(query.astype(np.float32))
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        q = q / norm
        n_capped = min(n, len(self.chunk_ids))
        _, idxs = self.index.search(q, n_capped)
        return [self.chunk_ids[int(i)] for i in idxs[0] if int(i) >= 0]


def build_dense_index_from_vectors(chunk_ids: tuple[str, ...], vectors: NDArray[np.float32]) -> DenseIndex:
    import faiss

    if vectors.ndim != 2 or vectors.shape[0] != len(chunk_ids):
        raise ValueError(f"vector shape {vectors.shape} does not match chunk_ids length {len(chunk_ids)}")
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return DenseIndex(chunk_ids=chunk_ids, index=index, vectors=vectors.astype(np.float32))


def save_dense_index(idx: DenseIndex, npy_path: str | Path, faiss_path: str | Path) -> None:
    import faiss

    Path(npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, idx.vectors.astype(np.float16))
    faiss.write_index(idx.index, str(faiss_path))


def load_dense_index(npy_path: str | Path, faiss_path: str | Path, *, chunk_ids: tuple[str, ...]) -> DenseIndex:
    import faiss

    vectors = np.load(npy_path).astype(np.float32)
    index = faiss.read_index(str(faiss_path))
    return DenseIndex(chunk_ids=chunk_ids, index=index, vectors=vectors)
