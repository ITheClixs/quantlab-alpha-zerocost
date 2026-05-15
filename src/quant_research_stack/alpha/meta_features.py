from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def hash_input_dataframe(df: pl.DataFrame) -> str:
    payload = df.write_ipc(file=None).getvalue()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MetaFeatureCache:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return Path(self.root) / f"{safe}.npy"

    def get(self, key: str) -> NDArray[np.float64] | None:
        p = self._path(key)
        if not p.exists():
            return None
        return np.load(p)

    def put(self, key: str, arr: NDArray[np.float64]) -> None:
        np.save(self._path(key), arr)


def finbert_logits_cached(
    texts: list[str],
    cache: MetaFeatureCache,
    cache_key: str,
    runner: Callable[[list[str]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    out = runner(texts)
    cache.put(cache_key, out)
    return out
