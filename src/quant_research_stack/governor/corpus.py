from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Chunk:
    id: str
    source_type: str
    source_path: str
    chunk_index: int
    text: str
    sha256: str
    n_words: int


class CorpusIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._by_id: dict[str, Chunk] = {c.id: c for c in chunks}
        joined = "\n".join(f"{c.id}|{c.sha256}" for c in sorted(chunks, key=lambda x: x.id))
        self._sha256 = hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._by_id

    def __getitem__(self, key: str) -> Chunk:
        return self._by_id[key]

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._by_id.values())

    @property
    def sha256(self) -> str:
        return self._sha256


def load_corpus(parquet_dir: str | Path) -> CorpusIndex:
    root = Path(parquet_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shards under {root}")
    df = pl.read_parquet(files)
    chunks = [
        Chunk(
            id=str(row["id"]),
            source_type=str(row["source_type"]),
            source_path=str(row["source_path"]),
            chunk_index=int(row["chunk_index"]),
            text=str(row["text"]),
            sha256=str(row["sha256"]),
            n_words=int(row["n_words"]),
        )
        for row in df.iter_rows(named=True)
    ]
    return CorpusIndex(chunks)
