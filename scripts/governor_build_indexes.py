from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

from quant_research_stack.governor.bm25_index import build_bm25_index, save_bm25_index
from quant_research_stack.governor.corpus import load_corpus
from quant_research_stack.governor.dense_index import build_dense_index_from_vectors, save_dense_index

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25 + dense + reranker indexes for the governor.")
    p.add_argument("--config", default="configs/governor.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    parquet_dir = Path(cfg["corpus"]["parquet_dir"])
    index_dir = Path(cfg["retrieval"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Loading corpus from {parquet_dir}")
    corpus = load_corpus(parquet_dir)
    console.print(f"Corpus has {len(corpus)} chunks; sha256 prefix={corpus.sha256[:16]}")

    bm25 = build_bm25_index(corpus)
    save_bm25_index(bm25, index_dir / "bm25_index.pkl")
    console.print(f"Saved BM25 index to {index_dir / 'bm25_index.pkl'}")

    embedding_dir = Path(cfg["retrieval"]["embedding_model_dir"])
    console.print(f"Encoding {len(corpus)} chunks with {embedding_dir}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(embedding_dir))
    chunk_ids = tuple(c.id for c in corpus)
    texts = [c.text for c in corpus]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    dense = build_dense_index_from_vectors(chunk_ids, vecs)
    save_dense_index(dense, index_dir / "dense_index.npy", index_dir / "dense_index.faiss")
    console.print(f"Saved dense index ({vecs.shape}) to {index_dir}")

    metadata = {
        "corpus_sha": corpus.sha256,
        "n_chunks": len(corpus),
        "embedding_model_id": cfg["retrieval"]["embedding_model_id"],
        "vector_dim": int(vecs.shape[1]),
    }
    (index_dir / "index_metadata.json").write_text(json.dumps(metadata, indent=2))
    console.print(f"Wrote {index_dir / 'index_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
