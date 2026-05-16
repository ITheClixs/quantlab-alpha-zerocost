# Runbook: Governor index rebuild

## When to run
- Corpus content changed (new chunks added to `data/processed/research/parquet/`).
- `s2_govern.py` refuses to start with "stale index" (its check compares the corpus
  SHA in `models/governor/index_metadata.json` to the current Parquet SHA).
- After upgrading `FinLang/finance-embeddings-investopedia` or the cross-encoder model.

## Steps
1. Stop any running `s2_govern.py` daemon (SIGINT).
2. Run: `make governor-build-indexes`.
3. Wait ~1–2 min on the M4 (BM25 pickle + dense vectors + faiss index).
4. Verify: `cat models/governor/index_metadata.json` — `corpus_sha` matches the new
   Parquet, `bm25_path`, `dense_npy_path`, `faiss_path` all exist on disk.
5. Restart `s2_govern.py`.

## Disk footprint
- BM25 pickle: ~2 MB
- Dense .npy (1581 × 768 × float16): ~2.4 MB
- FAISS IndexFlatIP: ~5 MB
- Cross-encoder model: ~80 MB (downloaded once into `models/huggingface/`)

## Failure modes
- FinLang model missing: re-run `scripts/download_hf_artifacts.py --types model`.
- FAISS install missing: re-run `uv sync --extra dev --extra llm`.
