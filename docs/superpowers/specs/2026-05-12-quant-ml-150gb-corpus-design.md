# QuantLab 150 GB Corpus — Design

Date: 2026-05-12
Project: QuantLab (`/Users/dmr/MachineLearning`)
Target host: MacBook Air M4, 24 GB unified memory, macOS, ~617 GB free disk

## 1. Goal

Build a local, reproducible quantitative-finance and machine-learning corpus of approximately 150 GB on disk. The corpus is composed of:

- Hugging Face datasets (finance + general ML/NLP/DL)
- Kaggle competition data and Kaggle datasets
- Hugging Face models (time-series forecasters, finance embeddings, sentiment, small-to-30B-class GGUF LLMs)
- Open research papers, plus derivative trainable formats (chunked JSONL, sharded Parquet, instruction-format Q&A)

The corpus must contain no semantic duplicates and no irrelevant entries. Every entry must carry a recorded `purpose`, `group`, and topic tag set. The resulting workspace must be reusable as a training corpus in subsequent sessions.

## 2. Non-goals

- No live trading, no broker connections.
- No training of 12B–14B (or larger) models from scratch.
- No full fine-tuning of large LLMs locally.
- No random train-test split for financial time-series data.
- No future data in features at any point downstream of this corpus.
- No git-tracking of raw data — only manifests, scripts, configs, and reports are versioned.

## 3. Bucket allocation (~150 GB total)

| Bucket | Budget GB | Contents |
|---|---|---|
| HF datasets | ~50 | Existing quant manifest + expanded NLP/DL/code/reasoning/general-ML coverage |
| Kaggle | ~40 | JPX, Optiver (both), G-Research Crypto, Jane Street (both), Two-Sigma News, Winton, 9000-tickers, US OHLCV, and selected datasets |
| Models | ~40 | Existing 7 HF models + small NLP + small code/reasoning + 1–2 GGUF 14B–30B Q4_K_M variants + FinGPT LoRA + MLX-community 4-bit Apple Silicon variants |
| Papers + converted | ~20 | ~80–120 open arXiv PDFs (~5 GB) → JSONL chunks (~5 GB) → sharded Parquet (~5 GB) → instruction-format Q&A pairs (~5 GB) |

Bucket totals are soft caps enforced by the size-aware downloader. If a bucket underflows, the downloader rebalances overflow into adjacent buckets up to a 165 GB hard ceiling.

## 4. Topic taxonomy

Every manifest entry declares one or more topics from a fixed set:

```text
microstructure      # order book, OFI, micro-price
orderflow           # trade signs, queue dynamics
crypto              # BTCUSDT and other crypto OHLCV / orderbook / trades
equity              # S&P, NASDAQ, JPX, intraday equity
options             # implied volatility, options chains
sentiment           # FinBERT-style sentiment corpora
news                # FNSPID, Twitter financial news, 10-Ks, earnings transcripts
instruction         # instruction-tuned datasets, quant-finance Q&A
code                # code generation / programming corpora
reasoning           # math, multi-step reasoning datasets
vision              # CIFAR, ImageNet subsets (deep-learning baseline coverage)
general_nlp         # WikiText, OpenWebText subsets, IMDB
time_series         # foundation time-series corpora and forecasting benchmarks
llm_agent           # LLM trading agents and simulated markets
gpu_parallel        # GPU acceleration, parallel computing, CUDA papers/datasets
```

Each topic is capped at three primary sources; further candidates must carry a written `purpose` field justifying their addition.

## 5. Repository layout (delta from current)

```text
configs/
  stack.yaml                                # bump max_total_gb 100 → 150
manifests/
  datasets.yaml                             # +20–30 HF entries spanning the topics above
  models.yaml                               # +5–8 entries up to 30B Q4 GGUF
  papers.yaml                               # +50–80 paper entries beyond current ~22
  kaggle.yaml                               # NEW
scripts/
  download_hf_artifacts.py                  # existing, unchanged behaviorally
  download_papers.py                        # existing, unchanged behaviorally
  prepare_research_corpus.py                # existing, unchanged behaviorally
  prepare_market_data.py                    # existing, unchanged behaviorally
  download_kaggle_artifacts.py              # NEW — budget-aware Kaggle downloader
  paper_corpus_to_parquet.py                # NEW — JSONL → sharded Parquet
  paper_corpus_to_instructions.py           # NEW — chunk → instruction-format Q&A
  dedupe_and_verify.py                      # NEW — content-hash dedup + inventory
src/quant_research_stack/
  artifacts.py                              # existing utilities
  kaggle_artifacts.py                       # NEW — Kaggle manifest loader, size estimator
pyproject.toml                              # +kaggle, +duckdb
```

No files are deleted. No previously-working behavior is altered.

## 6. New code: contracts

### 6.1 `manifests/kaggle.yaml`

Mirrors the structure of `datasets.yaml`:

```yaml
schema_version: 1
description: Candidate Kaggle competitions and datasets for the quant corpus.
defaults:
  resource_type: competition       # or 'dataset'
  enabled: true
items:
  - id: jpx-tokyo-stock-exchange-prediction
    resource_type: competition
    group: equity_jpx
    priority: 10
    topics: [equity]
    license_hint: kaggle_competition
    purpose: "Cross-sectional equity ranking benchmark."
  - id: jakewright/9000-tickers-of-stock-market-data-full-history
    resource_type: dataset
    group: equity_ohlcv_us
    priority: 200
    topics: [equity]
    license_hint: cc0
    purpose: "Wide US equity OHLCV history."
  # ... and so on
```

Each entry carries an `expected_max_gb` cap; the downloader skips any entry that exceeds its cap after estimating the remote zip size.

### 6.2 `scripts/download_kaggle_artifacts.py`

Same surface as `download_hf_artifacts.py`:

```text
--config configs/stack.yaml
--manifest manifests/kaggle.yaml
--dry-run
--max-gb 40
--sort {size,priority}
--report reports/kaggle_download_plan.json
```

Behavior:

1. Read manifest; for each enabled item, estimate size by calling `kaggle competitions list --search` or `kaggle datasets metadata`, falling back to a HEAD on the download URL.
2. Sort by size or priority; mark `decision in {download, skip_present, skip_budget, skip_unknown_size, skip_rules_not_accepted}`.
3. Write the plan to JSON.
4. If not dry-run: call `kaggle competitions download -c <id>` or `kaggle datasets download -d <id>`, unzip with `unzip -n`, and record the final size.
5. On `403` from a competition, surface the message to accept the rules at the competition URL and continue with the next item.

### 6.3 `scripts/paper_corpus_to_parquet.py`

Reads `data/processed/research/research_corpus.jsonl` (output of existing `prepare_research_corpus.py`) and writes Parquet shards under `data/processed/research/parquet/` partitioned by `source_type` and shard index, target shard size 256 MB, compression `zstd`. Schema:

```text
id: string (PK)
source_type: string
source_path: string
chunk_index: int32
text: string
sha256: string
n_words: int32
```

### 6.4 `scripts/paper_corpus_to_instructions.py`

Generates instruction-format Q&A from each chunk using a small local LLM. The default model is `Qwen/Qwen2.5-0.5B-Instruct`, which is added to `manifests/models.yaml` as part of step 11.2 (model manifest extension) and therefore downloaded before this script runs. If `Qwen2.5-0.5B-Instruct` is unavailable for any reason, the script falls back to `roneneldan/TinyStories-33M` for extraction-style outputs and logs a degraded-mode warning. The script never starts without at least one of these two models present locally.

Output: JSONL at `data/processed/research/instructions.jsonl`, schema:

```text
id: string
source_chunk_id: string
prompt: string         # e.g. "Explain the role of order-flow imbalance in mid-price prediction."
response: string       # generated from chunk; capped at 200 words
model_id: string
generated_at: ISO-8601 string
```

Constraints:

- MPS device when available, batch_size = 1, max_new_tokens = 256.
- Skip chunks shorter than `min_chunk_words` from `configs/stack.yaml`.
- Generate at most 3 Q&A pairs per chunk.
- The script is idempotent: it skips chunks whose `source_chunk_id` already has a record in the output file.

### 6.5 `scripts/dedupe_and_verify.py`

Walks `data/raw/` and `data/processed/`, computes file SHA256 for every artifact, and emits `reports/corpus_inventory.json`:

```text
{
  "generated_at": "...",
  "total_size_bytes": ...,
  "by_bucket": {"hf_datasets": ..., "kaggle": ..., "models": ..., "papers": ...},
  "items": [
    {"path": "...", "size_bytes": ..., "sha256": "...", "bucket": "...", "topic_tags": ["..."]},
    ...
  ],
  "duplicates": [{"sha256": "...", "paths": ["...", "..."]}]
}
```

For each duplicate group, the first occurrence wins by manifest priority; later occurrences are emitted to `reports/duplicates_to_remove.json` rather than auto-deleted. The user runs an explicit cleanup pass.

### 6.6 `src/quant_research_stack/kaggle_artifacts.py`

Pure functions, no I/O at import time:

```python
@dataclass(frozen=True)
class KaggleItem:
    id: str
    resource_type: str  # "competition" | "dataset"
    group: str
    priority: int
    topics: tuple[str, ...]
    purpose: str
    license_hint: str | None
    expected_max_gb: float | None
    enabled: bool

def load_kaggle_items(manifest_path: Path) -> list[KaggleItem]: ...
def estimate_remote_size(item: KaggleItem) -> tuple[int | None, str | None]: ...
def local_path_for(item: KaggleItem, root: Path) -> Path: ...
```

## 7. Budget enforcement and rebalance

- `configs/stack.yaml: artifact_budget.max_total_gb: 150`
- `reserve_gb_for_processed_outputs: 12` (room for paper Parquet + instructions JSONL)
- Each bucket downloader respects its own bucket cap from `configs/stack.yaml: bucket_budgets:`
- If a bucket finishes underfilled by more than 5 GB, the next downloader gets a one-time top-up; never exceed 165 GB hard ceiling.

`configs/stack.yaml` gains:

```yaml
artifact_budget:
  max_total_gb: 150
  reserve_gb_for_processed_outputs: 12
  default_sort: size
  hard_ceiling_gb: 165

bucket_budgets:
  hf_datasets: 50
  kaggle: 40
  models: 40
  papers_and_derived: 20
```

## 8. No-overlap strategy

Three layers of deduplication, applied in this order:

1. **Manifest-level**: each `group` allows at most one primary item plus at most two alternates with `purpose` justification. Reviewed at the time of writing the manifest.
2. **Topic-level**: each topic capped at three primary sources. Enforced by `dedupe_and_verify.py` which emits a warning, not an error.
3. **Content-hash**: file SHA256 across `data/`. Duplicates surfaced for manual review.

## 9. Authentication and gated assets

- Kaggle: user provides `~/.kaggle/kaggle.json` before the Kaggle step runs. The downloader exits early with a clear message if it is missing.
- Hugging Face: user runs `huggingface-cli login` before the HF step runs. Public artifacts work without login but gated models (Llama-3 family, some 14B+ GGUFs) require a token.
- For competitions requiring rule acceptance, the downloader reports the competition URL and continues.

## 10. Apple Silicon and 24 GB RAM safety

- Streaming reads for HF datasets larger than 5 GB.
- Q&A generation uses Qwen2.5-0.5B with batch_size = 1 on MPS.
- 30B GGUF artifacts are downloaded only; not loaded into RAM during this session.
- At most one LLM is held in memory at any time during conversion steps.
- Long-running downloads use `resume_download=True` so an interrupted run continues without re-downloading partial files.

## 11. Execution order (operator-facing)

1. Bump `configs/stack.yaml` to 150 GB and add `bucket_budgets`.
2. Extend `manifests/datasets.yaml`, `manifests/models.yaml`, `manifests/papers.yaml`.
3. Create `manifests/kaggle.yaml`.
4. Implement `src/quant_research_stack/kaggle_artifacts.py`.
5. Implement `scripts/download_kaggle_artifacts.py`.
6. Implement `scripts/paper_corpus_to_parquet.py`.
7. Implement `scripts/paper_corpus_to_instructions.py`.
8. Implement `scripts/dedupe_and_verify.py`.
9. Add `kaggle`, `duckdb` to `pyproject.toml` dependencies; refresh lock.
10. Run dry-runs for HF, Kaggle, papers; commit the three plan reports.
11. Wait for the user to (a) provide `~/.kaggle/kaggle.json` and (b) run `huggingface-cli login`.
12. Execute downloads in this order: papers → HF datasets → models (smallest first) → Kaggle → large GGUFs.
13. Run paper → JSONL → Parquet → instructions conversion.
14. Run `dedupe_and_verify.py`; commit `reports/corpus_inventory.json`.
15. Commit manifests, scripts, configs, reports. Data directories remain git-ignored.

## 12. Testing

- Unit tests for `kaggle_artifacts.load_kaggle_items` (valid manifest, missing field, disabled item).
- Unit tests for chunking math in `paper_corpus_to_parquet.py` shard-size logic.
- Unit tests for SHA256 grouping in `dedupe_and_verify.py`.
- Smoke test: run all dry-run plans end-to-end on an empty `data/` to ensure scripts produce well-formed reports without network errors.
- No integration tests hit the network. Network-bound steps are excluded from `pytest` and run manually.

`PYTHONPATH=src pytest -q` and `ruff check src scripts` must both pass before declaring the task complete.

## 13. Completion criteria

Done means:

- `du -sh /Users/dmr/MachineLearning/data /Users/dmr/MachineLearning/models` reports a combined size in the range 140–160 GB.
- `reports/corpus_inventory.json` exists, with no duplicate-hash groups containing more than one item.
- All four bucket totals are within ±5 GB of their target.
- `pytest -q` and `ruff check` pass.
- Manifests, scripts, configs, and reports are committed; data directories are not.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle competition refuses download due to unaccepted rules | Downloader surfaces the URL and skips with `skip_rules_not_accepted`; no fatal exit. |
| HF model size estimate is unknown for a repo | Item is marked `skip_unknown_size` and surfaced for manual decision; budget is not overrun. |
| Q&A generation OOMs on M4 | batch_size 1, max_new_tokens 256, MPS fallback to CPU on error, skip-on-failure. |
| Disk fills past 165 GB hard ceiling | Hard ceiling enforced inside the downloader; remaining items skipped with `skip_hard_ceiling`. |
| arXiv blocks scripted downloads | Existing downloader is single-threaded and uses default UA; if blocked, fall back to manual seeding from a pre-cached mirror. |
| Future quant-finance rules in CLAUDE.md violated by derivative data | Paper Q&A pairs are flagged as instructional text only, never used as trading signals. |
