# Quant Research LLM Workspace

This repository builds a local-first quantitative trading research system whose primary interface is a large language model, but whose claims are accepted only through leakage-safe market benchmarks.

The target model role is not a generic chatbot. It is a local quantitative researcher that reads market data, retrieves papers, proposes structured signals, explains why a signal may work, and then lets a Jane Street-style benchmark decide whether the signal has value.

## Operating Model

The system is split into three layers:

1. **Market layer**: OHLCV, order-book, return, volatility, depth, imbalance, and Jane Street tabular features.
2. **LLM research layer**: local GGUF inference, paper retrieval, structured signal generation, hypothesis critique, and mentoring-style explanations.
3. **Validation layer**: time-ordered folds, weighted zero-mean R2, leakage checks, baselines, and generated reports.

The LLM may be first in the workflow, but it is never trusted without validation. A signal becomes useful only after it improves an out-of-sample metric against simple baselines.

## Local Constraints

- No paid APIs.
- No cloud training jobs.
- No real-money broker integration.
- Free downloads from Kaggle, Hugging Face, arXiv, and open sources are allowed.
- Total local artifacts must stay under `150 GB`.
- 20B-class LLMs are used through quantized local inference, not full local training.

The primary 20B-class target is a quantized GGUF model such as `bartowski/Mistral-Small-Instruct-2409-GGUF`. The installed `TheBloke/finance-LLM-13B-GGUF` remains the fallback model when the 22B runtime is unavailable.

## Current Layout

```text
configs/                  stack, paths, budgets, model/runtime settings
manifests/                Hugging Face, Kaggle, and paper manifests
scripts/                  download and preparation entrypoints
src/quant_research_stack/ reusable implementation modules
tests/                    unit tests and fixture-level validation
data/raw/                 ignored downloaded source data
data/processed/           ignored train-ready parquet/jsonl outputs
models/huggingface/       ignored local model snapshots and GGUFs
reports/                  generated plans, inventories, metrics, and benchmark reports
```

## Recreate Data And Models

Dry-run before any large operation:

```bash
uv run python scripts/report_artifact_budget.py
uv run python scripts/download_hf_artifacts.py --dry-run --sort size
uv run python scripts/download_kaggle_artifacts.py --dry-run
```

Prepare existing market and research corpora:

```bash
uv run python scripts/prepare_market_data.py
uv run python scripts/prepare_orderbook_data.py
uv run python scripts/prepare_research_corpus.py
```

Run the local Jane Street benchmark after the Kaggle dataset is present:

```bash
uv run python scripts/run_jane_street_benchmark.py --sample-rows 200000
```

## Benchmark Target

The first formal benchmark target is `jane-street-real-time-market-data-forecasting`. The local harness evaluates `responder_6` with weighted zero-mean R2 and time-ordered validation. The goal is to iteratively improve the score, not to claim rank without a reproducible local measurement.

## Safety Standard

This repository is for research and competition-style benchmarking. It does not claim profitability. Any trading bot integration must come after repeatable out-of-sample validation, cost/slippage modeling, risk limits, and paper-trading simulation.

## Corpus Reproduction

The target on-disk corpus is approximately 150 GB. It splits across four buckets (see `configs/stack.yaml::bucket_budgets`):

| Bucket | Budget | Source |
|---|---|---|
| `hf_datasets` | 50 GB | `manifests/datasets.yaml` |
| `kaggle` | 40 GB | `manifests/kaggle.yaml` (requires `~/.kaggle/kaggle.json`) |
| `models` | 40 GB | `manifests/models.yaml` |
| `papers_and_derived` | 20 GB | `manifests/papers.yaml` + chunked JSONL + Parquet shards + Q&A pairs |

Full reproduction sequence:

```bash
uv sync --extra dev --extra llm

# Optional but required for gated models and Kaggle competitions:
huggingface-cli login
# Place Kaggle token at ~/.kaggle/kaggle.json (chmod 600), accept competition rules on Kaggle website.

# Papers and derived training formats
PYTHONPATH=src uv run python scripts/download_papers.py
PYTHONPATH=src uv run python scripts/prepare_research_corpus.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_parquet.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py

# Hugging Face datasets and models (priority order, size-capped)
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types dataset --max-gb 50
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --max-gb 40

# Kaggle competitions and datasets
PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py

# Final inventory and dedup report
PYTHONPATH=src uv run python scripts/dedupe_and_verify.py
```

The final report at `reports/corpus_inventory.json` records SHA256, size, bucket, and duplicate groups for every file across the four buckets.
