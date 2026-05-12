# QuantLab 150 GB Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 150 GB on-disk corpus of HF datasets, Kaggle data, HF models (up to 30B Q4 GGUF), open research papers, and paper-derived trainable formats inside `/Users/dmr/MachineLearning`, with no semantic duplicates and a full inventory report.

**Architecture:** Extend three existing manifests, add one new Kaggle manifest, add four new scripts and one new module. All downloaders share the same budget-aware pattern as the existing `download_hf_artifacts.py`. Paper conversion produces JSONL → Parquet shards → instruction-format Q&A. A final dedup pass produces `reports/corpus_inventory.json`.

**Tech Stack:** Python 3.11, `huggingface_hub`, `kaggle`, `polars`, `pyarrow`, `pypdf`, `transformers`, `torch` (MPS), `pyyaml`, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-05-12-quant-ml-150gb-corpus-design.md`

---

## File Structure

**New files:**

```text
manifests/kaggle.yaml
src/quant_research_stack/kaggle_artifacts.py
scripts/download_kaggle_artifacts.py
scripts/paper_corpus_to_parquet.py
scripts/paper_corpus_to_instructions.py
scripts/dedupe_and_verify.py
tests/__init__.py
tests/test_kaggle_artifacts.py
tests/test_paper_corpus_to_parquet.py
tests/test_dedupe_and_verify.py
tests/test_manifests_well_formed.py
```

**Modified files:**

```text
configs/stack.yaml                # budget bump + bucket_budgets
manifests/datasets.yaml           # extend with NLP/DL/code/reasoning entries
manifests/models.yaml             # extend with small instruct + 14–30B GGUFs
manifests/papers.yaml             # extend with ~50–80 new entries
pyproject.toml                    # add kaggle, duckdb, pytest deps
```

**Existing modules reused without modification:**

```text
src/quant_research_stack/artifacts.py   # read_yaml, write_json, ManifestItem, etc.
scripts/download_hf_artifacts.py
scripts/download_papers.py
scripts/prepare_research_corpus.py
```

---

## Task 1: Bump stack budget to 150 GB and add bucket budgets

**Files:**
- Modify: `configs/stack.yaml`

- [ ] **Step 1: Update budget and add bucket_budgets section**

Replace the `artifact_budget` block and add a `bucket_budgets` block. Final file should read:

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

paths:
  raw_hf_root: data/raw/huggingface
  raw_paper_root: data/raw/papers
  raw_kaggle_root: data/raw/kaggle
  processed_market_root: data/processed/market
  processed_research_root: data/processed/research
  model_root: models/huggingface
  reports_root: reports

market_preparation:
  horizons:
    - 1
    - 5
    - 15
    - 60
  volatility_windows:
    - 5
    - 20
    - 60
  triple_barrier:
    profit_take_return: 0.002
    stop_loss_return: -0.002
    max_horizon: 60

research_corpus:
  chunk_words: 420
  chunk_overlap_words: 80
  min_chunk_words: 80
  parquet_shard_target_mb: 256
  instruction_max_per_chunk: 3
  instruction_max_new_tokens: 256
  instruction_primary_model: Qwen/Qwen2.5-0.5B-Instruct
  instruction_fallback_model: roneneldan/TinyStories-33M
```

- [ ] **Step 2: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('configs/stack.yaml'))"`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add configs/stack.yaml
git commit -m "feat: bump artifact budget to 150 GB with bucket caps"
```

---

## Task 2: Add pytest + ruff dev deps and kaggle/duckdb runtime deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `pyproject.toml` dependencies and add a dev group**

Final `pyproject.toml`:

```toml
[project]
name = "quant-research-stack"
version = "0.1.0"
description = "Local quantitative finance dataset, model, and paper-corpus workspace."
requires-python = ">=3.11"
dependencies = [
    "datasets>=3.0.0",
    "duckdb>=1.0.0",
    "huggingface-hub>=0.30.0",
    "kaggle>=1.6.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "polars>=1.6.0",
    "pyarrow>=16.0.0",
    "pypdf>=4.3.0",
    "pyyaml>=6.0.1",
    "requests>=2.32.0",
    "rich>=13.7.0",
    "scikit-learn>=1.4.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.5.0",
]
llm = [
    "torch>=2.3.0",
    "transformers>=4.42.0",
    "accelerate>=0.30.0",
]

[tool.uv]
package = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]
```

- [ ] **Step 2: Install the new dev deps**

Run: `cd /Users/dmr/MachineLearning && uv sync --extra dev`
Expected: kaggle, duckdb, pytest, ruff installed; exit code 0.

- [ ] **Step 3: Verify ruff and pytest work**

Run: `cd /Users/dmr/MachineLearning && uv run ruff --version && uv run pytest --version`
Expected: both print version strings.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add kaggle, duckdb runtime deps and pytest, ruff dev deps"
```

---

## Task 3: Create `tests/__init__.py` and a manifest well-formedness test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_manifests_well_formed.py`

- [ ] **Step 1: Create empty package marker**

Create `tests/__init__.py` with a single blank line.

- [ ] **Step 2: Write the failing test**

Create `tests/test_manifests_well_formed.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILES = [
    REPO_ROOT / "manifests" / "datasets.yaml",
    REPO_ROOT / "manifests" / "models.yaml",
    REPO_ROOT / "manifests" / "papers.yaml",
    REPO_ROOT / "manifests" / "kaggle.yaml",
]


@pytest.mark.parametrize("manifest_path", MANIFEST_FILES, ids=lambda p: p.name)
def test_manifest_loads(manifest_path: Path) -> None:
    assert manifest_path.exists(), f"missing manifest: {manifest_path}"
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1


@pytest.mark.parametrize(
    "manifest_path,key",
    [
        (REPO_ROOT / "manifests" / "datasets.yaml", "datasets"),
        (REPO_ROOT / "manifests" / "models.yaml", "models"),
        (REPO_ROOT / "manifests" / "papers.yaml", "papers"),
        (REPO_ROOT / "manifests" / "kaggle.yaml", "items"),
    ],
    ids=lambda x: str(x),
)
def test_manifest_has_entries(manifest_path: Path, key: str) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    items = data.get(key) or []
    assert isinstance(items, list)
    assert len(items) > 0


def test_no_duplicate_ids_within_manifest() -> None:
    for manifest_path, key in [
        (REPO_ROOT / "manifests" / "datasets.yaml", "datasets"),
        (REPO_ROOT / "manifests" / "models.yaml", "models"),
        (REPO_ROOT / "manifests" / "kaggle.yaml", "items"),
    ]:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        ids = [item["id"] for item in data.get(key, []) if "id" in item]
        assert len(ids) == len(set(ids)), f"duplicate ids in {manifest_path}"
```

- [ ] **Step 3: Run the test, expect failure on kaggle.yaml**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_manifests_well_formed.py -v`
Expected: failures for `kaggle.yaml` (file missing). Tests for `datasets.yaml`, `models.yaml`, `papers.yaml` should pass.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_manifests_well_formed.py
git commit -m "test: add manifest well-formedness tests (kaggle.yaml RED)"
```

---

## Task 4: Create `manifests/kaggle.yaml`

**Files:**
- Create: `manifests/kaggle.yaml`

- [ ] **Step 1: Write the manifest**

Create `manifests/kaggle.yaml`:

```yaml
schema_version: 1
description: >
  Kaggle competitions and datasets for the QuantLab 150 GB corpus. The downloader
  estimates sizes via kaggle CLI metadata where possible and respects the kaggle
  bucket budget from configs/stack.yaml.

defaults:
  resource_type: competition
  enabled: true
  expected_max_gb: 12.0

items:
  - id: jpx-tokyo-stock-exchange-prediction
    resource_type: competition
    group: equity_jpx
    priority: 10
    topics: [equity, time_series]
    license_hint: kaggle_competition
    expected_max_gb: 2.0
    purpose: "Cross-sectional equity ranking benchmark used as first milestone in CLAUDE.md."

  - id: stock-market-signal-predict-next-day-returns
    resource_type: competition
    group: equity_signal
    priority: 20
    topics: [equity]
    license_hint: kaggle_competition
    expected_max_gb: 1.0
    purpose: "Compact next-day-return supervised problem for baseline plumbing."

  - id: g-research-crypto-forecasting
    resource_type: competition
    group: crypto_forecasting
    priority: 30
    topics: [crypto, time_series]
    license_hint: kaggle_competition
    expected_max_gb: 6.0
    purpose: "Minute-bar crypto forecasting with cross-asset structure."

  - id: optiver-trading-at-the-close
    resource_type: competition
    group: microstructure_close
    priority: 40
    topics: [microstructure, equity]
    license_hint: kaggle_competition
    expected_max_gb: 4.0
    purpose: "Closing auction microstructure features."

  - id: optiver-realized-volatility-prediction
    resource_type: competition
    group: realized_volatility
    priority: 50
    topics: [microstructure, options]
    license_hint: kaggle_competition
    expected_max_gb: 5.0
    purpose: "Realized volatility from book snapshots."

  - id: jane-street-market-prediction
    resource_type: competition
    group: jane_street_classic
    priority: 60
    topics: [equity, microstructure]
    license_hint: kaggle_competition
    expected_max_gb: 6.0
    purpose: "Utility-based decision modeling benchmark."

  - id: jane-street-real-time-market-data-forecasting
    resource_type: competition
    group: jane_street_realtime
    priority: 70
    topics: [equity, microstructure, time_series]
    license_hint: kaggle_competition
    expected_max_gb: 10.0
    purpose: "Real-time market data forecasting with explicit cost modeling."

  - id: the-winton-stock-market-challenge
    resource_type: competition
    group: equity_intraday_features
    priority: 80
    topics: [equity, time_series]
    license_hint: kaggle_competition
    expected_max_gb: 1.0
    purpose: "Classic intraday return prediction benchmark."

  - id: jakewright/9000-tickers-of-stock-market-data-full-history
    resource_type: dataset
    group: equity_ohlcv_us
    priority: 200
    topics: [equity]
    license_hint: cc0
    expected_max_gb: 4.0
    purpose: "Wide US equity OHLCV history for cross-sectional features."

  - id: asadullahcreative/us-stock-market-historical-ohlcv-dataset
    resource_type: dataset
    group: equity_ohlcv_us_alt
    priority: 210
    topics: [equity]
    license_hint: unknown
    expected_max_gb: 2.0
    purpose: "Alternate US OHLCV dataset for cross-validation."

  - id: dgawlik/nyse
    resource_type: dataset
    group: equity_nyse_fundamentals
    priority: 220
    topics: [equity]
    license_hint: cc0
    expected_max_gb: 0.5
    purpose: "NYSE fundamentals + prices for fundamental factors."
```

- [ ] **Step 2: Run the manifest tests, expect all to pass now**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_manifests_well_formed.py -v`
Expected: all four manifest files load, all four have entries, no duplicate ids.

- [ ] **Step 3: Commit**

```bash
git add manifests/kaggle.yaml
git commit -m "feat: add manifests/kaggle.yaml with curated quant competitions and datasets"
```

---

## Task 5: Extend `manifests/datasets.yaml`

**Files:**
- Modify: `manifests/datasets.yaml`

- [ ] **Step 1: Append new entries**

Append the following entries to the `datasets:` list in `manifests/datasets.yaml` (do not remove existing entries):

```yaml
  - id: roneneldan/TinyStories
    group: general_nlp
    priority: 900
    license_hint: cdla-sharing-1.0
    topics: [general_nlp]
    purpose: "Small synthetic story corpus for tiny-LM experiments and NLP plumbing."

  - id: Salesforce/wikitext
    group: general_nlp
    priority: 910
    license_hint: cc-by-sa-3.0
    topics: [general_nlp]
    purpose: "WikiText-103 for general language modeling baselines."

  - id: stanfordnlp/imdb
    group: general_nlp_sentiment
    priority: 920
    license_hint: other
    topics: [general_nlp, sentiment]
    purpose: "IMDB sentiment baseline for NLP plumbing."

  - id: uoft-cs/cifar10
    group: vision_baseline
    priority: 930
    license_hint: mit
    topics: [vision]
    allow_patterns:
      - "*.parquet"
      - "*.json"
      - "*.md"
      - "README*"
    purpose: "CIFAR-10 vision baseline for general DL plumbing."

  - id: takala/financial_phrasebank
    group: finance_sentiment
    priority: 940
    license_hint: cc-by-nc-sa-3.0
    topics: [sentiment, news]
    purpose: "Financial PhraseBank — small high-quality finance sentiment labels."

  - id: zeroshot/twitter-financial-news-sentiment
    group: finance_sentiment
    priority: 950
    license_hint: mit
    topics: [sentiment, news]
    purpose: "Twitter finance sentiment labels."

  - id: zeroshot/twitter-financial-news-topic
    group: finance_topic
    priority: 960
    license_hint: mit
    topics: [news]
    purpose: "Twitter finance topic labels."

  - id: FinGPT/fingpt-sentiment-train
    group: finance_sentiment_instruction
    priority: 970
    license_hint: mit
    topics: [sentiment, instruction]
    purpose: "FinGPT sentiment training set in instruction form."

  - id: TheFinAI/fingpt-fiqa_qa
    group: finance_qa
    priority: 980
    license_hint: mit
    topics: [instruction]
    purpose: "FiQA financial Q&A pairs."

  - id: TheFinAI/flare-fpb
    group: finance_benchmark
    priority: 990
    license_hint: cc-by-nc-4.0
    topics: [sentiment]
    purpose: "FLARE FPB finance benchmark split."

  - id: Open-Orca/OpenOrca
    group: reasoning_instruction
    priority: 1000
    license_hint: mit
    topics: [reasoning, instruction]
    allow_patterns:
      - "*.parquet"
      - "*.json"
      - "README*"
    purpose: "Broad instruction-following corpus for reasoning models."

  - id: hendrycks/competition_math
    group: math_reasoning
    priority: 1010
    license_hint: mit
    topics: [reasoning]
    purpose: "Competition math problems with step-by-step solutions."

  - id: glaive-ai/glaive-code-assistant
    group: code_instruction
    priority: 1020
    license_hint: apache-2.0
    topics: [code, instruction]
    allow_patterns:
      - "*.parquet"
      - "*.json"
      - "README*"
    purpose: "Code assistant Q&A for coding LLM tuning experiments."

  - id: bigcode/the-stack-smol
    group: code_corpus_small
    priority: 1030
    license_hint: other
    topics: [code]
    allow_patterns:
      - "*.parquet"
      - "*.json"
      - "README*"
    purpose: "Small slice of The Stack for code modeling baselines."

  - id: HuggingFaceH4/MATH-500
    group: math_reasoning_bench
    priority: 1040
    license_hint: mit
    topics: [reasoning]
    purpose: "MATH-500 benchmark slice."

  - id: gsm8k
    group: math_reasoning_bench
    priority: 1050
    license_hint: mit
    topics: [reasoning]
    purpose: "Grade-school math word problems for reasoning evals."

  - id: monash_tsf
    group: time_series_benchmark
    priority: 1060
    license_hint: cc-by-4.0
    topics: [time_series]
    purpose: "Monash time-series forecasting archive."

  - id: ETT
    group: time_series_benchmark
    priority: 1070
    license_hint: cc-by-4.0
    topics: [time_series]
    purpose: "Electricity Transformer Temperature standard forecasting benchmark."
```

- [ ] **Step 2: Verify manifest still parses and tests pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_manifests_well_formed.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add manifests/datasets.yaml
git commit -m "feat: extend datasets manifest with NLP, code, reasoning, vision, time-series entries"
```

---

## Task 6: Extend `manifests/models.yaml`

**Files:**
- Modify: `manifests/models.yaml`

- [ ] **Step 1: Append new entries**

Append to the `models:` list (do not remove existing entries):

```yaml
  - id: Qwen/Qwen2.5-0.5B-Instruct
    group: small_instruct
    priority: 500
    expected_params: 500000000
    license_hint: apache-2.0
    topics: [instruction, code, reasoning]
    purpose: "Tiny instruction LLM used as default Q&A generator for paper instructions."

  - id: roneneldan/TinyStories-33M
    group: tiny_lm_fallback
    priority: 510
    expected_params: 33000000
    license_hint: cdla-sharing-1.0
    topics: [general_nlp]
    purpose: "Fallback model for paper Q&A when Qwen is unavailable."

  - id: sentence-transformers/all-MiniLM-L6-v2
    group: retrieval_embeddings
    priority: 520
    expected_params: 22700000
    license_hint: apache-2.0
    topics: [general_nlp]
    purpose: "Compact sentence embedder for retrieval baselines."

  - id: ProsusAI/finbert
    group: sentiment_features
    priority: 530
    expected_params: 110000000
    license_hint: apache-2.0
    topics: [sentiment, news]
    purpose: "FinBERT financial sentiment classifier."

  - id: distilbert/distilbert-base-uncased
    group: general_nlp_baseline
    priority: 540
    expected_params: 67000000
    license_hint: apache-2.0
    topics: [general_nlp]
    purpose: "Small distilled BERT for NLP plumbing."

  - id: Qwen/Qwen2.5-Coder-1.5B
    group: code_small
    priority: 550
    expected_params: 1500000000
    license_hint: apache-2.0
    topics: [code]
    purpose: "Compact code LLM for local experiments."

  - id: mlx-community/Qwen2.5-7B-Instruct-4bit
    group: apple_silicon_llm
    priority: 600
    expected_params: 7000000000
    license_hint: apache-2.0
    topics: [instruction]
    purpose: "MLX-optimized 7B instruct for Apple Silicon."

  - id: bartowski/Qwen2.5-14B-Instruct-GGUF
    group: large_llm_gguf
    priority: 700
    expected_params: 14000000000
    license_hint: apache-2.0
    topics: [instruction, reasoning]
    allow_patterns:
      - "*Q4_K_M*.gguf"
      - "*q4_k_m*.gguf"
      - "README*"
      - "LICENSE*"
    purpose: "14B Qwen2.5 instruct GGUF Q4_K_M; ~9 GB on disk."

  - id: bartowski/Mistral-Small-Instruct-2409-GGUF
    group: large_llm_gguf
    priority: 710
    expected_params: 22000000000
    license_hint: mrl
    topics: [instruction]
    allow_patterns:
      - "*Q4_K_M*.gguf"
      - "*q4_k_m*.gguf"
      - "README*"
      - "LICENSE*"
    purpose: "22B Mistral Small Instruct GGUF Q4_K_M; ~13 GB on disk."

  - id: bartowski/Yi-1.5-34B-Chat-GGUF
    group: very_large_llm_gguf
    priority: 800
    expected_params: 34000000000
    license_hint: apache-2.0
    topics: [instruction, reasoning]
    allow_patterns:
      - "*Q4_K_M*.gguf"
      - "*q4_k_m*.gguf"
      - "README*"
      - "LICENSE*"
    purpose: "34B Yi 1.5 chat GGUF Q4_K_M; ~20 GB on disk. Pushes the M4 24 GB envelope at inference."
```

- [ ] **Step 2: Verify**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_manifests_well_formed.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add manifests/models.yaml
git commit -m "feat: extend models manifest with Qwen instruct, FinBERT, embeddings, 14B/22B/34B GGUFs"
```

---

## Task 7: Extend `manifests/papers.yaml`

**Files:**
- Modify: `manifests/papers.yaml`

- [ ] **Step 1: Append new paper entries**

Append to the `papers:` list (do not remove existing entries):

```yaml
  - title: "Attention Is All You Need"
    source: "arxiv"
    arxiv_id: "1706.03762"
    year: 2017
    tags: ["transformer", "attention", "deep_learning"]
    training_category: "deep_learning_foundations"
    priority: 300
    open_text: true
    purpose: "Foundational transformer architecture reference."

  - title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    source: "arxiv"
    arxiv_id: "1810.04805"
    year: 2018
    tags: ["bert", "pretraining", "nlp"]
    training_category: "nlp_foundations"
    priority: 310
    open_text: true
    purpose: "BERT foundation for FinBERT lineage."

  - title: "Deep Residual Learning for Image Recognition"
    source: "arxiv"
    arxiv_id: "1512.03385"
    year: 2015
    tags: ["resnet", "computer_vision", "deep_learning"]
    training_category: "deep_learning_foundations"
    priority: 320
    open_text: true
    purpose: "ResNet foundation reference."

  - title: "Adam: A Method for Stochastic Optimization"
    source: "arxiv"
    arxiv_id: "1412.6980"
    year: 2014
    tags: ["optimizer", "training", "deep_learning"]
    training_category: "deep_learning_foundations"
    priority: 330
    open_text: true
    purpose: "Adam optimizer reference."

  - title: "LoRA: Low-Rank Adaptation of Large Language Models"
    source: "arxiv"
    arxiv_id: "2106.09685"
    year: 2021
    tags: ["peft", "lora", "fine_tuning"]
    training_category: "training_methods"
    priority: 340
    open_text: true
    purpose: "LoRA reference for small local adapter experiments."

  - title: "QLoRA: Efficient Finetuning of Quantized LLMs"
    source: "arxiv"
    arxiv_id: "2305.14314"
    year: 2023
    tags: ["qlora", "quantization", "fine_tuning"]
    training_category: "training_methods"
    priority: 350
    open_text: true
    purpose: "QLoRA reference for memory-efficient fine-tuning."

  - title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    source: "arxiv"
    arxiv_id: "2205.14135"
    year: 2022
    tags: ["attention", "gpu", "efficiency"]
    training_category: "gpu_parallel"
    priority: 360
    open_text: true
    purpose: "GPU attention efficiency reference."

  - title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
    source: "arxiv"
    arxiv_id: "1909.08053"
    year: 2019
    tags: ["model_parallelism", "gpu", "large_language_models"]
    training_category: "gpu_parallel"
    priority: 370
    open_text: true
    purpose: "Model parallelism reference."

  - title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    source: "arxiv"
    arxiv_id: "1910.02054"
    year: 2019
    tags: ["zero", "memory", "distributed_training"]
    training_category: "gpu_parallel"
    priority: 380
    open_text: true
    purpose: "ZeRO memory optimization reference."

  - title: "Mixed Precision Training"
    source: "arxiv"
    arxiv_id: "1710.03740"
    year: 2017
    tags: ["mixed_precision", "fp16", "training"]
    training_category: "gpu_parallel"
    priority: 390
    open_text: true
    purpose: "Mixed precision training reference."

  - title: "Reformer: The Efficient Transformer"
    source: "arxiv"
    arxiv_id: "2001.04451"
    year: 2020
    tags: ["transformer", "efficiency"]
    training_category: "deep_learning_foundations"
    priority: 400
    open_text: true
    purpose: "Efficient transformer reference."

  - title: "Longformer: The Long-Document Transformer"
    source: "arxiv"
    arxiv_id: "2004.05150"
    year: 2020
    tags: ["transformer", "long_context"]
    training_category: "deep_learning_foundations"
    priority: 410
    open_text: true
    purpose: "Long-context transformer reference."

  - title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    source: "arxiv"
    arxiv_id: "2010.11929"
    year: 2020
    tags: ["vit", "vision_transformer"]
    training_category: "vision_foundations"
    priority: 420
    open_text: true
    purpose: "Vision Transformer reference."

  - title: "GPT-3: Language Models are Few-Shot Learners"
    source: "arxiv"
    arxiv_id: "2005.14165"
    year: 2020
    tags: ["gpt", "large_language_models", "few_shot"]
    training_category: "llm_foundations"
    priority: 430
    open_text: true
    purpose: "Few-shot LLM reference."

  - title: "Chinchilla: Training Compute-Optimal Large Language Models"
    source: "arxiv"
    arxiv_id: "2203.15556"
    year: 2022
    tags: ["scaling_laws", "training"]
    training_category: "llm_foundations"
    priority: 440
    open_text: true
    purpose: "Compute-optimal scaling reference."

  - title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    source: "arxiv"
    arxiv_id: "2305.18290"
    year: 2023
    tags: ["dpo", "rlhf", "alignment"]
    training_category: "alignment"
    priority: 450
    open_text: true
    purpose: "DPO reference for finance-DPO models in our model manifest."

  - title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    source: "arxiv"
    arxiv_id: "2312.00752"
    year: 2023
    tags: ["state_space", "sequence_modeling"]
    training_category: "sequence_modeling"
    priority: 460
    open_text: true
    purpose: "SSM/Mamba reference."

  - title: "Chronos: Learning the Language of Time Series"
    source: "arxiv"
    arxiv_id: "2403.07815"
    year: 2024
    tags: ["time_series", "foundation_model"]
    training_category: "time_series_foundation"
    priority: 470
    open_text: true
    purpose: "Underpins amazon/chronos-2 in our model manifest."

  - title: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
    source: "arxiv"
    arxiv_id: "2210.02186"
    year: 2022
    tags: ["time_series", "deep_learning"]
    training_category: "time_series_foundation"
    priority: 480
    open_text: true
    purpose: "Strong time-series modeling reference."

  - title: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
    source: "arxiv"
    arxiv_id: "2012.07436"
    year: 2020
    tags: ["time_series", "transformer"]
    training_category: "time_series_foundation"
    priority: 490
    open_text: true
    purpose: "Long-sequence time-series transformer."

  - title: "Reinforcement Learning for Optimal Market Making"
    source: "arxiv"
    arxiv_id: "1810.06593"
    year: 2018
    tags: ["reinforcement_learning", "market_making"]
    training_category: "rl_trading"
    priority: 500
    open_text: true
    purpose: "RL market making reference."

  - title: "Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review"
    source: "arxiv"
    arxiv_id: "2106.00123"
    year: 2021
    tags: ["reinforcement_learning", "quant_trading", "review"]
    training_category: "rl_trading"
    priority: 510
    open_text: true
    purpose: "RL trading review."

  - title: "Practical Deep Reinforcement Learning Approach for Stock Trading"
    source: "arxiv"
    arxiv_id: "1811.07522"
    year: 2018
    tags: ["reinforcement_learning", "stock_trading"]
    training_category: "rl_trading"
    priority: 520
    open_text: true
    purpose: "Practical RL trading reference."

  - title: "FinBERT: A Pretrained Language Model for Financial Communications"
    source: "arxiv"
    arxiv_id: "2006.08097"
    year: 2020
    tags: ["finbert", "finance", "nlp"]
    training_category: "finance_nlp"
    priority: 530
    open_text: true
    purpose: "FinBERT foundation paper for our ProsusAI/finbert model."

  - title: "FinGPT: Open-Source Financial Large Language Models"
    source: "arxiv"
    arxiv_id: "2306.06031"
    year: 2023
    tags: ["fingpt", "finance_llm"]
    training_category: "finance_llm"
    priority: 540
    open_text: true
    purpose: "FinGPT reference for finance LLM lineage."

  - title: "BloombergGPT: A Large Language Model for Finance"
    source: "arxiv"
    arxiv_id: "2303.17564"
    year: 2023
    tags: ["bloomberg", "finance_llm"]
    training_category: "finance_llm"
    priority: 550
    open_text: true
    purpose: "Finance-domain LLM design reference."

  - title: "FNSPID: A Comprehensive Financial News Dataset in Time Series"
    source: "arxiv"
    arxiv_id: "2402.06698"
    year: 2024
    tags: ["news", "time_series", "dataset"]
    training_category: "finance_news"
    priority: 560
    open_text: true
    purpose: "FNSPID dataset paper for our HF dataset entry."

  - title: "Statistical Arbitrage in the U.S. Equities Market"
    source: "arxiv"
    arxiv_id: "0911.2392"
    year: 2008
    tags: ["statistical_arbitrage", "equity"]
    training_category: "stat_arb"
    priority: 570
    open_text: true
    purpose: "Stat-arb baseline reference."

  - title: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
    source: "doi"
    doi: "10.1093/rfs/hhj020"
    year: 2006
    tags: ["pairs_trading", "stat_arb"]
    training_category: "stat_arb"
    priority: 580
    open_text: false

  - title: "The Adaptive Markets Hypothesis"
    source: "open_reference"
    year: 2004
    tags: ["market_efficiency"]
    training_category: "market_theory"
    priority: 590
    open_text: true
    purpose: "Behavioral and adaptive markets reference."

  - title: "High-Frequency Trading and Price Discovery"
    source: "doi"
    doi: "10.1093/rfs/hhu032"
    year: 2014
    tags: ["high_frequency", "price_discovery"]
    training_category: "market_microstructure"
    priority: 600
    open_text: false

  - title: "An Empirical Behavioral Model of Liquidity and Volatility"
    source: "arxiv"
    arxiv_id: "0706.1271"
    year: 2007
    tags: ["liquidity", "volatility"]
    training_category: "market_microstructure"
    priority: 610
    open_text: true
    purpose: "Liquidity-volatility empirical reference."

  - title: "Risk and Return in High Frequency Trading"
    source: "arxiv"
    arxiv_id: "1605.06956"
    year: 2016
    tags: ["high_frequency", "risk_return"]
    training_category: "market_microstructure"
    priority: 620
    open_text: true
    purpose: "HFT risk/return reference."

  - title: "Generative Adversarial Networks"
    source: "arxiv"
    arxiv_id: "1406.2661"
    year: 2014
    tags: ["gan", "generative_models"]
    training_category: "generative_models"
    priority: 700
    open_text: true
    purpose: "GAN foundation reference."

  - title: "Denoising Diffusion Probabilistic Models"
    source: "arxiv"
    arxiv_id: "2006.11239"
    year: 2020
    tags: ["diffusion", "generative_models"]
    training_category: "generative_models"
    priority: 710
    open_text: true
    purpose: "DDPM foundation reference."

  - title: "Variational Inference with Normalizing Flows"
    source: "arxiv"
    arxiv_id: "1505.05770"
    year: 2015
    tags: ["normalizing_flows", "variational_inference"]
    training_category: "generative_models"
    priority: 720
    open_text: true
    purpose: "Normalizing flows reference."

  - title: "Conditional Time-Series Generation with GANs"
    source: "arxiv"
    arxiv_id: "1706.02633"
    year: 2017
    tags: ["gan", "time_series"]
    training_category: "generative_time_series"
    priority: 730
    open_text: true
    purpose: "Time-series GAN reference."

  - title: "Tuning-Free Generation of Realistic Synthetic Tabular Data"
    source: "arxiv"
    arxiv_id: "2407.07116"
    year: 2024
    tags: ["tabular", "synthetic_data"]
    training_category: "tabular_synthesis"
    priority: 740
    open_text: true
    purpose: "Synthetic tabular generation reference."

  - title: "XGBoost: A Scalable Tree Boosting System"
    source: "arxiv"
    arxiv_id: "1603.02754"
    year: 2016
    tags: ["gbm", "tabular"]
    training_category: "tabular_baselines"
    priority: 800
    open_text: true
    purpose: "XGBoost reference for our tabular finance baseline."

  - title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
    source: "open_reference"
    year: 2017
    tags: ["gbm", "tabular"]
    training_category: "tabular_baselines"
    priority: 810
    open_text: true
    purpose: "LightGBM reference for our tabular finance baseline."

  - title: "CatBoost: Unbiased Boosting with Categorical Features"
    source: "arxiv"
    arxiv_id: "1706.09516"
    year: 2017
    tags: ["gbm", "tabular", "categorical"]
    training_category: "tabular_baselines"
    priority: 820
    open_text: true
    purpose: "CatBoost reference for our tabular finance baseline."
```

- [ ] **Step 2: Verify**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_manifests_well_formed.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add manifests/papers.yaml
git commit -m "feat: extend papers manifest with DL/NLP/RL/GPU/time-series references"
```

---

## Task 8: Add `KaggleItem` dataclass and loader (`src/quant_research_stack/kaggle_artifacts.py`)

**Files:**
- Create: `src/quant_research_stack/kaggle_artifacts.py`
- Create: `tests/test_kaggle_artifacts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kaggle_artifacts.py`:

```python
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from quant_research_stack.kaggle_artifacts import (
    KaggleItem,
    load_kaggle_items,
    local_path_for,
    safe_kaggle_dir_name,
)


def write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "kaggle.yaml"
    path.write_text(dedent(body), encoding="utf-8")
    return path


def test_load_kaggle_items_parses_competition(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        defaults:
          resource_type: competition
          enabled: true
          expected_max_gb: 5.0
        items:
          - id: jpx-tokyo-stock-exchange-prediction
            group: equity_jpx
            priority: 10
            topics: [equity]
            purpose: test
            license_hint: kaggle_competition
        """,
    )
    items = load_kaggle_items(manifest)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, KaggleItem)
    assert item.id == "jpx-tokyo-stock-exchange-prediction"
    assert item.resource_type == "competition"
    assert item.priority == 10
    assert item.topics == ("equity",)
    assert item.expected_max_gb == 5.0
    assert item.enabled is True


def test_load_kaggle_items_parses_dataset(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        items:
          - id: jakewright/9000-tickers-of-stock-market-data-full-history
            resource_type: dataset
            group: equity_ohlcv_us
            priority: 200
            topics: [equity]
            purpose: test
            license_hint: cc0
            expected_max_gb: 4.0
            enabled: true
        """,
    )
    items = load_kaggle_items(manifest)
    assert len(items) == 1
    assert items[0].resource_type == "dataset"


def test_load_kaggle_items_skips_disabled_when_filtered(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        defaults:
          resource_type: competition
        items:
          - id: a
            group: g
            priority: 1
            topics: [equity]
            purpose: t
            enabled: false
          - id: b
            group: g
            priority: 2
            topics: [equity]
            purpose: t
            enabled: true
        """,
    )
    all_items = load_kaggle_items(manifest)
    enabled = [item for item in all_items if item.enabled]
    assert {item.id for item in all_items} == {"a", "b"}
    assert {item.id for item in enabled} == {"b"}


def test_safe_kaggle_dir_name_replaces_separators() -> None:
    assert safe_kaggle_dir_name("jakewright/9000-tickers-of-stock-market-data-full-history") == "jakewright__9000-tickers-of-stock-market-data-full-history"
    assert safe_kaggle_dir_name("jpx-tokyo-stock-exchange-prediction") == "jpx-tokyo-stock-exchange-prediction"


def test_local_path_for_competition_uses_competitions_subdir(tmp_path: Path) -> None:
    item = KaggleItem(
        id="jpx-tokyo-stock-exchange-prediction",
        resource_type="competition",
        group="equity_jpx",
        priority=10,
        topics=("equity",),
        purpose="t",
        license_hint=None,
        expected_max_gb=None,
        enabled=True,
    )
    path = local_path_for(item, tmp_path)
    assert path == tmp_path / "competitions" / "jpx-tokyo-stock-exchange-prediction"


def test_local_path_for_dataset_uses_datasets_subdir(tmp_path: Path) -> None:
    item = KaggleItem(
        id="jakewright/9000-tickers-of-stock-market-data-full-history",
        resource_type="dataset",
        group="equity_ohlcv_us",
        priority=200,
        topics=("equity",),
        purpose="t",
        license_hint="cc0",
        expected_max_gb=4.0,
        enabled=True,
    )
    path = local_path_for(item, tmp_path)
    assert path == tmp_path / "datasets" / "jakewright__9000-tickers-of-stock-market-data-full-history"


def test_load_kaggle_items_rejects_invalid_resource_type(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        """
        schema_version: 1
        items:
          - id: bad
            resource_type: notebook
            group: g
            priority: 1
            topics: [equity]
            purpose: t
        """,
    )
    with pytest.raises(ValueError, match="resource_type"):
        load_kaggle_items(manifest)
```

- [ ] **Step 2: Run test, expect import failure**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_kaggle_artifacts.py -v`
Expected: ImportError or ModuleNotFoundError on `quant_research_stack.kaggle_artifacts`.

- [ ] **Step 3: Write minimal implementation**

Create `src/quant_research_stack/kaggle_artifacts.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_research_stack.artifacts import read_yaml


VALID_RESOURCE_TYPES = frozenset({"competition", "dataset"})


def safe_kaggle_dir_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", item_id)


@dataclass(frozen=True)
class KaggleItem:
    id: str
    resource_type: str
    group: str
    priority: int
    topics: tuple[str, ...]
    purpose: str
    license_hint: str | None
    expected_max_gb: float | None
    enabled: bool


def _merge(defaults: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(raw)
    return merged


def load_kaggle_items(manifest_path: str | Path) -> list[KaggleItem]:
    manifest = read_yaml(manifest_path)
    defaults = manifest.get("defaults", {}) or {}
    items: list[KaggleItem] = []
    for raw in manifest.get("items", []) or []:
        merged = _merge(defaults, raw)
        resource_type = merged.get("resource_type", "competition")
        if resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError(f"Invalid resource_type {resource_type!r} for item {merged.get('id')!r}")
        expected_max_gb = merged.get("expected_max_gb")
        items.append(
            KaggleItem(
                id=str(merged["id"]),
                resource_type=resource_type,
                group=str(merged.get("group", "ungrouped")),
                priority=int(merged.get("priority", 9999)),
                topics=tuple(merged.get("topics", []) or []),
                purpose=str(merged.get("purpose", "")),
                license_hint=merged.get("license_hint"),
                expected_max_gb=float(expected_max_gb) if expected_max_gb is not None else None,
                enabled=bool(merged.get("enabled", True)),
            )
        )
    return items


def local_path_for(item: KaggleItem, root: str | Path) -> Path:
    base = Path(root)
    if item.resource_type == "competition":
        return base / "competitions" / item.id
    return base / "datasets" / safe_kaggle_dir_name(item.id)
```

- [ ] **Step 4: Run test, expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_kaggle_artifacts.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check src/quant_research_stack/kaggle_artifacts.py tests/test_kaggle_artifacts.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/kaggle_artifacts.py tests/test_kaggle_artifacts.py
git commit -m "feat: add KaggleItem loader and path helpers with unit tests"
```

---

## Task 9: Create `scripts/download_kaggle_artifacts.py`

**Files:**
- Create: `scripts/download_kaggle_artifacts.py`

- [ ] **Step 1: Write the script**

Create `scripts/download_kaggle_artifacts.py`:

```python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from quant_research_stack.artifacts import GB, bytes_to_gb, folder_size, read_yaml, write_json
from quant_research_stack.kaggle_artifacts import (
    KaggleItem,
    load_kaggle_items,
    local_path_for,
)


console = Console()


def kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def has_credentials() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def estimate_size_bytes(item: KaggleItem) -> tuple[int | None, str | None]:
    if item.expected_max_gb is None:
        return None, "no expected_max_gb in manifest"
    return int(item.expected_max_gb * GB), None


def kaggle_download_cmd(item: KaggleItem, dest: Path) -> list[str]:
    if item.resource_type == "competition":
        return ["kaggle", "competitions", "download", "-c", item.id, "-p", str(dest)]
    return ["kaggle", "datasets", "download", "-d", item.id, "-p", str(dest), "--unzip"]


def unzip_competition_artifacts(dest: Path) -> None:
    for zip_path in dest.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(dest)
        zip_path.unlink()


def execute_download(item: KaggleItem, dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = kaggle_download_cmd(item, dest)
    console.print(f"[bold]Downloading[/bold] {item.resource_type} {item.id} -> {dest}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    record: dict[str, Any] = {
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        if "403" in proc.stderr or "Forbidden" in proc.stderr:
            record["status"] = "skip_rules_not_accepted"
            return record
        record["status"] = "error"
        return record
    if item.resource_type == "competition":
        unzip_competition_artifacts(dest)
    record["status"] = "downloaded"
    record["final_size_bytes"] = folder_size(dest)
    record["final_size_gb"] = bytes_to_gb(record["final_size_bytes"])
    return record


def build_plan(items: list[KaggleItem], config: dict[str, Any], force: bool, sort: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = Path(config["paths"]["raw_kaggle_root"])
    bucket_budget_gb = float(config["bucket_budgets"]["kaggle"])
    remaining = int(bucket_budget_gb * GB)
    planned: list[dict[str, Any]] = []
    for item in items:
        if not item.enabled:
            continue
        dest = local_path_for(item, base)
        local_size = folder_size(dest)
        estimated, est_err = estimate_size_bytes(item)
        row: dict[str, Any] = {
            "id": item.id,
            "resource_type": item.resource_type,
            "group": item.group,
            "priority": item.priority,
            "topics": list(item.topics),
            "purpose": item.purpose,
            "license_hint": item.license_hint,
            "expected_max_gb": item.expected_max_gb,
            "estimated_size_bytes": estimated,
            "estimated_size_gb": bytes_to_gb(estimated),
            "local_dir": str(dest),
            "local_size_bytes": local_size,
            "local_size_gb": bytes_to_gb(local_size),
            "estimate_error": est_err,
            "decision": "download",
        }
        if local_size > 0 and not force:
            row["decision"] = "skip_present"
            planned.append(row)
            continue
        if estimated is None:
            row["decision"] = "skip_unknown_size"
            planned.append(row)
            continue
        if estimated > remaining:
            row["decision"] = "skip_budget"
            planned.append(row)
            continue
        remaining -= estimated
        planned.append(row)

    if sort == "size":
        planned.sort(key=lambda row: (row["estimated_size_bytes"] is None, row["estimated_size_bytes"] or 0, row["priority"]))
    else:
        planned.sort(key=lambda row: (row["priority"], row["estimated_size_bytes"] is None, row["estimated_size_bytes"] or 0))

    summary = {
        "bucket_budget_gb": bucket_budget_gb,
        "planned_download_gb": bytes_to_gb(sum((row["estimated_size_bytes"] or 0) for row in planned if row["decision"] == "download")),
        "remaining_bucket_budget_gb": bytes_to_gb(remaining),
        "sort": sort,
    }
    return summary, planned


def print_plan(summary: dict[str, Any], planned: list[dict[str, Any]]) -> None:
    table = Table(title="Kaggle Artifact Plan")
    table.add_column("Decision")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Group")
    table.add_column("GB", justify="right")
    table.add_column("License")
    for row in planned:
        table.add_row(
            row["decision"],
            row["resource_type"],
            row["id"],
            row["group"],
            "?" if row["estimated_size_gb"] is None else f"{row['estimated_size_gb']:.2f}",
            str(row.get("license_hint") or ""),
        )
    console.print(table)
    console.print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-aware Kaggle competition/dataset downloader.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--manifest", default="manifests/kaggle.yaml")
    parser.add_argument("--sort", choices=["size", "priority"], default="priority")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default="reports/kaggle_download_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    items = load_kaggle_items(args.manifest)
    summary, planned = build_plan(items, config, force=args.force, sort=args.sort)
    write_json(args.report, {"summary": summary, "items": planned})
    print_plan(summary, planned)

    if args.dry_run:
        console.print(f"Dry run only. Wrote {args.report}")
        return 0

    if not kaggle_available():
        console.print("[red]`kaggle` CLI not found on PATH. Install with `uv pip install kaggle`.[/red]")
        return 2
    if not has_credentials():
        console.print("[red]Missing ~/.kaggle/kaggle.json. Place your Kaggle API token there.[/red]")
        return 3

    base = Path(config["paths"]["raw_kaggle_root"])
    base.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for row in planned:
        if row["decision"] != "download":
            results.append(row)
            continue
        item = next((it for it in items if it.id == row["id"] and it.resource_type == row["resource_type"]), None)
        if item is None:
            row["status"] = "error"
            row["error"] = "manifest item not found for plan row"
            results.append(row)
            continue
        outcome = execute_download(item, local_path_for(item, base))
        row.update(outcome)
        results.append(row)
    write_json(args.report, {"summary": summary, "items": results})
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "src")
    sys.exit(main())
```

- [ ] **Step 2: Smoke test in dry-run mode**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --dry-run`
Expected: prints a plan table, writes `reports/kaggle_download_plan.json`, exit code 0. Total planned ≤ 40 GB.

- [ ] **Step 3: Verify plan JSON is well-formed**

Run: `cd /Users/dmr/MachineLearning && python -c "import json; data = json.load(open('reports/kaggle_download_plan.json')); print(data['summary']); assert len(data['items']) >= 11"`
Expected: prints summary; assertion passes.

- [ ] **Step 4: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check scripts/download_kaggle_artifacts.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_kaggle_artifacts.py reports/kaggle_download_plan.json
git commit -m "feat: add budget-aware kaggle downloader with dry-run plan"
```

---

## Task 10: Create `scripts/paper_corpus_to_parquet.py`

**Files:**
- Create: `scripts/paper_corpus_to_parquet.py`
- Create: `tests/test_paper_corpus_to_parquet.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_paper_corpus_to_parquet.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from scripts.paper_corpus_to_parquet import (
    iter_jsonl_records,
    record_to_row,
    write_parquet_shards,
)


def test_iter_jsonl_records_yields_dicts(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    records = list(iter_jsonl_records(jsonl))
    assert records == [{"a": 1}, {"a": 2}]


def test_record_to_row_computes_sha256_and_word_count() -> None:
    record = {
        "id": "paper_pdf:foo.pdf:0",
        "source_type": "paper_pdf",
        "source_path": "foo.pdf",
        "chunk_index": 0,
        "text": "hello world test",
    }
    row = record_to_row(record)
    assert row["id"] == "paper_pdf:foo.pdf:0"
    assert row["source_type"] == "paper_pdf"
    assert row["n_words"] == 3
    assert row["sha256"] == hashlib.sha256("hello world test".encode("utf-8")).hexdigest()


def test_write_parquet_shards_writes_at_least_one_shard(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"id-{i}",
            "source_type": "paper_pdf",
            "source_path": "x.pdf",
            "chunk_index": i,
            "text": "x " * 200,
            "sha256": hashlib.sha256((str(i) * 16).encode()).hexdigest(),
            "n_words": 200,
        }
        for i in range(50)
    ]
    out_dir = tmp_path / "out"
    written = write_parquet_shards(rows, out_dir, shard_target_mb=1)
    assert len(written) >= 1
    df = pl.concat([pl.read_parquet(path) for path in written])
    assert df.height == 50
    assert {"id", "source_type", "source_path", "chunk_index", "text", "sha256", "n_words"} <= set(df.columns)


def test_write_parquet_shards_empty_input_writes_nothing(tmp_path: Path) -> None:
    written = write_parquet_shards([], tmp_path / "out", shard_target_mb=256)
    assert written == []
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_paper_corpus_to_parquet.py -v`
Expected: ImportError on `scripts.paper_corpus_to_parquet`.

- [ ] **Step 3: Write the script**

Create `scripts/paper_corpus_to_parquet.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import polars as pl
from rich.console import Console

from quant_research_stack.artifacts import read_yaml


console = Console()


def iter_jsonl_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text", "")
    return {
        "id": str(record["id"]),
        "source_type": str(record.get("source_type", "")),
        "source_path": str(record.get("source_path", "")),
        "chunk_index": int(record.get("chunk_index", 0)),
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_words": len(text.split()),
    }


def _row_bytes(row: dict[str, Any]) -> int:
    return sum(len(str(value).encode("utf-8")) for value in row.values())


def write_parquet_shards(rows: Iterable[dict[str, Any]], out_dir: str | Path, shard_target_mb: int) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_bytes = shard_target_mb * 1024 * 1024
    buffer: list[dict[str, Any]] = []
    buffer_bytes = 0
    shard_paths: list[Path] = []
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes, shard_index
        if not buffer:
            return
        df = pl.DataFrame(buffer)
        path = out / f"shard_{shard_index:05d}.parquet"
        df.write_parquet(path, compression="zstd")
        shard_paths.append(path)
        shard_index += 1
        buffer = []
        buffer_bytes = 0

    for row in rows:
        buffer.append(row)
        buffer_bytes += _row_bytes(row)
        if buffer_bytes >= target_bytes:
            flush()
    flush()
    return shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert chunked research JSONL to Parquet shards.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shard-target-mb", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    corpus_cfg = config.get("research_corpus", {})
    input_path = Path(args.input or Path(paths["processed_research_root"]) / "research_corpus.jsonl")
    output_dir = Path(args.output_dir or Path(paths["processed_research_root"]) / "parquet")
    shard_target_mb = int(args.shard_target_mb or corpus_cfg.get("parquet_shard_target_mb", 256))

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}. Run scripts/prepare_research_corpus.py first.[/red]")
        return 2

    rows = (record_to_row(rec) for rec in iter_jsonl_records(input_path))
    shards = write_parquet_shards(rows, output_dir, shard_target_mb)
    console.print(f"Wrote {len(shards)} parquet shards to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_paper_corpus_to_parquet.py -v`
Expected: all pass.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check scripts/paper_corpus_to_parquet.py tests/test_paper_corpus_to_parquet.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/paper_corpus_to_parquet.py tests/test_paper_corpus_to_parquet.py
git commit -m "feat: add paper_corpus_to_parquet with sharding and sha256/word-count rows"
```

---

## Task 11: Create `scripts/paper_corpus_to_instructions.py`

**Files:**
- Create: `scripts/paper_corpus_to_instructions.py`

This script does not get unit tests because it depends on a downloaded model and live MPS device; it has a fast `--dry-run` path that prints the inferred model choice and skips generation.

- [ ] **Step 1: Write the script**

Create `scripts/paper_corpus_to_instructions.py`:

```python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console

from quant_research_stack.artifacts import read_yaml, safe_repo_id


console = Console()


PROMPT_TEMPLATES = [
    "Summarize the key claim of this passage in 3 sentences.",
    "What is the main quantitative-finance concept introduced here, and how is it operationalized?",
    "Generate one well-formed exam-style question and answer pair grounded in this passage.",
]


@dataclass(frozen=True)
class ModelChoice:
    repo_id: str
    local_dir: Path
    mode: str  # "primary" | "fallback"


def find_model(primary_repo: str, fallback_repo: str, model_root: Path) -> ModelChoice | None:
    primary_dir = model_root / safe_repo_id(primary_repo)
    if primary_dir.exists() and any(primary_dir.iterdir()):
        return ModelChoice(repo_id=primary_repo, local_dir=primary_dir, mode="primary")
    fallback_dir = model_root / safe_repo_id(fallback_repo)
    if fallback_dir.exists() and any(fallback_dir.iterdir()):
        return ModelChoice(repo_id=fallback_repo, local_dir=fallback_dir, mode="fallback")
    return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def already_generated_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(record.get("source_chunk_id", "")))
    return done


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_with_transformers(
    model_choice: ModelChoice,
    chunks: Iterable[dict[str, Any]],
    output_path: Path,
    max_per_chunk: int,
    max_new_tokens: int,
) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"Loading {model_choice.repo_id} ({model_choice.mode}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_choice.local_dir)
    model = AutoModelForCausalLM.from_pretrained(model_choice.local_dir).to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            chunk_id = str(chunk.get("id", ""))
            text = chunk.get("text", "")
            if not text:
                continue
            for index, template in enumerate(PROMPT_TEMPLATES[:max_per_chunk]):
                prompt = f"Passage:\n{text}\n\nInstruction: {template}\nResponse:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                record = {
                    "id": f"{chunk_id}#qa{index}",
                    "source_chunk_id": chunk_id,
                    "prompt": template,
                    "response": response,
                    "model_id": model_choice.repo_id,
                    "model_mode": model_choice.mode,
                    "generated_at": now_iso(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instruction-format Q&A from chunked research JSONL.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None, help="If set, process only the first N chunks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    corpus_cfg = config.get("research_corpus", {})
    input_path = Path(args.input or Path(paths["processed_research_root"]) / "research_corpus.jsonl")
    output_path = Path(args.output or Path(paths["processed_research_root"]) / "instructions.jsonl")
    model_root = Path(paths["model_root"])
    max_per_chunk = int(corpus_cfg.get("instruction_max_per_chunk", 3))
    max_new_tokens = int(corpus_cfg.get("instruction_max_new_tokens", 256))
    primary = corpus_cfg.get("instruction_primary_model", "Qwen/Qwen2.5-0.5B-Instruct")
    fallback = corpus_cfg.get("instruction_fallback_model", "roneneldan/TinyStories-33M")

    if not input_path.exists():
        console.print(f"[red]Input not found: {input_path}.[/red]")
        return 2

    choice = find_model(primary, fallback, model_root)
    if choice is None:
        console.print(f"[red]Neither {primary} nor {fallback} is present under {model_root}.[/red]")
        return 3
    if choice.mode == "fallback":
        console.print(f"[yellow]Using fallback model {choice.repo_id} (primary {primary} unavailable).[/yellow]")

    if args.dry_run:
        console.print(f"Would use model: {choice.repo_id} ({choice.mode}) at {choice.local_dir}")
        console.print(f"Would read chunks from {input_path}; write Q&A to {output_path}")
        return 0

    done = already_generated_ids(output_path)
    def chunks_iter() -> Iterable[dict[str, Any]]:
        seen = 0
        for record in iter_jsonl(input_path):
            if str(record.get("id", "")) in done:
                continue
            yield record
            seen += 1
            if args.limit_chunks is not None and seen >= args.limit_chunks:
                return

    count = generate_with_transformers(choice, chunks_iter(), output_path, max_per_chunk, max_new_tokens)
    console.print(f"Wrote {count} Q&A records to {output_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "src")
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check scripts/paper_corpus_to_instructions.py`
Expected: no errors.

- [ ] **Step 3: Dry-run (will fail until the input JSONL exists; that's the expected red state at this point)**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py --dry-run`
Expected: exits with code 2 and prints `Input not found: data/processed/research/research_corpus.jsonl.` This will turn green after Task 16 produces the JSONL.

- [ ] **Step 4: Commit**

```bash
git add scripts/paper_corpus_to_instructions.py
git commit -m "feat: add paper_corpus_to_instructions with idempotent JSONL appends and MPS fallback"
```

---

## Task 12: Create `scripts/dedupe_and_verify.py`

**Files:**
- Create: `scripts/dedupe_and_verify.py`
- Create: `tests/test_dedupe_and_verify.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dedupe_and_verify.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.dedupe_and_verify import (
    build_inventory,
    classify_bucket,
    file_sha256,
    group_duplicates,
)


def test_file_sha256_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"hello")
    assert file_sha256(path) == hashlib.sha256(b"hello").hexdigest()


def test_classify_bucket_handles_known_roots(tmp_path: Path) -> None:
    roots = {
        "hf_datasets": tmp_path / "data" / "raw" / "huggingface",
        "kaggle": tmp_path / "data" / "raw" / "kaggle",
        "models": tmp_path / "models" / "huggingface",
        "papers_and_derived": tmp_path / "data" / "raw" / "papers",
    }
    assert classify_bucket(roots["hf_datasets"] / "x" / "y.parquet", roots) == "hf_datasets"
    assert classify_bucket(roots["models"] / "x" / "config.json", roots) == "models"
    assert classify_bucket(roots["papers_and_derived"] / "x.pdf", roots) == "papers_and_derived"
    assert classify_bucket(tmp_path / "outside" / "x.bin", roots) == "other"


def test_group_duplicates_collapses_same_hash() -> None:
    items = [
        {"path": "a", "sha256": "X", "size_bytes": 1, "bucket": "hf_datasets"},
        {"path": "b", "sha256": "X", "size_bytes": 1, "bucket": "kaggle"},
        {"path": "c", "sha256": "Y", "size_bytes": 1, "bucket": "models"},
    ]
    dups = group_duplicates(items)
    assert dups == [{"sha256": "X", "paths": ["a", "b"]}]


def test_build_inventory_walks_files(tmp_path: Path) -> None:
    hf = tmp_path / "data" / "raw" / "huggingface" / "ds1"
    hf.mkdir(parents=True)
    (hf / "a.parquet").write_bytes(b"abc")
    (hf / "b.parquet").write_bytes(b"abc")
    models = tmp_path / "models" / "huggingface" / "m1"
    models.mkdir(parents=True)
    (models / "config.json").write_bytes(b"unique")

    roots = {
        "hf_datasets": tmp_path / "data" / "raw" / "huggingface",
        "kaggle": tmp_path / "data" / "raw" / "kaggle",
        "models": tmp_path / "models" / "huggingface",
        "papers_and_derived": tmp_path / "data" / "raw" / "papers",
    }
    inventory = build_inventory(roots)
    assert inventory["total_size_bytes"] == 3 + 3 + 6
    assert inventory["by_bucket"]["hf_datasets"] == 6
    assert inventory["by_bucket"]["models"] == 6
    assert len(inventory["items"]) == 3
    assert len(inventory["duplicates"]) == 1
```

- [ ] **Step 2: Run, expect import failure**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_dedupe_and_verify.py -v`
Expected: ImportError on `scripts.dedupe_and_verify`.

- [ ] **Step 3: Write the script**

Create `scripts/dedupe_and_verify.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from quant_research_stack.artifacts import bytes_to_gb, read_yaml, write_json


console = Console()


SHA_CHUNK = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def classify_bucket(path: Path, roots: dict[str, Path]) -> str:
    resolved = path.resolve()
    for bucket, root in roots.items():
        try:
            resolved.relative_to(root.resolve())
            return bucket
        except ValueError:
            continue
    return "other"


def group_duplicates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(item["sha256"], []).append(item["path"])
    return [{"sha256": sha, "paths": sorted(paths)} for sha, paths in groups.items() if len(paths) > 1]


def build_inventory(roots: dict[str, Path]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    by_bucket: dict[str, int] = {bucket: 0 for bucket in roots}
    by_bucket["other"] = 0
    for bucket, root in roots.items():
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            size = file_path.stat().st_size
            sha = file_sha256(file_path)
            items.append(
                {
                    "path": str(file_path),
                    "size_bytes": size,
                    "sha256": sha,
                    "bucket": bucket,
                }
            )
            by_bucket[bucket] += size
    total = sum(by_bucket.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_size_bytes": total,
        "total_size_gb": bytes_to_gb(total),
        "by_bucket": by_bucket,
        "by_bucket_gb": {bucket: bytes_to_gb(size) for bucket, size in by_bucket.items()},
        "items": items,
        "duplicates": group_duplicates(items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk corpus roots, compute SHA256, emit inventory + duplicates.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--report", default="reports/corpus_inventory.json")
    parser.add_argument("--duplicates-report", default="reports/duplicates_to_remove.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    paths = config["paths"]
    roots = {
        "hf_datasets": Path(paths["raw_hf_root"]),
        "kaggle": Path(paths["raw_kaggle_root"]),
        "models": Path(paths["model_root"]),
        "papers_and_derived": Path(paths["raw_paper_root"]),
    }
    inventory = build_inventory(roots)
    write_json(args.report, inventory)
    write_json(args.duplicates_report, {"duplicates": inventory["duplicates"]})
    console.print(f"Wrote {args.report} ({inventory['total_size_gb']} GB across {len(inventory['items'])} files).")
    console.print(f"Duplicate groups: {len(inventory['duplicates'])}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_dedupe_and_verify.py -v`
Expected: all four tests pass.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check scripts/dedupe_and_verify.py tests/test_dedupe_and_verify.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/dedupe_and_verify.py tests/test_dedupe_and_verify.py
git commit -m "feat: add dedupe_and_verify with SHA256 inventory and duplicate grouping"
```

---

## Task 13: Run all dry-run plans and commit reports

**Files:**
- Modify: `reports/hf_download_plan.json`
- Create/Modify: `reports/kaggle_download_plan.json`
- Create/Modify: `reports/paper_downloads_plan.json`

- [ ] **Step 1: HF dry-run**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --dry-run --max-gb 90 --types dataset model --sort priority --report reports/hf_download_plan.json`
Expected: prints plan table; writes report; exit 0; `planned_download_gb` ≤ 90.

- [ ] **Step 2: Kaggle dry-run**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --dry-run --report reports/kaggle_download_plan.json`
Expected: prints plan; writes report; exit 0; `planned_download_gb` ≤ 40.

- [ ] **Step 3: Papers dry-run**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_papers.py --dry-run --report reports/paper_downloads_plan.json`
Expected: prints plan; writes report; exit 0; status `would_download` or `present` for every open-text item.

- [ ] **Step 4: Inspect the three plans**

Run: `cd /Users/dmr/MachineLearning && python - <<'PY'
import json
for label, path in [("HF", "reports/hf_download_plan.json"), ("Kaggle", "reports/kaggle_download_plan.json"), ("Papers", "reports/paper_downloads_plan.json")]:
    data = json.load(open(path))
    summary = data.get("summary") or {"papers": len(data.get("papers", []))}
    print(label, summary)
PY`
Expected: prints the three summaries.

- [ ] **Step 5: Commit reports**

```bash
git add reports/hf_download_plan.json reports/kaggle_download_plan.json reports/paper_downloads_plan.json
git commit -m "chore: refresh download dry-run plans (HF, Kaggle, papers)"
```

---

## Task 14: Operator action — credentials

This task does not modify the repo. It blocks the next download tasks.

- [ ] **Step 1: User places Kaggle token**

The user runs (outside this session):
```bash
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
Verify with: `cd /Users/dmr/MachineLearning && uv run kaggle --version`
Expected: prints a version like `Kaggle API 1.6.x`.

- [ ] **Step 2: User accepts Kaggle competition rules**

For each competition in `manifests/kaggle.yaml`, the user visits the URL `https://www.kaggle.com/competitions/<id>/rules` and clicks "I Understand and Accept". Without this, downloads return 403.

- [ ] **Step 3: User logs into Hugging Face**

The user runs:
```bash
cd /Users/dmr/MachineLearning && uv run huggingface-cli login
```
And accepts model licenses on the HF web UI for any gated 14B+ models in the manifest.

- [ ] **Step 4: Re-run dry-runs to confirm credentials**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --dry-run --report reports/kaggle_download_plan.json`
Expected: same as Task 13 Step 2, but now `kaggle --version` works.

---

## Task 15: Download papers

**Files:**
- Generated: `data/raw/papers/**`
- Modify: `reports/paper_downloads.json`

- [ ] **Step 1: Execute paper downloads**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_papers.py --report reports/paper_downloads.json`
Expected: prints status table; writes report; exit 0. Most open-text arXiv items end in status `downloaded` or `present`.

- [ ] **Step 2: Verify total paper size**

Run: `cd /Users/dmr/MachineLearning && du -sh data/raw/papers`
Expected: between 0.5 GB and 5 GB.

- [ ] **Step 3: Re-run chunker to produce JSONL**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/prepare_research_corpus.py`
Expected: prints `Wrote N research chunks to data/processed/research/research_corpus.jsonl` with N ≥ 1000.

- [ ] **Step 4: Commit the refreshed paper plan**

```bash
git add reports/paper_downloads.json
git commit -m "chore: record paper download results"
```

---

## Task 16: Convert papers → Parquet shards → instruction Q&A

**Files:**
- Generated: `data/processed/research/parquet/shard_NNNNN.parquet`
- Generated: `data/processed/research/instructions.jsonl`

- [ ] **Step 1: Produce Parquet shards**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/paper_corpus_to_parquet.py`
Expected: prints `Wrote N parquet shards to data/processed/research/parquet`.

- [ ] **Step 2: Spot-check one shard**

Run: `cd /Users/dmr/MachineLearning && python - <<'PY'
import polars as pl
from pathlib import Path
shards = sorted(Path('data/processed/research/parquet').glob('shard_*.parquet'))
df = pl.read_parquet(shards[0])
print(df.shape, df.columns)
print(df.head(2))
PY`
Expected: prints the shape (N rows, 7 columns) and the expected schema.

- [ ] **Step 3: Generate Q&A on a limited slice first (smoke)**

Requires the primary or fallback model to have been downloaded (Task 17 covers this). If a model is already present (e.g. `roneneldan_TinyStories-33M` from a prior session), proceed; otherwise jump to Task 17 first, then return.

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py --limit-chunks 5`
Expected: prints loaded model line; writes ~15 records to `data/processed/research/instructions.jsonl`.

- [ ] **Step 4: Generate full Q&A corpus**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py`
Expected: appends remaining records; final file size 0.5–5 GB depending on chunk count.

- [ ] **Step 5: Commit nothing — outputs are git-ignored under `data/`. Move on.**

No commit required for data outputs.

---

## Task 17: Download HF models (small first, GGUFs last)

**Files:**
- Generated: `models/huggingface/**`
- Modify: `reports/hf_download_plan.json`

- [ ] **Step 1: Download small instruct + retrieval first (priority sort, models only, 12 GB cap)**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --sort priority --max-gb 12 --report reports/hf_download_plan_models_small.json`
Expected: downloads Qwen2.5-0.5B-Instruct, TinyStories-33M, all-MiniLM-L6-v2, FinBERT, DistilBERT, Qwen2.5-Coder-1.5B. Disk delta 4–10 GB.

- [ ] **Step 2: Download Apple Silicon MLX 7B**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --sort priority --max-gb 8 --report reports/hf_download_plan_models_mlx.json`
Expected: downloads `mlx-community/Qwen2.5-7B-Instruct-4bit` if budget permits. Disk delta 3–5 GB.

- [ ] **Step 3: Download 14B and 22B GGUFs**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --sort priority --max-gb 25 --report reports/hf_download_plan_models_gguf.json`
Expected: downloads only Q4_K_M shards via the manifest `allow_patterns`. Disk delta ~20 GB.

- [ ] **Step 4: Download 34B GGUF last**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --sort priority --max-gb 22 --report reports/hf_download_plan_models_34b.json`
Expected: downloads Yi-1.5-34B Q4_K_M only. Disk delta ~20 GB.

- [ ] **Step 5: Verify total model size**

Run: `cd /Users/dmr/MachineLearning && du -sh models/huggingface`
Expected: 35–45 GB.

- [ ] **Step 6: Commit refreshed reports**

```bash
git add reports/hf_download_plan_models_*.json
git commit -m "chore: record HF model download phases"
```

---

## Task 18: Download HF datasets

**Files:**
- Generated: `data/raw/huggingface/**`
- Modify: `reports/hf_download_plan.json`

- [ ] **Step 1: Download priority-sorted datasets under 50 GB cap**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types dataset --sort priority --max-gb 50 --report reports/hf_download_plan_datasets.json`
Expected: walks the dataset manifest and downloads enabled entries until 50 GB is exhausted. Skipped entries are recorded with `skip_budget` or `skip_unknown_size`.

- [ ] **Step 2: Verify total dataset size**

Run: `cd /Users/dmr/MachineLearning && du -sh data/raw/huggingface`
Expected: 45–55 GB.

- [ ] **Step 3: Commit**

```bash
git add reports/hf_download_plan_datasets.json
git commit -m "chore: record HF dataset download phase"
```

---

## Task 19: Download Kaggle artifacts

**Files:**
- Generated: `data/raw/kaggle/**`
- Modify: `reports/kaggle_download_plan.json`

- [ ] **Step 1: Execute Kaggle downloads**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --sort priority --report reports/kaggle_download_plan.json`
Expected: downloads competitions/datasets up to the 40 GB bucket budget. Any 403 entries get `skip_rules_not_accepted` and the user must accept on the Kaggle website before re-running.

- [ ] **Step 2: Verify Kaggle directory size**

Run: `cd /Users/dmr/MachineLearning && du -sh data/raw/kaggle`
Expected: 30–45 GB.

- [ ] **Step 3: Commit**

```bash
git add reports/kaggle_download_plan.json
git commit -m "chore: record Kaggle download phase"
```

---

## Task 20: Final dedupe + inventory

**Files:**
- Generated: `reports/corpus_inventory.json`
- Generated: `reports/duplicates_to_remove.json`

- [ ] **Step 1: Build inventory**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/dedupe_and_verify.py`
Expected: prints two lines (size + dup-group count). Writes `reports/corpus_inventory.json` and `reports/duplicates_to_remove.json`.

- [ ] **Step 2: Verify total in 140–160 GB window**

Run: `cd /Users/dmr/MachineLearning && python - <<'PY'
import json
data = json.load(open("reports/corpus_inventory.json"))
gb = data["total_size_gb"]
print("Total GB:", gb)
print("By bucket GB:", data["by_bucket_gb"])
print("Duplicate groups:", len(data["duplicates"]))
assert 140 <= gb <= 165, f"corpus size out of target: {gb}"
PY`
Expected: GB in window, assertion passes.

- [ ] **Step 3: If duplicates exist, list them for manual review**

Run: `cd /Users/dmr/MachineLearning && python - <<'PY'
import json
dups = json.load(open("reports/duplicates_to_remove.json"))["duplicates"]
for group in dups[:20]:
    print(group["sha256"][:12], "x", len(group["paths"]))
    for p in group["paths"]:
        print("  ", p)
PY`
Expected: empty or short list. If any group is large, decide manually which path to remove and re-run dedupe; do not auto-delete.

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Run ruff**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check src scripts tests`
Expected: no errors.

- [ ] **Step 6: Commit the inventory**

```bash
git add reports/corpus_inventory.json reports/duplicates_to_remove.json
git commit -m "chore: produce final corpus inventory and duplicate report"
```

---

## Task 21: Final README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a "Corpus" section to README**

Append this section to the existing `README.md`:

```markdown

## Corpus

Reproduce the 150 GB corpus by running:

```bash
uv sync --extra dev --extra llm
# Place Kaggle token at ~/.kaggle/kaggle.json (chmod 600)
huggingface-cli login   # optional but required for gated models

PYTHONPATH=src uv run python scripts/download_papers.py
PYTHONPATH=src uv run python scripts/prepare_research_corpus.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_parquet.py
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --max-gb 40
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types dataset --max-gb 50
PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py
PYTHONPATH=src uv run python scripts/dedupe_and_verify.py
```

The total on-disk corpus targets 140–160 GB. Manifests under `manifests/` define every artifact; budgets and bucket caps live in `configs/stack.yaml`. The final report at `reports/corpus_inventory.json` records SHA256, size, bucket, and duplicate groups.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Corpus section with reproduction commands"
```

---

## Self-review

Spec coverage:

- §3 bucket allocation → Tasks 1, 5, 6, 7, 8 (manifest sources) + 15, 17, 18, 19 (execution)
- §4 topic taxonomy → topics fields in Tasks 4 (Kaggle), 5 (HF datasets), 6 (HF models)
- §5 repository layout delta → Tasks 1 (config), 4 (kaggle.yaml), 5 (datasets.yaml), 6 (models.yaml), 7 (papers.yaml), 8 (kaggle_artifacts.py), 9 (download_kaggle_artifacts.py), 10 (paper_corpus_to_parquet.py), 11 (paper_corpus_to_instructions.py), 12 (dedupe_and_verify.py), 2 (pyproject)
- §6 contracts → covered by Tasks 4, 8, 9, 10, 11, 12
- §7 budget enforcement → Task 1
- §8 no-overlap → Task 12 (dedupe), plus manifest group/topic discipline in 4–7
- §9 authentication → Task 14
- §10 Apple Silicon safety → Task 11 (batch=1, MPS), Task 17 (smallest-first)
- §11 execution order → Tasks 1–21 follow the spec order
- §12 testing → Tasks 3, 8, 10, 12 add unit tests; Task 20 runs `pytest -q` and `ruff`
- §13 completion criteria → Task 20 asserts 140–160 GB and runs the full suite
- §14 risks → addressed in Tasks 9 (skip_rules_not_accepted), 11 (model fallback), 12 (dedupe report not auto-delete)

Placeholder scan: no TBD, TODO, "later", "similar to". Every code step shows full code.

Type consistency: `KaggleItem` fields used identically in `kaggle_artifacts.py`, tests, and `download_kaggle_artifacts.py`. `ModelChoice.local_dir` resolved via `safe_repo_id(repo_id)` matches the directory naming used by the existing `download_hf_artifacts.py` (which uses `snapshot_download` with `local_dir=target/safe_repo_id(item.id)` per the existing pattern in `models/huggingface/`). `record_to_row` schema matches what `dedupe_and_verify.py` consumes from the on-disk Parquet (it walks file bytes, not Parquet rows, so no cross-task type coupling needed).
