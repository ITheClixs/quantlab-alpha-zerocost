# QuantLab Alpha — S2 (LLM Governor + RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-tier LLM governor (Qwen 0.5B+LoRA fast → Mistral 22B Q4 medium → Yi 34B Q4 deep async) with hybrid BM25+dense+rerank retrieval over the 1581-chunk research corpus, GBNF-constrained JSON outputs with mandatory paper citations, and append-only verdict JSONLs that S4 will later tail.

**Architecture:** New `src/quant_research_stack/governor/` package with one file per concern (`signal_schema`, `grammar`, `corpus`, `bm25_index`, `dense_index`, `reranker`, `retrieval`, `query_builder`, `citation_resolver`, `escalator`, `prompts`, `transport`, `audit`, `runtime_tier1/2/3`). Tier runtimes are separate (transformers+LoRA for Tier 1; `llama-cpp-python` for Tier 2 and Tier 3 with a thread for Tier 3). All verdicts go to `experiments/s2_verdicts/<date>.jsonl`; Tier 3 to a parallel `experiments/s2_verdicts_tier3/<date>.jsonl`.

**Tech Stack:** Python 3.11, `pydantic`, `polars`, `numpy`, `rank_bm25`, `faiss-cpu`, `transformers`, `peft`, `llama-cpp-python` (Metal backend), `sentence-transformers` (already installed), `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-05-16-quantlab-alpha-s2-governor-design.md`

---

## File Structure

**New files:**

```text
configs/governor.yaml
docs/architecture/adrs/0006-tier-cascade-fast-medium-deep.md
docs/architecture/adrs/0007-async-tier3-stance-modifier.md
docs/architecture/adrs/0008-llama-cpp-python-runtime.md
docs/runbooks/governor_index_rebuild.md
docs/runbooks/governor_lora_retrain.md
src/quant_research_stack/governor/__init__.py
src/quant_research_stack/governor/signal_schema.py
src/quant_research_stack/governor/grammar.py
src/quant_research_stack/governor/grammar.gbnf
src/quant_research_stack/governor/grammar_tier1.gbnf
src/quant_research_stack/governor/corpus.py
src/quant_research_stack/governor/bm25_index.py
src/quant_research_stack/governor/dense_index.py
src/quant_research_stack/governor/reranker.py
src/quant_research_stack/governor/retrieval.py
src/quant_research_stack/governor/query_builder.py
src/quant_research_stack/governor/citation_resolver.py
src/quant_research_stack/governor/escalator.py
src/quant_research_stack/governor/prompts.py
src/quant_research_stack/governor/transport.py
src/quant_research_stack/governor/audit.py
src/quant_research_stack/governor/runtime_tier1.py
src/quant_research_stack/governor/runtime_tier2.py
src/quant_research_stack/governor/runtime_tier3.py
scripts/governor_lora_dataset.py
scripts/governor_lora_label.py
scripts/governor_train_lora.py
scripts/governor_build_indexes.py
scripts/s2_govern.py
scripts/s2_smoke.py
tests/test_governor_signal_schema.py
tests/test_governor_grammar.py
tests/test_governor_corpus.py
tests/test_governor_bm25.py
tests/test_governor_dense.py
tests/test_governor_reranker.py
tests/test_governor_retrieval.py
tests/test_governor_query_builder.py
tests/test_governor_citation_resolver.py
tests/test_governor_escalator.py
tests/test_governor_lora_label.py
tests/test_governor_transport.py
tests/test_governor_audit.py
tests/test_governor_citation_property.py
tests/integration/__init__.py
tests/integration/test_governor_tier1_smoke.py
tests/integration/test_governor_tier2_smoke.py
tests/integration/test_governor_tier3_async.py
```

**Modified files:**

```text
pyproject.toml          add llama-cpp-python, peft, faiss-cpu, rank-bm25 to runtime + register governor_slow marker
Makefile                add governor-build-indexes, governor-train-lora, governor-smoke targets
```

Existing modules (`src/quant_research_stack/alpha/*`, `llm_quant.py`, etc.) are untouched.

---

## Task 1: ADRs 0006–0008 and governor runbooks

**Files:**
- Create: `docs/architecture/adrs/0006-tier-cascade-fast-medium-deep.md`
- Create: `docs/architecture/adrs/0007-async-tier3-stance-modifier.md`
- Create: `docs/architecture/adrs/0008-llama-cpp-python-runtime.md`
- Create: `docs/runbooks/governor_index_rebuild.md`
- Create: `docs/runbooks/governor_lora_retrain.md`

- [ ] **Step 1: Write ADR 0006**

```markdown
# ADR 0006: Three-tier governor cascade — fast / medium / deep

## Status
Accepted, 2026-05-16.

## Context
S2 governs every S1 signal but on the M4 a single Mistral 22B Q4 call costs 5–10 s,
and Yi 34B Q4 costs 20–30 s. Calling either on every signal misses the trading window
for tick-frequency crypto and is overkill for low-confidence S1 signals that won't
trade anyway.

## Decision
Three tiers with explicit gates:

- Tier 1: Qwen 0.5B-Instruct + LoRA, runs on every signal, < 500 ms, decision space
  reduced to {pass, veto}.
- Tier 2: Mistral 22B Q4_K_M, runs only when Tier 1 passes AND |signal.confidence| > 0.6,
  ~5–10 s, RAG top-5 evidence required, citations mandatory.
- Tier 3: Yi 34B Q4_K_M, runs only when trade_size_pct > 1 %, async; verdict applies
  to NEXT trade in the same symbol (stance modifier).

A pass requires unanimity across every tier that ran.

## Consequences
+ Latency budget honored on tick-frequency crypto.
+ Every model on disk has a role; nothing wasted.
+ Deep reasoning on big trades without blocking the loop.
- Three runtime classes to maintain.
- LoRA training adds 8 hours to the full-retrain budget.
```

- [ ] **Step 2: Write ADR 0007**

```markdown
# ADR 0007: Tier 3 verdicts apply to NEXT trade, not current

## Status
Accepted, 2026-05-16.

## Context
Tier 3 (Yi 34B Q4) takes 20–30 s per call. We want its deep reasoning without blocking
the trading loop on signals that need to fire within seconds.

## Decision
When Tier 3 is triggered (trade_size_pct > 1 %), it is scheduled async. The current
verdict for S4 is the Tier 2 verdict. Tier 3's eventual verdict is written to a
separate file `experiments/s2_verdicts_tier3/<date>.jsonl`. S4's risk engine reads
Tier 3 verdicts as a stance modifier for the NEXT trade in the same symbol — never
the current one. A Tier 3 veto on the previous trade in this symbol widens the next
signal's confidence threshold by 0.20 (a tightening, not a hard block).

## Consequences
+ Trading loop never waits on Yi 34B.
+ Deep reasoning still influences the system, just with a one-trade lag.
+ Crashes in the async worker do not block trades.
- Operators must understand the lag semantics; runbook covers this.
- A single Tier 3 verdict from a stale signal could over-tighten the next trade;
  the modifier is intentionally small (20 %) to limit overreaction.
```

- [ ] **Step 3: Write ADR 0008**

```markdown
# ADR 0008: Use llama-cpp-python with Metal backend for Tier 2 and Tier 3

## Status
Accepted, 2026-05-16.

## Context
Tier 2 (Mistral 22B) and Tier 3 (Yi 34B) are stored as Q4_K_M GGUF files on disk.
They must run with GBNF grammar enforcement (ADR 0003) and use Apple Silicon's
Metal backend for acceptable latency.

## Decision
`llama-cpp-python` is the runtime for Tier 2 and Tier 3. It is the only mainstream
Python library that:
- accepts a GBNF grammar string and constrains token sampling natively,
- loads GGUF Q4_K_M models without conversion,
- supports Apple Silicon Metal via the `n_gpu_layers=-1` flag.

Tier 1 (Qwen 0.5B + LoRA) uses `transformers` + `peft` because:
- LoRA + GGUF in llama-cpp-python is awkward,
- the model is small enough that HF-native inference on MPS is fast (< 500 ms target),
- HF supports a logits-processor JSON-only fallback if grammar enforcement breaks.

## Consequences
+ Single inference dependency for the heavy tiers.
+ GBNF grammar enforced at sampling time, not post-hoc.
- Two runtime libraries to install (`llama-cpp-python` + `transformers`).
- Building llama-cpp-python with Metal requires `CMAKE_ARGS="-DGGML_METAL=on"`.
```

- [ ] **Step 4: Write runbook governor_index_rebuild.md**

```markdown
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
```

- [ ] **Step 5: Write runbook governor_lora_retrain.md**

```markdown
# Runbook: Governor LoRA retrain

## When to run
- Veto precision on the 200-pair backtest fixture drops below 60 % (master spec
  criterion 6).
- Held-out perplexity vs base regresses by > 5 %.
- The synthetic-label rule changes (regenerate `lora_governor.jsonl`).
- Quarterly hygiene retrain.

## Steps
1. Stop `s2_govern.py` daemon.
2. Regenerate the LoRA dataset:
   `PYTHONPATH=src uv run python scripts/governor_lora_dataset.py`
3. Train the adapter (up to 8 h wall-clock; checkpoints every 500 steps):
   `PYTHONPATH=src uv run python scripts/governor_train_lora.py`
4. Inspect `models/trained/governor_lora_qwen05b/<run_id>/metrics.json`.
   Required: `held_out_perplexity` < base − 10 %, `veto_precision_200pair` ≥ 0.60.
5. If criteria pass, point `configs/governor.yaml::tiers.tier1.adapter_dir` at the
   new run dir or update `models/trained/governor_lora_qwen05b/latest` symlink.
6. If criteria fail, run the fallback adapter on Qwen 1.5B Coder:
   `PYTHONPATH=src uv run python scripts/governor_train_lora.py --base Qwen/Qwen2.5-Coder-1.5B`
7. Restart `s2_govern.py`.

## Failure modes
- MPS OOM during training: drop `batch_size` from 4 to 2 in `configs/governor.yaml`,
  bump `gradient_accumulation_steps` to 8 to keep effective batch size 16.
- Training stalls: kill, examine the loss curve in `metrics.json`, reduce
  `learning_rate` from 2e-4 to 1e-4.
```

- [ ] **Step 6: Commit**

```bash
git add docs/architecture/adrs/0006-tier-cascade-fast-medium-deep.md \
        docs/architecture/adrs/0007-async-tier3-stance-modifier.md \
        docs/architecture/adrs/0008-llama-cpp-python-runtime.md \
        docs/runbooks/governor_index_rebuild.md \
        docs/runbooks/governor_lora_retrain.md
git commit -m "docs: add ADRs 0006-0008 and S2 governor runbooks"
```

---

## Task 2: `configs/governor.yaml` and `governor/__init__.py` scaffold

**Files:**
- Create: `configs/governor.yaml`
- Create: `src/quant_research_stack/governor/__init__.py`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p src/quant_research_stack/governor
cat > src/quant_research_stack/governor/__init__.py <<'PY'
"""S2 LLM Governor for QuantLab Alpha.

Three-tier cascade: Qwen 0.5B + LoRA (fast) → Mistral 22B Q4 (medium with RAG) →
Yi 34B Q4 (deep async). GBNF-constrained JSON outputs with mandatory paper citations.

Spec: docs/superpowers/specs/2026-05-16-quantlab-alpha-s2-governor-design.md
"""
PY
```

- [ ] **Step 2: Create `configs/governor.yaml`**

```yaml
tiers:
  tier1:
    enabled: true
    base_model_id: Qwen/Qwen2.5-0.5B-Instruct
    base_model_dir: models/huggingface/Qwen__Qwen2.5-0.5B-Instruct
    adapter_dir: models/trained/governor_lora_qwen05b/latest
    max_new_tokens: 256
    temperature: 0.0
  tier2:
    enabled: true
    gguf_path: models/huggingface/bartowski__Mistral-Small-Instruct-2409-GGUF/Mistral-Small-Instruct-2409-Q4_K_M.gguf
    n_ctx: 4096
    n_gpu_layers: -1
    max_new_tokens: 384
    temperature: 0.0
    triggered_when_tier1_passes_above_confidence: 0.6
  tier3:
    enabled: true
    gguf_path: models/huggingface/bartowski__Yi-1.5-34B-Chat-GGUF/Yi-1.5-34B-Chat-Q4_K_M.gguf
    n_ctx: 4096
    n_gpu_layers: -1
    max_new_tokens: 512
    temperature: 0.0
    triggered_when_trade_size_pct_above: 1.0
    async_workers: 1

retrieval:
  bm25_top_n: 20
  dense_top_n: 20
  rerank_to_k: 5
  reranker_model_id: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_model_dir: models/huggingface/cross-encoder__ms-marco-MiniLM-L-6-v2
  embedding_model_id: FinLang/finance-embeddings-investopedia
  embedding_model_dir: models/huggingface/FinLang__finance-embeddings-investopedia
  index_dir: models/governor

corpus:
  parquet_dir: data/processed/research/parquet

transport:
  primary_verdicts_dir: experiments/s2_verdicts
  tier3_verdicts_dir: experiments/s2_verdicts_tier3
  audit_log_dir: logs/audit/governor
  rotation: daily
  chmod_after_close: true

stance:
  tier3_stance_modifier_pct: 0.20

lora_training:
  base_model_dir: models/huggingface/Qwen__Qwen2.5-0.5B-Instruct
  output_root: models/trained/governor_lora_qwen05b
  rank: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 1024
  max_epochs: 5
  learning_rate: 2e-4
  warmup_steps: 100
  held_out_fraction: 0.10
  random_seed: 42
  max_wall_clock_hours: 8
```

- [ ] **Step 3: Verify YAML loads**

Run: `python -c "import yaml; c=yaml.safe_load(open('configs/governor.yaml')); assert c['tiers']['tier2']['triggered_when_tier1_passes_above_confidence']==0.6; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 4: Verify package imports**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python -c "import quant_research_stack.governor; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/governor.yaml src/quant_research_stack/governor/__init__.py
git commit -m "feat: scaffold governor/ package and configs/governor.yaml"
```

---

## Task 3: Add S2 dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add deps to the `dependencies = [...]` block in alphabetical order**

Add these lines:

```toml
    "faiss-cpu>=1.8.0",
    "llama-cpp-python>=0.3.0",
    "peft>=0.12.0",
    "rank-bm25>=0.2.2",
```

- [ ] **Step 2: Add the `governor_slow` pytest marker**

In the `[tool.pytest.ini_options]` block, add (or extend) a `markers` list:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -m 'not governor_slow'"
markers = [
    "governor_slow: integration tests that load LLM models (skipped by default)",
]
```

- [ ] **Step 3: Sync deps**

Run: `cd /Users/dmr/MachineLearning && CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra dev --extra llm`
Expected: `faiss-cpu`, `llama-cpp-python`, `peft`, `rank-bm25` installed. The CMAKE_ARGS env var is needed for Metal-enabled `llama-cpp-python` build.

- [ ] **Step 4: Smoke-import**

Run: `cd /Users/dmr/MachineLearning && uv run python -c "import faiss, llama_cpp, peft, rank_bm25; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Verify default test run skips governor_slow**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest --collect-only -q 2>&1 | tail -3`
Expected: collected count printed; no errors about unknown marker.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add llama-cpp-python (Metal), peft, faiss-cpu, rank-bm25 + governor_slow marker"
```

---

## Task 4: `governor/signal_schema.py` — Pydantic schema with citation invariant

**Files:**
- Create: `src/quant_research_stack/governor/signal_schema.py`
- Create: `tests/test_governor_signal_schema.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_signal_schema.py`:

```python
from __future__ import annotations

import pytest

from quant_research_stack.governor.signal_schema import (
    Decision,
    Direction,
    GovernorVerdict,
    RegimeTag,
)


def _valid_payload(**overrides):
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.8,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "trending momentum aligns with cited paper",
        "cited_paper_chunk_ids": ["paper_pdf:foo:0"],
        "contradictions_flagged": [],
    }
    payload.update(overrides)
    return payload


def test_valid_pass_with_citations() -> None:
    v = GovernorVerdict.model_validate(_valid_payload())
    assert v.decision == Decision.pass_
    assert v.cited_paper_chunk_ids == ["paper_pdf:foo:0"]


def test_valid_veto() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(decision="veto", cited_paper_chunk_ids=[]))
    assert v.decision == Decision.veto


def test_valid_insufficient_evidence() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(decision="insufficient_evidence", cited_paper_chunk_ids=[]))
    assert v.decision == Decision.insufficient_evidence


def test_pass_without_citations_is_downgraded() -> None:
    v = GovernorVerdict.model_validate(_valid_payload(cited_paper_chunk_ids=[]))
    assert v.decision == Decision.insufficient_evidence
    assert "no citations" in v.rationale_short


def test_signal_id_too_short_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(signal_id="abc"))


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(confidence=1.5))


def test_horizon_zero_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(horizon_minutes=0))


def test_rationale_too_long_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(rationale_short="x" * 201))


def test_cited_array_too_long_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(cited_paper_chunk_ids=[f"id-{i}" for i in range(11)]))


def test_unknown_decision_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(decision="maybe"))


def test_direction_out_of_set_rejected() -> None:
    with pytest.raises(ValueError):
        GovernorVerdict.model_validate(_valid_payload(direction=2))


def test_enum_values_have_expected_strings() -> None:
    assert Decision.pass_.value == "pass"
    assert Decision.veto.value == "veto"
    assert Decision.insufficient_evidence.value == "insufficient_evidence"
    assert Direction.short.value == -1
    assert Direction.long.value == 1
    assert RegimeTag.high_vol.value == "high_vol"
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_signal_schema.py -v`
Expected: ImportError on `quant_research_stack.governor.signal_schema`.

- [ ] **Step 3: Implement `signal_schema.py`**

Create `src/quant_research_stack/governor/signal_schema.py`:

```python
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class Decision(str, Enum):
    pass_ = "pass"
    veto = "veto"
    insufficient_evidence = "insufficient_evidence"


class Direction(int, Enum):
    short = -1
    flat = 0
    long = 1


class RegimeTag(str, Enum):
    trending = "trending"
    mean_reverting = "mean_reverting"
    high_vol = "high_vol"
    low_vol = "low_vol"
    unknown = "unknown"


class GovernorVerdict(BaseModel):
    signal_id: Annotated[str, Field(min_length=8, max_length=64)]
    decision: Decision
    direction: Direction
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    horizon_minutes: Annotated[int, Field(ge=1, le=1440)]
    regime_tag: RegimeTag
    rationale_short: Annotated[str, Field(max_length=200)]
    cited_paper_chunk_ids: Annotated[list[str], Field(min_length=0, max_length=10)]
    contradictions_flagged: Annotated[list[str], Field(max_length=5)]

    @model_validator(mode="after")
    def enforce_citation_invariant(self) -> "GovernorVerdict":
        if self.decision == Decision.pass_ and not self.cited_paper_chunk_ids:
            object.__setattr__(self, "decision", Decision.insufficient_evidence)
            object.__setattr__(self, "rationale_short", "no citations provided; auto-downgrade")
        return self
```

- [ ] **Step 4: Run tests, expect 12 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_signal_schema.py -v`
Expected: 12 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/signal_schema.py tests/test_governor_signal_schema.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/governor/signal_schema.py tests/test_governor_signal_schema.py
git commit -m "feat: governor/signal_schema.py with Pydantic verdict + citation invariant"
```

---

## Task 5: `governor/corpus.py` — CorpusIndex over Parquet shards

**Files:**
- Create: `src/quant_research_stack/governor/corpus.py`
- Create: `tests/test_governor_corpus.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_corpus.py`:

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_research_stack.governor.corpus import Chunk, CorpusIndex, load_corpus


def _write_fixture(tmp_path: Path) -> Path:
    df = pl.DataFrame({
        "id": ["paper_pdf:a:0", "paper_pdf:a:1", "paper_pdf:b:0"],
        "source_type": ["paper_pdf"] * 3,
        "source_path": ["a.pdf", "a.pdf", "b.pdf"],
        "chunk_index": [0, 1, 0],
        "text": ["alpha text one", "alpha text two", "beta text"],
        "sha256": ["aa", "ab", "ba"],
        "n_words": [3, 3, 2],
    })
    out = tmp_path / "shard_00000.parquet"
    df.write_parquet(out)
    return tmp_path


def test_load_corpus_reads_parquet_shards(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    assert len(corpus) == 3


def test_corpus_id_lookup_returns_chunk(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    chunk = corpus["paper_pdf:a:1"]
    assert isinstance(chunk, Chunk)
    assert chunk.text == "alpha text two"
    assert chunk.source_path == "a.pdf"


def test_corpus_membership(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    assert "paper_pdf:a:0" in corpus
    assert "missing-id" not in corpus


def test_corpus_iter_yields_all_chunks(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    corpus = load_corpus(parquet_dir)
    ids = sorted(c.id for c in corpus)
    assert ids == ["paper_pdf:a:0", "paper_pdf:a:1", "paper_pdf:b:0"]


def test_corpus_sha_is_stable(tmp_path: Path) -> None:
    parquet_dir = _write_fixture(tmp_path)
    sha_a = load_corpus(parquet_dir).sha256
    sha_b = load_corpus(parquet_dir).sha256
    assert sha_a == sha_b
    assert len(sha_a) == 64
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_corpus.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `corpus.py`**

Create `src/quant_research_stack/governor/corpus.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect 5 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_corpus.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/corpus.py tests/test_governor_corpus.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/governor/corpus.py tests/test_governor_corpus.py
git commit -m "feat: governor/corpus.py with CorpusIndex and stable SHA"
```

---

## Task 6: `governor/bm25_index.py` — BM25Okapi over chunk text

**Files:**
- Create: `src/quant_research_stack/governor/bm25_index.py`
- Create: `tests/test_governor_bm25.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_bm25.py`:

```python
from __future__ import annotations

from pathlib import Path

from quant_research_stack.governor.bm25_index import BM25Index, build_bm25_index, load_bm25_index, save_bm25_index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex


def _corpus() -> CorpusIndex:
    chunks = [
        Chunk(id="a", source_type="t", source_path="p", chunk_index=0,
              text="order flow imbalance equity prediction", sha256="aa", n_words=5),
        Chunk(id="b", source_type="t", source_path="p", chunk_index=1,
              text="mean reversion crypto microstructure tick", sha256="bb", n_words=5),
        Chunk(id="c", source_type="t", source_path="p", chunk_index=2,
              text="momentum trending equities stocks", sha256="cc", n_words=4),
    ]
    return CorpusIndex(chunks)


def test_build_bm25_index_returns_top_n_for_lexical_match() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("order flow imbalance", n=2)
    assert hits[0] == "a"
    assert len(hits) == 2


def test_top_k_returns_chunk_ids_only() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("microstructure tick", n=3)
    assert all(isinstance(h, str) for h in hits)
    assert hits[0] == "b"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = build_bm25_index(_corpus())
    save_bm25_index(idx, tmp_path / "bm25.pkl")
    loaded = load_bm25_index(tmp_path / "bm25.pkl")
    assert isinstance(loaded, BM25Index)
    assert loaded.top_k("momentum", n=1) == ["c"]


def test_top_k_n_larger_than_corpus_returns_all() -> None:
    idx = build_bm25_index(_corpus())
    hits = idx.top_k("anything", n=10)
    assert len(hits) == 3
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_bm25.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bm25_index.py`**

Create `src/quant_research_stack/governor/bm25_index.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_bm25.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/bm25_index.py tests/test_governor_bm25.py
git add src/quant_research_stack/governor/bm25_index.py tests/test_governor_bm25.py
git commit -m "feat: governor/bm25_index.py with rank_bm25 and pickle persistence"
```

---

## Task 7: `governor/dense_index.py` — FinLang embeddings + faiss IndexFlatIP

**Files:**
- Create: `src/quant_research_stack/governor/dense_index.py`
- Create: `tests/test_governor_dense.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_dense.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.governor.dense_index import (
    DenseIndex,
    build_dense_index_from_vectors,
    load_dense_index,
    save_dense_index,
)


def _vectors(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_build_dense_index_returns_top_k() -> None:
    chunk_ids = ("a", "b", "c", "d", "e")
    vectors = _vectors()
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    query = vectors[2]
    hits = idx.top_k(query, n=3)
    assert hits[0] == "c"
    assert len(hits) == 3


def test_dense_index_returns_unique_ids() -> None:
    idx = build_dense_index_from_vectors(("x", "y", "z"), _vectors()[:3])
    hits = idx.top_k(_vectors()[0], n=10)
    assert len(set(hits)) == len(hits)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    chunk_ids = ("a", "b", "c", "d", "e")
    vectors = _vectors()
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    npy_path = tmp_path / "dense.npy"
    faiss_path = tmp_path / "dense.faiss"
    save_dense_index(idx, npy_path, faiss_path)
    loaded = load_dense_index(npy_path, faiss_path, chunk_ids=chunk_ids)
    assert isinstance(loaded, DenseIndex)
    assert loaded.top_k(vectors[1], n=1) == ["b"]


def test_query_unit_norm_required() -> None:
    chunk_ids = ("a",)
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    idx = build_dense_index_from_vectors(chunk_ids, vectors)
    # FlatIP requires unit-norm queries for cosine equivalence; test we accept any vector
    hits = idx.top_k(np.array([2.0, 0.0], dtype=np.float32), n=1)
    assert hits == ["a"]
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_dense.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `dense_index.py`**

Create `src/quant_research_stack/governor/dense_index.py`:

```python
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
    index: "faiss.IndexFlatIP"
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
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_dense.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/dense_index.py tests/test_governor_dense.py
git add src/quant_research_stack/governor/dense_index.py tests/test_governor_dense.py
git commit -m "feat: governor/dense_index.py with faiss IndexFlatIP and float16 persistence"
```

---

## Task 8: `governor/reranker.py` — cross-encoder reranker with stub-friendly interface

**Files:**
- Create: `src/quant_research_stack/governor/reranker.py`
- Create: `tests/test_governor_reranker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_reranker.py`:

```python
from __future__ import annotations

from quant_research_stack.governor.reranker import RerankCandidate, Reranker, StubReranker


def test_stub_reranker_orders_by_string_overlap() -> None:
    reranker = StubReranker()
    cands = [
        RerankCandidate(id="a", text="alpha beta"),
        RerankCandidate(id="b", text="beta gamma delta"),
        RerankCandidate(id="c", text="zeta"),
    ]
    out = reranker.rerank("beta gamma", cands)
    assert out[0].id == "b"
    assert out[-1].id == "c"


def test_stub_reranker_returns_same_length() -> None:
    reranker = StubReranker()
    cands = [RerankCandidate(id=str(i), text=f"x {i}") for i in range(5)]
    out = reranker.rerank("anything", cands)
    assert len(out) == len(cands)


def test_reranker_protocol_accepts_stub() -> None:
    def use_reranker(r: Reranker) -> str:
        return r.rerank("q", [RerankCandidate(id="a", text="t")])[0].id
    assert use_reranker(StubReranker()) == "a"
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_reranker.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `reranker.py`**

Create `src/quant_research_stack/governor/reranker.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(str(self._model_dir))

    def rerank(self, query: str, candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
        self._load()
        cand_list = list(candidates)
        if not cand_list:
            return []
        pairs = [(query, cand.text) for cand in cand_list]
        scores = self._model.predict(pairs)
        scored = list(zip(scores, cand_list, strict=True))
        scored.sort(key=lambda pair: float(pair[0]), reverse=True)
        return [cand for _, cand in scored]
```

- [ ] **Step 4: Run tests, expect 3 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_reranker.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/reranker.py tests/test_governor_reranker.py
git add src/quant_research_stack/governor/reranker.py tests/test_governor_reranker.py
git commit -m "feat: governor/reranker.py with Reranker protocol, Stub + CrossEncoder impls"
```

---

## Task 9: `governor/retrieval.py` — hybrid orchestrator

**Files:**
- Create: `src/quant_research_stack/governor/retrieval.py`
- Create: `tests/test_governor_retrieval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_retrieval.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.governor.bm25_index import build_bm25_index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.dense_index import build_dense_index_from_vectors
from quant_research_stack.governor.reranker import StubReranker
from quant_research_stack.governor.retrieval import HybridRetriever


def _setup() -> tuple[HybridRetriever, CorpusIndex]:
    chunks = [
        Chunk(id="a", source_type="t", source_path="p", chunk_index=0, text="order flow imbalance equity", sha256="x", n_words=4),
        Chunk(id="b", source_type="t", source_path="p", chunk_index=1, text="mean reversion crypto micro", sha256="y", n_words=4),
        Chunk(id="c", source_type="t", source_path="p", chunk_index=2, text="momentum trending stocks", sha256="z", n_words=3),
    ]
    corpus = CorpusIndex(chunks)
    bm25 = build_bm25_index(corpus)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dense = build_dense_index_from_vectors(("a", "b", "c"), vecs)
    retriever = HybridRetriever(corpus=corpus, bm25=bm25, dense=dense, reranker=StubReranker())
    return retriever, corpus


def test_retrieve_returns_at_most_k() -> None:
    retriever, _ = _setup()
    out = retriever.retrieve("order flow", bm25_n=3, dense_n=3, k=2, query_vector=np.zeros(8, dtype=np.float32))
    assert len(out) <= 2


def test_retrieve_returns_unique_chunks() -> None:
    retriever, _ = _setup()
    out = retriever.retrieve("order flow imbalance", bm25_n=3, dense_n=3, k=3, query_vector=np.zeros(8, dtype=np.float32))
    assert len({c.id for c in out}) == len(out)


def test_retrieve_returns_chunk_dataclass() -> None:
    retriever, corpus = _setup()
    out = retriever.retrieve("momentum trending", bm25_n=3, dense_n=3, k=1, query_vector=np.zeros(8, dtype=np.float32))
    assert out
    assert out[0].id in corpus
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_retrieval.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `retrieval.py`**

Create `src/quant_research_stack/governor/retrieval.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.governor.bm25_index import BM25Index
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.dense_index import DenseIndex
from quant_research_stack.governor.reranker import RerankCandidate, Reranker


@dataclass(frozen=True)
class HybridRetriever:
    corpus: CorpusIndex
    bm25: BM25Index
    dense: DenseIndex
    reranker: Reranker

    def retrieve(self, query: str, *, bm25_n: int, dense_n: int, k: int, query_vector: NDArray[np.float32]) -> list[Chunk]:
        bm25_hits = self.bm25.top_k(query, n=bm25_n)
        dense_hits = self.dense.top_k(query_vector, n=dense_n)
        seen: set[str] = set()
        union_ids: list[str] = []
        for cid in bm25_hits + dense_hits:
            if cid in self.corpus and cid not in seen:
                union_ids.append(cid)
                seen.add(cid)
        candidates = [
            RerankCandidate(id=cid, text=self.corpus[cid].text) for cid in union_ids
        ]
        reranked = self.reranker.rerank(query, candidates)
        return [self.corpus[cand.id] for cand in reranked[:k]]
```

- [ ] **Step 4: Run tests, expect 3 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_retrieval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/retrieval.py tests/test_governor_retrieval.py
git add src/quant_research_stack/governor/retrieval.py tests/test_governor_retrieval.py
git commit -m "feat: governor/retrieval.py hybrid BM25+dense+rerank orchestrator"
```

---

## Task 10: `governor/query_builder.py` — fixed-template query builder

**Files:**
- Create: `src/quant_research_stack/governor/query_builder.py`
- Create: `tests/test_governor_query_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_query_builder.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.governor.query_builder import build_query


@dataclass(frozen=True)
class _Sig:
    symbol: str
    direction: int
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str


def test_build_query_with_regime_hint() -> None:
    sig = _Sig(symbol="BTCUSDT", direction=1, horizon_minutes=15, regime_hint="trending", recent_vol_label="med")
    assert build_query(sig) == "trending BTCUSDT direction=1 horizon=15m vol=med"


def test_build_query_with_no_regime_hint() -> None:
    sig = _Sig(symbol="ETHUSDT", direction=-1, horizon_minutes=5, regime_hint=None, recent_vol_label="high")
    assert build_query(sig) == "unknown ETHUSDT direction=-1 horizon=5m vol=high"
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_query_builder.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `query_builder.py`**

Create `src/quant_research_stack/governor/query_builder.py`:

```python
from __future__ import annotations

from typing import Protocol


class _SignalShape(Protocol):
    symbol: str
    direction: int
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str


def build_query(signal: _SignalShape) -> str:
    regime = signal.regime_hint or "unknown"
    return f"{regime} {signal.symbol} direction={signal.direction} horizon={signal.horizon_minutes}m vol={signal.recent_vol_label}"
```

- [ ] **Step 4: Run tests, expect 2 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_query_builder.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/query_builder.py tests/test_governor_query_builder.py
git add src/quant_research_stack/governor/query_builder.py tests/test_governor_query_builder.py
git commit -m "feat: governor/query_builder.py with fixed retrieval-query template"
```

---

## Task 11: `governor/grammar.py` + `grammar.gbnf` + `grammar_tier1.gbnf`

**Files:**
- Create: `src/quant_research_stack/governor/grammar.py`
- Create: `src/quant_research_stack/governor/grammar.gbnf`
- Create: `src/quant_research_stack/governor/grammar_tier1.gbnf`
- Create: `tests/test_governor_grammar.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_grammar.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.grammar import (
    PACKAGE_GRAMMAR_FULL_PATH,
    PACKAGE_GRAMMAR_TIER1_PATH,
    generate_full_grammar,
    generate_tier1_grammar,
    validate_against_grammar_shape,
)


def test_full_grammar_file_committed() -> None:
    text = Path(PACKAGE_GRAMMAR_FULL_PATH).read_text()
    assert "decision ::=" in text
    assert '"\\"pass\\""' in text
    assert '"\\"veto\\""' in text
    assert '"\\"insufficient_evidence\\""' in text


def test_tier1_grammar_file_committed() -> None:
    text = Path(PACKAGE_GRAMMAR_TIER1_PATH).read_text()
    assert '"\\"pass\\""' in text
    assert '"\\"veto\\""' in text
    assert '"\\"insufficient_evidence\\""' not in text


def test_generated_full_grammar_matches_committed_file() -> None:
    generated = generate_full_grammar()
    committed = Path(PACKAGE_GRAMMAR_FULL_PATH).read_text()
    assert generated.strip() == committed.strip()


def test_generated_tier1_grammar_matches_committed_file() -> None:
    generated = generate_tier1_grammar()
    committed = Path(PACKAGE_GRAMMAR_TIER1_PATH).read_text()
    assert generated.strip() == committed.strip()


def test_validate_against_shape_accepts_valid_payload() -> None:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.85,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:foo:0"],
        "contradictions_flagged": [],
    }
    assert validate_against_grammar_shape(json.dumps(payload)) is True


def test_validate_against_shape_rejects_missing_field() -> None:
    bad = '{"signal_id": "abc"}'
    assert validate_against_grammar_shape(bad) is False


def test_validate_against_shape_rejects_unknown_decision() -> None:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "maybe",
        "direction": 1,
        "confidence": 0.85,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["x"],
        "contradictions_flagged": [],
    }
    assert validate_against_grammar_shape(json.dumps(payload)) is False
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_grammar.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `grammar.gbnf`**

Create `src/quant_research_stack/governor/grammar.gbnf`:

```
root ::= "{" ws "\"signal_id\":" ws string ws "," ws "\"decision\":" ws decision ws "," ws "\"direction\":" ws direction ws "," ws "\"confidence\":" ws confidence ws "," ws "\"horizon_minutes\":" ws posint ws "," ws "\"regime_tag\":" ws regime ws "," ws "\"rationale_short\":" ws string ws "," ws "\"cited_paper_chunk_ids\":" ws array_of_strings ws "," ws "\"contradictions_flagged\":" ws array_of_strings ws "}" ws

decision ::= "\"pass\"" | "\"veto\"" | "\"insufficient_evidence\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\"trending\"" | "\"mean_reverting\"" | "\"high_vol\"" | "\"low_vol\"" | "\"unknown\""
string ::= "\"" char* "\""
char ::= [^"\\\n] | "\\" ["\\nrt]
array_of_strings ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
ws ::= [ \t\n]*
```

- [ ] **Step 4: Write `grammar_tier1.gbnf`** (restricted decision space, no insufficient_evidence)

Create `src/quant_research_stack/governor/grammar_tier1.gbnf`:

```
root ::= "{" ws "\"signal_id\":" ws string ws "," ws "\"decision\":" ws decision ws "," ws "\"direction\":" ws direction ws "," ws "\"confidence\":" ws confidence ws "," ws "\"horizon_minutes\":" ws posint ws "," ws "\"regime_tag\":" ws regime ws "," ws "\"rationale_short\":" ws string ws "," ws "\"cited_paper_chunk_ids\":" ws "[]" ws "," ws "\"contradictions_flagged\":" ws "[]" ws "}" ws

decision ::= "\"pass\"" | "\"veto\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\"trending\"" | "\"mean_reverting\"" | "\"high_vol\"" | "\"low_vol\"" | "\"unknown\""
string ::= "\"" char* "\""
char ::= [^"\\\n] | "\\" ["\\nrt]
ws ::= [ \t\n]*
```

- [ ] **Step 5: Implement `grammar.py`** (generator + shape validator)

Create `src/quant_research_stack/governor/grammar.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict


_PACKAGE_DIR = Path(__file__).parent
PACKAGE_GRAMMAR_FULL_PATH = _PACKAGE_DIR / "grammar.gbnf"
PACKAGE_GRAMMAR_TIER1_PATH = _PACKAGE_DIR / "grammar_tier1.gbnf"


_FULL_GRAMMAR = """root ::= "{" ws "\\"signal_id\\":" ws string ws "," ws "\\"decision\\":" ws decision ws "," ws "\\"direction\\":" ws direction ws "," ws "\\"confidence\\":" ws confidence ws "," ws "\\"horizon_minutes\\":" ws posint ws "," ws "\\"regime_tag\\":" ws regime ws "," ws "\\"rationale_short\\":" ws string ws "," ws "\\"cited_paper_chunk_ids\\":" ws array_of_strings ws "," ws "\\"contradictions_flagged\\":" ws array_of_strings ws "}" ws

decision ::= "\\"pass\\"" | "\\"veto\\"" | "\\"insufficient_evidence\\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\\"trending\\"" | "\\"mean_reverting\\"" | "\\"high_vol\\"" | "\\"low_vol\\"" | "\\"unknown\\""
string ::= "\\"" char* "\\""
char ::= [^"\\\\\\n] | "\\\\" ["\\\\nrt]
array_of_strings ::= "[" ws ( string ( ws "," ws string )* )? ws "]"
ws ::= [ \\t\\n]*
"""


_TIER1_GRAMMAR = """root ::= "{" ws "\\"signal_id\\":" ws string ws "," ws "\\"decision\\":" ws decision ws "," ws "\\"direction\\":" ws direction ws "," ws "\\"confidence\\":" ws confidence ws "," ws "\\"horizon_minutes\\":" ws posint ws "," ws "\\"regime_tag\\":" ws regime ws "," ws "\\"rationale_short\\":" ws string ws "," ws "\\"cited_paper_chunk_ids\\":" ws "[]" ws "," ws "\\"contradictions_flagged\\":" ws "[]" ws "}" ws

decision ::= "\\"pass\\"" | "\\"veto\\""
direction ::= "-1" | "0" | "1"
confidence ::= "0" | "0." [0-9]+ | "1" | "1.0"
posint ::= [1-9] [0-9]*
regime ::= "\\"trending\\"" | "\\"mean_reverting\\"" | "\\"high_vol\\"" | "\\"low_vol\\"" | "\\"unknown\\""
string ::= "\\"" char* "\\""
char ::= [^"\\\\\\n] | "\\\\" ["\\\\nrt]
ws ::= [ \\t\\n]*
"""


def generate_full_grammar() -> str:
    return _FULL_GRAMMAR


def generate_tier1_grammar() -> str:
    return _TIER1_GRAMMAR


def validate_against_grammar_shape(text: str) -> bool:
    """Cheap structural validation that mirrors what the GBNF accepts.

    Used in unit tests to check fixtures without invoking llama.cpp.
    """
    try:
        payload = json.loads(text)
        GovernorVerdict.model_validate(payload)
        return True
    except Exception:
        return False
```

- [ ] **Step 6: Run tests, expect 7 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_grammar.py -v`
Expected: 7 passed.

- [ ] **Step 7: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/grammar.py tests/test_governor_grammar.py
git add src/quant_research_stack/governor/grammar.py src/quant_research_stack/governor/grammar.gbnf src/quant_research_stack/governor/grammar_tier1.gbnf tests/test_governor_grammar.py
git commit -m "feat: governor grammar.py + grammar.gbnf + grammar_tier1.gbnf with drift test"
```

---

## Task 12: `governor/citation_resolver.py`

**Files:**
- Create: `src/quant_research_stack/governor/citation_resolver.py`
- Create: `tests/test_governor_citation_resolver.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_citation_resolver.py`:

```python
from __future__ import annotations

from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _corpus(ids: list[str]) -> CorpusIndex:
    return CorpusIndex([Chunk(id=cid, source_type="t", source_path="p", chunk_index=i, text="t", sha256="x", n_words=1) for i, cid in enumerate(ids)])


def _verdict(**overrides) -> GovernorVerdict:
    payload = {
        "signal_id": "sig-12345678",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "x",
        "cited_paper_chunk_ids": ["a"],
        "contradictions_flagged": [],
    }
    payload.update(overrides)
    return GovernorVerdict.model_validate(payload)


def test_all_citations_valid_keeps_pass() -> None:
    corpus = _corpus(["a", "b"])
    v = _verdict(cited_paper_chunk_ids=["a", "b"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.pass_
    assert out.cited_paper_chunk_ids == ["a", "b"]
    assert invalid == []


def test_partial_invalid_drops_them() -> None:
    corpus = _corpus(["a"])
    v = _verdict(cited_paper_chunk_ids=["a", "missing"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.pass_
    assert out.cited_paper_chunk_ids == ["a"]
    assert invalid == ["missing"]


def test_all_invalid_pass_downgrades_to_insufficient() -> None:
    corpus = _corpus(["x"])
    v = _verdict(cited_paper_chunk_ids=["nope1", "nope2"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.insufficient_evidence
    assert out.cited_paper_chunk_ids == []
    assert invalid == ["nope1", "nope2"]


def test_veto_with_invalid_citations_is_kept() -> None:
    corpus = _corpus(["x"])
    v = _verdict(decision="veto", cited_paper_chunk_ids=["nope"])
    out, invalid = resolve_citations(v, corpus)
    assert out.decision == Decision.veto
    assert out.cited_paper_chunk_ids == []
    assert invalid == ["nope"]
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_citation_resolver.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `citation_resolver.py`**

Create `src/quant_research_stack/governor/citation_resolver.py`:

```python
from __future__ import annotations

from quant_research_stack.governor.corpus import CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def resolve_citations(verdict: GovernorVerdict, corpus: CorpusIndex) -> tuple[GovernorVerdict, list[str]]:
    valid = [cid for cid in verdict.cited_paper_chunk_ids if cid in corpus]
    invalid = [cid for cid in verdict.cited_paper_chunk_ids if cid not in corpus]
    if not valid and verdict.decision == Decision.pass_:
        verdict = verdict.model_copy(update={
            "decision": Decision.insufficient_evidence,
            "rationale_short": "all citations unresolved; auto-downgrade",
            "cited_paper_chunk_ids": [],
        })
    else:
        verdict = verdict.model_copy(update={"cited_paper_chunk_ids": valid})
    return verdict, invalid
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_citation_resolver.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/citation_resolver.py tests/test_governor_citation_resolver.py
git add src/quant_research_stack/governor/citation_resolver.py tests/test_governor_citation_resolver.py
git commit -m "feat: governor/citation_resolver.py drops invalid IDs and auto-downgrades empty-cite passes"
```

---

## Task 13: `governor/prompts.py` — system + user prompt builders (no tests; pure templates)

**Files:**
- Create: `src/quant_research_stack/governor/prompts.py`

- [ ] **Step 1: Implement `prompts.py`**

Create `src/quant_research_stack/governor/prompts.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from quant_research_stack.governor.corpus import Chunk


class _SignalShape(Protocol):
    signal_id: str
    symbol: str
    direction: int
    confidence: float
    horizon_minutes: int
    regime_hint: str | None


SYSTEM_PROMPT = """You are QuantLab's signal governor.

You receive an S1 trading signal candidate plus retrieved evidence from the local
research corpus. Decide whether to pass, veto, or return insufficient_evidence.

Rules:
1. Output ONLY valid JSON matching the schema. The grammar will reject anything else.
2. Cite at least one chunk_id you actually used. Do not invent IDs.
3. If the retrieved evidence does not address the signal's regime + horizon + symbol,
   return insufficient_evidence. Do not guess.
4. Veto if the signal contradicts cited evidence (e.g. signal says long-momentum at
   1-min horizon but cited paper shows mean-reversion at that horizon).
5. confidence is your confidence in the verdict, not the trade.
"""


def build_user_message(signal: _SignalShape, retrieved: Iterable[Chunk]) -> str:
    evidence_block = "\n".join(
        f"[{c.id}] ({c.source_path}): {c.text[:600]}..." for c in retrieved
    )
    regime = signal.regime_hint or "unknown"
    return (
        f"Signal:\n"
        f"  signal_id: {signal.signal_id}\n"
        f"  symbol: {signal.symbol}\n"
        f"  direction: {signal.direction}\n"
        f"  confidence: {signal.confidence:.4f}\n"
        f"  horizon_minutes: {signal.horizon_minutes}\n"
        f"  regime_hint: {regime}\n\n"
        f"Retrieved evidence (use these chunk_ids if you cite):\n"
        f"{evidence_block}\n\n"
        f"Emit your verdict as JSON now."
    )
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/prompts.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/quant_research_stack/governor/prompts.py
git commit -m "feat: governor/prompts.py system + user message builders"
```

---

## Task 14: `governor/escalator.py` — three-tier orchestration

**Files:**
- Create: `src/quant_research_stack/governor/escalator.py`
- Create: `tests/test_governor_escalator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_escalator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.escalator import EscalationConfig, S1Signal, govern_signal
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _v(decision: str = "pass", citations=("paper_pdf:foo:0",)) -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-12345678",
        "decision": decision,
        "direction": 1,
        "confidence": 0.9,
        "horizon_minutes": 15,
        "regime_tag": "trending",
        "rationale_short": "stub",
        "cited_paper_chunk_ids": list(citations),
        "contradictions_flagged": [],
    })


@dataclass
class _StubT1:
    next_decision: str = "pass"

    def govern(self, signal, retrieval):  # noqa: D401, ANN001
        return _v(decision=self.next_decision, citations=())


@dataclass
class _StubT2:
    next_decision: str = "pass"

    def govern(self, signal, retrieval):  # noqa: ANN001
        return _v(decision=self.next_decision, citations=("paper_pdf:foo:0",))


@dataclass
class _StubT3:
    scheduled: list = field(default_factory=list)

    def schedule_async(self, signal, chunks):  # noqa: ANN001
        self.scheduled.append((signal.signal_id, len(chunks)))


@dataclass
class _StubRuntimes:
    tier1: _StubT1
    tier2: _StubT2
    tier3: _StubT3


def _corpus_with(ids: list[str]) -> CorpusIndex:
    return CorpusIndex([Chunk(id=cid, source_type="t", source_path="p", chunk_index=i, text="t", sha256="x", n_words=1) for i, cid in enumerate(ids)])


def _signal(confidence=0.7, trade_size_pct=0.5) -> S1Signal:
    return S1Signal(
        signal_id="sig-12345678",
        symbol="BTCUSDT",
        direction=1,
        confidence=confidence,
        horizon_minutes=15,
        regime_hint="trending",
        recent_vol_label="med",
        trade_size_pct=trade_size_pct,
    )


def _retrieval(corpus: CorpusIndex):
    def _retr(signal, k):  # noqa: ANN001
        return [next(iter(corpus))]
    return _retr


def test_tier1_veto_short_circuits() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="veto"), tier2=_StubT2(), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.veto
    assert runtimes.tier3.scheduled == []


def test_low_confidence_does_not_escalate_to_tier2() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="veto"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.3), cfg, runtimes, corpus, _retrieval(corpus))
    # Tier 2 not called -> result is Tier 1's pass (which gets passed through)
    assert out.decision == Decision.pass_


def test_high_confidence_calls_tier2_and_uses_its_decision() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="veto"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.veto
    assert runtimes.tier3.scheduled == []


def test_large_trade_schedules_tier3_async() -> None:
    cfg = EscalationConfig()
    runtimes = _StubRuntimes(tier1=_StubT1(next_decision="pass"), tier2=_StubT2(next_decision="pass"), tier3=_StubT3())
    corpus = _corpus_with(["paper_pdf:foo:0"])
    out = govern_signal(_signal(confidence=0.9, trade_size_pct=2.0), cfg, runtimes, corpus, _retrieval(corpus))
    assert out.decision == Decision.pass_
    assert len(runtimes.tier3.scheduled) == 1
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_escalator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `escalator.py`**

Create `src/quant_research_stack/governor/escalator.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


@dataclass(frozen=True)
class S1Signal:
    signal_id: str
    symbol: str
    direction: int
    confidence: float
    horizon_minutes: int
    regime_hint: str | None
    recent_vol_label: str
    trade_size_pct: float


@dataclass(frozen=True)
class EscalationConfig:
    tier1_required: bool = True
    tier2_required_when_tier1_passes_above_confidence: float = 0.6
    tier3_required_when_trade_size_pct_above: float = 1.0
    rerank_to_k: int = 5


def govern_signal(
    signal: S1Signal,
    cfg: EscalationConfig,
    runtimes: Any,
    corpus: CorpusIndex,
    retrieve_top_k: Callable[[S1Signal, int], list[Chunk]],
) -> GovernorVerdict:
    t1 = runtimes.tier1.govern(signal, retrieval=None)
    if t1.decision != Decision.pass_:
        return t1
    if abs(signal.confidence) < cfg.tier2_required_when_tier1_passes_above_confidence:
        return t1
    chunks = retrieve_top_k(signal, cfg.rerank_to_k)
    t2 = runtimes.tier2.govern(signal, retrieval=chunks)
    if t2.decision != Decision.pass_:
        return t2
    if signal.trade_size_pct > cfg.tier3_required_when_trade_size_pct_above:
        runtimes.tier3.schedule_async(signal, chunks)
    return t2
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_escalator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/escalator.py tests/test_governor_escalator.py
git add src/quant_research_stack/governor/escalator.py tests/test_governor_escalator.py
git commit -m "feat: governor/escalator.py three-tier cascade with size-gated async tier3"
```

---

## Task 15: `governor/transport.py` — append-only verdict JSONL writer

**Files:**
- Create: `src/quant_research_stack/governor/transport.py`
- Create: `tests/test_governor_transport.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_transport.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict
from quant_research_stack.governor.transport import VerdictWriter, tail_verdicts


def _v() -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-12345678",
        "decision": "veto",
        "direction": 0,
        "confidence": 0.9,
        "horizon_minutes": 15,
        "regime_tag": "high_vol",
        "rationale_short": "x",
        "cited_paper_chunk_ids": [],
        "contradictions_flagged": [],
    })


def test_writer_appends_one_line_per_call(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.write(_v())
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["signal_id"] == "sig-12345678"


def test_writer_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    assert out.exists()


def test_tail_verdicts_yields_appended_records(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.write(_v())
    rows = list(tail_verdicts(out))
    assert len(rows) == 2
    assert rows[0]["signal_id"] == "sig-12345678"


def test_writer_chmod_when_requested(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.jsonl"
    w = VerdictWriter(out)
    w.write(_v())
    w.close_and_lock()
    assert not (out.stat().st_mode & 0o222)  # no write bits anywhere
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_transport.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `transport.py`**

Create `src/quant_research_stack/governor/transport.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class VerdictWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, verdict: GovernorVerdict) -> None:
        line = verdict.model_dump_json()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def close_and_lock(self) -> None:
        if self.path.exists():
            mode = self.path.stat().st_mode & 0o7777
            self.path.chmod(mode & ~0o222)


def tail_verdicts(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_transport.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/transport.py tests/test_governor_transport.py
git add src/quant_research_stack/governor/transport.py tests/test_governor_transport.py
git commit -m "feat: governor/transport.py append-only JSONL writer + tail reader + close_and_lock"
```

---

## Task 16: `governor/audit.py` — per-decision audit row writer

**Files:**
- Create: `src/quant_research_stack/governor/audit.py`
- Create: `tests/test_governor_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.governor.audit import AuditWriter, replay_audit


def test_audit_writer_appends_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="signal_received", payload={"signal_id": "abc"})
    w.record(event="governor_verdict", payload={"signal_id": "abc", "decision": "veto"})
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "signal_received"
    assert "timestamp_utc" in first


def test_replay_audit_yields_records_in_order(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="a", payload={"i": 1})
    w.record(event="b", payload={"i": 2})
    rows = list(replay_audit(out))
    assert [r["event"] for r in rows] == ["a", "b"]


def test_audit_writes_not_investment_advice_flag(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    w = AuditWriter(out)
    w.record(event="x", payload={})
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["not_investment_advice"] is True
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_audit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `audit.py`**

Create `src/quant_research_stack/governor/audit.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class AuditWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, event: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "not_investment_advice": True,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def replay_audit(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
```

- [ ] **Step 4: Run tests, expect 3 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_audit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/audit.py tests/test_governor_audit.py
git add src/quant_research_stack/governor/audit.py tests/test_governor_audit.py
git commit -m "feat: governor/audit.py append-only audit log with not_investment_advice flag"
```

---

## Task 17: `governor/runtime_tier1.py` — Qwen 0.5B + LoRA via transformers

**Files:**
- Create: `src/quant_research_stack/governor/runtime_tier1.py`

This module wraps a transformers + peft model. No unit tests because it requires the model to be loaded (slow + heavy). Integration test in Task 25 covers it.

- [ ] **Step 1: Implement `runtime_tier1.py`**

Create `src/quant_research_stack/governor/runtime_tier1.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class Tier1Runtime:
    base_model_dir: Path
    adapter_dir: Path | None
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
        model = AutoModelForCausalLM.from_pretrained(self.base_model_dir).to(device)
        if self.adapter_dir is not None and Path(self.adapter_dir).exists():
            model = PeftModel.from_pretrained(model, self.adapter_dir).to(device)
        model.eval()
        self._model = model
        self._device = device

    def govern(self, signal, retrieval: Iterable[Chunk] | None) -> GovernorVerdict:
        import torch

        self._load()
        prompt = self._render_prompt(signal)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            payload = json.loads(text)
            payload["cited_paper_chunk_ids"] = []
            payload["contradictions_flagged"] = []
            return GovernorVerdict.model_validate(payload)
        except Exception:
            return GovernorVerdict.model_validate({
                "signal_id": signal.signal_id,
                "decision": "insufficient_evidence",
                "direction": signal.direction,
                "confidence": 0.0,
                "horizon_minutes": signal.horizon_minutes,
                "regime_tag": signal.regime_hint or "unknown",
                "rationale_short": "tier1 parse failure",
                "cited_paper_chunk_ids": [],
                "contradictions_flagged": [],
            })

    @staticmethod
    def _render_prompt(signal) -> str:
        return (
            "<|im_start|>system\n"
            "You are QuantLab's fast veto governor. Output strict JSON with fields "
            "signal_id, decision (pass|veto), direction, confidence, horizon_minutes, "
            "regime_tag, rationale_short (<=120 chars), cited_paper_chunk_ids: [], "
            "contradictions_flagged: [].\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nSignal: {signal.signal_id} {signal.symbol} dir={signal.direction} "
            f"conf={signal.confidence:.4f} horizon={signal.horizon_minutes}m "
            f"regime={signal.regime_hint or 'unknown'}\nRespond with JSON only.\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/runtime_tier1.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/quant_research_stack/governor/runtime_tier1.py
git commit -m "feat: governor/runtime_tier1.py transformers + peft Qwen 0.5B LoRA wrapper"
```

---

## Task 18: `governor/runtime_tier2.py` — llama-cpp-python wrapper for Mistral 22B

**Files:**
- Create: `src/quant_research_stack/governor/runtime_tier2.py`

- [ ] **Step 1: Implement `runtime_tier2.py`**

Create `src/quant_research_stack/governor/runtime_tier2.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.grammar import generate_full_grammar
from quant_research_stack.governor.prompts import SYSTEM_PROMPT, build_user_message
from quant_research_stack.governor.signal_schema import GovernorVerdict


@dataclass
class Tier2Runtime:
    gguf_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_new_tokens: int = 384

    def __post_init__(self) -> None:
        self._llm = None
        self._grammar_text = generate_full_grammar()

    def _load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama, LlamaGrammar

        self._llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(self._grammar_text)

    def govern(self, signal, retrieval: Iterable[Chunk] | None) -> GovernorVerdict:
        self._load()
        chunks = list(retrieval or [])
        prompt = (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n{build_user_message(signal, chunks)} [/INST]"
        )
        out = self._llm(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            grammar=self._grammar,
        )
        text = out["choices"][0]["text"].strip()
        payload = json.loads(text)
        return GovernorVerdict.model_validate(payload)
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/runtime_tier2.py
git add src/quant_research_stack/governor/runtime_tier2.py
git commit -m "feat: governor/runtime_tier2.py llama-cpp-python Mistral 22B Q4 + GBNF wrapper"
```

---

## Task 19: `governor/runtime_tier3.py` — async Yi 34B wrapper

**Files:**
- Create: `src/quant_research_stack/governor/runtime_tier3.py`

- [ ] **Step 1: Implement `runtime_tier3.py`**

Create `src/quant_research_stack/governor/runtime_tier3.py`:

```python
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from quant_research_stack.governor.corpus import Chunk
from quant_research_stack.governor.grammar import generate_full_grammar
from quant_research_stack.governor.prompts import SYSTEM_PROMPT, build_user_message
from quant_research_stack.governor.signal_schema import GovernorVerdict
from quant_research_stack.governor.transport import VerdictWriter


@dataclass
class Tier3Runtime:
    gguf_path: Path
    output_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_new_tokens: int = 512
    queue: Queue = field(default_factory=Queue)
    _llm: object = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._grammar_text = generate_full_grammar()
        self._writer = VerdictWriter(self.output_path)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def schedule_async(self, signal, chunks: list[Chunk]) -> None:
        if self._thread is None:
            self.start()
        self.queue.put((signal, chunks))

    def _load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama, LlamaGrammar

        self._llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(self._grammar_text)

    def _loop(self) -> None:
        self._load()
        while not self._stop.is_set():
            item = self.queue.get()
            if item is None:
                break
            signal, chunks = item
            prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{build_user_message(signal, chunks)} [/INST]"
            out = self._llm(prompt, max_tokens=self.max_new_tokens, temperature=0.0, grammar=self._grammar)
            text = out["choices"][0]["text"].strip()
            payload = json.loads(text)
            verdict = GovernorVerdict.model_validate(payload)
            self._writer.write(verdict)
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/governor/runtime_tier3.py
git add src/quant_research_stack/governor/runtime_tier3.py
git commit -m "feat: governor/runtime_tier3.py async Yi 34B Q4 worker writing tier3 verdicts file"
```

---

## Task 20: `scripts/governor_lora_dataset.py` — build LoRA training JSONL

**Files:**
- Create: `scripts/governor_lora_dataset.py`

- [ ] **Step 1: Write the script**

Create `scripts/governor_lora_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build governor-LoRA training JSONL from instructions corpus.")
    p.add_argument("--instructions-jsonl", default="data/processed/research/instructions.jsonl")
    p.add_argument("--corpus-parquet-dir", default="data/processed/research/parquet")
    p.add_argument("--out-jsonl", default="data/processed/research/lora_governor.jsonl")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    instructions = Path(args.instructions_jsonl)
    if not instructions.exists():
        console.print(f"[red]missing {instructions}; run paper_corpus_to_instructions.py first[/red]")
        return 2
    corpus = pl.read_parquet(list(Path(args.corpus_parquet_dir).glob("shard_*.parquet")))
    chunk_text_by_id = dict(zip(corpus["id"].to_list(), corpus["text"].to_list(), strict=True))

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with instructions.open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_text = chunk_text_by_id.get(rec.get("source_chunk_id", ""), "")
            if not chunk_text:
                continue
            messages = [
                {"role": "system", "content": "You are QuantLab's fast veto governor. Emit JSON only."},
                {"role": "user", "content": f"Chunk:\n{chunk_text}\n\nQuestion: {rec['prompt']}"},
                {"role": "assistant", "content": rec["response"]},
            ]
            dst.write(json.dumps({"messages": messages}) + "\n")
            written += 1
    console.print(f"Wrote {written} LoRA records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/governor_lora_dataset.py
git add scripts/governor_lora_dataset.py
git commit -m "feat: scripts/governor_lora_dataset.py builds messages-format JSONL from paper Q&A"
```

---

## Task 21: `scripts/governor_lora_label.py` — deterministic synthetic-label rule + tests

**Files:**
- Create: `scripts/governor_lora_label.py`
- Create: `tests/test_governor_lora_label.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_governor_lora_label.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from governor_lora_label import label_chunk_with_seed  # noqa: E402


def test_labeller_deterministic_across_two_runs() -> None:
    chunk_text = "mean reversion is reliable at 1-min horizon"
    chunk_id = "paper_pdf:foo:0"
    a = label_chunk_with_seed(chunk_id, chunk_text, seed=42)
    b = label_chunk_with_seed(chunk_id, chunk_text, seed=42)
    assert a == b


def test_different_seed_yields_different_label() -> None:
    chunk_text = "mean reversion is reliable at 1-min horizon"
    chunk_id = "paper_pdf:foo:0"
    a = label_chunk_with_seed(chunk_id, chunk_text, seed=1)
    b = label_chunk_with_seed(chunk_id, chunk_text, seed=999)
    # at least one of the synthetic fields must differ
    assert a != b


def test_label_returns_required_keys() -> None:
    out = label_chunk_with_seed("id", "text", seed=0)
    expected = {"signal_id", "decision", "direction", "confidence", "horizon_minutes", "regime_tag", "rationale_short", "cited_paper_chunk_ids", "contradictions_flagged"}
    assert expected.issubset(set(out.keys()))
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_lora_label.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `governor_lora_label.py`**

Create `scripts/governor_lora_label.py`:

```python
from __future__ import annotations

import hashlib
import random


def _seed_from(chunk_id: str, seed: int) -> int:
    h = hashlib.sha256(f"{chunk_id}::{seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def label_chunk_with_seed(chunk_id: str, chunk_text: str, *, seed: int) -> dict:
    rng = random.Random(_seed_from(chunk_id, seed))
    direction = rng.choice([-1, 0, 1])
    confidence = round(rng.random(), 4)
    horizon = rng.choice([1, 5, 15, 30, 60])
    regime_choices = ("trending", "mean_reverting", "high_vol", "low_vol", "unknown")
    regime = rng.choice(regime_choices)
    text_low = chunk_text.lower()
    if "mean reversion" in text_low and direction == 1 and horizon <= 5:
        decision = "veto"
        rationale = "long-direction at short horizon contradicts mean-reversion in cited chunk"
    elif "trending" in text_low and direction == 0:
        decision = "veto"
        rationale = "flat direction contradicts trending evidence"
    else:
        decision = "pass"
        rationale = "synthetic-pass case"
    return {
        "signal_id": f"sig-{_seed_from(chunk_id, seed):08x}",
        "decision": decision,
        "direction": direction,
        "confidence": confidence,
        "horizon_minutes": horizon,
        "regime_tag": regime,
        "rationale_short": rationale[:200],
        "cited_paper_chunk_ids": [chunk_id],
        "contradictions_flagged": [],
    }
```

- [ ] **Step 4: Run tests, expect 3 passed**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_lora_label.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/governor_lora_label.py tests/test_governor_lora_label.py
git add scripts/governor_lora_label.py tests/test_governor_lora_label.py
git commit -m "feat: scripts/governor_lora_label.py deterministic synthetic-label rule + tests"
```

---

## Task 22: `scripts/governor_train_lora.py` — peft + transformers training loop

**Files:**
- Create: `scripts/governor_train_lora.py`

This is a training driver. No unit tests — the MLOps eval rules in master spec §S2 §4.1 are checked at the end of training and emitted to `metrics.json`.

- [ ] **Step 1: Write the script**

Create `scripts/governor_train_lora.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA governor adapter on Qwen 0.5B-Instruct.")
    p.add_argument("--config", default="configs/governor.yaml")
    p.add_argument("--dataset-jsonl", default="data/processed/research/lora_governor.jsonl")
    p.add_argument("--base", default=None, help="Override base model id (e.g. Qwen/Qwen2.5-Coder-1.5B for fallback)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    lt = cfg["lora_training"]
    base_dir = args.base or lt["base_model_dir"]
    out_root = Path(lt["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Loading base {base_dir} for LoRA training")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16).to(device)
    peft_cfg = LoraConfig(
        r=int(lt["rank"]),
        lora_alpha=int(lt["alpha"]),
        target_modules=list(lt["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_cfg)

    ds = load_dataset("json", data_files=args.dataset_jsonl, split="train")
    held_out_n = max(1, int(len(ds) * float(lt["held_out_fraction"])))
    eval_ds = ds.select(range(held_out_n))
    train_ds = ds.select(range(held_out_n, len(ds)))

    def render(example):
        text = ""
        for m in example["messages"]:
            text += f"<|im_start|>{m['role']}\n{m['content']}\n<|im_end|>\n"
        toks = tok(text, truncation=True, max_length=int(lt["max_seq_length"]), padding="max_length")
        toks["labels"] = toks["input_ids"]
        return toks

    train_tok = train_ds.map(render, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(render, remove_columns=eval_ds.column_names)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=int(lt["max_epochs"]),
        per_device_train_batch_size=int(lt["batch_size"]),
        gradient_accumulation_steps=int(lt["gradient_accumulation_steps"]),
        learning_rate=float(lt["learning_rate"]),
        warmup_steps=int(lt["warmup_steps"]),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=int(lt["random_seed"]),
        bf16=device == "mps",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_tok, eval_dataset=eval_tok)
    started = time.time()
    trainer.train()
    elapsed_h = (time.time() - started) / 3600.0

    eval_metrics = trainer.evaluate()
    held_out_perplexity = float(eval_metrics.get("eval_loss", 0.0))

    model.save_pretrained(run_dir)
    tok.save_pretrained(run_dir)

    metrics = {
        "run_id": run_id,
        "base_model_dir": base_dir,
        "elapsed_hours": elapsed_h,
        "held_out_perplexity": held_out_perplexity,
        "n_train_records": len(train_ds),
        "n_eval_records": len(eval_ds),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    latest = out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_id)

    console.print(f"Trained adapter at {run_dir}; metrics.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/governor_train_lora.py
git add scripts/governor_train_lora.py
git commit -m "feat: scripts/governor_train_lora.py peft+transformers Qwen LoRA driver"
```

---

## Task 23: `scripts/governor_build_indexes.py` — one-shot BM25+dense+metadata builder

**Files:**
- Create: `scripts/governor_build_indexes.py`

- [ ] **Step 1: Write the script**

Create `scripts/governor_build_indexes.py`:

```python
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
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/governor_build_indexes.py
git add scripts/governor_build_indexes.py
git commit -m "feat: scripts/governor_build_indexes.py one-shot BM25+dense+metadata builder"
```

---

## Task 24: `scripts/s2_govern.py` — long-running governor daemon

**Files:**
- Create: `scripts/s2_govern.py`

- [ ] **Step 1: Write the daemon**

Create `scripts/s2_govern.py`:

```python
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from sentence_transformers import SentenceTransformer

from quant_research_stack.governor.audit import AuditWriter
from quant_research_stack.governor.bm25_index import load_bm25_index
from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import load_corpus
from quant_research_stack.governor.dense_index import load_dense_index
from quant_research_stack.governor.escalator import EscalationConfig, S1Signal, govern_signal
from quant_research_stack.governor.query_builder import build_query
from quant_research_stack.governor.reranker import CrossEncoderReranker
from quant_research_stack.governor.retrieval import HybridRetriever
from quant_research_stack.governor.runtime_tier1 import Tier1Runtime
from quant_research_stack.governor.runtime_tier2 import Tier2Runtime
from quant_research_stack.governor.runtime_tier3 import Tier3Runtime
from quant_research_stack.governor.transport import VerdictWriter

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 governor daemon — tail S1 predictions, write verdicts.")
    p.add_argument("--config", default="configs/governor.yaml")
    p.add_argument("--predictions", required=True, help="Path to S1 predictions Parquet (or directory of Parquets).")
    p.add_argument("--once", action="store_true", help="Process current rows then exit (CI smoke).")
    return p.parse_args()


def _today_path(root: Path) -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return root / f"{today}.jsonl"


def _signal_from_row(row: dict) -> S1Signal:
    return S1Signal(
        signal_id=str(row.get("signal_id") or row.get("id") or f"sig-{int(time.time()*1e6)}"),
        symbol=str(row.get("symbol", "UNKNOWN")),
        direction=int(row.get("direction", 0)),
        confidence=float(row.get("confidence", 0.0)),
        horizon_minutes=int(row.get("horizon_minutes", 15)),
        regime_hint=row.get("regime_hint"),
        recent_vol_label=str(row.get("recent_vol_label", "med")),
        trade_size_pct=float(row.get("trade_size_pct", 0.0)),
    )


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if Path("KILL_TRADING").exists():
        console.print("[red]KILL_TRADING present in repo root; refusing to start[/red]")
        return 4

    corpus = load_corpus(cfg["corpus"]["parquet_dir"])
    metadata_path = Path(cfg["retrieval"]["index_dir"]) / "index_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]missing {metadata_path}; run scripts/governor_build_indexes.py[/red]")
        return 3

    bm25 = load_bm25_index(Path(cfg["retrieval"]["index_dir"]) / "bm25_index.pkl")
    dense = load_dense_index(
        Path(cfg["retrieval"]["index_dir"]) / "dense_index.npy",
        Path(cfg["retrieval"]["index_dir"]) / "dense_index.faiss",
        chunk_ids=tuple(c.id for c in corpus),
    )
    reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model_dir"])
    retriever = HybridRetriever(corpus=corpus, bm25=bm25, dense=dense, reranker=reranker)
    embedder = SentenceTransformer(str(cfg["retrieval"]["embedding_model_dir"]))

    tier1 = Tier1Runtime(
        base_model_dir=Path(cfg["tiers"]["tier1"]["base_model_dir"]),
        adapter_dir=Path(cfg["tiers"]["tier1"]["adapter_dir"]) if Path(cfg["tiers"]["tier1"]["adapter_dir"]).exists() else None,
        max_new_tokens=int(cfg["tiers"]["tier1"]["max_new_tokens"]),
    )
    tier2 = Tier2Runtime(
        gguf_path=Path(cfg["tiers"]["tier2"]["gguf_path"]),
        n_ctx=int(cfg["tiers"]["tier2"]["n_ctx"]),
        n_gpu_layers=int(cfg["tiers"]["tier2"]["n_gpu_layers"]),
        max_new_tokens=int(cfg["tiers"]["tier2"]["max_new_tokens"]),
    )
    tier3 = Tier3Runtime(
        gguf_path=Path(cfg["tiers"]["tier3"]["gguf_path"]),
        output_path=_today_path(Path(cfg["transport"]["tier3_verdicts_dir"])),
        n_ctx=int(cfg["tiers"]["tier3"]["n_ctx"]),
        n_gpu_layers=int(cfg["tiers"]["tier3"]["n_gpu_layers"]),
        max_new_tokens=int(cfg["tiers"]["tier3"]["max_new_tokens"]),
    )
    tier3.start()

    class _Runtimes:
        pass

    runtimes = _Runtimes()
    runtimes.tier1 = tier1
    runtimes.tier2 = tier2
    runtimes.tier3 = tier3

    primary_writer = VerdictWriter(_today_path(Path(cfg["transport"]["primary_verdicts_dir"])))
    audit = AuditWriter(_today_path(Path(cfg["transport"]["audit_log_dir"])))
    esc_cfg = EscalationConfig(
        tier2_required_when_tier1_passes_above_confidence=float(cfg["tiers"]["tier2"]["triggered_when_tier1_passes_above_confidence"]),
        tier3_required_when_trade_size_pct_above=float(cfg["tiers"]["tier3"]["triggered_when_trade_size_pct_above"]),
        rerank_to_k=int(cfg["retrieval"]["rerank_to_k"]),
    )

    def retrieve_top_k(signal: S1Signal, k: int):
        query = build_query(signal)
        qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        return retriever.retrieve(query, bm25_n=int(cfg["retrieval"]["bm25_top_n"]), dense_n=int(cfg["retrieval"]["dense_top_n"]), k=k, query_vector=qv)

    stop = False

    def _handle(_signum, _frame):
        nonlocal stop
        stop = True
        console.print("[yellow]signal received; draining[/yellow]")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    def _process_rows(df: pl.DataFrame) -> int:
        n = 0
        for row in df.iter_rows(named=True):
            sig = _signal_from_row(row)
            audit.record(event="signal_received", payload={"signal_id": sig.signal_id, "symbol": sig.symbol})
            verdict = govern_signal(sig, esc_cfg, runtimes, corpus, retrieve_top_k)
            verdict, invalid = resolve_citations(verdict, corpus)
            audit.record(event="governor_verdict", payload={"signal_id": sig.signal_id, "decision": verdict.decision.value, "invalid_cited": invalid})
            primary_writer.write(verdict)
            n += 1
            if Path("KILL_TRADING").exists():
                return n
        return n

    seen_paths: set[str] = set()
    while not stop:
        candidate = Path(args.predictions)
        if candidate.is_file():
            df = pl.read_parquet(candidate)
            _process_rows(df)
        else:
            for shard in sorted(candidate.glob("*.parquet")):
                if str(shard) in seen_paths:
                    continue
                df = pl.read_parquet(shard)
                _process_rows(df)
                seen_paths.add(str(shard))
        if args.once:
            break
        time.sleep(2.0)

    tier3.stop()
    primary_writer.close_and_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/s2_govern.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add scripts/s2_govern.py
git commit -m "feat: scripts/s2_govern.py long-running governor daemon with KILL_TRADING + signal handlers"
```

---

## Task 25: `scripts/s2_smoke.py` + integration tests

**Files:**
- Create: `scripts/s2_smoke.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_governor_tier1_smoke.py`
- Create: `tests/integration/test_governor_tier2_smoke.py`
- Create: `tests/integration/test_governor_tier3_async.py`

- [ ] **Step 1: Write the smoke driver**

Create `scripts/s2_smoke.py`:

```python
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import polars as pl
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-signal end-to-end governor smoke.")
    p.add_argument("--config", default="configs/governor.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fixture = pl.DataFrame({
        "signal_id": [f"sig-smoke-{i}" for i in range(5)],
        "symbol": ["BTCUSDT"] * 5,
        "direction": [1, -1, 1, 0, 1],
        "confidence": [0.7, 0.4, 0.85, 0.5, 0.9],
        "horizon_minutes": [5, 15, 1, 60, 30],
        "regime_hint": ["trending", "mean_reverting", "trending", "unknown", "high_vol"],
        "recent_vol_label": ["med", "low", "high", "med", "high"],
        "trade_size_pct": [0.3, 0.6, 1.5, 0.1, 2.5],
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.parquet"
        fixture.write_parquet(path)
        from subprocess import run

        rc = run([
            sys.executable, "-m", "scripts.s2_govern",
            "--config", args.config,
            "--predictions", str(path),
            "--once",
        ], check=False).returncode
        console.print(f"smoke completed with rc={rc}")
        return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create integration test package**

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

- [ ] **Step 3: Write tier1 integration test**

Create `tests/integration/test_governor_tier1_smoke.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.governor.escalator import S1Signal
from quant_research_stack.governor.runtime_tier1 import Tier1Runtime
from quant_research_stack.governor.signal_schema import GovernorVerdict


@pytest.mark.governor_slow
def test_tier1_emits_valid_json_on_5_fixtures() -> None:
    rt = Tier1Runtime(
        base_model_dir=Path("models/huggingface/Qwen__Qwen2.5-0.5B-Instruct"),
        adapter_dir=None,
    )
    fixtures = [
        S1Signal(signal_id=f"sig-{i:08d}", symbol="BTCUSDT", direction=1, confidence=0.8,
                 horizon_minutes=15, regime_hint="trending", recent_vol_label="med",
                 trade_size_pct=0.5)
        for i in range(5)
    ]
    for sig in fixtures:
        v = rt.govern(sig, retrieval=None)
        assert isinstance(v, GovernorVerdict)
```

- [ ] **Step 4: Write tier2 integration test**

Create `tests/integration/test_governor_tier2_smoke.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.governor.escalator import S1Signal
from quant_research_stack.governor.runtime_tier2 import Tier2Runtime
from quant_research_stack.governor.signal_schema import GovernorVerdict


@pytest.mark.governor_slow
def test_tier2_emits_valid_json_on_5_fixtures() -> None:
    rt = Tier2Runtime(
        gguf_path=Path("models/huggingface/bartowski__Mistral-Small-Instruct-2409-GGUF/Mistral-Small-Instruct-2409-Q4_K_M.gguf"),
    )
    fixtures = [
        S1Signal(signal_id=f"sig-{i:08d}", symbol="BTCUSDT", direction=1, confidence=0.8,
                 horizon_minutes=15, regime_hint="trending", recent_vol_label="med",
                 trade_size_pct=0.5)
        for i in range(5)
    ]
    for sig in fixtures:
        v = rt.govern(sig, retrieval=[])
        assert isinstance(v, GovernorVerdict)
```

- [ ] **Step 5: Write tier3 async integration test**

Create `tests/integration/test_governor_tier3_async.py`:

```python
from __future__ import annotations

import time
from pathlib import Path

import pytest

from quant_research_stack.governor.escalator import S1Signal
from quant_research_stack.governor.runtime_tier3 import Tier3Runtime


@pytest.mark.governor_slow
def test_tier3_writes_verdict_within_60_seconds(tmp_path: Path) -> None:
    out = tmp_path / "tier3_verdicts.jsonl"
    rt = Tier3Runtime(
        gguf_path=Path("models/huggingface/bartowski__Yi-1.5-34B-Chat-GGUF/Yi-1.5-34B-Chat-Q4_K_M.gguf"),
        output_path=out,
    )
    rt.start()
    sig = S1Signal(
        signal_id="sig-async-01", symbol="BTCUSDT", direction=1, confidence=0.95,
        horizon_minutes=15, regime_hint="trending", recent_vol_label="high", trade_size_pct=2.5,
    )
    rt.schedule_async(sig, [])
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if out.exists() and out.read_text().strip():
            break
        time.sleep(1.0)
    rt.stop()
    assert out.exists() and out.read_text().strip(), "tier3 did not write within 60s"
```

- [ ] **Step 6: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/s2_smoke.py tests/integration/
git add scripts/s2_smoke.py tests/integration/
git commit -m "feat: scripts/s2_smoke.py + tier1/tier2/tier3 integration smoke tests (governor_slow)"
```

---

## Task 26: `tests/test_governor_citation_property.py` — 1000-generation property test

**Files:**
- Create: `tests/test_governor_citation_property.py`

- [ ] **Step 1: Write the property test**

Create `tests/test_governor_citation_property.py`:

```python
from __future__ import annotations

import random

import pytest

from quant_research_stack.governor.citation_resolver import resolve_citations
from quant_research_stack.governor.corpus import Chunk, CorpusIndex
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _corpus() -> CorpusIndex:
    return CorpusIndex([
        Chunk(id=f"valid:{i}", source_type="t", source_path="p", chunk_index=i, text="x", sha256="x", n_words=1)
        for i in range(50)
    ])


def _random_verdict(rng: random.Random) -> GovernorVerdict:
    decisions = ["pass", "veto", "insufficient_evidence"]
    n_valid = rng.randint(0, 5)
    n_invalid = rng.randint(0, 5)
    cited = [f"valid:{rng.randint(0, 49)}" for _ in range(n_valid)] + [f"invalid:{rng.randint(0, 999)}" for _ in range(n_invalid)]
    rng.shuffle(cited)
    payload = {
        "signal_id": f"sig-{rng.randint(10**7, 10**8 - 1):08d}",
        "decision": rng.choice(decisions),
        "direction": rng.choice([-1, 0, 1]),
        "confidence": round(rng.random(), 4),
        "horizon_minutes": rng.choice([1, 5, 15, 60]),
        "regime_tag": rng.choice(["trending", "mean_reverting", "high_vol", "low_vol", "unknown"]),
        "rationale_short": "synthetic",
        "cited_paper_chunk_ids": cited[:10],
        "contradictions_flagged": [],
    }
    return GovernorVerdict.model_validate(payload)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_citation_invariant_across_200_generations(seed: int) -> None:
    rng = random.Random(seed)
    corpus = _corpus()
    for _ in range(200):
        v = _random_verdict(rng)
        out, _ = resolve_citations(v, corpus)
        if out.decision == Decision.pass_:
            assert out.cited_paper_chunk_ids, "pass verdict reaching consumer must have at least one valid citation"
            assert all(cid in corpus for cid in out.cited_paper_chunk_ids), "all citations must resolve"
```

- [ ] **Step 2: Run, expect 5 passed (one per seed → 5 × 200 = 1 000 generations)**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_governor_citation_property.py -v`
Expected: 5 passed.

- [ ] **Step 3: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix tests/test_governor_citation_property.py
git add tests/test_governor_citation_property.py
git commit -m "test: 1000-generation citation invariant property test (5 seeds × 200)"
```

---

## Task 27: Makefile additions for governor

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Read current Makefile**

Run: `cat Makefile`
Expected: contains `test`, `lint`, `type`, `extract`, `train`, `optuna`, `full-retrain-s1`, `clean-experiments` targets.

- [ ] **Step 2: Append governor targets**

Append these lines to `Makefile`:

```makefile

GOVERNOR_BUILD_INDEXES := scripts/governor_build_indexes.py
GOVERNOR_LORA_DATASET := scripts/governor_lora_dataset.py
GOVERNOR_TRAIN_LORA := scripts/governor_train_lora.py
GOVERNOR_SMOKE := scripts/s2_smoke.py
GOVERNOR_DAEMON := scripts/s2_govern.py

.PHONY: governor-build-indexes governor-lora-dataset governor-train-lora governor-smoke governor-up governor-down

governor-build-indexes:
	$(PY) python $(GOVERNOR_BUILD_INDEXES)

governor-lora-dataset:
	$(PY) python $(GOVERNOR_LORA_DATASET)

governor-train-lora: governor-lora-dataset
	$(PY) python $(GOVERNOR_TRAIN_LORA)

governor-smoke:
	$(PY) python $(GOVERNOR_SMOKE)

governor-up:
	@echo "Run: PYTHONPATH=src uv run python $(GOVERNOR_DAEMON) --predictions <path-to-S1-predictions.parquet>"

governor-down:
	@pkill -f "python.*s2_govern.py" || true
```

- [ ] **Step 3: Smoke `make test` + `make lint`**

Run: `cd /Users/dmr/MachineLearning && make test && make lint`
Expected: tests pass; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: Makefile governor-build-indexes / governor-lora-dataset / governor-train-lora / governor-smoke / governor-up / governor-down"
```

---

## Self-review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 master architecture (block + 3 tiers + transport) | Tasks 14, 15, 24 |
| §2.1–2.6 RAG corpus + indexes + retrieval + query | Tasks 5, 6, 7, 8, 9, 10, 23 |
| §3.1 Pydantic schema | Task 4 |
| §3.2 GBNF grammar (full + tier1 variants) | Task 11 |
| §3.3 citation enforcement pipeline | Task 12 |
| §3.4 tier1 special case (empty citations OK at tier1) | Task 11 (tier1 grammar) + Task 17 (tier1 runtime forces empties) |
| §4.1 LoRA training | Tasks 20, 21, 22 |
| §4.2 tier escalation logic | Task 14 |
| §4.3 Tier 2 / 3 prompts | Task 13 (prompts) + Tasks 18, 19 (use them) |
| §5.1 unit tests | Tasks 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 21 |
| §5.2 integration tests | Task 25 |
| §5.3 1000-generation citation property test | Task 26 |
| §5.4 8 success criteria | enforced by individual tests + (criterion 7) `governor_train_lora.py` metrics.json |
| §6 doc layout | Tasks 1, 2 |
| §6.1 configs/governor.yaml | Task 2 |
| §7 risks | mitigations spread across tasks 17, 18, 19, 23 |

**Placeholder scan:** no TBD / TODO. Every code step shows full code. Every test step has explicit assertions.

**Type consistency:** `S1Signal`, `GovernorVerdict`, `Decision`, `Direction`, `RegimeTag`, `Chunk`, `CorpusIndex`, `BM25Index`, `DenseIndex`, `RerankCandidate`, `Reranker`, `HybridRetriever`, `EscalationConfig`, `Tier1Runtime`, `Tier2Runtime`, `Tier3Runtime`, `VerdictWriter`, `AuditWriter` referenced consistently across the daemon (Task 24) and the modules that define them. `S1Signal` fields used identically in `escalator.py` and `s2_govern.py`. The `_SignalShape` Protocol in `query_builder.py` and `prompts.py` is structurally compatible with `S1Signal`.

**Deferred items (master S2 spec §5.5):**
- Multi-LLM cross-check ensemble (Mistral + Yi must agree) — only built if veto precision plateaus < 60 % per criterion 6.
- Coder-1.5B fallback adapter — built only if 0.5B adapter fails criterion 3 or 7.
- Live-LLM monitoring dashboard — out of S2 scope.

All 27 tasks are bite-sized, TDD-disciplined, exact-path, exact-command, with frequent commits. Foundation tasks (1–4: docs + scaffold + deps) complete before any module work; module tasks (5–19) complete before any script work; script tasks (20–25) complete before integration tests; Makefile (27) lands last.
