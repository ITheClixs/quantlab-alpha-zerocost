# QuantLab Alpha — S2 (LLM Governor + RAG) Design

**Date:** 2026-05-16
**Status:** Approved (sections 1–5 walked through with operator)
**Project:** QuantLab Alpha (`/Users/dmr/MachineLearning`)
**Master spec:** `docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md` (§3.1, §6.2, ADRs 0001 / 0003 / 0005)
**Predecessor implementation plan:** `docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md`

S2 is the LLM Governor layer of the QuantLab Alpha platform. It never originates trades. For every S1 signal that fires, S2 decides `pass` / `veto` / `insufficient_evidence`, with mandatory paper-chunk citations whenever it passes. Outputs are constrained-JSON via GBNF grammar (ADR 0003) and citation-required (ADR 0005). The runtime is `llama-cpp-python` so all governor models can run from the local GGUF files we already have on disk.

---

## 1. Master Architecture

```text
S1 prediction file
  experiments/alpha_s1/<run_id>/predictions.parquet
        │
        ▼
  scripts/s2_govern.py  (long-running daemon, OS signal-graceful)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                      S2 GOVERNOR PIPELINE                     │
│                                                               │
│   pending_signal_queue (file-tail of new predictions)         │
│         │                                                     │
│         ▼                                                     │
│   ┌──────────────────┐                                        │
│   │ Tier 1: fast     │  Qwen2.5-0.5B-Instruct + LoRA          │
│   │ (always runs)    │  GBNF JSON, < 500 ms                   │
│   │                  │  output: pass / veto / insufficient    │
│   └──────────────────┘                                        │
│         │ pass + |confidence| > 0.6                           │
│         ▼                                                     │
│   ┌──────────────────┐                                        │
│   │ Tier 2: medium   │  Mistral-Small-Instruct 22B Q4_K_M     │
│   │ (conditional)    │  GBNF JSON + RAG top-5 evidence        │
│   │                  │  ~5-10 s, citations required           │
│   └──────────────────┘                                        │
│         │ trade_size_pct > 1.0                                │
│         ▼                                                     │
│   ┌──────────────────┐                                        │
│   │ Tier 3: deep     │  Yi-1.5-34B-Chat Q4_K_M                │
│   │ (async)          │  ~20-30 s, applies to NEXT trade       │
│   │                  │  contradictions check across papers    │
│   └──────────────────┘                                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  experiments/s2_verdicts/<YYYY-MM-DD>.jsonl
  experiments/s2_verdicts_tier3/<YYYY-MM-DD>.jsonl
  (append-only, one verdict per line, S4 tails these files)
```

**Verdict precedence.** If any tier returns `veto` or `insufficient_evidence`, the trade is blocked. `pass` requires unanimity across every tier that ran for that signal.

**File-based transport contract.**

- **Input.** Tail `experiments/alpha_s1/<latest>/predictions.parquet` rows incrementally via Polars `scan_parquet`, or a configurable watch directory of additional signal JSONL files.
- **Output (primary).** Append-only JSONL at `experiments/s2_verdicts/YYYY-MM-DD.jsonl`. One record per signal. After end-of-day rotation, file is `chmod a-w` so it cannot be modified after closing (matches master spec §5.3 audit-immutability rule).
- **Output (Tier 3 async).** Append-only JSONL at `experiments/s2_verdicts_tier3/YYYY-MM-DD.jsonl`. S4 reads this as a stance modifier for *next* trades only.
- **Ordering.** Each verdict carries a `signal_id` (UUIDv4 assigned by S2 on receipt) and `s1_run_id` (the S1 experiment run that produced the prediction).

**Runtime: `llama-cpp-python`.** It is the only mainstream Python library with native GBNF grammar enforcement and runs the GGUF models we have without conversion. Built with Metal backend for MPS acceleration on the M4. The `transformers` path is reserved for the LoRA-adapter Tier 1 model only, where GGUF + LoRA is awkward and HF-native is simpler.

---

## 2. RAG Corpus and Hybrid Retrieval

### 2.1 Corpus shape (already on disk)

```text
data/processed/research/research_corpus.jsonl   1 581 source chunks (~5 MB)
data/processed/research/parquet/                1 Parquet shard (1.2 MB), same 1 581 chunks
data/processed/research/instructions.jsonl      4 742 LoRA training records (Q&A from chunks)
data/raw/papers/<category>/*.pdf                48 source arXiv PDFs
```

The 1 581 chunks are the **retrieval unit**. Each row carries `id`, `source_type`, `source_path`, `chunk_index`, `text`, `sha256`, `n_words`. The `id` column (e.g. `paper_pdf:lob_prediction/arxiv_1808.03668.pdf:7`) is what S2 cites. IDs are stable across reruns of `prepare_research_corpus.py`.

### 2.2 Components

```text
src/quant_research_stack/governor/
  corpus.py              CorpusIndex: id → (text, sha256, source_path); in-memory dict for O(1) lookup
  bm25_index.py          BM25Okapi over tokenized chunk text; pickled
  dense_index.py         FinLang/finance-embeddings-investopedia → 768-dim float16; faiss-cpu IndexFlatIP
  reranker.py            cross-encoder/ms-marco-MiniLM-L-6-v2; called over <=40 candidates
  retrieval.py           Hybrid orchestrator: BM25 ∪ dense → reranker → top-k
  query_builder.py       Build retrieval query string from S1Signal
  embeddings_cache.py    Reuse the existing alpha/meta_features.MetaFeatureCache pattern
```

### 2.3 Index build (one-time, cached, re-run on corpus SHA change)

1. Load all 1 581 chunks from `data/processed/research/parquet/`.
2. **BM25 index.** Tokenize each chunk (lowercase + word-split, no stemming), build `rank_bm25.BM25Okapi`, pickle to `models/governor/bm25_index.pkl` (~2 MB).
3. **Dense index.** Encode every chunk with `FinLang/finance-embeddings-investopedia` (already on disk under `models/huggingface/FinLang__finance-embeddings-investopedia/`) → `models/governor/dense_index.npy` shape `(1581, 768)`, dtype `float16` (~2.4 MB) + `models/governor/dense_index.faiss` (`IndexFlatIP`).
4. **Reranker.** Download `cross-encoder/ms-marco-MiniLM-L-6-v2` once into `models/huggingface/cross-encoder__ms-marco-MiniLM-L-6-v2/` (~80 MB).

Total index footprint on disk: ~85 MB including the reranker download. Build time: ~1–2 min on the M4. Re-index trigger: SHA over the corpus rows changes (recorded in `models/governor/index_metadata.json`).

### 2.4 Retrieve(query, k=5)

```python
def retrieve(query: str, k: int = 5) -> list[Chunk]:
    bm25_hits = bm25_index.top_k(query, n=20)        # list[ChunkId]
    dense_hits = dense_index.top_k(query, n=20)      # list[ChunkId]
    candidates = unique(bm25_hits + dense_hits)      # ≤ 40
    reranked = reranker.rerank(query, candidates)    # cross-encoder scores
    return reranked[:k]                              # top-k with rerank scores
```

### 2.5 Query construction

For each pending S1 signal, the query string is built by `governor/query_builder.py::build_query(signal)`:

```python
def build_query(signal: S1Signal) -> str:
    return f"{signal.regime_hint or 'unknown'} {signal.symbol} direction={signal.direction} horizon={signal.horizon_minutes}m vol={signal.recent_vol_label}"
```

The exact template is fixed and a unit test asserts byte-equality of output for a fixed input. No model calls during query construction.

### 2.6 Latency budget

```text
BM25 top-20:           ~5 ms
Dense top-20:          ~30 ms (768-dim FlatIP over 1 581 vectors)
Cross-encoder rerank:  ~150 ms (40 pairs)
─────────────────────────────────
Total retrieve():      ~200 ms
```

Within the 5–15 s Tier 2 latency budget. Tier 1 skips RAG entirely (the LoRA adapter is supposed to memorize the patterns, not retrieve them).

---

## 3. GBNF Grammar, Pydantic Schema, Citation Enforcement

The single most important hallucination control. The grammar is the source of truth — schemas exist to validate after generation, but generation cannot produce malformed output by construction.

### 3.1 Pydantic schema (the contract)

```python
# src/quant_research_stack/governor/signal_schema.py

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
    signal_id: str = Field(min_length=8, max_length=64)
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

The `model_validator` enforces the citation invariant post-generation: any `pass` without citations is automatically downgraded to `insufficient_evidence` (per ADR 0005).

### 3.2 GBNF grammar (the prevention)

llama.cpp accepts a GBNF grammar string and constrains token sampling to that grammar. Non-grammar tokens are physically impossible to sample.

```
# src/quant_research_stack/governor/grammar.gbnf

root ::= "{" ws
  "\"signal_id\":" ws string ws "," ws
  "\"decision\":" ws decision ws "," ws
  "\"direction\":" ws direction ws "," ws
  "\"confidence\":" ws confidence ws "," ws
  "\"horizon_minutes\":" ws posint ws "," ws
  "\"regime_tag\":" ws regime ws "," ws
  "\"rationale_short\":" ws string ws "," ws
  "\"cited_paper_chunk_ids\":" ws array_of_strings ws "," ws
  "\"contradictions_flagged\":" ws array_of_strings ws
"}" ws

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

The grammar is generated from the Pydantic schema by `governor/grammar.py::pydantic_to_gbnf(schema)` so the two stay in sync. A unit test diffs the generated grammar against `grammar.gbnf` and fails on drift.

### 3.3 Citation enforcement pipeline

Three layers stack to make hallucinated citations impossible to act on:

| Layer | Check | Action on fail |
|---|---|---|
| 1. Grammar | `cited_paper_chunk_ids` is well-formed JSON array of strings | sampling cannot produce malformed |
| 2. Schema | `model_validator` downgrades empty-citation `pass` → `insufficient_evidence` | auto-rewrite |
| 3. Resolver | Each cited ID must resolve to a row in `corpus.id` (in-memory CorpusIndex) | drop unresolved IDs; if all unresolved, downgrade to `insufficient_evidence` and audit `governor_citation_invalid` |

```python
# src/quant_research_stack/governor/citation_resolver.py

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

The `invalid` list is recorded in the verdict's audit row but never reaches S4.

### 3.4 Tier 1 special case (Qwen 0.5B + LoRA)

Tier 1 is a fast veto gate. It skips RAG, runs the GBNF grammar, but produces `cited_paper_chunk_ids: []` legitimately — its decision space is reduced to `{pass, veto}` only (the LoRA adapter is trained on the labeled paper Q&A pairs to recognize known regime / direction combos). The invariant for Tier 1: a `pass` from Tier 1 is only a *gate*, not an authorization — it must still survive Tier 2 to reach S4. So Tier 1 outputs are not subject to the citation downgrade rule.

The grammar for Tier 1 uses a restricted variant: `decision ::= "\"pass\"" | "\"veto\""` and `cited_paper_chunk_ids ::= "[]"`. This variant lives at `governor/grammar_tier1.gbnf` and is also generated from the schema (with the Tier 1 restriction flag).

---

## 4. LoRA Training and Tier Escalation Logic

### 4.1 LoRA training

**Base model.** `Qwen/Qwen2.5-0.5B-Instruct` (already on disk at `models/huggingface/Qwen__Qwen2.5-0.5B-Instruct/`). The 1.5B Coder variant is held back for a possible second adapter if veto precision is < 60 %.

**Training data.**

```text
Source: data/processed/research/instructions.jsonl  (4 742 records)

Each record currently:
  {"id": "...#qa0", "source_chunk_id": "...", "prompt": "Summarize...",
   "response": "...", "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "generated_at": "..."}

Pre-process into governor-style instruction pairs:
  scripts/governor_lora_dataset.py reads instructions.jsonl, joins with the source chunk,
  and emits data/processed/research/lora_governor.jsonl with shape:
    {"messages": [
       {"role": "system",  "content": "You are QuantLab's signal governor. ..."},
       {"role": "user",    "content": <chunk text + synthetic signal context>},
       {"role": "assistant","content": <verbatim JSON verdict>}
     ]}
```

**Synthetic signal context** is generated deterministically per chunk: a fake S1 signal is fabricated with `direction`, `confidence`, `horizon_minutes` drawn from a seeded RNG, then a target verdict is computed by rule (e.g. if the chunk talks about mean-reversion at 1-min horizon and the synthetic signal direction matches mean-reversion's expected sign at that horizon, emit `pass`; otherwise `veto`). Rule code lives in `scripts/governor_lora_label.py` and is **the only synthetic-data artifact in the system** — its determinism is asserted by a unit test.

**Training mechanics.**

| Setting | Value |
|---|---|
| Library | `peft` + `transformers` |
| Adapter type | LoRA, `r=16`, `alpha=32`, target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Quantization during training | `bf16` on MPS; CPU fallback if MPS errors |
| Batch size | 4 (gradient accumulation 4 → effective 16) |
| Max sequence length | 1024 |
| Epochs | up to 5 with early stopping on held-out perplexity |
| Held-out split | 10 % stratified by source chunk |
| Learning rate | 2e-4 with cosine decay, warmup 100 steps |
| Wall-clock budget (master spec §4.3) | up to 8 h |
| Output | `models/trained/governor_lora_qwen05b/<run_id>/` containing `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, `metrics.json`, `eval_examples.jsonl` |

**Training-time eval.**

1. Held-out perplexity ≤ baseline perplexity − 10 %.
2. Veto precision on a curated 200-pair backtest fixture ≥ 60 % (master spec criterion 6).
3. Citation array is always `[]` in Tier 1 outputs (grammar enforces this).
4. Adapter footprint < 100 MB.

If any of (1)–(4) fails, a second adapter is trained on `Qwen/Qwen2.5-Coder-1.5B` with the same data. Selection rule: highest veto precision wins, ties broken by lower latency.

### 4.2 Tier escalation logic

```python
# src/quant_research_stack/governor/escalator.py

@dataclass(frozen=True)
class EscalationConfig:
    tier1_required: bool = True
    tier2_required_when_tier1_passes_above_confidence: float = 0.6
    tier3_required_when_trade_size_pct_above: float = 1.0

def govern_signal(signal: S1Signal, cfg: EscalationConfig, runtimes: TierRuntimes, corpus: CorpusIndex) -> GovernorVerdict:
    # Tier 1: always
    t1 = runtimes.tier1.govern(signal, retrieval=None)
    if t1.decision != Decision.pass_:
        return _record(t1, tiers_run=["tier1"])

    # Tier 2: gated on Tier 1 confidence
    if abs(signal.confidence) < cfg.tier2_required_when_tier1_passes_above_confidence:
        return _record(t1, tiers_run=["tier1"])

    chunks_for_t2 = retrieve_top_k(signal, k=5, corpus=corpus)
    t2 = runtimes.tier2.govern(signal, retrieval=chunks_for_t2)
    if t2.decision != Decision.pass_:
        return _record(t2, tiers_run=["tier1", "tier2"])

    # Tier 3: gated on trade size — runs ASYNC; current verdict is t2,
    # t3 verdict applies to the NEXT trade for this symbol
    if signal.trade_size_pct > cfg.tier3_required_when_trade_size_pct_above:
        runtimes.tier3.schedule_async(signal, chunks_for_t2)

    return _record(t2, tiers_run=["tier1", "tier2"])
```

**Async Tier 3 contract.** Yi 34B Q4 takes 20–30 s per call. Blocking would miss the trading window. Tier 3 publishes its verdict to a separate `experiments/s2_verdicts_tier3/<date>.jsonl`. S4's risk engine reads Tier 3 verdicts as a stance modifier for the *next* trade in the same symbol — never the current one. This decouples the deep-reasoning model from the trading loop.

**Verdict precedence at S4.** S4 reads only the primary verdicts JSONL. If Tier 3 has flagged the previous trade in this symbol with `veto`, S4 widens the next signal's confidence threshold by 0.2 (a tightening, not a hard block) — captured in `configs/governor.yaml::tier3_stance_modifier`.

### 4.3 Tier 2 / Tier 3 prompt template

A single template lives in `governor/prompts.py`:

```python
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

def build_user_message(signal: S1Signal, retrieved: list[Chunk]) -> str:
    return f"""Signal:
  signal_id: {signal.signal_id}
  symbol: {signal.symbol}
  direction: {signal.direction}
  confidence: {signal.confidence:.4f}
  horizon_minutes: {signal.horizon_minutes}
  regime_hint: {signal.regime_hint}

Retrieved evidence (use these chunk_ids if you cite):
""" + "\n".join(f"[{c.id}] ({c.source_path}): {c.text[:600]}..." for c in retrieved) + """

Emit your verdict as JSON now."""
```

The `[{c.id}]` formatting makes valid IDs visually salient to the model and discourages invention.

---

## 5. Testing, Success Criteria, and What's Deferred

### 5.1 Unit tests (deterministic, no model invocations)

```text
tests/test_governor_signal_schema.py    Pydantic field validation, citation invariant downgrade
tests/test_governor_grammar.py          GBNF generator output diffs match committed grammar.gbnf;
                                        50 corrupt JSON fixtures rejected; 50 valid fixtures accepted
tests/test_governor_corpus.py           CorpusIndex builds from Parquet, .id lookup is O(1),
                                        SHA matches input
tests/test_governor_bm25.py             top_k returns expected docs for synthetic query;
                                        index pickle round-trips
tests/test_governor_dense.py            FinLang vectors normalized; faiss IndexFlatIP returns
                                        cosine-equivalent ranking; persistence round-trips
tests/test_governor_reranker.py         cross-encoder rerank changes order on a known case
                                        (uses a fake reranker stub that scores by string overlap)
tests/test_governor_retrieval.py        Hybrid pipeline returns ≤ 5 unique chunks; works on empty BM25
tests/test_governor_query_builder.py    template emits expected fixed string for fixed input
tests/test_governor_citation_resolver.py    unresolved IDs dropped; pass with 0 valid → insufficient
tests/test_governor_escalator.py        Tier 1 veto stops; low confidence stops; high confidence + size
                                        triggers tier3.schedule_async; uses fake runtimes
tests/test_governor_lora_label.py       deterministic synthetic labeller produces identical output
                                        across two seeded runs
tests/test_governor_transport.py        write/append/tail of verdicts JSONL is append-only;
                                        chmod a-w simulated; race-safe append via fcntl
tests/test_governor_audit.py            every decision writes one audit row; replay reproduces state
```

### 5.2 Integration tests (model-touching, marked `@pytest.mark.governor_slow`)

```text
tests/integration/test_governor_tier1_smoke.py    Qwen 0.5B + LoRA produces valid JSON on 5 fixtures
tests/integration/test_governor_tier2_smoke.py    Mistral 22B Q4 produces valid JSON on 5 fixtures
tests/integration/test_governor_tier3_async.py    Yi 34B Q4 verdict lands in tier3 file within 60 s
```

The slow tests are skipped by default (`pytest -m "not governor_slow"`) and only run by `make governor-smoke`.

### 5.3 Citation invariant assertion (1 000 generations)

A property test in `tests/test_governor_citation_property.py` runs 1 000 random verdict candidates through `resolve_citations` (fake corpus, mixed valid / invalid IDs) and asserts: zero outputs reaching the final `_record()` call have a `pass` decision with empty or all-invalid citations. This is the hard invariant from master spec criterion 5.

### 5.4 Success criteria (S2-specific)

| # | Criterion | Measurable how | Threshold |
|---|---|---|---|
| 1 | Grammar enforced | LLM outputs failing JSON parse | 0 in 10 000 |
| 2 | Citation invariant | Verdicts forwarded with `pass` + 0 valid citations | 0 |
| 3 | Veto precision | Of S2-vetoed signals, share that would have lost money on a 200-pair labeled fixture | ≥ 60 % |
| 4 | Tier 1 latency p99 | Qwen 0.5B + LoRA single-signal time | < 500 ms on M4 |
| 5 | Tier 2 latency p99 | Mistral 22B Q4 + retrieval + rerank | < 15 s on M4 |
| 6 | Tier 3 async lag | Time from signal arrival to Tier 3 verdict file write | < 60 s on M4 |
| 7 | LoRA quality | Held-out perplexity vs base | base − 10 % |
| 8 | Audit replay | Replay of 24 h verdicts JSONL produces identical decision sequence | byte-identical |

Criteria 1–8 gate completion of S2 implementation. Criterion 3 (veto precision) gates LoRA adapter selection (Qwen 0.5B vs Coder 1.5B fallback).

### 5.5 What this spec defers

These belong in the **S4 spec** when it's brainstormed:

- How S4 reads verdicts JSONL with `KILL_TRADING` semantics.
- Stance modifier from Tier 3 to S4's risk engine.
- Two-person review of `configs/governor.yaml` once `live_shadow`.

These belong in a possible **S2.1 spec** later:

- Multi-LLM cross-check ensemble (Mistral + Yi must agree). Reserved if veto precision plateaus at ~60 %.
- Adapter on `Qwen2.5-Coder-1.5B`. Built only if 0.5B adapter fails criterion 3 or 7.
- Live-LLM monitoring dashboard.

---

## 6. Repository Documentation Layout (delta from S1)

```text
configs/
  governor.yaml                                    NEW — tier escalation thresholds, retrieval params,
                                                   tier3_stance_modifier
src/quant_research_stack/governor/
  __init__.py                                      NEW
  signal_schema.py                                 NEW
  grammar.py                                       NEW   (generates grammar.gbnf from schema)
  grammar.gbnf                                     NEW   (committed; tested against generator)
  grammar_tier1.gbnf                               NEW   (restricted variant for Tier 1)
  corpus.py                                        NEW
  bm25_index.py                                    NEW
  dense_index.py                                   NEW
  reranker.py                                      NEW
  retrieval.py                                     NEW
  query_builder.py                                 NEW
  citation_resolver.py                             NEW
  escalator.py                                     NEW
  prompts.py                                       NEW
  runtime_tier1.py                                 NEW   (transformers + LoRA adapter loader)
  runtime_tier2.py                                 NEW   (llama-cpp-python wrapper for Mistral 22B)
  runtime_tier3.py                                 NEW   (llama-cpp-python wrapper for Yi 34B; async)
  transport.py                                     NEW   (append-only JSONL writer + tail reader)
  audit.py                                         NEW   (per-decision audit row)
scripts/
  governor_lora_dataset.py                         NEW   (build LoRA training JSONL from instructions.jsonl)
  governor_lora_label.py                           NEW   (deterministic synthetic verdict labeller)
  governor_train_lora.py                           NEW   (peft + transformers training loop)
  governor_build_indexes.py                        NEW   (one-shot index builder)
  s2_govern.py                                     NEW   (long-running governor daemon)
  s2_smoke.py                                      NEW   (5-signal end-to-end smoke for `make governor-smoke`)
docs/architecture/adrs/
  0006-tier-cascade-fast-medium-deep.md            NEW
  0007-async-tier3-stance-modifier.md              NEW
  0008-llama-cpp-python-runtime.md                 NEW
docs/runbooks/
  governor_index_rebuild.md                        NEW
  governor_lora_retrain.md                         NEW
Makefile                                           MODIFY — add governor-build-indexes, governor-train-lora,
                                                   governor-smoke, governor-up, governor-down targets
```

No existing files are deleted. The S1 module surface is untouched.

### 6.1 `configs/governor.yaml`

```yaml
tiers:
  tier1:
    enabled: true
    base_model_id: Qwen/Qwen2.5-0.5B-Instruct
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
  embedding_model_id: FinLang/finance-embeddings-investopedia
  index_dir: models/governor

transport:
  primary_verdicts_dir: experiments/s2_verdicts
  tier3_verdicts_dir: experiments/s2_verdicts_tier3
  audit_log_dir: logs/audit/governor
  rotation: daily
  chmod_after_close: true

stance:
  tier3_stance_modifier_pct: 0.20
```

---

## 7. Risks This Spec Carries

| Risk | Mitigation |
|---|---|
| Veto precision on Tier 1 stuck below 60 % even after LoRA on 0.5B | Section 4.1 fallback adapter on Qwen2.5-Coder-1.5B; spec already lists this in 5.5 deferred. |
| `llama-cpp-python` GBNF performance regression on a Metal build | Tier 1 (transformers + logits-processor JSON-only fallback) provides a parallel path; Tier 2/3 fall back to CPU inference if Metal misbehaves (slower but correct). |
| Cross-encoder reranker download fails | Skip reranker; use BM25 ∪ dense unique union as top-k. Captured by a `--no-rerank` flag; logged as a degraded mode. |
| Async Tier 3 worker dies | Worker restart on next signal; no recovery of dropped Tier 3 verdicts is attempted (they only modify next-trade stance, not current). |
| LoRA training hits the 8-hour budget without converging | Log failure, fall back to base Qwen 0.5B without adapter for Tier 1 (Tier 1 still benefits from grammar enforcement; veto precision will be lower but the system is operational). |
| Corpus changes invalidate cached citations | `index_metadata.json` records the corpus SHA; index rebuild auto-triggered when SHA changes; running `s2_govern.py` refuses to start with a stale index. |
| S4 not yet built; verdicts JSONL has no consumer | Verdicts file is still useful as an audit trail and for offline backtesting of S2's veto precision; S4 will tail the same file with no API changes. |

---

## 8. Spec Doc Transition

After approval of this spec:

1. Inline self-review (placeholder scan, internal consistency, scope, ambiguity).
2. Spec committed.
3. Operator reviews the written spec.
4. On approval, `superpowers:writing-plans` produces the detailed S2 implementation plan.

The S2 plan will follow the same TDD-discipline structure as the S1 plan — one task per file pair, failing test → impl → green → ruff → commit, with foundation tasks first (corpus + indexes) before any LLM-touching tasks. ADRs 0006–0008 land as Task 1.
