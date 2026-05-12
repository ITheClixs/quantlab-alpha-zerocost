# CLAUDE.md

This file defines how coding agents should behave in this repository.

## Project Goal

Build a local-first large-language-model quantitative research system for market prediction experiments. The system should be original, rigorous, and benchmark-driven: an LLM proposes structured signals and explanations, while local benchmark code decides whether those signals are useful.

The first benchmark anchor is Kaggle `jane-street-real-time-market-data-forecasting`.

## Hard Constraints

```text
Local Mac execution only.
No paid APIs.
No cloud jobs.
No real-money broker integration.
No random financial time-series splits.
No target leakage.
No future-derived features.
No full local training of 20B-class models.
No artifact footprint above 150 GB.
```

Free downloads from Kaggle, Hugging Face, arXiv, and open web sources are allowed only when the artifact budget remains below the configured ceiling.

## Actual Repository Shape

Reusable code lives under:

```text
src/quant_research_stack/
```

Primary ignored artifacts live under:

```text
data/raw/
data/processed/
models/huggingface/
reports/
```

Do not create a parallel `src/quantlab/` package unless the repository is intentionally migrated.

## Preferred Implementation Order

```text
1. Documentation and architecture alignment.
2. Artifact budget accounting.
3. Free-data downloaders with dry-run mode.
4. Jane Street 2024 ingestion and scoring harness.
5. Local LLM runtime wrapper.
6. RAG over paper/research chunks.
7. Structured LLM signal schema and parser.
8. Baseline benchmark reports.
9. Iterative score improvements.
10. Paper-trading simulator only after benchmark validation.
```

## LLM-First Policy

The LLM is the primary research interface. It should:

```text
retrieve research
summarize market context
propose signal hypotheses
emit structured prediction JSON
explain feature relevance
criticize leakage and overfitting risk
mentor the terminal workflow
```

The LLM must not be trusted by default. Invalid JSON, missing citations, missing feature evidence, or unsupported confidence must be rejected.

## Local Model Policy

Use quantized local inference for large models:

```text
primary target: 22B-class GGUF when downloaded and runnable
fallback target: installed 13B finance GGUF
small helpers: finance embeddings, FinBERT, Chronos/Kronos/TimeMoE-style local models
```

20B-class local models may be used for inference, prompt/RAG augmentation, and signal generation. Full local training is out of scope on the Mac. Smaller adapters, distilled models, or benchmark-specific tabular predictors may be trained locally when tests and runtime allow.

## Jane Street Benchmark Rules

For `jane-street-real-time-market-data-forecasting`:

```text
target responder: responder_6
primary metric: weighted zero-mean R2
validation: time-ordered folds
baseline floor: constant/zero predictor and simple local model
report path: reports/jane_street_benchmark.json
```

Never shuffle dates. Never use responder columns from future rows as features. Never present a public leaderboard claim unless produced by an actual Kaggle submission.

## Testing Commands

Run before completion:

```bash
uv run python -m compileall scripts src
uv run pytest -q
uv run ruff check src tests scripts
```

If `ruff` reports pre-existing style issues outside the edited area, document them rather than hiding them.

## Completion Criteria

Before calling work complete:

```text
docs reflect current architecture
tests pass
budget report works
benchmark harness has fixture coverage
LLM parser has fixture coverage
reports are reproducible
git status is clean or unrelated user changes are explicitly noted
```
