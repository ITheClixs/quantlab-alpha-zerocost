# CLAUDE.md

This file defines how coding agents should behave in this repository.

## Project Goal

Build a local-first large-language-model quantitative research system for market prediction experiments. The system should be original, rigorous, and benchmark-driven: an LLM proposes structured signals and explanations, while local benchmark code decides whether those signals are useful.
```text
QuantLab
```

Project goal:

```text
Build a rigorous local quantitative finance machine learning research system on Apple Silicon.
```

Target machine:

```text
MacBook Air M4
24 GB unified memory
macOS
Apple Silicon MPS available through PyTorch
MLX available for local LLM inference
```

Primary pipeline:

```text
raw data
  -> cleaned panel data
  -> features
  -> labels
  -> walk-forward validation
  -> model training
  -> predictions
  -> backtest
  -> paper trading simulator
```

---

## 1. Absolute project rules

 Some Rules

```text
1. Do not use random train-test split for financial time series.
2. Do not use future data in features.
3. Do not fit scalers, imputers, encoders, or normalizers on validation or test data.
4. Do not evaluate only accuracy.
5. Do not report a strategy without transaction costs and turnover.
9. Do not full-fine-tune a 12B to 14B parameter model locally.
10. Do not download massive datasets without checking disk and user intent.
```

---

## 2. Preferred implementation order

Implement in this order:

```text
1. Project scaffolding
2. Environment validation
3. Dataset download scripts
4. Raw to Parquet conversion
5. Canonical panel schema
6. Feature generation
7. Label generation
8. Walk-forward split
9. Leakage tests
10. Ridge baseline
11. LightGBM baseline
12. Cross-sectional backtest
13. Report generation
14. G-Research crypto pipeline
15. Optiver pipeline
16. Jane Street pipeline
17. FinBERT sentiment features
18. Small PyTorch MLP
19. Sequence models
20. Paper trading simulator
21. Optional MLX local LLM assistant
```

---

## 3. Repository structure

Expected structure:

```text
QuantLab/
  data/
    raw/
    processed/
    features/
    labels/
    splits/
    backtests/
    paper_trading/
  models/
    tree/
    torch/
    sentiment/
    embeddings/
    llm/
    mlx/
  notebooks/
  reports/
  experiments/
  logs/
  config/
  scripts/
  src/
    quantlab/
      data/
      features/
      labels/
      splits/
      models/
      backtest/
      paper/
      execution/
      research/
      utils/
  tests/
```

Do not put important project logic only in notebooks. Notebooks may be used for exploration, but reusable logic must live under `src/quantlab`.

---

## 4. Coding style

Use:

```text
Python 3.11
type hints
dataclasses or pydantic for configuration
Polars or DuckDB for large preprocessing
Pandas for model interface when necessary
Parquet for intermediate storage
Joblib for tree model artifacts
Torch save for neural model artifacts
JSON for metrics
YAML for configs
```

Prefer:

```text
small composable modules
deterministic functions
explicit config
clear file paths
reproducible outputs
```

Avoid:

```text
hidden global state
implicit notebook variables
hardcoded absolute paths except $HOME/QuantLab
large monolithic scripts
silent exception swallowing
```

---

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

The result must be reproducible from a clean repository using documented commands.

new:

now it is time for you to code a large language model for quantitative trading and quantitative research
  with the inspiration of current models that you have downloaded in this repo. The model that you will code
  will be medium to big sized and will then be integrated into a trading bot (ca. 20B-params) which will
  detect signals and upside downs given the order flow or any ingestion of a market data. For this purpose
  please reedit the README.md and AGENTS.md and also CLAUDE.md and also enhance them in a way that they are
  way more efficient. The model needs to be capable of  JaneStreet kaggle competition market prediction and
  needs to rank really high in that purpose. you are free to use any of your skills and HF and paper search
  skills for that purpose and also your other skills such as /superpowers:writing-plans or your plan mode
