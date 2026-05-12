# AGENTS.md

This repository is a local-only Quant Research LLM system. Agents must build reusable code under `src/quant_research_stack/`, keep large artifacts out of git, and treat every model output as untrusted until benchmarked.

## Global Rules

```text
No random split for financial time series.
No future information in features.
No target leakage.
No scaler, imputer, encoder, or normalizer fitted on validation/test data.
No strategy report without transaction costs, turnover, and drawdown.
No paid API, paid data feed, or cloud training job.
No real-money broker integration.
No full local training of 20B-class models.
No hidden notebook-only logic.
No artifact downloads that push the workspace above 150 GB.
```

## Agent: Documentation Architect

Owns `README.md`, `AGENTS.md`, `CLAUDE.md`, and command documentation.

Must keep documentation aligned with the actual package name, paths, commands, artifact cap, and local-only constraints. Must remove stale references to `src/quantlab/` unless that package is actually created.

## Agent: Data Engineer

Owns data acquisition and preparation for Hugging Face, Kaggle, arXiv, and open datasets.

Required outputs:

```text
data/raw/
data/processed/
reports/*download*.json
reports/*preparation*.json
```

Must run budget checks before downloading. Must preserve raw data and create deterministic processed parquet/jsonl outputs.

## Agent: LLM Quant Engineer

Owns the local LLM-first research layer.

Responsibilities:

```text
configure local GGUF inference
retrieve paper chunks and market context
generate structured JSON signal hypotheses
reject malformed or uncited outputs
log prompt, model, response, and parser status
fall back from 22B GGUF to installed 13B GGUF when needed
```

The LLM may propose predictions, explanations, and feature hypotheses. It must not be treated as a validated trading system until the validation layer shows improvement.

## Agent: Jane Street Benchmark Engineer

Owns `jane-street-real-time-market-data-forecasting` ingestion, scoring, and local reports.

Required behavior:

```text
load train/lags/test-style parquet files
validate required columns
create time-ordered folds
score responder_6 with weighted zero-mean R2
compare LLM signals against simple baselines
write reports/jane_street_benchmark.json
```

Must not use test-row future responders as features. Must not shuffle dates.

## Agent: Validation Engineer

Owns leakage controls, split logic, and acceptance criteria.

Required tests:

```text
forward labels only use future rows for targets
features use only information available at prediction time
folds are monotonic in time
weighted zero-mean R2 matches hand-computed fixtures
LLM JSON outputs are schema-validated before use
```

## Agent: Report Engineer

Owns experiment and benchmark reporting.

Every report must include:

```text
dataset
artifact sizes
date range or partition range
features used
target definition
split method
model or LLM runtime
metric values
baseline comparison
limitations
next action
```

## Done Definition

A task is done only when code runs, tests pass, outputs are written where expected, limitations are documented, and any significant code changes are committed and pushed.
