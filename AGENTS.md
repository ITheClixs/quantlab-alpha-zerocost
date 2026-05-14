# AGENTS.md

This file defines agent roles, responsibilities, and task boundaries for the QuantLab Alpha repository.

The repository builds a commercial-grade alpha generation and execution platform.

Core pipeline:

```text
raw data
  -> cleaned panel data
  -> features (+ foundation-model meta-features + sentiment embeddings)
  -> labels
  -> walk-forward purged-embargoed validation
  -> S1 base learners + stacking
  -> S2 LLM governor (GBNF JSON, citations required)
  -> S4 execution gated by QUANTLAB_STAGE
  -> audit log + position book
```

## 0. Cross-agent invariant

No agent may write or modify code under `brokers/*_live.py` or `configs/promotion.yaml`
unless explicitly assigned by the operator with a signed `docs/runbooks/stage_change.md`
commit. Promotion to a higher stage is human-only.

## 1. Global agent rules

```text
No random split for financial time series.
No future information in features.
No target leakage.
No scaler or imputer fitted on validation or test data.
No backtest without costs.
No real-money trading outside QUANTLAB_STAGE=live.
No in-process stage self-promotion.
No full fine-tuning >=12B models locally.
No training >=12B models from scratch.
No hidden notebook-only logic.
No LLM output accepted without >=1 cited_paper_chunk_id.
```

All reusable code must live under `src/quant_research_stack/`.
All experiment outputs must live under `experiments/`.

## 2. Agent: Data Engineer

Kept verbatim from prior AGENTS.md.

## 3. Agent: Feature Engineer

Kept verbatim from prior AGENTS.md.

## 4. Agent: Label Engineer

Kept verbatim from prior AGENTS.md.

## 5. Agent: Validation Engineer

Kept verbatim from prior AGENTS.md.

## 6. Agent: Tabular Alpha Engineer (S1) — new

Owns: `src/quant_research_stack/alpha/` and `experiments/alpha_s1/`.

Must produce:
```text
fold-stable weighted zero-mean R^2 improvement vs ridge baseline
out-of-fold predictions usable by the stacking meta-learner
fold-stable feature importance (>=3 of 5 folds agree)
inference contract: predict(row: pl.DataFrame) -> tuple[float, float], <1 ms per row
artifacts in experiments/alpha_s1/<run_id>/
```

Must not:
```text
place orders
modify governor schema
edit risk configs or promotion configs
use random splits
fit scalers on validation
```

## 7. Agent: LLM Governor Engineer (S2) — new (built in S2 plan)

Owns: `src/quant_research_stack/governor/`.

Must produce:
```text
GBNF grammar enforcing JSON schema
LoRA adapters on <=7B base models
RAG retrieval index over data/processed/research/parquet
veto-precision metric on backtested signals
```

Must not:
```text
bypass JSON schema constraints
accept LLM outputs without cited_paper_chunk_ids
originate trades directly (S2 is veto-only)
```

## 8. Agent: Data Feeds + Broker Adapter Engineer (S3) — new (built in S3 plan)

Owns: `src/quant_research_stack/feeds/`, `src/quant_research_stack/brokers/`.

Must produce:
```text
typed FeedAdapter and BrokerAdapter protocol implementations
recorder + replayer parity (one-hour live recording replayed 100x produces same trace)
null_broker + every *_paper.py pass the same broker contract test
```

Must not:
```text
skip live recording (every tick must be recorded to data/live/)
ship a *_live.py broker without a corresponding *_paper.py first
ship a feed adapter without a fixture-based parser test
```

## 9. Agent: Execution + Risk Engineer (S4) — new (built in S4 plan)

Owns: `src/quant_research_stack/execution/`.

Must produce:
```text
kill switch tested in CI for every trigger
three-stage env-var gating
audit log integrity (replay-byte-identical invariant)
broker reconciliation every minute
```

Must not:
```text
allow in-process stage promotion
weaken risk caps without two-person review
import brokers/*_live.py from anywhere outside execution/router.py
```

## 10. Agent: Tabular Model Engineer (renamed from Model Engineer)

Kept verbatim from prior AGENTS.md §6, with the additional rule that
all new tabular models live under `src/quant_research_stack/alpha/models/`.

## 11. Agent: Backtesting Engineer

Kept verbatim from prior AGENTS.md §7.

## 12. Agent: NLP Engineer

Kept verbatim from prior AGENTS.md §8.

## 13. Agent: Report Engineer

Kept verbatim from prior AGENTS.md §11, with reports landing under
`experiments/alpha_s1/<run_id>/report.md`.

## 14. First milestone assignment

Build the S1 tabular Jane Street predictor per
`docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md`.

Completion target: weighted zero-mean R^2 >= 0.012 on the permanent holdout,
reproducible from a clean clone in <= 4 days wall-clock.

## 15. Done definition

A task is not done unless:

```text
code runs
tests pass
ruff and mypy clean
outputs are saved under experiments/alpha_s1/<run_id>/
metrics are recorded
report is written
limitations are documented
audit log row written (or smoke-test audit log if S4 not yet built)
```
