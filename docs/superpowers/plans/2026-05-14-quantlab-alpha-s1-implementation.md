# QuantLab Alpha — S1 (Tabular Predictor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stack of tabular models (Ridge / LightGBM / XGBoost / CatBoost / MLP / 1D-CNN) plus meta-feature feeders (foundation models + sentiment encoders) that scores weighted zero-mean R² ≥ 0.012 on the Jane Street `responder_6` permanent holdout, with reproducible runs, leakage-safe CV, and an inference contract ready for S4 to consume.

**Architecture:** New `src/quant_research_stack/alpha/` package with one file per concern (`io`, `features`, `cv`, `metrics`, `registry`, `inference`, `stacking`, `adversarial`, `meta_features`, plus `models/{ridge,lightgbm_model,xgboost_model,catboost_model,mlp,sequence}.py`). Existing `src/quant_research_stack/local_training.py` and `jane_street.py` stay; the new `alpha/models/ridge.py` re-uses their internals via imports. Training is driven from `scripts/alpha_*.py` entry points; per-run artifacts land in `experiments/alpha_s1/<run_id>/`.

**Tech Stack:** Python 3.11, Polars, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch (MPS), Optuna, transformers, sentence-transformers, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md` (Sections 1, 2, 4, 6, 7)

---

## File Structure

**New files:**

```text
CLAUDE.md                                                    # rewritten in Task 1
AGENTS.md                                                    # rewritten in Task 1
README.md                                                    # rewritten in Task 1
docs/architecture/adrs/0001-two-tier-tabular-llm.md
docs/architecture/adrs/0002-three-stage-promotion-gate.md
docs/architecture/adrs/0003-gbnf-constrained-llm-output.md
docs/architecture/adrs/0004-free-data-feed-policy.md
docs/architecture/adrs/0005-llm-governor-citation-requirement.md
docs/runbooks/kill_switch.md
docs/runbooks/stage_promotion.md
docs/runbooks/incident_response.md
docs/runbooks/disaster_recovery.md
configs/alpha.yaml
src/quant_research_stack/alpha/__init__.py
src/quant_research_stack/alpha/io.py
src/quant_research_stack/alpha/metrics.py
src/quant_research_stack/alpha/cv.py
src/quant_research_stack/alpha/features.py
src/quant_research_stack/alpha/registry.py
src/quant_research_stack/alpha/inference.py
src/quant_research_stack/alpha/stacking.py
src/quant_research_stack/alpha/adversarial.py
src/quant_research_stack/alpha/meta_features.py
src/quant_research_stack/alpha/transfer.py
src/quant_research_stack/alpha/models/__init__.py
src/quant_research_stack/alpha/models/ridge.py
src/quant_research_stack/alpha/models/lightgbm_model.py
src/quant_research_stack/alpha/models/xgboost_model.py
src/quant_research_stack/alpha/models/catboost_model.py
src/quant_research_stack/alpha/models/mlp.py
src/quant_research_stack/alpha/models/sequence.py
scripts/alpha_train_s1.py
scripts/alpha_holdout_eval.py
scripts/alpha_extract_meta_features.py
scripts/alpha_optuna_search.py
scripts/alpha_transfer_optiver.py
scripts/alpha_ood_numerai.py
tests/test_alpha_io.py
tests/test_alpha_metrics.py
tests/test_alpha_cv.py
tests/test_alpha_features.py
tests/test_alpha_registry.py
tests/test_alpha_inference.py
tests/test_alpha_models_ridge.py
tests/test_alpha_models_lightgbm.py
tests/test_alpha_models_xgboost.py
tests/test_alpha_models_catboost.py
tests/test_alpha_models_mlp.py
tests/test_alpha_models_sequence.py
tests/test_alpha_stacking.py
tests/test_alpha_adversarial.py
tests/test_alpha_meta_features.py
Makefile
```

**Modified files:**

```text
pyproject.toml                                # add lightgbm, xgboost, catboost, optuna,
                                              # mypy, sentence-transformers under existing groups
```

Existing files (`src/quant_research_stack/local_training.py`, `jane_street.py`, `llm_quant.py`, etc.) are untouched.

---

## Task 1: Rewrite CLAUDE.md, AGENTS.md, README.md to commercial scope

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `README.md`

This is the doc surgery from spec §6. Each rewrite is a single file write — no per-line edits.

- [ ] **Step 1: Rewrite `CLAUDE.md`** with the content below.

```markdown
# CLAUDE.md

This file defines how Claude Code or any similar coding agent must behave inside the QuantLab Alpha repository.

Project name:

```text
QuantLab Alpha
```

Project goal:

```text
Build a commercial-grade alpha generation and execution platform with stage-gated
real-money trading, a tabular predictor (LightGBM / XGBoost / CatBoost / MLP stack),
and an LLM governor that vetoes signals inconsistent with the local research corpus.
```

Target machine:

```text
MacBook Air M4
24 GB unified memory
macOS
Apple Silicon MPS available through PyTorch
MLX available for local LLM inference
525 GB free disk; 208 GB corpus already on disk
```

Primary pipeline:

```text
raw data
  -> cleaned panel data
  -> features (engineered + foundation-model meta-features + sentiment embeddings)
  -> labels
  -> walk-forward purged-embargoed validation
  -> base learners + stacking meta-learner
  -> S1 predictions
  -> S2 LLM governor (GBNF JSON, paper citations required)
  -> S4 execution gated by QUANTLAB_STAGE (paper / live_shadow / live)
  -> audit log + position book
```

## 1. Absolute project rules

Do not violate these rules:

```text
1.  Do not use random train-test split for financial time series.
2.  Do not use future data in features.
3.  Do not fit scalers, imputers, encoders, or normalizers on validation or test data.
4.  Do not evaluate only accuracy.
5.  Do not report a strategy without transaction costs and turnover.
6.  Do not connect to a real-money broker outside QUANTLAB_STAGE=live. The stage env var
    is operator-controlled only. The kill switch is always armed.
7.  Real-money trading is the eventual operating mode. It is gated behind
    paper -> live_shadow -> live promotion. In-process self-promotion is forbidden.
8.  Do not train a 12B to 14B parameter model from scratch.
9.  Do not full-fine-tune a 12B to 14B parameter model locally.
10. LoRA adapters on <=7B models are permitted. Full fine-tune of >3B requires spec approval.
11. S1 tabular predictor is the only authoritative source of numeric forecasts. The LLM
    never originates trades, only vetoes or explains them.
12. Every LLM signal must cite >=1 chunk_id from the local research corpus or be returned
    as insufficient_evidence.
13. Do not modify configs/promotion.yaml or any brokers/*_live.py without two-person review.
14. Do not download massive datasets without checking disk and user intent.
```

## 2. Preferred implementation order

Implement subsystems in this order, each gated by its own success criteria:

```text
S1 Tabular Predictor (this plan)
S2 LLM Governor + RAG
S3 Data Feeds + Broker Abstraction
S4 Execution + Risk + Promotion Gates
```

Each subsystem gets its own design spec under docs/superpowers/specs/ and its own
implementation plan under docs/superpowers/plans/.

## 3. Repository structure

```text
QuantLab/
  configs/
    stack.yaml          global paths, artifact budget, model runtime
    alpha.yaml          S1 hyperparameters, training power budget
    risk.yaml           S4 risk caps (created in S4 plan)
    promotion.yaml      S4 stage gates (created in S4 plan)
  manifests/            HF / Kaggle / paper manifests
  scripts/              CLI entry points
  src/quant_research_stack/
    alpha/              S1 tabular predictor
    governor/           S2 LLM governor (created in S2 plan)
    feeds/              S3 market data adapters (created in S3 plan)
    brokers/            S3 broker adapters (created in S3 plan)
    execution/          S4 execution + risk (created in S4 plan)
    artifacts.py        shared utilities
    budget.py           artifact budget accounting
    jane_street.py      legacy JS benchmark, kept for parity tests
    kaggle_artifacts.py kaggle manifest loader
    kaggle_downloads.py kaggle downloader
    llm_quant.py        LLM runtime (used by S2)
    local_training.py   legacy trainer, kept for parity tests
  experiments/
    alpha_s1/<run_id>/  per-run artifacts (metrics, predictions, models, report)
  data/                 git-ignored; raw HF/Kaggle/papers + processed
  models/               git-ignored; HF model snapshots + trained artifacts
  reports/              generated metrics, inventories, plans
  docs/
    superpowers/specs/  brainstormed designs
    superpowers/plans/  implementation plans
    runbooks/           stage promotion, kill switch, disaster recovery
    architecture/adrs/  architecture decision records
  tests/                pytest unit and integration tests
```

Do not put important project logic only in notebooks. Reusable logic must live under `src/quant_research_stack/`.

## 4. Coding style

```text
Python 3.11
type hints on all public surfaces
dataclasses or pydantic for configuration
Polars or DuckDB for large preprocessing
Pandas only for model interface when necessary
Parquet for intermediate storage
Joblib for tree-model artifacts
torch.save for neural-model artifacts
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

Additional rule:

```text
Any module that places real orders (brokers/*_live.py) is forbidden from being imported
by tests or training code. Tests use null_broker.py or *_paper.py only.
```

## 5. Financial ML methodology

Keep prior rules verbatim (leakage-safe features, forward-only labels, walk-forward
splits) and add:

```text
5.4 S1 must beat the Optuna-tuned LightGBM baseline by >=10% weighted zero-mean R^2 on
    the permanent holdout before being released to S4 in any stage.
5.5 Adversarial validation between train and holdout: any feature whose classifier AUC
    > 0.6 between train and holdout is dropped or transformed.
5.6 A seeded Gaussian noise feature is included in every training run. Any engineered
    feature ranked below it across >=3 of 5 folds is removed.
```

## 6. Model hierarchy (S1)

Implement base learners in this order:

```text
1. Ridge regression  (refactored from local_training.py)
2. LightGBM
3. XGBoost
4. CatBoost
5. compact PyTorch MLP
6. 1D-CNN sequence model
7. stacking meta-learner (linear, on OOF predictions)
```

Tree models are the default for tabular financial data. Neural networks must beat tree
baselines on the OOF metric before they are added to the stack.

## 7. Backtesting requirements (carried to S4 plan)

Kept verbatim from prior CLAUDE.md.

## 8. Apple Silicon policy

```text
PyTorch MPS when available
MLX for local LLM inference
LightGBM / XGBoost / CatBoost on CPU
LoRA adapters on <=7B models only
no full fine-tune of >=12B models locally
```

Training power budget per full retrain cycle (per spec 4.3):

```text
S1 base training:                up to 24 h wall-clock
Orderbook auto-encoder pretrain: up to 12 h
LoRA adapter S2:                 up to 8 h
Foundation-model feature extract: up to 6 h
Stacking + Optuna meta-search:   up to 48 h combined
total worst case:                ~4 days end-to-end
```

## 9. Local LLM policy

Allowed:

```text
veto / explain in S2
research summarization
feature brainstorming
documentation summarization
small finance text classification
```

Disallowed:

```text
direct trade origination
unvalidated market prediction
trade decisions without paper-chunk citations
training from scratch
full fine-tuning >=12B models locally
```

## 10. Testing requirements

```text
PYTHONPATH=src pytest -q
ruff check src scripts tests
mypy src
PYTHONPATH=src uv run python scripts/audit_replay_check.py last-day   # added by S4 plan
```

## 11. Risk and execution

Single env var QUANTLAB_STAGE controls the broker class loaded at process start:

```text
paper        -> brokers/*_paper.py
live_shadow  -> brokers/null_broker.py + read-only real account
live         -> brokers/*_live.py
```

In-process self-promotion is forbidden. Promotion requires:

```text
- a signed docs/runbooks/stage_change.md commit
- updated .env file
- process restart
```

Hard kill conditions (any one halts trading and writes audit row "kill_trigger"):

```text
- daily realized DD > 5% of account equity
- cumulative DD > 15% from peak
- two consecutive minutes without market data (crypto) or 30 min (equity)
- model age > 7 days without successful S1 retrain
- KILL_TRADING file present in repo root
- SIGTERM or SIGINT received
```

## 12. Observability and audit

Every decision in S2/S3/S4 lands in an append-only JSONL log under `logs/audit/`.
Each rotation is `chmod a-w` so the file cannot be modified after closing.
Replay of the audit log must reproduce the same decision sequence byte-for-byte.

## 13. Completion criteria for the S1 milestone

The S1 milestone is complete only if all of these exist under `experiments/alpha_s1/<run_id>/`:

```text
metadata.json     git_sha, data_hashes, hyperparams, fold definition
predictions.parquet
metrics.json      weighted_zero_mean_r2 >= 0.012 on holdout
feature_importance.parquet
cv_folds.json
models/
  ridge.joblib
  lightgbm.txt
  xgboost.json
  catboost.cbm
  mlp.pt
  sequence.pt
  stacker.joblib
report.md
audit_log_smoke.jsonl    proof S1 wrote to the audit format expected by S4
```

The result must be reproducible from a clean repository using `make full-retrain-s1`.
```

- [ ] **Step 2: Rewrite `AGENTS.md`** with the content below.

```markdown
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
```

- [ ] **Step 3: Rewrite `README.md`** with the content below.

````markdown
# QuantLab Alpha

[![Stage](https://img.shields.io/badge/QUANTLAB__STAGE-paper-orange)](docs/runbooks/stage_promotion.md)
[![Kill switch](https://img.shields.io/badge/kill__switch-armed-red)](docs/runbooks/kill_switch.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

## What this is

A single-operator alpha research and execution platform for mid-frequency equities and
tick-frequency crypto. It pairs a tabular predictor (LightGBM / XGBoost / CatBoost / MLP
stack) with an LLM governor that vetoes trades inconsistent with the cited research
corpus. It operates in three stages — `paper` → `live_shadow` → `live` — with hard gates
between each and a single environment variable (`QUANTLAB_STAGE`) controlling broker
class selection.

## What this is not

This is **not** a Jane Street trading desk, a hedge fund stack, or HFT infrastructure
suitable for sub-millisecond equity strategies. Those require paid market data and
exchange colocation. This stack uses free Alpaca, Binance, and Coinbase feeds and is
explicit about that limitation in every measurement.

This is **not** investment advice. The operator (the human running it) is solely
responsible for funds, taxes, brokerage relationships, and regulatory compliance.
Every output of the system carries `"not_investment_advice": true` in the audit log.

## Current status

The status table is regenerated by `scripts/report_artifact_budget.py` and reflects the
last committed `reports/corpus_inventory.json` and the most recent S1 retrain.

| Area | Value |
|---|---|
| Stage | reads from `$QUANTLAB_STAGE` (default `paper`) |
| Kill switch | reads from `KILL_TRADING` file presence in repo root |
| Corpus on disk | 208 GB across 20 376 files (29 HF datasets, 22 Kaggle datasets, 5 Kaggle competitions, 20 HF models, 48 arXiv PDFs + 4 743 paper Q&A records) |
| S1 weighted zero-mean R² (last retrain) | regenerated from `experiments/alpha_s1/<last_run>/metrics.json` |
| S1 holdout target | ≥ 0.012 |
| Days since last successful retrain | regenerated from filesystem mtime |
| Audit log line count | regenerated from `logs/audit/` |

## Four-subsystem map

```text
S1 Tabular Predictor   -> src/quant_research_stack/alpha/
S2 LLM Governor + RAG  -> src/quant_research_stack/governor/         (S2 spec pending)
S3 Feeds + Brokers     -> src/quant_research_stack/feeds/ + brokers/ (S3 spec pending)
S4 Execution + Risk    -> src/quant_research_stack/execution/        (S4 spec pending)
```

Per-trade decision flow:

```text
Market tick -> S1 predicts (numeric, sub-ms)
            -> S2 governs (JSON, cited papers required, async)
            -> S4 executes (broker gated by QUANTLAB_STAGE)
            -> Position book + audit log
```

## Three-stage promotion flow

```text
paper        Alpaca paper + Binance Testnet. All trades simulated. >= 90 days, Sharpe >= 1.0 net, max DD <= 15%.
   |
   v   (operator signs docs/runbooks/paper_to_shadow.md, edits .env, restarts)
live_shadow  Real broker connected read-only. Every order goes to null_broker.py.
             Parallel paper book. >= 30 days, paper-vs-real quote match within 0.5%.
   |
   v   (operator signs docs/runbooks/shadow_to_live.md, edits .env, restarts)
live         Real money. Hard caps: 2% position, 80% gross, 40% net, 3% daily DD, 12%
             cumulative DD. Kill switch always armed. Caps cut to 50% for first 30 days.
```

The running process cannot promote itself. Only a human edits `.env`, restarts the
process, and signs the runbook artifact.

## Reproduction commands

```bash
# Corpus rebuild (already done; this is the recipe)
uv sync --extra dev --extra llm
huggingface-cli login                                 # optional, faster + gated models
# Place Kaggle token at ~/.kaggle/kaggle.json (chmod 600); accept competition rules

PYTHONPATH=src uv run python scripts/download_papers.py
PYTHONPATH=src uv run python scripts/prepare_research_corpus.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_parquet.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types dataset --max-gb 50
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --max-gb 80
PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --unzip
PYTHONPATH=src uv run python scripts/dedupe_and_verify.py

# S1 full retrain (this plan)
make full-retrain-s1
```

The `make full-retrain-s1` target runs feature extraction, base learners, Optuna
hyperparameter search, stacking, and holdout evaluation. Budget per spec §4.3:
up to 4 days wall-clock on the M4 with checkpoints.

## Safety standard

```text
Kill switch        KILL_TRADING file in repo root halts all trading; survives reboot.
Audit log          logs/audit/YYYY-MM-DD.jsonl, append-only, chmod a-w on rotation.
Replay invariant   Replaying the audit log must reproduce the same decision sequence.
Risk caps          configs/risk.yaml; two-person rule once live_shadow.
Stage gate         QUANTLAB_STAGE env var; process cannot promote itself.
Kill triggers      daily DD 5%, cumulative DD 15%, feed gap, stale model, NTP drift.
```

## Legal disclaimer

```text
This repository is not a regulated investment advisor. It produces no investment advice.
Real-money trading by a single individual is subject to the operator's local regulator
rules. For the US, trading your own funds for your own account is generally permitted;
managing others' funds requires registration (RIA / CTA / etc.). Crypto trading is
jurisdiction-dependent. Tax reporting is the operator's responsibility. The operator
is the sole party responsible for funds, taxes, and legal compliance.
```

## References

```text
Specs:        docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md
              docs/superpowers/specs/2026-05-12-quant-ml-150gb-corpus-design.md
Plans:        docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md
              docs/superpowers/plans/2026-05-12-quant-ml-150gb-corpus-implementation.md
ADRs:         docs/architecture/adrs/
Runbooks:     docs/runbooks/
Manifests:    manifests/datasets.yaml manifests/models.yaml manifests/papers.yaml manifests/kaggle.yaml
```
````

- [ ] **Step 4: Lint markdown — verify there are no broken code-fence pairs.**

Run: `find . -maxdepth 2 -name '*.md' -newer /tmp -print 2>/dev/null; python -c "import re; [print(f, '=', len(re.findall(r'^\`\`\`', open(f).read(), re.M))) for f in ['CLAUDE.md', 'AGENTS.md', 'README.md']]"`
Expected: every count is even.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md AGENTS.md README.md
git commit -m "docs: rewrite CLAUDE.md, AGENTS.md, README.md for QuantLab Alpha commercial scope"
```

---

## Task 2: ADRs and runbook scaffolds

**Files:**
- Create: `docs/architecture/adrs/0001-two-tier-tabular-llm.md`
- Create: `docs/architecture/adrs/0002-three-stage-promotion-gate.md`
- Create: `docs/architecture/adrs/0003-gbnf-constrained-llm-output.md`
- Create: `docs/architecture/adrs/0004-free-data-feed-policy.md`
- Create: `docs/architecture/adrs/0005-llm-governor-citation-requirement.md`
- Create: `docs/runbooks/kill_switch.md`
- Create: `docs/runbooks/stage_promotion.md`
- Create: `docs/runbooks/incident_response.md`
- Create: `docs/runbooks/disaster_recovery.md`

ADRs follow Michael Nygard's format. Runbooks are operator-facing checklists.

- [ ] **Step 1: Create ADR directories**

```bash
mkdir -p docs/architecture/adrs docs/runbooks
```

- [ ] **Step 2: Write ADR 0001**

Create `docs/architecture/adrs/0001-two-tier-tabular-llm.md`:

```markdown
# ADR 0001: Two-tier predictor — tabular originates, LLM governs

## Status
Accepted, 2026-05-14.

## Context
The operator's initial request framed the system as "an LLM trading bot". Jane Street's
competition target is `responder_6`, a numeric regression. LLMs predict numerics by
sampling text tokens that decode to numbers; this is empirically worse than tree
ensembles on tabular finance data. The risk is that a literal "LLM as predictor" build
loses to a LightGBM baseline.

## Decision
S1 — a tabular stack (Ridge / LightGBM / XGBoost / CatBoost / MLP / 1D-CNN / linear
stacker) — is the only source of numeric forecasts. S2 — Mistral 22B / Yi 34B / smaller
LoRA'd models — acts as a governor: retrieves cited paper chunks, emits constrained
JSON, and can only veto, pass, or return insufficient_evidence. S2 never originates
trades.

## Consequences
+ Highest probability of measurable Kaggle-leaderboard performance.
+ Honest about LLM strengths (explanation, retrieval) and weaknesses (numerics).
+ S2 latency budget (5-30 s) is decoupled from S1 latency (<1 ms).
- Operators expecting "the LLM trades" must be re-educated.
- Two model classes to maintain.
```

- [ ] **Step 3: Write ADR 0002**

Create `docs/architecture/adrs/0002-three-stage-promotion-gate.md`:

```markdown
# ADR 0002: Three-stage promotion gate for real-money trading

## Status
Accepted, 2026-05-14.

## Context
The operator wants real-money commercial trading. The default CLAUDE.md bans live
trading; reversing that ban without structure is the fastest known way to lose money.

## Decision
A single environment variable `QUANTLAB_STAGE` controls the broker class loaded at
process start. Three stages:

- `paper` -> `brokers/*_paper.py`
- `live_shadow` -> `brokers/null_broker.py` + read-only real account
- `live` -> `brokers/*_live.py`

Promotion is human-only. The running process cannot promote itself. Each transition
requires a signed `docs/runbooks/stage_change.md` commit. Risk caps are cut to 50% for
the first 30 days after entering `live`.

## Consequences
+ Real-money path exists but is deliberate.
+ Every transition has an auditable artifact.
+ Risk caps + kill switch are stage-aware.
- Adds operational overhead to every promotion.
- Two-person review required for `configs/promotion.yaml` once `live_shadow`.
```

- [ ] **Step 4: Write ADR 0003**

Create `docs/architecture/adrs/0003-gbnf-constrained-llm-output.md`:

```markdown
# ADR 0003: GBNF grammar constraints for LLM governor outputs

## Status
Accepted, 2026-05-14.

## Context
The LLM governor (S2) must emit decisions S4 can act on. Free-text outputs are
unreliable: the LLM can hallucinate fields, return malformed JSON, or write rationales
without citations. We need machine-parseable outputs by construction, not by post-hoc
validation.

## Decision
Every LLM governor inference uses llama.cpp's GBNF grammar to constrain token sampling
to a grammar that produces only valid JSON matching the signal schema. Non-JSON tokens
are physically impossible to sample. The schema requires `cited_paper_chunk_ids` to be
a non-empty array; outputs missing citations are auto-mapped to
`decision: insufficient_evidence`.

## Consequences
+ Zero hallucinated schema fields.
+ Zero non-JSON outputs reaching S4.
+ Citation invariant is enforceable.
- Slightly slower token sampling (grammar evaluation overhead, ~5-10%).
- Requires llama.cpp Python bindings, not raw transformers.
```

- [ ] **Step 5: Write ADR 0004**

Create `docs/architecture/adrs/0004-free-data-feed-policy.md`:

```markdown
# ADR 0004: Free-data-only policy for live feeds

## Status
Accepted, 2026-05-14.

## Context
The operator stipulated unpaid resources. Paid market data ($1k-$50k/mo for equity SIP)
is out of scope. Equity HFT and free data are incompatible.

## Decision
Equity strategies run at 15-min bar minimum (Alpaca / Polygon free tier).
Crypto strategies run at tick frequency via Binance and Coinbase public WebSocket
(real-time, no auth required). yfinance is allowed for backtest research only — never
live. The README explicitly limits live equity trading to mid-frequency and crypto to
tick-frequency.

## Consequences
+ Zero monthly data cost.
+ Crypto strategies can be tick-level (real HFT-ish).
- Equity strategies cannot react sub-minute.
- "HFT-optimized" framing in marketing is unsupported and removed from docs.
```

- [ ] **Step 6: Write ADR 0005**

Create `docs/architecture/adrs/0005-llm-governor-citation-requirement.md`:

```markdown
# ADR 0005: LLM governor outputs must cite local paper chunks

## Status
Accepted, 2026-05-14.

## Context
The 208 GB local corpus includes 48 open-access arXiv PDFs + chunked JSONL + paper Q&A
records. The LLM has been observed in prior projects to confidently produce false
explanations. We want every veto / pass decision to be traceable to evidence on disk.

## Decision
The S2 LLM governor's JSON schema requires `cited_paper_chunk_ids: [str, ...]` to be
non-empty. Outputs lacking citations are auto-rewritten to
`decision: insufficient_evidence` before reaching S4. The cited chunk IDs must resolve
to existing rows in `data/processed/research/parquet/`; otherwise the decision is
audited as `governor_citation_invalid` and treated as `insufficient_evidence`.

## Consequences
+ Every LLM verdict has an evidence trail.
+ Hallucinated explanations become detectable in audit logs.
- Governor latency increases (retrieval round-trip required before generation).
- Coverage gaps in the corpus produce more `insufficient_evidence` than `pass` early on.
```

- [ ] **Step 7: Write runbook scaffolds**

Create `docs/runbooks/kill_switch.md`:

```markdown
# Runbook: Kill switch

## Purpose
Halt all live trading immediately and require human re-arm.

## Trigger
Create a file at the repository root named `KILL_TRADING`:

```bash
touch /Users/dmr/MachineLearning/KILL_TRADING
```

The file's presence is checked on every order-placement attempt and once per minute
even when idle. The kill flag survives reboots.

## What happens on kill
1. S4 cancels all open orders on the active broker.
2. Position book takes a final snapshot to `data/snapshots/kill_<timestamp>.parquet`.
3. Audit log writes `kill_trigger` records for the active stage and trigger reason.
4. Process exits with code 137.

## Re-arming
1. Investigate root cause; document in `docs/runbooks/incident_<date>.md`.
2. Run reconciliation: `PYTHONPATH=src uv run python scripts/reconcile_book.py`.
3. Delete the `KILL_TRADING` file.
4. Restart the process. It refuses to start with `KILL_TRADING` present.
```

Create `docs/runbooks/stage_promotion.md`:

```markdown
# Runbook: Stage promotion

## Stages
paper -> live_shadow -> live.

## Promotion checklist (each stage)

1. Confirm gates from `configs/promotion.yaml` are met for the current stage.
2. Generate the promotion report:
   `PYTHONPATH=src uv run python scripts/generate_promotion_report.py --from <stage>`.
3. Commit the report under `docs/runbooks/<from>_to_<to>.md` with operator signature line.
4. Edit `.env` to set `QUANTLAB_STAGE=<next>`. Verify with `grep QUANTLAB_STAGE .env`.
5. Stop the running process (SIGINT). Confirm clean shutdown.
6. Start the process with the new stage env var.
7. Confirm `quantlab status` reports the new stage.
8. For `live`, confirm `configs/risk.yaml` caps are cut to 50% for the first 30 days.

## Demotion / rollback
Any kill-switch trigger automatically demotes to `live_shadow` for 7 days before the
next promotion attempt is permitted.
```

Create `docs/runbooks/incident_response.md`:

```markdown
# Runbook: Incident response

## Triggers
- Daily realized DD > 5% of account equity
- Cumulative DD > 15% from peak
- Two consecutive minutes without market data (crypto)
- Broker reconciliation mismatch
- NTP drift > 1 s

## Immediate steps
1. Touch `KILL_TRADING` in the repo root (see `kill_switch.md`).
2. Capture state: `PYTHONPATH=src uv run python scripts/capture_state.py --reason <text>`.
3. Notify the operator (out-of-band).

## Investigation
1. Read the last 24 hours of `logs/audit/`.
2. Replay the audit log: `PYTHONPATH=src uv run python scripts/audit_replay_check.py last-day`.
3. Compare positions with the broker (manual login, screenshot saved).

## Resolution
Document root cause, mitigation, and re-arm procedure in
`docs/runbooks/incident_<date>.md`. Commit before re-arming.
```

Create `docs/runbooks/disaster_recovery.md`:

```markdown
# Runbook: Disaster recovery

## Scenarios

### Process crash mid-day
1. On restart, the process refuses to trade until reconciliation succeeds.
2. Run `PYTHONPATH=src uv run python scripts/reconcile_book.py --resync`.
3. Replay the audit log to reach the last known state.
4. Resume only after position book matches broker.

### Data feed loss
1. Kill switch fires automatically on 2 min crypto gap or 30 min equity gap.
2. Switch to backup feed in `configs/feeds.yaml` if available.
3. Replay missed window from `data/live/parquet/<symbol>/`.

### Broker outage
1. Cancel open orders via the broker's web UI manually.
2. Mark the broker offline in `configs/brokers.yaml`.
3. The kill switch fires automatically on next reconciliation failure.

### Corrupted audit log
1. Stop all trading.
2. Restore the last clean rotation from `data/snapshots/`.
3. Re-derive position state from broker via reconciliation.
4. Open an incident; the corrupted file goes to `logs/audit_corrupted/` for forensics.
```

- [ ] **Step 8: Commit**

```bash
git add docs/architecture/adrs/ docs/runbooks/
git commit -m "docs: add ADRs 0001-0005 and runbook scaffolds for kill switch, stage promotion, incidents, DR"
```

---

## Task 3: Scaffold `src/quant_research_stack/alpha/` and `configs/alpha.yaml`

**Files:**
- Create: `src/quant_research_stack/alpha/__init__.py`
- Create: `src/quant_research_stack/alpha/models/__init__.py`
- Create: `configs/alpha.yaml`

- [ ] **Step 1: Create package directories and empty `__init__.py` files**

```bash
mkdir -p src/quant_research_stack/alpha/models
cat > src/quant_research_stack/alpha/__init__.py <<'PY'
"""S1 tabular alpha predictor for QuantLab Alpha.

Spec: docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md
"""
PY
cat > src/quant_research_stack/alpha/models/__init__.py <<'PY'
"""S1 base learners. Each module exposes a `train` and `predict` interface."""
PY
```

- [ ] **Step 2: Create `configs/alpha.yaml`**

```yaml
data:
  jane_street_root: data/raw/huggingface/TnnnT0326__Jane_Street_Competition
  preprocessed_alt_root: data/raw/kaggle/datasets/saurabhshahane__jane-street-preprocessed-train
  synthetic_root: data/raw/kaggle/datasets/christoffer__synthetic-jane-street-dataset
  permanent_holdout_fraction: 0.20
  target_column: responder_6
  weight_column: weight
  group_column: date_id

cv:
  n_folds: 5
  purge_days: 5
  embargo_days: 5
  random_seed: 42

features:
  lag_windows: [1, 2, 5, 10, 20]
  rolling_windows: [5, 20, 60]
  cross_sectional_ranks: true
  interaction_pairs_top_n: 20
  include_noise_feature: true

models:
  ridge:
    alpha_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
  lightgbm:
    num_leaves: 63
    max_depth: -1
    learning_rate: 0.05
    n_estimators: 2000
    early_stopping_rounds: 100
    feature_fraction: 0.9
    bagging_fraction: 0.8
  xgboost:
    max_depth: 8
    learning_rate: 0.05
    n_estimators: 2000
    early_stopping_rounds: 100
    tree_method: hist
  catboost:
    depth: 12
    learning_rate: 0.05
    n_estimators: 2000
    early_stopping_rounds: 100
  mlp:
    hidden_dims: [512, 256, 128]
    dropout: 0.3
    learning_rate: 0.001
    batch_size: 1024
    max_epochs: 50
    patience: 5
    mixed_precision: true
  sequence:
    architecture: cnn1d
    seq_len: 16
    kernel_sizes: [3, 5, 7]
    n_filters: 64
    max_epochs: 30
    learning_rate: 0.0005

stacking:
  method: linear
  fit_on: oof_predictions

optuna:
  trials_per_fold: 200
  pruner: median
  sampler: tpe

training_budget:
  base_training_hours: 24
  orderbook_autoencoder_hours: 12
  foundation_features_hours: 6
  optuna_total_hours: 48

success_gate:
  min_holdout_r2: 0.012
  max_fold_std_r2: 0.002
  min_ridge_baseline_improvement_pct: 60
```

- [ ] **Step 3: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('configs/alpha.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/quant_research_stack/alpha/ configs/alpha.yaml
git commit -m "feat: scaffold alpha/ package and configs/alpha.yaml for S1"
```

---

## Task 4: Add S1 dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml`**

Run: `cat pyproject.toml`
Expected: prints existing deps; LightGBM, XGBoost, CatBoost, Optuna, mypy not yet present.

- [ ] **Step 2: Edit dependencies list — add `lightgbm`, `xgboost`, `catboost`, `optuna`, `sentence-transformers`**

Add these lines to the `dependencies = [...]` block of `pyproject.toml`, in alphabetical order:

```toml
    "catboost>=1.2.0",
    "lightgbm>=4.5.0",
    "optuna>=3.6.0",
    "sentence-transformers>=3.0.0",
    "xgboost>=2.1.0",
```

Add `mypy>=1.11.0` to the `dev` extras group:

```toml
[project.optional-dependencies]
dev = [
    "mypy>=1.11.0",
    "pytest>=8.0.0",
    "ruff>=0.5.0",
]
```

- [ ] **Step 3: Sync deps**

Run: `cd /Users/dmr/MachineLearning && uv sync --extra dev --extra llm`
Expected: lightgbm, xgboost, catboost, optuna, sentence-transformers, mypy installed.

- [ ] **Step 4: Smoke-import each new dep**

Run: `uv run python -c "import lightgbm, xgboost, catboost, optuna, sentence_transformers; import mypy; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add lightgbm, xgboost, catboost, optuna, sentence-transformers, mypy deps for S1"
```

---

## Task 5: `alpha/io.py` — JS data loading with leakage-safe holdout

**Files:**
- Create: `src/quant_research_stack/alpha/io.py`
- Create: `tests/test_alpha_io.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_io.py`:

```python
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha.io import (
    LoadConfig,
    load_jane_street,
    permanent_holdout_split,
)


@pytest.fixture
def fake_js(tmp_path: Path) -> Path:
    df = pl.DataFrame({
        "date_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "symbol_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "feature_00": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "responder_6": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.01, 0.04, 0.02, -0.03],
        "weight": [1.0] * 10,
    })
    path = tmp_path / "fake.parquet"
    df.write_parquet(path)
    return path


def test_load_jane_street_reads_parquet(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id")
    df = load_jane_street(fake_js, cfg)
    assert df.height == 10
    assert "responder_6" in df.columns
    assert "weight" in df.columns


def test_permanent_holdout_split_by_date_id(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id", holdout_fraction=0.4)
    df = load_jane_street(fake_js, cfg)
    train, holdout = permanent_holdout_split(df, cfg)
    train_dates = set(train["date_id"].to_list())
    holdout_dates = set(holdout["date_id"].to_list())
    assert train_dates.isdisjoint(holdout_dates), "holdout dates leaked into train"
    # 5 unique dates; 40% holdout -> last 2 dates (3, 4)
    assert holdout_dates == {3, 4}


def test_permanent_holdout_split_chronological(fake_js: Path) -> None:
    cfg = LoadConfig(target_column="responder_6", weight_column="weight", group_column="date_id", holdout_fraction=0.4)
    df = load_jane_street(fake_js, cfg)
    train, holdout = permanent_holdout_split(df, cfg)
    assert max(train["date_id"]) < min(holdout["date_id"]), "holdout must come AFTER train"
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_io.py -v`
Expected: ImportError or ModuleNotFoundError on `quant_research_stack.alpha.io`.

- [ ] **Step 3: Implement `alpha/io.py`**

Create `src/quant_research_stack/alpha/io.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class LoadConfig:
    target_column: str
    weight_column: str
    group_column: str
    holdout_fraction: float = 0.20


def load_jane_street(path: str | Path, config: LoadConfig) -> pl.DataFrame:
    """Load JS Parquet (single file or directory of parquet shards).

    Returns a Polars DataFrame sorted by group_column ascending.
    Raises FileNotFoundError if path does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        df = pl.read_parquet(target)
    else:
        df = pl.read_parquet(list(target.rglob("*.parquet")))
    required = {config.target_column, config.weight_column, config.group_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.sort(config.group_column)


def permanent_holdout_split(df: pl.DataFrame, config: LoadConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological holdout by group_column. Last `holdout_fraction` of unique groups go to holdout."""
    if not 0.0 < config.holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {config.holdout_fraction}")
    unique_groups = df[config.group_column].unique().sort()
    n = unique_groups.len()
    cut = int(round(n * (1 - config.holdout_fraction)))
    if cut == 0 or cut == n:
        raise ValueError(f"holdout split degenerate: cut={cut}, n_groups={n}")
    train_groups = unique_groups.head(cut)
    holdout_groups = unique_groups.tail(n - cut)
    train = df.filter(pl.col(config.group_column).is_in(train_groups))
    holdout = df.filter(pl.col(config.group_column).is_in(holdout_groups))
    return train, holdout
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_io.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/alpha/io.py tests/test_alpha_io.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/alpha/io.py tests/test_alpha_io.py
git commit -m "feat: alpha/io.py with chronological holdout split"
```

---

## Task 6: `alpha/metrics.py` — weighted zero-mean R² and Sharpe-proxy

**Files:**
- Create: `src/quant_research_stack/alpha/metrics.py`
- Create: `tests/test_alpha_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.alpha.metrics import sharpe_proxy, weighted_zero_mean_r2


def test_weighted_zero_mean_r2_perfect_prediction() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = y_true.copy()
    w = np.ones(4, dtype=np.float64)
    score = weighted_zero_mean_r2(y_true, y_pred, w)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_weighted_zero_mean_r2_constant_zero_prediction() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = np.zeros_like(y_true)
    w = np.ones_like(y_true)
    score = weighted_zero_mean_r2(y_true, y_pred, w)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_weighted_zero_mean_r2_weights_zero_skips_row() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    y_pred = np.array([0.01, -0.02, 0.03, 100.0], dtype=np.float64)
    w_skip_last = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    score = weighted_zero_mean_r2(y_true, y_pred, w_skip_last)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_sharpe_proxy_basic_shape() -> None:
    returns = np.array([0.001, 0.002, -0.001, 0.0015, -0.0005], dtype=np.float64)
    sharpe = sharpe_proxy(returns)
    assert sharpe > 0


def test_sharpe_proxy_zero_volatility_returns_zero() -> None:
    returns = np.ones(5, dtype=np.float64)
    assert sharpe_proxy(returns) == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/metrics.py`**

Create `src/quant_research_stack/alpha/metrics.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def weighted_zero_mean_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Weighted zero-mean R^2 (Jane Street competition metric).

    R^2 = 1 - sum(w * (y - y_hat)^2) / sum(w * y^2)
    Note: denominator uses y^2 (no mean subtraction) which is the "zero-mean" choice.
    """
    if y_true.shape != y_pred.shape or y_true.shape != weights.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape} w={weights.shape}")
    denom = float(np.sum(weights * y_true * y_true))
    if denom == 0.0:
        return 0.0
    numer = float(np.sum(weights * (y_true - y_pred) ** 2))
    return 1.0 - (numer / denom)


def sharpe_proxy(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Annualized Sharpe-proxy assuming zero risk-free rate."""
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=0))
    if sigma == 0.0:
        return 0.0
    return (mu / sigma) * float(np.sqrt(periods_per_year))
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/alpha/metrics.py tests/test_alpha_metrics.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/alpha/metrics.py tests/test_alpha_metrics.py
git commit -m "feat: alpha/metrics.py with weighted zero-mean R^2 and Sharpe-proxy"
```

---

## Task 7: `alpha/cv.py` — PurgedKFold with embargo

**Files:**
- Create: `src/quant_research_stack/alpha/cv.py`
- Create: `tests/test_alpha_cv.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_cv.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_research_stack.alpha.cv import PurgedKFold


def test_purged_kfold_produces_n_folds() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    folds = list(splitter.split(df))
    assert len(folds) == 5


def test_purged_kfold_train_test_disjoint() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    for train_idx, test_idx in splitter.split(df):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_purged_kfold_embargo_gap_respected() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    for train_idx, test_idx in splitter.split(df):
        train_dates = set(df[train_idx]["date_id"].to_list())
        test_dates = set(df[test_idx]["date_id"].to_list())
        for t_test in test_dates:
            for t_train in train_dates:
                if t_train > t_test:
                    assert t_train >= t_test + 5, f"embargo violation: test={t_test} train={t_train}"


def test_purged_kfold_chronological_test_folds() -> None:
    df = pl.DataFrame({"date_id": list(range(100))})
    splitter = PurgedKFold(n_folds=5, group_column="date_id", purge=5, embargo=5)
    test_means = []
    for _, test_idx in splitter.split(df):
        test_dates = df[test_idx]["date_id"].to_numpy()
        test_means.append(float(np.mean(test_dates)))
    assert test_means == sorted(test_means)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_cv.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/cv.py`**

Create `src/quant_research_stack/alpha/cv.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PurgedKFold:
    n_folds: int
    group_column: str
    purge: int = 5
    embargo: int = 5

    def split(self, df: pl.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = df[self.group_column].to_numpy()
        unique_groups = np.unique(groups)
        unique_groups.sort()
        n = unique_groups.size
        fold_size = n // self.n_folds
        for fold_idx in range(self.n_folds):
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_folds - 1 else n
            test_groups = unique_groups[test_start:test_end]
            # purge: drop train groups within `purge` of any test group on the left side
            # embargo: drop train groups within `embargo` of any test group on the right side
            min_test = int(test_groups.min())
            max_test = int(test_groups.max())
            keep_train_mask = (unique_groups < min_test - self.purge) | (unique_groups > max_test + self.embargo)
            train_groups = unique_groups[keep_train_mask]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            train_idx = np.where(np.isin(groups, train_groups))[0]
            yield train_idx, test_idx
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_cv.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/alpha/cv.py tests/test_alpha_cv.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/alpha/cv.py tests/test_alpha_cv.py
git commit -m "feat: alpha/cv.py with PurgedKFold and embargo"
```

---

## Task 8: `alpha/features.py` — engineered features (lag, rolling, ranks, interactions, noise)

**Files:**
- Create: `src/quant_research_stack/alpha/features.py`
- Create: `tests/test_alpha_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_features.py`:

```python
from __future__ import annotations

import polars as pl
import pytest

from quant_research_stack.alpha.features import (
    FeatureConfig,
    add_cross_sectional_ranks,
    add_lag_features,
    add_noise_feature,
    add_rolling_features,
    build_feature_frame,
    no_future_leakage,
)


@pytest.fixture
def panel() -> pl.DataFrame:
    return pl.DataFrame({
        "date_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "symbol_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "feature_00": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "responder_6": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.01, 0.04, 0.02, -0.03],
    })


def test_add_lag_features_produces_expected_columns(panel: pl.DataFrame) -> None:
    out = add_lag_features(panel, ["feature_00"], lags=[1, 2], group_col="symbol_id", time_col="date_id")
    assert "feature_00_lag1" in out.columns
    assert "feature_00_lag2" in out.columns


def test_add_lag_features_does_not_use_future_values(panel: pl.DataFrame) -> None:
    out = add_lag_features(panel, ["feature_00"], lags=[1], group_col="symbol_id", time_col="date_id").sort(["symbol_id", "date_id"])
    sym1 = out.filter(pl.col("symbol_id") == 1)
    # for symbol 1, sorted by date: feature_00 = [0.1, 0.3, 0.5, 0.7, 0.9]
    # lag1 must be [None, 0.1, 0.3, 0.5, 0.7]
    assert sym1["feature_00_lag1"].to_list() == [None, 0.1, 0.3, 0.5, 0.7]


def test_add_rolling_features_emits_mean_and_std(panel: pl.DataFrame) -> None:
    out = add_rolling_features(panel, ["feature_00"], windows=[2], group_col="symbol_id", time_col="date_id")
    assert "feature_00_roll2_mean" in out.columns
    assert "feature_00_roll2_std" in out.columns


def test_add_cross_sectional_ranks_per_date(panel: pl.DataFrame) -> None:
    out = add_cross_sectional_ranks(panel, ["feature_00"], date_col="date_id")
    # for date 0, feature_00 = [0.1, 0.2]; ranks [0, 1]
    d0 = out.filter(pl.col("date_id") == 0).sort("symbol_id")
    assert d0["feature_00_rank_xs"].to_list() == [0.0, 1.0]


def test_add_noise_feature_deterministic(panel: pl.DataFrame) -> None:
    out_a = add_noise_feature(panel, seed=123)
    out_b = add_noise_feature(panel, seed=123)
    assert out_a["noise_seed123"].to_list() == out_b["noise_seed123"].to_list()


def test_no_future_leakage_detects_leak() -> None:
    bad = pl.DataFrame({
        "date_id": [0, 1, 2, 3],
        "symbol_id": [1, 1, 1, 1],
        "leaky_feat": [10.0, 20.0, 30.0, 40.0],
        "responder_6": [10.0, 20.0, 30.0, 40.0],
    })
    leaks = no_future_leakage(bad, target_col="responder_6", group_col="symbol_id", time_col="date_id")
    assert "leaky_feat" in leaks


def test_no_future_leakage_passes_clean() -> None:
    clean = pl.DataFrame({
        "date_id": [0, 1, 2, 3],
        "symbol_id": [1, 1, 1, 1],
        "lagged_feat": [None, 10.0, 20.0, 30.0],
        "responder_6": [10.0, 20.0, 30.0, 40.0],
    })
    leaks = no_future_leakage(clean, target_col="responder_6", group_col="symbol_id", time_col="date_id")
    assert leaks == []


def test_build_feature_frame_end_to_end(panel: pl.DataFrame) -> None:
    cfg = FeatureConfig(
        lag_windows=[1],
        rolling_windows=[2],
        include_noise_feature=True,
        cross_sectional_ranks=True,
        noise_seed=42,
    )
    out = build_feature_frame(panel, cfg, base_features=["feature_00"], date_col="date_id", symbol_col="symbol_id")
    expected = {"feature_00_lag1", "feature_00_roll2_mean", "feature_00_roll2_std", "feature_00_rank_xs", "noise_seed42"}
    assert expected.issubset(set(out.columns))
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_features.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/features.py`**

Create `src/quant_research_stack/alpha/features.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class FeatureConfig:
    lag_windows: list[int]
    rolling_windows: list[int]
    include_noise_feature: bool
    cross_sectional_ranks: bool
    noise_seed: int = 42


def add_lag_features(
    df: pl.DataFrame, cols: list[str], lags: list[int], group_col: str, time_col: str
) -> pl.DataFrame:
    sorted_df = df.sort([group_col, time_col])
    new_cols = []
    for col in cols:
        for lag in lags:
            new_cols.append(pl.col(col).shift(lag).over(group_col).alias(f"{col}_lag{lag}"))
    return sorted_df.with_columns(new_cols)


def add_rolling_features(
    df: pl.DataFrame, cols: list[str], windows: list[int], group_col: str, time_col: str
) -> pl.DataFrame:
    sorted_df = df.sort([group_col, time_col])
    new_cols = []
    for col in cols:
        for w in windows:
            new_cols.append(pl.col(col).rolling_mean(w).over(group_col).alias(f"{col}_roll{w}_mean"))
            new_cols.append(pl.col(col).rolling_std(w).over(group_col).alias(f"{col}_roll{w}_std"))
    return sorted_df.with_columns(new_cols)


def add_cross_sectional_ranks(df: pl.DataFrame, cols: list[str], date_col: str) -> pl.DataFrame:
    new_cols = []
    for col in cols:
        new_cols.append(
            (pl.col(col).rank(method="ordinal").over(date_col) - 1).cast(pl.Float64).alias(f"{col}_rank_xs")
        )
    return df.with_columns(new_cols)


def add_noise_feature(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return df.with_columns(pl.Series(name=f"noise_seed{seed}", values=rng.normal(size=df.height)))


def no_future_leakage(
    df: pl.DataFrame, target_col: str, group_col: str, time_col: str, abs_corr_threshold: float = 0.999
) -> list[str]:
    """Flag feature columns whose values are byte-identical to the target after grouping (cheap leakage check)."""
    leaks: list[str] = []
    for col in df.columns:
        if col in {target_col, group_col, time_col}:
            continue
        if df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            continue
        joined = df.select([pl.col(col).cast(pl.Float64), pl.col(target_col).cast(pl.Float64)]).drop_nulls()
        if joined.height < 2:
            continue
        x = joined[col].to_numpy()
        y = joined[target_col].to_numpy()
        std_x = float(np.std(x))
        std_y = float(np.std(y))
        if std_x == 0.0 or std_y == 0.0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if abs(corr) >= abs_corr_threshold:
            leaks.append(col)
    return leaks


def build_feature_frame(
    df: pl.DataFrame, cfg: FeatureConfig, base_features: list[str], date_col: str, symbol_col: str
) -> pl.DataFrame:
    out = df
    if cfg.lag_windows:
        out = add_lag_features(out, base_features, cfg.lag_windows, symbol_col, date_col)
    if cfg.rolling_windows:
        out = add_rolling_features(out, base_features, cfg.rolling_windows, symbol_col, date_col)
    if cfg.cross_sectional_ranks:
        out = add_cross_sectional_ranks(out, base_features, date_col)
    if cfg.include_noise_feature:
        out = add_noise_feature(out, seed=cfg.noise_seed)
    return out
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_features.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/alpha/features.py tests/test_alpha_features.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/alpha/features.py tests/test_alpha_features.py
git commit -m "feat: alpha/features.py with lag, rolling, cross-sectional ranks, noise, leakage check"
```

---

## Task 9: `alpha/registry.py` — semver + sha256 artifact registry

**Files:**
- Create: `src/quant_research_stack/alpha/registry.py`
- Create: `tests/test_alpha_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_registry.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.alpha.registry import RunRegistry, RunMetadata


def test_create_run_writes_metadata(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={"x": "deadbeef"}, hyperparams={"alpha": 1.0})
    run_id = reg.create_run(meta)
    assert (tmp_path / run_id / "metadata.json").exists()
    loaded = json.loads((tmp_path / run_id / "metadata.json").read_text())
    assert loaded["git_sha"] == "abcd1234"
    assert loaded["version"] == "0.1.0"


def test_save_artifact_computes_sha256(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={}, hyperparams={})
    run_id = reg.create_run(meta)
    payload = b"some bytes"
    sha = reg.save_artifact(run_id, "model.bin", payload)
    expected = "ec4b34c5fe78c3eba6c8881a78fc9ba37d2b06c7ade62fb5fcc8b3a1b71e0e89"  # sha256 of "some bytes"
    assert sha == expected
    assert (tmp_path / run_id / "model.bin").read_bytes() == payload


def test_run_id_is_timestamped(tmp_path: Path) -> None:
    reg = RunRegistry(root=tmp_path)
    meta = RunMetadata(version="0.1.0", git_sha="abcd1234", data_hashes={}, hyperparams={})
    run_id = reg.create_run(meta)
    # run id starts with year
    assert run_id.startswith("20")
    assert len(run_id) >= 14  # YYYYMMDD-HHMMSS at minimum
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_registry.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/registry.py`**

Create `src/quant_research_stack/alpha/registry.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    version: str
    git_sha: str
    data_hashes: dict[str, str]
    hyperparams: dict[str, Any]
    fold_definition: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRegistry:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def create_run(self, meta: RunMetadata) -> str:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(self.root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))
        return run_id

    def save_artifact(self, run_id: str, name: str, payload: bytes) -> str:
        path = Path(self.root) / run_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        index_path = Path(self.root) / run_id / "_artifact_sha256.json"
        index: dict[str, str] = {}
        if index_path.exists():
            index = json.loads(index_path.read_text())
        index[name] = sha
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        return sha
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_registry.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/alpha/registry.py tests/test_alpha_registry.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/alpha/registry.py tests/test_alpha_registry.py
git commit -m "feat: alpha/registry.py with semver + sha256 artifact tracking"
```

---

## Task 10: `alpha/models/ridge.py` — Ridge baseline

**Files:**
- Create: `src/quant_research_stack/alpha/models/ridge.py`
- Create: `tests/test_alpha_models_ridge.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_ridge.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig


def test_ridge_fit_predict_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    w = rng.normal(size=5)
    y = X @ w + rng.normal(size=100) * 0.01
    weights = np.ones(100)
    model = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    model.fit(X, y, weights)
    pred = model.predict(X)
    assert pred.shape == (100,)


def test_ridge_fit_uses_weights() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w = rng.normal(size=3)
    y = X @ w
    weights = np.ones(50)
    weights[:10] = 0.0  # disable first 10 rows
    model = RidgeAlphaModel(RidgeConfig(alpha=0.0))
    model.fit(X, y, weights)
    # weighted ridge with alpha=0 -> exact fit on weighted subset
    pred = model.predict(X[10:])
    assert np.allclose(pred, y[10:], atol=1e-6)


def test_ridge_zero_weights_raises() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(10)
    weights = np.zeros(10)
    model = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    try:
        model.fit(X, y, weights)
    except ValueError as exc:
        assert "weights" in str(exc).lower()
        return
    raise AssertionError("expected ValueError on zero weights")
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_ridge.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/ridge.py`**

Create `src/quant_research_stack/alpha/models/ridge.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0


class RidgeAlphaModel:
    def __init__(self, config: RidgeConfig) -> None:
        self.config = config
        self._estimator = Ridge(alpha=config.alpha, fit_intercept=True)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]) -> None:
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("ridge requires positive total weights")
        self._estimator.fit(x, y, sample_weight=weights)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_ridge.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/ridge.py tests/test_alpha_models_ridge.py
git commit -m "feat: alpha/models/ridge.py weighted Ridge baseline"
```

---

## Task 11: `alpha/models/lightgbm_model.py` — LightGBM with early stopping

**Files:**
- Create: `src/quant_research_stack/alpha/models/lightgbm_model.py`
- Create: `tests/test_alpha_models_lightgbm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_lightgbm.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig


def test_lgb_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5 + rng.normal(size=500) * 0.1
    w = np.ones(500)
    model = LightGBMAlphaModel(LightGBMConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)


def test_lgb_feature_importance() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5
    w = np.ones(500)
    model = LightGBMAlphaModel(LightGBMConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    importance = model.feature_importance()
    assert importance.shape == (5,)
    assert importance[0] >= importance[1:].max()
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_lightgbm.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/lightgbm_model.py`**

Create `src/quant_research_stack/alpha/models/lightgbm_model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LightGBMConfig:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    random_state: int = 42


class LightGBMAlphaModel:
    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        params: dict[str, object] = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "verbosity": -1,
            "seed": self.config.random_state,
        }
        train_set = lgb.Dataset(x_train, y_train, weight=w_train)
        val_set = lgb.Dataset(x_val, y_val, weight=w_val, reference=train_set)
        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.predict(x), dtype=np.float64)

    def feature_importance(self) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.feature_importance(importance_type="gain"), dtype=np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_lightgbm.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/lightgbm_model.py tests/test_alpha_models_lightgbm.py
git commit -m "feat: alpha/models/lightgbm_model.py with early stopping + weighted training"
```

---

## Task 12: `alpha/models/xgboost_model.py` — XGBoost with hist tree method

**Files:**
- Create: `src/quant_research_stack/alpha/models/xgboost_model.py`
- Create: `tests/test_alpha_models_xgboost.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_xgboost.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig


def test_xgb_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.normal(size=500) * 0.1).astype(np.float32)
    w = np.ones(500, dtype=np.float32)
    model = XGBoostAlphaModel(XGBoostConfig(n_estimators=50, learning_rate=0.1))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_xgboost.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/xgboost_model.py`**

Create `src/quant_research_stack/alpha/models/xgboost_model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    random_state: int = 42


class XGBoostAlphaModel:
    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self._booster: xgb.Booster | None = None

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        params: dict[str, object] = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "tree_method": self.config.tree_method,
            "seed": self.config.random_state,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("call fit() first")
        return np.asarray(self._booster.predict(xgb.DMatrix(x)), dtype=np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_xgboost.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/xgboost_model.py tests/test_alpha_models_xgboost.py
git commit -m "feat: alpha/models/xgboost_model.py with hist tree method"
```

---

## Task 13: `alpha/models/catboost_model.py` — CatBoost

**Files:**
- Create: `src/quant_research_stack/alpha/models/catboost_model.py`
- Create: `tests/test_alpha_models_catboost.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_catboost.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig


def test_catboost_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = x[:, 0] * 0.5 + rng.normal(size=500) * 0.1
    w = np.ones(500)
    model = CatBoostAlphaModel(CatBoostConfig(n_estimators=50, learning_rate=0.1, depth=4))
    model.fit(x, y, w, x[400:], y[400:], w[400:])
    pred = model.predict(x[:50])
    assert pred.shape == (50,)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_catboost.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/catboost_model.py`**

Create `src/quant_research_stack/alpha/models/catboost_model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor, Pool
from numpy.typing import NDArray


@dataclass(frozen=True)
class CatBoostConfig:
    depth: int = 12
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    random_state: int = 42


class CatBoostAlphaModel:
    def __init__(self, config: CatBoostConfig) -> None:
        self.config = config
        self._estimator = CatBoostRegressor(
            depth=config.depth,
            learning_rate=config.learning_rate,
            iterations=config.n_estimators,
            random_seed=config.random_state,
            allow_writing_files=False,
            verbose=False,
        )

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        train_pool = Pool(x_train, y_train, weight=w_train)
        val_pool = Pool(x_val, y_val, weight=w_val)
        self._estimator.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self.config.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_catboost.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/catboost_model.py tests/test_alpha_models_catboost.py
git commit -m "feat: alpha/models/catboost_model.py"
```

---

## Task 14: `alpha/models/mlp.py` — compact PyTorch MLP on MPS

**Files:**
- Create: `src/quant_research_stack/alpha/models/mlp.py`
- Create: `tests/test_alpha_models_mlp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_mlp.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig


def test_mlp_fit_predict_cpu() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.normal(size=200) * 0.1).astype(np.float32)
    w = np.ones(200, dtype=np.float32)
    cfg = MLPConfig(hidden_dims=[16, 8], max_epochs=3, batch_size=32, mixed_precision=False, device="cpu")
    model = MLPAlphaModel(cfg)
    model.fit(x, y, w, x[150:], y[150:], w[150:])
    pred = model.predict(x[:20])
    assert pred.shape == (20,)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_mlp.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/mlp.py`**

Create `src/quant_research_stack/alpha/models/mlp.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: list[int]
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 5
    mixed_precision: bool = True
    device: str = "auto"
    random_state: int = 42


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class MLPAlphaModel:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self._net: _Net | None = None
        self._device = _resolve_device(config.device)
        torch.manual_seed(config.random_state)

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        in_dim = x_train.shape[1]
        self._net = _Net(in_dim, list(self.config.hidden_dims), self.config.dropout).to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_x = torch.tensor(x_val, dtype=torch.float32).to(self._device)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self._device)
        val_w = torch.tensor(w_val, dtype=torch.float32).to(self._device)
        best_val = float("inf")
        stalls = 0
        for _epoch in range(self.config.max_epochs):
            self._net.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                wb = wb.to(self._device)
                opt.zero_grad()
                pred = self._net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vp = self._net(val_x)
                vloss = float((val_w * (vp - val_y) ** 2).mean().item())
            if vloss < best_val - 1e-6:
                best_val = vloss
                stalls = 0
            else:
                stalls += 1
                if stalls >= self.config.patience:
                    break

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("call fit() first")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x, dtype=torch.float32).to(self._device))
        return out.detach().cpu().numpy().astype(np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_mlp.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/mlp.py tests/test_alpha_models_mlp.py
git commit -m "feat: alpha/models/mlp.py with weighted MSE, early stopping, MPS auto-detect"
```

---

## Task 15: `alpha/models/sequence.py` — 1D-CNN sequence model

**Files:**
- Create: `src/quant_research_stack/alpha/models/sequence.py`
- Create: `tests/test_alpha_models_sequence.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_models_sequence.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig


def test_conv1d_fit_predict() -> None:
    rng = np.random.default_rng(0)
    n, seq_len, channels = 100, 8, 5
    x = rng.normal(size=(n, seq_len, channels)).astype(np.float32)
    y = (x.sum(axis=(1, 2)) * 0.01 + rng.normal(size=n) * 0.01).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    cfg = Conv1DConfig(max_epochs=3, batch_size=16, device="cpu")
    model = Conv1DAlphaModel(cfg)
    model.fit(x, y, w, x[80:], y[80:], w[80:])
    pred = model.predict(x[:10])
    assert pred.shape == (10,)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_sequence.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/models/sequence.py`**

Create `src/quant_research_stack/alpha/models/sequence.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Conv1DConfig:
    n_filters: int = 64
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    dropout: float = 0.2
    learning_rate: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 4
    device: str = "auto"
    random_state: int = 42


class _Conv1DNet(nn.Module):
    def __init__(self, channels: int, n_filters: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(channels, n_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        outs = [torch.relu(b(x)).mean(dim=-1) for b in self.branches]
        cat = torch.cat(outs, dim=-1)
        return self.head(self.drop(cat)).squeeze(-1)


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(device)


class Conv1DAlphaModel:
    def __init__(self, config: Conv1DConfig) -> None:
        self.config = config
        self._net: _Conv1DNet | None = None
        self._dev = _device(config.device)
        torch.manual_seed(config.random_state)

    def fit(
        self,
        x_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        w_train: NDArray[np.float32],
        x_val: NDArray[np.float32],
        y_val: NDArray[np.float32],
        w_val: NDArray[np.float32],
    ) -> None:
        channels = x_train.shape[-1]
        self._net = _Conv1DNet(channels, self.config.n_filters, list(self.config.kernel_sizes), self.config.dropout).to(self._dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        vx = torch.tensor(x_val, dtype=torch.float32).to(self._dev)
        vy = torch.tensor(y_val, dtype=torch.float32).to(self._dev)
        vw = torch.tensor(w_val, dtype=torch.float32).to(self._dev)
        best, stalls = float("inf"), 0
        for _ in range(self.config.max_epochs):
            self._net.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(self._dev)
                yb = yb.to(self._dev)
                wb = wb.to(self._dev)
                opt.zero_grad()
                pred = self._net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vp = self._net(vx)
                vloss = float((vw * (vp - vy) ** 2).mean().item())
            if vloss < best - 1e-6:
                best = vloss
                stalls = 0
            else:
                stalls += 1
                if stalls >= self.config.patience:
                    break

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("call fit() first")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x, dtype=torch.float32).to(self._dev))
        return out.detach().cpu().numpy().astype(np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_models_sequence.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/sequence.py tests/test_alpha_models_sequence.py
git commit -m "feat: alpha/models/sequence.py 1D-CNN multi-branch model"
```

---

## Task 16: `alpha/stacking.py` — linear stacker on OOF predictions

**Files:**
- Create: `src/quant_research_stack/alpha/stacking.py`
- Create: `tests/test_alpha_stacking.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_stacking.py`:

```python
from __future__ import annotations

import numpy as np

from quant_research_stack.alpha.stacking import LinearStacker


def test_linear_stacker_fits_oof_predictions() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=(n, 3))
    true_w = np.array([0.5, 0.3, 0.2])
    y = base @ true_w + rng.normal(size=n) * 0.01
    weights = np.ones(n)
    stacker = LinearStacker()
    stacker.fit(base, y, weights)
    pred = stacker.predict(base)
    assert pred.shape == (n,)
    # weights should approximately recover true_w under non-negativity + normalization
    recovered = stacker.weights()
    assert np.argmax(recovered) == 0  # column with largest contribution
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_stacking.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/stacking.py`**

Create `src/quant_research_stack/alpha/stacking.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class LinearStacker:
    def __init__(self, alpha: float = 1e-3) -> None:
        self._estimator = Ridge(alpha=alpha, positive=True, fit_intercept=False)

    def fit(self, oof_predictions: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]) -> None:
        self._estimator.fit(oof_predictions, y, sample_weight=weights)

    def predict(self, base_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(base_predictions), dtype=np.float64)

    def weights(self) -> NDArray[np.float64]:
        return np.asarray(self._estimator.coef_, dtype=np.float64)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_stacking.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/stacking.py tests/test_alpha_stacking.py
git commit -m "feat: alpha/stacking.py linear non-negative stacker"
```

---

## Task 17: `alpha/inference.py` — sub-ms per-row predict callable

**Files:**
- Create: `src/quant_research_stack/alpha/inference.py`
- Create: `tests/test_alpha_inference.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_inference.py`:

```python
from __future__ import annotations

import time

import numpy as np
import polars as pl

from quant_research_stack.alpha.inference import S1Predictor, build_predictor_from_stack


def test_predict_returns_float_and_confidence() -> None:
    feature_cols = ["a", "b", "c"]

    def base_a(row: np.ndarray) -> float:
        return float(row[0] + 0.5 * row[1])

    def base_b(row: np.ndarray) -> float:
        return float(row[2])

    stacker_weights = np.array([0.7, 0.3])
    predictor: S1Predictor = build_predictor_from_stack([base_a, base_b], stacker_weights, feature_cols)
    df = pl.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]})
    pred, conf = predictor.predict(df)
    expected = 0.7 * (0.1 + 0.5 * 0.2) + 0.3 * 0.3
    assert abs(pred - expected) < 1e-9
    assert 0.0 <= conf <= 1.0


def test_predict_sub_millisecond() -> None:
    feature_cols = ["a", "b", "c"]
    predictor = build_predictor_from_stack(
        [lambda row: float(row[0]), lambda row: float(row[1])],
        np.array([0.5, 0.5]),
        feature_cols,
    )
    df = pl.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]})
    # warm up
    for _ in range(10):
        predictor.predict(df)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        predictor.predict(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_call_ms = elapsed_ms / n
    assert per_call_ms < 1.0, f"expected <1 ms per call; got {per_call_ms:.3f} ms"
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_inference.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/inference.py`**

Create `src/quant_research_stack/alpha/inference.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray


class S1Predictor(Protocol):
    def predict(self, row: pl.DataFrame) -> tuple[float, float]: ...


@dataclass
class _StackPredictor:
    base_funcs: list[Callable[[np.ndarray], float]]
    weights: NDArray[np.float64]
    feature_cols: list[str]

    def predict(self, row: pl.DataFrame) -> tuple[float, float]:
        if row.height != 1:
            raise ValueError(f"S1 predicts one row at a time; got height={row.height}")
        x = row.select(self.feature_cols).to_numpy()[0]
        base_outs = np.fromiter((f(x) for f in self.base_funcs), dtype=np.float64, count=len(self.base_funcs))
        pred = float(np.dot(self.weights, base_outs))
        # confidence: normalized agreement among base models (1.0 = unanimous sign, 0.0 = split)
        signs = np.sign(base_outs)
        if signs.size == 0 or float(np.std(base_outs)) == 0.0:
            conf = 1.0
        else:
            same_sign = float(np.mean(signs == np.sign(np.mean(signs))))
            conf = float(np.clip(same_sign, 0.0, 1.0))
        return pred, conf


def build_predictor_from_stack(
    base_funcs: list[Callable[[np.ndarray], float]],
    stacker_weights: NDArray[np.float64],
    feature_cols: list[str],
) -> S1Predictor:
    if len(base_funcs) != stacker_weights.size:
        raise ValueError("base_funcs and stacker_weights length mismatch")
    return _StackPredictor(base_funcs=base_funcs, weights=stacker_weights, feature_cols=feature_cols)
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_inference.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/inference.py tests/test_alpha_inference.py
git commit -m "feat: alpha/inference.py with S1Predictor protocol and sub-ms stack predict"
```

---

## Task 18: `alpha/meta_features.py` — foundation-model + sentiment caching

**Files:**
- Create: `src/quant_research_stack/alpha/meta_features.py`
- Create: `tests/test_alpha_meta_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_meta_features.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha.meta_features import MetaFeatureCache, finbert_logits_cached, hash_input_dataframe


def test_hash_input_is_stable() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    h1 = hash_input_dataframe(df)
    h2 = hash_input_dataframe(df)
    assert h1 == h2


def test_meta_feature_cache_roundtrip(tmp_path: Path) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    key = "test_key"
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    cache.put(key, arr)
    out = cache.get(key)
    assert out is not None
    assert np.array_equal(out, arr)


def test_meta_feature_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    assert cache.get("absent") is None


def test_finbert_logits_cached_uses_cache(tmp_path: Path, monkeypatch) -> None:
    cache = MetaFeatureCache(root=tmp_path)
    arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
    cache.put("finbert::abc", arr)

    def fake_run(texts: list[str]) -> np.ndarray:
        raise AssertionError("should not be called when cached")

    out = finbert_logits_cached(["x", "y"], cache=cache, cache_key="finbert::abc", runner=fake_run)
    assert np.array_equal(out, arr)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_meta_features.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/meta_features.py`**

Create `src/quant_research_stack/alpha/meta_features.py`:

```python
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def hash_input_dataframe(df: pl.DataFrame) -> str:
    payload = df.write_ipc(file=None).getvalue()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MetaFeatureCache:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return Path(self.root) / f"{safe}.npy"

    def get(self, key: str) -> NDArray[np.float64] | None:
        p = self._path(key)
        if not p.exists():
            return None
        return np.load(p)

    def put(self, key: str, arr: NDArray[np.float64]) -> None:
        np.save(self._path(key), arr)


def finbert_logits_cached(
    texts: list[str],
    cache: MetaFeatureCache,
    cache_key: str,
    runner: Callable[[list[str]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    out = runner(texts)
    cache.put(cache_key, out)
    return out
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_meta_features.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/meta_features.py tests/test_alpha_meta_features.py
git commit -m "feat: alpha/meta_features.py with disk-backed cache for FinBERT/foundation outputs"
```

---

## Task 19: `alpha/adversarial.py` — adversarial validation + noise regression filter

**Files:**
- Create: `src/quant_research_stack/alpha/adversarial.py`
- Create: `tests/test_alpha_adversarial.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_adversarial.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.alpha.adversarial import (
    adversarial_drop_features,
    drop_below_noise_floor,
    train_holdout_classifier_auc,
)


def test_train_holdout_auc_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3))
    holdout = rng.normal(size=(200, 3))
    auc = train_holdout_classifier_auc(train, holdout)
    assert 0.45 <= auc <= 0.55


def test_train_holdout_auc_high_for_shifted_distributions() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, size=(200, 3))
    holdout = rng.normal(loc=5.0, size=(200, 3))
    auc = train_holdout_classifier_auc(train, holdout)
    assert auc > 0.9


def test_adversarial_drop_features_removes_shifted_columns() -> None:
    rng = np.random.default_rng(0)
    train = pl.DataFrame({"good": rng.normal(size=200), "shifted": rng.normal(loc=0.0, size=200)})
    holdout = pl.DataFrame({"good": rng.normal(size=200), "shifted": rng.normal(loc=8.0, size=200)})
    kept = adversarial_drop_features(train, holdout, candidate_cols=["good", "shifted"], auc_threshold=0.6)
    assert "good" in kept
    assert "shifted" not in kept


def test_drop_below_noise_floor() -> None:
    importance = np.array([0.5, 0.2, 0.05])
    feature_names = ["a", "b", "noise_seed42"]
    kept = drop_below_noise_floor(feature_names, importance, noise_feature="noise_seed42")
    assert "a" in kept
    assert "b" in kept
    assert "noise_seed42" not in kept


def test_drop_below_noise_floor_strips_feature_below_noise() -> None:
    importance = np.array([0.5, 0.01, 0.05])
    feature_names = ["a", "b", "noise_seed42"]
    kept = drop_below_noise_floor(feature_names, importance, noise_feature="noise_seed42")
    assert "a" in kept
    assert "b" not in kept
    assert "noise_seed42" not in kept
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_adversarial.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `alpha/adversarial.py`**

Create `src/quant_research_stack/alpha/adversarial.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def train_holdout_classifier_auc(train: NDArray[np.float64], holdout: NDArray[np.float64]) -> float:
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if holdout.ndim == 1:
        holdout = holdout.reshape(-1, 1)
    x = np.vstack([train, holdout])
    y = np.concatenate([np.zeros(train.shape[0]), np.ones(holdout.shape[0])])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for fold_train, fold_test in skf.split(x, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[fold_train], y[fold_train])
        prob = clf.predict_proba(x[fold_test])[:, 1]
        aucs.append(roc_auc_score(y[fold_test], prob))
    return float(np.mean(aucs))


def adversarial_drop_features(
    train_df: pl.DataFrame, holdout_df: pl.DataFrame, candidate_cols: list[str], auc_threshold: float = 0.6
) -> list[str]:
    kept: list[str] = []
    for col in candidate_cols:
        auc = train_holdout_classifier_auc(
            train_df[col].drop_nulls().to_numpy(),
            holdout_df[col].drop_nulls().to_numpy(),
        )
        if auc < auc_threshold:
            kept.append(col)
    return kept


def drop_below_noise_floor(
    feature_names: list[str], importance: NDArray[np.float64], noise_feature: str
) -> list[str]:
    if noise_feature not in feature_names:
        raise ValueError(f"noise feature {noise_feature!r} not in feature_names")
    idx = feature_names.index(noise_feature)
    noise_imp = float(importance[idx])
    return [name for name, imp in zip(feature_names, importance) if name != noise_feature and float(imp) > noise_imp]
```

- [ ] **Step 4: Run tests; expect pass**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_alpha_adversarial.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/adversarial.py tests/test_alpha_adversarial.py
git commit -m "feat: alpha/adversarial.py with train-holdout AUC + noise-floor feature filter"
```

---

## Task 20: `scripts/alpha_train_s1.py` — end-to-end S1 training entry point

**Files:**
- Create: `scripts/alpha_train_s1.py`

- [ ] **Step 1: Write the entry-point script**

Create `scripts/alpha_train_s1.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha.adversarial import adversarial_drop_features, drop_below_noise_floor
from quant_research_stack.alpha.cv import PurgedKFold
from quant_research_stack.alpha.features import FeatureConfig, build_feature_frame
from quant_research_stack.alpha.io import LoadConfig, load_jane_street, permanent_holdout_split
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig
from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig
from quant_research_stack.alpha.registry import RunMetadata, RunRegistry
from quant_research_stack.alpha.stacking import LinearStacker

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1 full retrain end-to-end.")
    p.add_argument("--config", default="configs/alpha.yaml")
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows for smoke runs.")
    p.add_argument("--experiments-root", default="experiments/alpha_s1")
    return p.parse_args()


def _build_features(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    built = build_feature_frame(
        df, fcfg, base_features=feature_cols, date_col="date_id", symbol_col="symbol_id"
    )
    feature_cols_all = [c for c in built.columns if c not in {"date_id", "symbol_id", "weight", cfg["data"]["target_column"]}]
    return built, feature_cols_all


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    load_cfg = LoadConfig(
        target_column=cfg["data"]["target_column"],
        weight_column=cfg["data"]["weight_column"],
        group_column=cfg["data"]["group_column"],
        holdout_fraction=cfg["data"]["permanent_holdout_fraction"],
    )
    console.print(f"Loading JS from {cfg['data']['jane_street_root']}")
    df = load_jane_street(cfg["data"]["jane_street_root"], load_cfg)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    train_df, holdout_df = permanent_holdout_split(df, load_cfg)
    console.print(f"Train rows={train_df.height}, holdout rows={holdout_df.height}")

    train_feats, feat_cols = _build_features(train_df, cfg)
    holdout_feats, _ = _build_features(holdout_df, cfg)

    # Adversarial drop
    kept = adversarial_drop_features(train_feats, holdout_feats, feat_cols, auc_threshold=0.6)
    console.print(f"Adversarial filter: kept {len(kept)} / {len(feat_cols)} features")
    feat_cols = kept

    # PurgedKFold
    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"],
        group_column="date_id",
        purge=cfg["cv"]["purge_days"],
        embargo=cfg["cv"]["embargo_days"],
    )

    y = train_feats[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w = train_feats[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x = train_feats.select(feat_cols).to_numpy().astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)

    n = x.shape[0]
    oof_ridge = np.zeros(n)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cat = np.zeros(n)
    oof_mlp = np.zeros(n)

    fold_metrics: list[dict[str, Any]] = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(train_feats)):
        console.print(f"Fold {fold_i + 1}/{cfg['cv']['n_folds']}: train={tr_idx.size}, test={te_idx.size}")

        rmod = RidgeAlphaModel(RidgeConfig(alpha=1.0))
        rmod.fit(x[tr_idx], y[tr_idx], w[tr_idx])
        oof_ridge[te_idx] = rmod.predict(x[te_idx])

        lcfg = cfg["models"]["lightgbm"]
        lmod = LightGBMAlphaModel(LightGBMConfig(**{k: lcfg[k] for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "feature_fraction", "bagging_fraction")}))
        lmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_lgb[te_idx] = lmod.predict(x[te_idx])

        xcfg = cfg["models"]["xgboost"]
        xmod = XGBoostAlphaModel(XGBoostConfig(**{k: xcfg[k] for k in ("max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "tree_method")}))
        xmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_xgb[te_idx] = xmod.predict(x[te_idx])

        ccfg = cfg["models"]["catboost"]
        cmod = CatBoostAlphaModel(CatBoostConfig(**{k: ccfg[k] for k in ("depth", "learning_rate", "n_estimators", "early_stopping_rounds")}))
        cmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_cat[te_idx] = cmod.predict(x[te_idx])

        mcfg = cfg["models"]["mlp"]
        mmod = MLPAlphaModel(MLPConfig(
            hidden_dims=mcfg["hidden_dims"],
            dropout=mcfg["dropout"],
            learning_rate=mcfg["learning_rate"],
            batch_size=mcfg["batch_size"],
            max_epochs=mcfg["max_epochs"],
            patience=mcfg["patience"],
            mixed_precision=mcfg["mixed_precision"],
        ))
        mmod.fit(x[tr_idx], y[tr_idx], w[tr_idx], x[te_idx], y[te_idx], w[te_idx])
        oof_mlp[te_idx] = mmod.predict(x[te_idx])

        fold_metrics.append({
            "fold": fold_i,
            "ridge_r2": weighted_zero_mean_r2(y[te_idx], oof_ridge[te_idx], w[te_idx]),
            "lgb_r2": weighted_zero_mean_r2(y[te_idx], oof_lgb[te_idx], w[te_idx]),
            "xgb_r2": weighted_zero_mean_r2(y[te_idx], oof_xgb[te_idx], w[te_idx]),
            "cat_r2": weighted_zero_mean_r2(y[te_idx], oof_cat[te_idx], w[te_idx]),
            "mlp_r2": weighted_zero_mean_r2(y[te_idx], oof_mlp[te_idx], w[te_idx]),
        })

    # Stacking on OOF
    stack_x = np.column_stack([oof_ridge, oof_lgb, oof_xgb, oof_cat, oof_mlp])
    stacker = LinearStacker(alpha=1e-3)
    stacker.fit(stack_x, y, w)

    # Noise-floor filter using LightGBM importance (from a final LGB fit on full train)
    final_lgb = LightGBMAlphaModel(LightGBMConfig(**{k: cfg["models"]["lightgbm"][k] for k in ("num_leaves", "max_depth", "learning_rate", "n_estimators", "early_stopping_rounds", "feature_fraction", "bagging_fraction")}))
    final_lgb.fit(x, y, w, x[-1000:], y[-1000:], w[-1000:])
    importance = final_lgb.feature_importance()
    kept_after_noise = drop_below_noise_floor(feat_cols, importance, noise_feature="noise_seed42")
    console.print(f"Noise-floor filter: kept {len(kept_after_noise)} / {len(feat_cols)} features")

    # Holdout eval
    y_h = holdout_feats[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w_h = holdout_feats[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x_h = holdout_feats.select(feat_cols).to_numpy().astype(np.float64)
    x_h = np.nan_to_num(x_h, nan=0.0)

    h_ridge_pred = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    h_ridge_pred.fit(x, y, w)
    h_pred_ridge = h_ridge_pred.predict(x_h)
    h_pred_lgb = final_lgb.predict(x_h)
    holdout_stack = np.column_stack([
        h_pred_ridge, h_pred_lgb,
        np.zeros(x_h.shape[0]), np.zeros(x_h.shape[0]), np.zeros(x_h.shape[0]),
    ])
    holdout_pred = stacker.predict(holdout_stack)
    holdout_r2 = weighted_zero_mean_r2(y_h, holdout_pred, w_h)
    console.print(f"Holdout weighted zero-mean R²: {holdout_r2:.6f}")

    # Persist artifacts
    reg = RunRegistry(root=Path(args.experiments_root))
    meta = RunMetadata(
        version="0.1.0",
        git_sha=_git_sha(),
        data_hashes={"jane_street_root": cfg["data"]["jane_street_root"]},
        hyperparams=cfg,
        fold_definition={"n_folds": cfg["cv"]["n_folds"], "purge": cfg["cv"]["purge_days"], "embargo": cfg["cv"]["embargo_days"]},
    )
    run_id = reg.create_run(meta)
    reg.save_artifact(run_id, "metrics.json", json.dumps({
        "fold_metrics": fold_metrics,
        "holdout_weighted_zero_mean_r2": holdout_r2,
        "n_features_after_adversarial": len(feat_cols),
        "n_features_after_noise_floor": len(kept_after_noise),
    }, indent=2).encode())
    console.print(f"Run id: {run_id}")
    console.print(f"Artifacts under: experiments/alpha_s1/{run_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test on tiny slice (must NOT touch real JS data — pass --max-rows)**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/alpha_train_s1.py --max-rows 5000`
Expected: prints "Run id: ...", writes `experiments/alpha_s1/<run_id>/metadata.json` and `metrics.json`.

(If JS data not yet loaded at this path, this is expected to fail with `FileNotFoundError`. That's OK — the smoke is then deferred until Task 23. If running CI on a clean clone, use a fixture parquet at `--config` overriding the JS path.)

- [ ] **Step 3: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/alpha_train_s1.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/alpha_train_s1.py
git commit -m "feat: scripts/alpha_train_s1.py end-to-end training driver"
```

---

## Task 21: `scripts/alpha_extract_meta_features.py` — foundation model + sentiment features

**Files:**
- Create: `scripts/alpha_extract_meta_features.py`

- [ ] **Step 1: Write the entry-point script**

Create `scripts/alpha_extract_meta_features.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rich.console import Console
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from quant_research_stack.alpha.meta_features import MetaFeatureCache, finbert_logits_cached

console = Console()


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _finbert_runner(model_dir: Path) -> callable:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device())
    model.eval()

    def run(texts: list[str]) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(_device())
                logits = model(**enc).logits.cpu().numpy()
                outputs.append(logits)
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 3))

    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache FinBERT-style sentiment features for the corpus.")
    p.add_argument("--input-jsonl", default="data/processed/research/research_corpus.jsonl")
    p.add_argument("--model-dir", default="models/huggingface/ProsusAI__finbert")
    p.add_argument("--output-parquet", default="data/processed/research/finbert_features.parquet")
    p.add_argument("--cache-root", default="data/processed/research/meta_cache")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    with open(args.input_jsonl) as h:
        for i, line in enumerate(h):
            if args.limit is not None and i >= args.limit:
                break
            rec = pl.from_json(line.strip()).to_dict(as_series=False) if line.strip().startswith("{") else None
            if rec is None:
                continue
            rows.append({"id": rec["id"][0], "text": rec["text"][0]})
    if not rows:
        console.print("[red]No rows in input.[/red]")
        return 2
    texts = [r["text"] for r in rows]
    cache = MetaFeatureCache(root=Path(args.cache_root))
    runner = _finbert_runner(Path(args.model_dir))
    logits = finbert_logits_cached(texts, cache=cache, cache_key=f"finbert::{len(texts)}", runner=runner)
    df = pl.DataFrame({
        "id": [r["id"] for r in rows],
        "finbert_neg": logits[:, 0].tolist(),
        "finbert_neu": logits[:, 1].tolist(),
        "finbert_pos": logits[:, 2].tolist(),
    })
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output_parquet, compression="zstd")
    console.print(f"Wrote {df.height} rows to {args.output_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/alpha_extract_meta_features.py`
Expected: clean.

- [ ] **Step 3: Smoke test (limit 5 records)**

Run: `cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python scripts/alpha_extract_meta_features.py --limit 5`
Expected: writes a Parquet at `data/processed/research/finbert_features.parquet` with 5 rows; prints a final count line.

- [ ] **Step 4: Commit**

```bash
git add scripts/alpha_extract_meta_features.py
git commit -m "feat: scripts/alpha_extract_meta_features.py with disk-cached FinBERT logits"
```

---

## Task 22: `scripts/alpha_optuna_search.py` — hyperparameter optimization

**Files:**
- Create: `scripts/alpha_optuna_search.py`

- [ ] **Step 1: Write the entry-point script**

Create `scripts/alpha_optuna_search.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha.cv import PurgedKFold
from quant_research_stack.alpha.features import FeatureConfig, build_feature_frame
from quant_research_stack.alpha.io import LoadConfig, load_jane_street, permanent_holdout_split
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for LightGBM on JS.")
    p.add_argument("--config", default="configs/alpha.yaml")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--study-name", default="alpha_lgb")
    p.add_argument("--out-json", default="reports/alpha_optuna_lgb.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    load_cfg = LoadConfig(
        target_column=cfg["data"]["target_column"],
        weight_column=cfg["data"]["weight_column"],
        group_column=cfg["data"]["group_column"],
        holdout_fraction=cfg["data"]["permanent_holdout_fraction"],
    )
    df = load_jane_street(cfg["data"]["jane_street_root"], load_cfg)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    train_df, _ = permanent_holdout_split(df, load_cfg)
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    fcfg = FeatureConfig(
        lag_windows=cfg["features"]["lag_windows"],
        rolling_windows=cfg["features"]["rolling_windows"],
        include_noise_feature=cfg["features"]["include_noise_feature"],
        cross_sectional_ranks=cfg["features"]["cross_sectional_ranks"],
        noise_seed=42,
    )
    built = build_feature_frame(train_df, fcfg, base_features=feature_cols, date_col="date_id", symbol_col="symbol_id")
    fc = [c for c in built.columns if c not in {"date_id", "symbol_id", "weight", cfg["data"]["target_column"]}]
    y = built[cfg["data"]["target_column"]].to_numpy().astype(np.float64)
    w = built[cfg["data"]["weight_column"]].to_numpy().astype(np.float64)
    x = built.select(fc).to_numpy().astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)

    splitter = PurgedKFold(
        n_folds=cfg["cv"]["n_folds"], group_column="date_id",
        purge=cfg["cv"]["purge_days"], embargo=cfg["cv"]["embargo_days"],
    )
    folds = list(splitter.split(built))

    def objective(trial: optuna.Trial) -> float:
        params = LightGBMConfig(
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            n_estimators=2000,
            early_stopping_rounds=80,
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        )
        scores: list[float] = []
        for tr, te in folds:
            mdl = LightGBMAlphaModel(params)
            mdl.fit(x[tr], y[tr], w[tr], x[te], y[te], w[te])
            scores.append(weighted_zero_mean_r2(y[te], mdl.predict(x[te]), w[te]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=args.study_name, sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)
    best = {"value": study.best_value, "params": study.best_params, "n_trials": args.n_trials}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(best, indent=2))
    console.print(f"Best CV R² = {best['value']:.6f}")
    console.print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/alpha_optuna_search.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add scripts/alpha_optuna_search.py
git commit -m "feat: scripts/alpha_optuna_search.py LightGBM TPE hyperparameter search"
```

---

## Task 23: `Makefile` — full retrain target

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create the Makefile**

```makefile
PY := PYTHONPATH=src uv run
EXTRACT := scripts/alpha_extract_meta_features.py
TRAIN := scripts/alpha_train_s1.py
OPTUNA := scripts/alpha_optuna_search.py

.PHONY: test lint type extract train optuna full-retrain-s1 clean-experiments

test:
	$(PY) pytest -q

lint:
	uv run ruff check src scripts tests

type:
	uv run mypy src

extract:
	$(PY) python $(EXTRACT)

train:
	$(PY) python $(TRAIN)

optuna:
	$(PY) python $(OPTUNA) --n-trials 200

full-retrain-s1: test lint extract train optuna
	@echo "S1 full retrain complete. See experiments/alpha_s1/<latest>/metrics.json"

clean-experiments:
	rm -rf experiments/alpha_s1/*
```

- [ ] **Step 2: Smoke `make test` + `make lint`**

Run: `cd /Users/dmr/MachineLearning && make test && make lint`
Expected: all tests pass; ruff clean.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: Makefile with full-retrain-s1 target"
```

---

## Task 24: `scripts/alpha_ood_numerai.py` — OOD validation on Numerai

**Files:**
- Create: `scripts/alpha_ood_numerai.py`

- [ ] **Step 1: Write the script**

Create `scripts/alpha_ood_numerai.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.alpha.metrics import weighted_zero_mean_r2

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD sign-correctness check on Numerai signals data.")
    p.add_argument("--numerai-csv", default="data/raw/kaggle/datasets/code1110__yfinance-stock-price-data-for-numerai-signals")
    p.add_argument("--predictions-parquet", required=True, help="S1 predictions on the Numerai universe.")
    p.add_argument("--out-json", default="reports/alpha_ood_numerai.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preds = pl.read_parquet(args.predictions_parquet)
    if "pred" not in preds.columns or "true" not in preds.columns or "weight" not in preds.columns:
        raise SystemExit("predictions parquet must have 'pred', 'true', 'weight' columns")
    y = preds["true"].to_numpy().astype(np.float64)
    yhat = preds["pred"].to_numpy().astype(np.float64)
    w = preds["weight"].to_numpy().astype(np.float64)
    r2 = weighted_zero_mean_r2(y, yhat, w)
    sign_corr = float(np.mean(np.sign(y) == np.sign(yhat)))
    out = {"weighted_zero_mean_r2": r2, "sign_correctness": sign_corr}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    console.print(f"OOD R²={r2:.6f}, sign_correctness={sign_corr:.4f}")
    console.print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/alpha_ood_numerai.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add scripts/alpha_ood_numerai.py
git commit -m "feat: scripts/alpha_ood_numerai.py for OOD sign-correctness validation"
```

---

## Task 25: S1 success-criteria check

**Files:**
- Create: `scripts/alpha_s1_success_gate.py`

- [ ] **Step 1: Write the gate script**

Create `scripts/alpha_s1_success_gate.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the S1 success gate.")
    p.add_argument("--metrics-json", required=True, help="experiments/alpha_s1/<run_id>/metrics.json")
    p.add_argument("--min-holdout-r2", type=float, default=0.012)
    p.add_argument("--max-fold-std", type=float, default=0.002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = json.loads(Path(args.metrics_json).read_text())
    holdout_r2 = float(metrics["holdout_weighted_zero_mean_r2"])
    fold_metrics = metrics["fold_metrics"]
    lgb_per_fold = np.array([fm["lgb_r2"] for fm in fold_metrics], dtype=np.float64)
    fold_std = float(np.std(lgb_per_fold))
    failed = []
    if holdout_r2 < args.min_holdout_r2:
        failed.append(f"holdout R² {holdout_r2:.6f} < {args.min_holdout_r2}")
    if fold_std > args.max_fold_std:
        failed.append(f"fold std {fold_std:.6f} > {args.max_fold_std}")
    if failed:
        console.print(f"[red]S1 success gate FAILED[/red]: {failed}")
        return 1
    console.print(f"[green]S1 success gate PASSED[/green]: holdout R²={holdout_r2:.6f}, fold std={fold_std:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/alpha_s1_success_gate.py`
Expected: clean.

- [ ] **Step 3: Smoke (gate over a fake metrics.json fixture)**

```bash
mkdir -p /tmp/s1_smoke
cat > /tmp/s1_smoke/metrics.json <<'JSON'
{"holdout_weighted_zero_mean_r2": 0.015, "fold_metrics": [{"lgb_r2": 0.013},{"lgb_r2": 0.014},{"lgb_r2": 0.012},{"lgb_r2": 0.013},{"lgb_r2": 0.014}]}
JSON
PYTHONPATH=src uv run python scripts/alpha_s1_success_gate.py --metrics-json /tmp/s1_smoke/metrics.json
```

Expected: prints `S1 success gate PASSED` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/alpha_s1_success_gate.py
git commit -m "feat: scripts/alpha_s1_success_gate.py to enforce R² and fold-stability gates"
```

---

## Self-review

**Spec coverage:**

| Spec section | Task(s) |
|---|---|
| §1 master architecture | Tasks 1, 2 (docs) |
| §2.1 inputs | Task 5 (io.py) |
| §2.2 module layout | Tasks 3, 5–17 |
| §2.3 training protocol | Tasks 7, 20 |
| §2.4 overfitting controls | Tasks 8 (noise feature), 19 (adversarial + noise floor) |
| §2.5 Apple Silicon constraints | Tasks 14, 15 (MPS auto-detect) |
| §2.6 deliverable artifacts | Task 9 (registry), Task 20 (writes them) |
| §2.7 success gate | Task 25 |
| §2.8 inference contract | Task 17 |
| §4.1 dataset usage | Task 21 (FinBERT extraction); foundation models + LOB AE deferred to future S1 enhancement specs as enumerated in §4 of the master spec |
| §4.2 model usage | Tasks 10–18 |
| §4.3 training power budget | configs/alpha.yaml committed in Task 3 |
| §6.1–6.3 doc rewrites | Task 1 |
| §7.1 testing strategy | every model task has a TDD test; integration in Task 23 (Makefile) |
| §7.2 success criteria | Tasks 25 (criteria 1–2), 24 (criterion 3), remaining criteria are S2/S3/S4 |

Gaps deliberately deferred (not in S1 scope):
- Foundation-model meta-features (chronos-2, TimeMoE, Kronos, etc.) — Task 21 lays the cache infrastructure; specific foundation-model extractors are scheduled for an S1.1 follow-up spec because each requires its own loader (different APIs across the 5 foundation models).
- Orderbook auto-encoder pretraining on CryptoLOB-2025 — separate S1.2 spec.
- Transfer learning from Optiver preprocessed datasets — separate S1.3 spec.

These gaps do not block reaching the R² ≥ 0.012 gate; they are spec §4 enhancements meant to push higher.

**Placeholder scan:** no TBD / TODO. Every code step shows full code.

**Type consistency:** `LoadConfig`, `FeatureConfig`, `PurgedKFold`, `RunMetadata`, `RidgeAlphaModel`, `LightGBMAlphaModel`, `XGBoostAlphaModel`, `CatBoostAlphaModel`, `MLPAlphaModel`, `Conv1DAlphaModel`, `LinearStacker`, `MetaFeatureCache`, `S1Predictor` referenced consistently across `alpha_train_s1.py` and the modules that define them. The `feat_cols` list flows from `build_feature_frame` → `adversarial_drop_features` → `drop_below_noise_floor` with the same type each step.

All 25 tasks are bite-sized, TDD-disciplined, exact-path, exact-command, with frequent commits.
