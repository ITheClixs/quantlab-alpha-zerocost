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
feature_cols.json ordered list + sha256
_artifact_sha256.json  sha256 over every artifact above
models/
  ridge.joblib
  lightgbm.txt
  lightgbm.config.json
  xgboost.json
  xgboost.config.json
  catboost.cbm
  catboost.config.json
  mlp.pt
  sequence.pt
  stacker.joblib
report.md
audit_log_smoke.jsonl    proof S1 wrote to the audit format expected by S4
```

The result must be reproducible from a clean repository using `make full-retrain-s1`.
