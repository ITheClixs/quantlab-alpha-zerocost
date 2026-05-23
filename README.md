# QuantLab Alpha

[![Stage](https://img.shields.io/badge/QUANTLAB__STAGE-paper-orange)](docs/runbooks/stage_promotion.md)
[![Kill switch](https://img.shields.io/badge/kill__switch-armed-red)](docs/runbooks/kill_switch.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing-and-verification)

## Abstract

QuantLab Alpha is a local-first alpha research platform for mid-frequency equities and tick-frequency crypto. Its central design decision is a two-layer separation between numeric prediction and language-model governance:

1. **S1 Tabular Predictor** estimates `responder_6` and related return targets with leakage-aware tabular machine learning.
2. **S2 LLM Governor** reviews trade candidates through a retrieval-augmented, citation-constrained local LLM pipeline.

The platform is not a claim of profitability. It is a research harness for testing whether open data, engineered market features, and local models can produce signals that pass out-of-sample validation before they are considered for paper trading. The current branch includes S1 alpha modules, S2 governor modules, S3 feed/broker adapters, S4 paper-stage execution/risk modules, platform ADRs, runbooks, and S4.1 paper-validation tooling.

## Research Thesis

The working thesis is:

> A local alpha stack should separate fast numerical prediction from slower evidence-based governance, and no trade candidate should pass unless the predictor, the governor, risk limits, and audit trail agree.

This gives four concrete research questions:

| ID | Research question | Repository evidence |
|---|---|---|
| RQ1 | Can tabular models improve weighted zero-mean R2 on Jane Street-style targets without leakage? | `src/quant_research_stack/alpha/`, `scripts/train_s1.py`, `configs/alpha.yaml` |
| RQ2 | Can fold-stable features survive adversarial validation and noise-feature controls? | `alpha/features.py`, `alpha/adversarial.py`, `scripts/alpha_s1_success_gate.py` |
| RQ3 | Can an LLM governor veto unsupported trades while being forced to cite local research chunks? | `src/quant_research_stack/governor/`, `configs/governor.yaml`, ADR 0005 |
| RQ4 | Can staged promotion prevent accidental live trading before paper and shadow evidence exists? | `docs/runbooks/`, ADR 0002, `QUANTLAB_STAGE` |

## Contributions

The latest commits move the project from a data/model workspace into a structured research platform:

1. **S1 alpha package** with feature engineering, purged cross-validation, base model wrappers, stacking, metrics, registry, inference, adversarial validation, and training scripts.
2. **S2 governor package** with stable corpus indexing, BM25 retrieval, FAISS dense retrieval, reranking, hybrid retrieval orchestration, fixed retrieval-query template, and Pydantic verdict schema.
3. **Operational modules and specifications** for a four-subsystem platform: S1 predictor, S2 governor, S3 feeds/brokers, and S4 execution/risk.
4. **Stage-gated safety model** using `paper`, `live_shadow`, and `live` promotion states.
5. **Local-only LLM design** using GGUF models and local research corpora rather than paid hosted inference.

## What This Is

QuantLab Alpha is a single-operator research and execution platform for:

- Jane Street-style tabular forecasting;
- open-data market and order-book research;
- feature stability testing;
- local LLM-governed trade vetting;
- paper-trading and future broker-gated execution.

It pairs a tabular predictor stack with an LLM governor that cannot approve a trade unless its verdict is valid JSON and, for pass decisions, backed by retrieved paper chunk citations.

## What This Is Not

This is not a hedge fund in a box, an HFT colocation stack, or a replacement for professional market data. Free feeds impose hard limits:

- crypto can use public Binance/Coinbase WebSocket streams;
- US equities on free tiers are delayed or coarse;
- yfinance-style sources are suitable for research and backtests, not live execution.

This repository is also not financial advice. The operator is responsible for funds, taxes, broker rules, and local regulation.

## Current Implementation Snapshot

| Subsystem | Current state |
|---|---|
| S1 Tabular Predictor | Implemented package under `src/quant_research_stack/alpha/`; includes features, CV, models, stacking, metrics, inference, registry, and scripts |
| S2 LLM Governor | Implemented foundation under `src/quant_research_stack/governor/`; includes corpus, BM25, dense index, reranker protocol, hybrid retrieval, schema, query builder |
| S3 Feeds + Brokers | Implemented package under `src/quant_research_stack/feeds/` and `src/quant_research_stack/brokers/`; includes public feed adapters, recorder/replayer, null broker, Alpaca paper, Binance testnet, and contract tests |
| S4 Execution + Risk | Implemented package under `src/quant_research_stack/execution/`; includes signal pairing, risk gate, sizing, broker routing, position book, reconciliation, kill switch, audit log, daemon, and promotion report tooling |
| S4.1 Paper Validation | Implemented package under `src/quant_research_stack/validation/` plus `scripts/tv_validation_report.py`; includes Alpaca/fixture forward bars, hit rate, governor block rate, net daily PnL, drawdown, rolling Sharpe, and per-signal parquet outputs |
| Stage | `QUANTLAB_STAGE`, default target is `paper` |
| Kill switch | `KILL_TRADING` file in repo root, documented in runbooks |
| S1 success target | holdout weighted zero-mean R2 at least `0.012`, fold standard deviation at most `0.002` |
| S2 retrieval | BM25 top-20, dense top-20, rerank to top-5 |
| S2 runtime plan | Tier 1 Qwen 0.5B LoRA, Tier 2 Mistral Small 22B GGUF, Tier 3 Yi 34B GGUF |

## System Architecture

```text
                    operator shell / runbooks / stage gate
                                  |
                                  v
        +---------------- structured signal bus ----------------+
        |                                                       |
        v                                                       v
S1 Tabular Predictor                                  S2 LLM Governor
alpha package                                         governor package
LightGBM/XGBoost/CatBoost/MLP                         BM25 + dense + rerank
purged CV + stacking                                  citation-required JSON
        |                                                       |
        +-----------------------+-------------------------------+
                                |
                                v
                       S4 Execution + Risk
                       paper -> live_shadow -> live
                       kill switch, audit log, caps
```

The intended per-trade flow is:

```text
Market tick or bar
  -> S1 emits numeric prediction and confidence
  -> S2 retrieves paper evidence and emits pass / veto / insufficient_evidence
  -> S4 applies stage, risk, stale-model, exposure, and kill-switch checks
  -> audit log records every decision and veto
```

## Data and Research Corpus

The platform is built around public or user-authenticated data:

| Source class | Role |
|---|---|
| Jane Street HF mirror | canonical local S1 target source in `configs/alpha.yaml` |
| Kaggle Jane Street variants | validation and augmentation sources when rules and auth allow |
| Hugging Face finance datasets | sentiment, instructions, filings, time series, and market data |
| arXiv/Paper corpus | S2 retrieval and LLM governor evidence |
| Local model store | GGUF LLMs, embeddings, time-series baselines, trained artifacts |

At platform-spec time, the local corpus was recorded as 208 GB across 20,376 files, including HF datasets, Kaggle datasets, Kaggle competitions, HF models, arXiv PDFs, and paper Q&A records. The reproducible source of truth remains the manifests under `manifests/` plus artifact reports under `reports/`.

## S1 Methodology: Tabular Alpha Predictor

S1 is the only subsystem that produces numeric forecasts. Its target is `responder_6`, and its primary metric is weighted zero-mean R2.

### Feature Construction

For a base feature `x_t`, lag features are:

```math
x_{t,l}^{lag} = x_{t-l}.
```

Plain English: the model sees past values of each feature, not future values.

Rolling means over a window `w` are:

```math
\mu_{t,w}(x) =
\frac{1}{w}
\sum_{i=0}^{w-1} x_{t-i}.
```

Plain English: this smooths recent feature history over `w` observations.

Rolling volatility is:

```math
\sigma_{t,w}(x) =
\sqrt{
    \frac{1}{w-1}
    \sum_{i=0}^{w-1}
    (x_{t-i} - \mu_{t,w}(x))^2
}.
```

Plain English: this measures how unstable the feature has been over the same lookback window.

Cross-sectional ranks are:

```math
\mathrm{rank}_{t,j}(x)
=
\frac{1}{M_t - 1}
\sum_{k=1}^{M_t}
\mathbf{1}\{x_{t,k} < x_{t,j}\}.
```

Plain English: instead of using only raw values, the model can ask where one symbol sits relative to other symbols at the same date.

A seeded noise feature is included as a negative control:

```math
\eta_i \sim N(0,1).
```

Plain English: real features that rank below random noise are suspicious and can be discarded.

### Leakage Control

For fold `j`, validation dates form an interval:

```math
V_j = [a_j, b_j].
```

With purge length `p` and embargo length `e`, training dates must satisfy:

```math
d < a_j - p
\quad
\vee
\quad
d > b_j + e.
```

Plain English: samples near the validation period are removed so labels and market regimes do not leak across the split.

### Weighted Zero-Mean R2

The Jane Street-style score is:

```math
R_w^2 =
1 -
\frac{\sum_i w_i (y_i - \hat{y}_i)^2}
     {\sum_i w_i y_i^2}.
```

Plain English: a score above zero means the model beats the baseline that predicts zero for every row.

### S1 Training Objective

The general weighted objective is:

```math
\hat{\theta}
=
\arg\min_{\theta}
[
    \sum_{i=1}^{N} w_i (y_i - f_{\theta}(x_i))^2
    + \lambda \Omega(\theta)
].
```

Plain English: the model is rewarded for accurate weighted predictions and penalized for excessive complexity.

### Base Learner Families

The current S1 design supports:

- Ridge regression as a linear baseline;
- LightGBM with early stopping on weighted R2;
- XGBoost with histogram tree method;
- CatBoost for categorical-symbol style effects;
- MLP with dropout and MPS/mixed-precision support;
- compact sequence models for short windows.

Gradient boosting is modeled as:

```math
F_m(x) = F_{m-1}(x) + \eta f_m(x).
```

Plain English: each tree adds a small correction to the previous ensemble prediction.

For squared error, the residual target at boosting step `m` is:

```math
r_{i,m} = y_i - F_{m-1}(x_i).
```

Plain English: every new tree learns what the prior trees still missed.

### Positive Linear Stacking

Let `Z` be the matrix of out-of-fold predictions from base models. A positive Ridge stacker solves:

```math
\hat{a}
=
\arg\min_{a}
[
    \sum_i w_i (y_i - Z_i a)^2
    + \alpha \sum_j a_j^2
],
\qquad
a_j \ge 0.
```

Plain English: the stacker blends models, penalizes unstable weights, and prevents a base model from being used with a negative sign.

The final stacked forecast is:

```math
\hat{y}_i^{stack} = \sum_j \hat{a}_j \hat{y}_{i,j}.
```

Plain English: final S1 output is a weighted blend of base model predictions.

### Success Gate

The S1 gate is defined in `configs/alpha.yaml`:

```text
holdout weighted zero-mean R2 >= 0.012
fold standard deviation of R2 <= 0.002
improvement over Ridge baseline >= 60%
```

This gate blocks downstream promotion if the model wins on only one lucky fold.

## S2 Methodology: LLM Governor and RAG

S2 never originates trades. It receives candidate S1 signals and returns one of:

```text
pass | veto | insufficient_evidence
```

A `pass` verdict must include citations to local research chunks unless it is a Tier 1 fast-gate verdict that still requires later Tier 2 review.

### Corpus Index

Each paper chunk is stored with:

```text
id, source_type, source_path, chunk_index, text, sha256, n_words
```

The corpus hash is:

```math
H(D) =
\mathrm{SHA256}
[
    (id_1, h_1),
    (id_2, h_2),
    ...,
    (id_N, h_N)
].
```

Plain English: if chunk IDs or chunk hashes change, the index metadata changes too.

### Hybrid Retrieval

S2 combines sparse and dense retrieval:

```math
C(q) = C_{BM25}(q) \cup C_{dense}(q).
```

Plain English: candidate evidence comes from both keyword overlap and vector similarity.

Dense retrieval uses cosine similarity:

```math
s_{dense}(q,c)
=
\frac{v_q^\top v_c}
      {\|v_q\|_2 \|v_c\|_2}.
```

Plain English: chunks are ranked by the angle between query and chunk embeddings.

The reranker produces the final evidence set:

```math
E_k(q) =
\mathrm{topk}_{c \in C(q)}
s_{rerank}(q,c).
```

Plain English: BM25 and dense search find candidates; the reranker sorts the best final citations.

### Fixed Query Template

The latest `governor/query_builder.py` commit fixes the retrieval query shape:

```text
<regime> <symbol> direction=<direction> horizon=<minutes>m vol=<recent_vol_label>
```

The template is intentionally deterministic so retrieval behavior is testable.

### Citation Invariant

The Pydantic verdict schema enforces:

```math
decision = pass
\quad
\Rightarrow
\quad
|citations| \ge 1.
```

Plain English: if the model says "pass" without citations, the schema downgrades the decision to `insufficient_evidence`.

Tier precedence is conservative:

```math
decision_{final} = pass
\quad
\Leftrightarrow
\quad
decision_1 = pass
\wedge
decision_2 = pass
\wedge
decision_3 \ne veto.
```

Plain English: any veto or insufficient-evidence result blocks the trade.

## S3 and S4 Execution Status

S3 and S4 now have paper-stage implementations. Their research contract remains:

1. feed adapters normalize public crypto and free-tier equity data;
2. broker adapters are stage-gated;
3. S4 is the only layer allowed to place orders;
4. every decision is written to an append-only audit log;
5. a file-based kill switch halts trading.

## Paper-Trading and Risk Equations

Position sizing should be volatility-scaled:

```math
q_t =
\mathrm{clip}
(
    k \frac{\hat{\mu}_{t,h}}{\hat{\sigma}_{t,h}^2 + \epsilon},
    -q_{max},
    q_{max}
).
```

Plain English: the position grows with expected return and shrinks when estimated risk rises.

Net paper PnL is:

```math
PnL_{t+1}
=
q_t (p_{t+1} - p_t)
- c |q_t - q_{t-1}|
- s_t |q_t - q_{t-1}|.
```

Plain English: the strategy earns or loses from the price move, then pays transaction costs and slippage when it changes position.

The research objective is risk-adjusted, not raw return:

```math
J(\pi)
=
E[R_t^{\pi}]
- \lambda_v Var(R_t^{\pi})
- \lambda_d DD(\pi)
- \lambda_u E[|q_t - q_{t-1}|].
```

Plain English: a strategy is penalized for volatility, drawdown, and excessive turnover even if gross return is positive.

## Stage Promotion Flow

```text
paper
  Alpaca paper + Binance Testnet. All trades simulated.
  Required evidence: at least 90 days, Sharpe >= 1.0 net, max DD <= 15%.

live_shadow
  Real broker connected read-only. Orders route to null_broker.py.
  Required evidence: at least 30 days, paper-vs-real quote match within 0.5%.

live
  Real money. Hard caps: 2% position, 80% gross, 40% net,
  3% daily drawdown, 12% cumulative drawdown.
```

The running process cannot promote itself. Only a human operator edits `.env`, restarts the process, and signs the runbook artifact.

## Reproduction Commands

```bash
uv sync --extra dev --extra llm
huggingface-cli login
# Place Kaggle token at ~/.kaggle/kaggle.json and accept competition rules.

PYTHONPATH=src uv run python scripts/download_papers.py
PYTHONPATH=src uv run python scripts/prepare_research_corpus.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_parquet.py
PYTHONPATH=src uv run python scripts/paper_corpus_to_instructions.py
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types dataset --max-gb 50
PYTHONPATH=src uv run python scripts/download_hf_artifacts.py --types model --max-gb 80
PYTHONPATH=src uv run python scripts/download_kaggle_artifacts.py --unzip
PYTHONPATH=src uv run python scripts/dedupe_and_verify.py

make full-retrain-s1
```

`make full-retrain-s1` runs tests, lint, feature extraction, S1 training, Optuna search, and final reporting.

## Testing and Verification

```bash
PYTHONPATH=src uv run pytest -q
uv run ruff check src scripts tests
uv run mypy src
```

The default pytest configuration skips `governor_slow` tests. Those are reserved for integration runs that load local LLM models.

## Safety Standard

```text
Kill switch        KILL_TRADING file in repo root halts all trading.
Audit log          logs/audit/YYYY-MM-DD.jsonl, append-only after rotation.
Replay invariant   Replaying the audit log must reproduce the same decision sequence.
Stage gate         QUANTLAB_STAGE controls broker class selection.
Risk caps          position, gross exposure, net exposure, daily DD, cumulative DD.
Citation rule      S2 pass verdicts require local paper chunk citations.
```

## Latest Commit Analysis

The newest commits indicate that the active development front is S2 Governor infrastructure:

| Commit theme | Meaning for README |
|---|---|
| `governor/query_builder.py` | README documents the fixed deterministic retrieval-query template |
| `governor/retrieval.py` | README documents the BM25 + dense + rerank retrieval cascade |
| `governor/reranker.py` | README separates retrieval candidates from final evidence ranking |
| `governor/dense_index.py` | README includes FAISS/cosine dense retrieval math |
| `governor/bm25_index.py` | README names sparse lexical retrieval as a first-stage evidence source |
| `governor/corpus.py` | README documents stable chunk IDs and corpus hashing |
| `governor/signal_schema.py` | README documents the citation invariant and auto-downgrade rule |
| `alpha_*` scripts and modules | README keeps S1 as the numeric prediction layer and documents training gates |

## Limitations

- S1 can be benchmarked locally, but a high score is not equivalent to a profitable live strategy.
- S2 can veto unsupported trades, but it does not prove that approved trades are profitable.
- S3/S4 are paper-stage modules. They are not live-trading approval, and live broker adapters remain intentionally blocked behind human promotion controls.
- S4.1 paper-validation reports require valid Alpaca data credentials, S1/S2 artifacts, and S4 fill audit logs before promotion gates should be treated as operational evidence.
- Free data limits make sub-millisecond equity HFT out of scope.
- Real-money use requires broker permissions, tax handling, regulatory review, and operator responsibility.

## References

```text
Master spec: docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md
S2 spec:     docs/superpowers/specs/2026-05-16-quantlab-alpha-s2-governor-design.md
S1 plan:     docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md
S2 plan:     docs/superpowers/plans/2026-05-16-quantlab-alpha-s2-governor-implementation.md
ADRs:        docs/architecture/adrs/
Runbooks:    docs/runbooks/
Manifests:   manifests/datasets.yaml manifests/models.yaml manifests/papers.yaml manifests/kaggle.yaml
```

## Legal Disclaimer

This repository is not a regulated investment advisor and produces no investment advice. It is a research system. Real-money trading is solely the operator's responsibility.
