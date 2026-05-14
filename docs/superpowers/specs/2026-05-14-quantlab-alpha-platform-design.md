# QuantLab Alpha Platform — Master Architecture Design

**Date:** 2026-05-14
**Status:** Approved (sections 1–7 walked through with operator)
**Project:** QuantLab Alpha (`/Users/dmr/MachineLearning`)
**Hardware target:** MacBook Air M4, 24 GB unified memory, ~525 GB free disk
**Corpus on disk at spec time:** 208 GB across 20 376 files (29 HF datasets, 22 Kaggle datasets, 5 official Kaggle competitions, 20 HF models, 48 arXiv PDFs + 4 743 paper Q&A records)

---

## 1. Master Architecture

QuantLab Alpha is a four-layer commercial alpha generation and execution platform. Each layer is independent, has its own spec, and is built in sequence. Promotion to real-money trading is gated behind three operational stages with explicit human sign-off.

```text
                                ┌────────────────────────────────────────┐
                                │           SHELL / OPERATOR             │
                                │  CLI + dashboards + kill-switch UI     │
                                └───────────────────┬────────────────────┘
                                                    │
     ┌────────────────────┬─────────────────────────┼──────────────────────┬─────────────────────┐
     │                    │                         │                      │                     │
  ┌──▼───────────┐  ┌─────▼────────┐       ┌────────▼─────────┐    ┌───────▼────────┐    ┌───────▼──────────┐
  │   S1         │  │   S2         │       │   S3             │    │   S4           │    │  Risk + State    │
  │ Tabular      │  │ LLM Governor │       │ Real-time data + │    │ Execution +    │    │  Position book,  │
  │ Predictor    │  │ + RAG over   │       │ broker abstraction│   │ paper / live-  │    │  PnL, kill switch│
  │ (LightGBM,   │  │ 208 GB corpus│       │ (Alpaca, IBKR,   │    │ shadow / real- │    │  daily DD limits │
  │  XGB,        │  │ + GBNF JSON  │       │ Binance, Coinbase│    │ money gates    │    │                  │
  │  CatBoost,   │  │ + LoRA finetune│     │ WebSocket)       │    │                │    │                  │
  │  MLP, stack) │  │ on paper Q&A │       │                  │    │                │    │                  │
  └──────┬───────┘  └──────┬───────┘       └─────────┬────────┘    └────────┬───────┘    └─────────┬────────┘
         │                 │                         │                      │                      │
         └─────────────────┴─────────────────────────┴──────────────────────┴──────────────────────┘
                                            structured signal bus
                                            (JSON-schema-validated)
```

**Per-trade two-tier decision flow:**

```text
Market tick / bar arrives
    │
    ▼
S1 tabular model emits numeric prediction + confidence
    │  if |prediction| > threshold
    ▼
S2 LLM governor:
    ├─ retrieves top-k paper chunks via BM25 + dense embedding hybrid
    ├─ emits constrained-JSON signal {direction, confidence, horizon, cited_paper_ids[]}
    ├─ MUST cite ≥ 1 paper chunk or returns insufficient_evidence
    └─ rejects if predicted regime contradicts cited papers' assumptions
    │  if both layers agree
    ▼
S4 execution:
    ├─ pre-trade risk checks (position, exposure, daily DD, feed freshness)
    ├─ routes to broker (paper / null_broker / live, depending on QUANTLAB_STAGE)
    └─ records to position book and audit log
```

**Latency budgets:**

| Layer | Budget |
|---|---|
| S1 tabular inference | sub-millisecond per row |
| S2 LLM governor | async; 5–30 s per trade-candidate; fast-path LoRA model can return veto in < 500 ms |
| S3 data | tick-level for crypto, 15-min bars for equities (free-feed constraint) |
| S4 execution | hundreds of ms (broker-latency-bound) |

---

## 2. S1 — Tabular Jane Street Predictor (detailed)

S1 is the *only* layer that produces numeric forecasts. Every other layer either filters S1's output or acts on it. S1 must be reproducible from a clean clone in under 4 days wall-clock on the M4.

### 2.1 Inputs (already on disk)

```text
data/raw/huggingface/TnnnT0326__Jane_Street_Competition/        8.2 GB  primary
data/raw/kaggle/datasets/saurabhshahane__jane-street-preprocessed-train/  5.9 GB
data/raw/kaggle/datasets/christoffer__synthetic-jane-street-dataset/      5.7 GB  augmentation
data/raw/kaggle/datasets/ravi20076__janestreetpublicv1/          0.2 GB
data/raw/kaggle/datasets/louise2001__janestreetimputeddata/      0.5 GB
data/raw/kaggle/competitions/jane-street-real-time-market-data-forecasting/  rule-gated; README only
```

The TnnnT0326 HF mirror is canonical. Kaggle datasets are validation and augmentation.

**Target:** `responder_6`. Loss: weighted MSE using the dataset's `weight` column. Reported metric: weighted zero-mean R² (the official competition metric) plus a Sharpe-proxy.

### 2.2 Module layout

```text
src/quant_research_stack/alpha/
  io.py             Load JS Parquet, split by date_id, no leakage
  features.py       Lag features, rolling stats per symbol, cross-sectional ranks, interactions
  cv.py             Time-series PurgedKFold by date_id with embargo
  models/
    ridge.py        L2 baseline (refactored from existing local_training.py)
    lightgbm.py     LGB with early stopping on weighted R²
    xgboost.py      XGB with histogram method
    catboost.py     CatBoost with categorical symbol embeddings
    mlp.py          Compact PyTorch MLP, MPS, batchnorm + dropout, mixed precision
    sequence.py     1D-CNN / compact Transformer encoder over short windows
  stacking.py       Linear meta-learner on out-of-fold predictions of all base models
  metrics.py        Weighted zero-mean R², Sharpe-proxy, hit rate, calibration plots
  inference.py      Polars-row predict(); ≤ 1 ms per row
  registry.py       Save/load artifacts to models/trained/<run>/ with semver + sha256
```

### 2.3 Training protocol

1. Walk-forward folds by `date_id` with 5-day purge + 5-day embargo (López de Prado).
2. Train each base model on each fold; predict held-out + later folds (OOF predictions).
3. Stacking meta-learner fitted only on OOF predictions (no leakage).
4. Final metric reported on a permanent holdout slice (last 20 % of `date_id`s, never touched during dev).
5. Each run writes `metrics.json`, `predictions.parquet`, `feature_importance.parquet`, `cv_folds.json`, `git_sha`, `data_hashes.json`.

### 2.4 S1 overfitting controls

- Adversarial validation between train and holdout — any feature with classifier AUC > 0.6 is dropped or transformed.
- Feature importance must agree across ≥ 3 of 5 folds; one-fold-only signals are discarded.
- A seeded Gaussian noise feature is included; any real feature ranked below it across ≥ 3 folds is removed.

### 2.5 Apple-Silicon-specific constraints

- LightGBM / XGB / CatBoost on CPU (GPU mode unavailable for CatBoost on MPS).
- MLP on MPS with `torch.float16`; CPU fallback per row if MPS errors.
- Polars streaming for JS rows; LightGBM full-cache capped at ~6 M rows per fold otherwise switched to streaming.

### 2.6 S1 deliverable artifacts

```text
experiments/alpha_s1/<run_id>/
  metadata.json             git_sha, data_hashes, hyperparams, fold definition
  predictions.parquet       row_id, fold, pred, true, weight
  metrics.json              weighted_zero_mean_r2, sharpe_proxy, hit_rate, fold_breakdown
  feature_importance.parquet
  models/
    ridge.joblib
    lightgbm.txt
    xgboost.json
    catboost.cbm
    mlp.pt
    stacker.joblib
  report.md
```

### 2.7 S1 hard success gate (blocks S2 work)

- Weighted zero-mean R² ≥ **0.012** on the permanent holdout (vs current 0.0075 Ridge baseline; +60 %).
- Improvement holds across all 5 folds (no single-fold luck).
- Full retrain reproducible in ≤ 4 days wall-clock on the M4.

### 2.8 S1 inference contract for S4

```python
class S1Predictor(Protocol):
    def predict(self, row: pl.DataFrame) -> tuple[float, float]:
        """Return (responder_6_estimate, confidence_score). Must complete in < 1 ms."""
```

---

## 3. S2, S3, S4 — Outlines

Each subsystem gets its own spec, plan, and implementation cycle later. The commitments below are what those future specs will lock down so the master architecture stays coherent.

### 3.1 S2 — LLM Governor + RAG

**Purpose:** Veto, explain, and contextualize signals that pass S1's threshold. Never originates a trade.

```text
src/quant_research_stack/governor/
  retrieval.py        BM25 + dense embedding hybrid over data/processed/research/parquet
  embeddings.py       sentence-transformers/all-MiniLM-L6-v2 for paper-chunk vectors
  llm_runtime.py      llama.cpp Python bindings, persistent per-session server
  gbnf_grammar.py     GBNF grammar enforcing the JSON schema below
  signal_schema.py    Pydantic models validating LLM output
  governor.py         Orchestrator: S1 → retrieval → LLM → JSON → veto/pass
  lora_adapter.py     LoRA fine-tuning on data/processed/research/instructions.jsonl
```

**Constrained output schema (GBNF enforced — non-JSON tokens are physically impossible):**

```json
{
  "signal_id": "uuid",
  "decision": "pass | veto | insufficient_evidence",
  "direction": -1,
  "confidence": 0.0,
  "horizon_minutes": 15,
  "regime_tag": "trending | mean_reverting | high_vol | low_vol | unknown",
  "rationale_short": "string ≤ 200 chars",
  "cited_paper_chunk_ids": ["chunk-id-1", "chunk-id-2"],
  "contradictions_flagged": ["string"]
}
```

Outputs missing ≥ 1 `cited_paper_chunk_id` are automatically replaced with `decision: insufficient_evidence`. S4 never sees raw text.

**LoRA fine-tune:**

- Base: `Qwen/Qwen2.5-0.5B-Instruct` first, then `Qwen/Qwen2.5-Coder-1.5B` if quality requires.
- Training data: 4 743 paper Q&A records + held-out test split.
- Adapter only; base model frozen.
- Eval: held-out Q&A accuracy + veto-precision ablation on real backtested signals.

**Model preference order:**

1. `bartowski/Mistral-Small-Instruct-2409-GGUF` Q4_K_M (primary, 22B-class).
2. `bartowski/Yi-1.5-34B-Chat-GGUF` Q4_K_M (deeper reasoning when latency allows).
3. `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (fast veto path, < 500 ms).

### 3.2 S3 — Real-time Data + Broker Abstraction

**Purpose:** A single typed interface between market reality and our models. Replaceable feeds, replaceable brokers.

```text
src/quant_research_stack/feeds/
  base.py             FeedAdapter protocol: subscribe(symbol), iterate() → Tick | Bar
  binance_ws.py       Binance public WebSocket (tick, free, no auth)
  coinbase_ws.py      Coinbase public WebSocket (tick, free, no auth)
  alpaca_rest.py      Alpaca free tier 15-min equity bars
  polygon_free.py     Polygon free tier daily / 15-min equity bars (backup)
  ccxt_unified.py     CCXT normalization across crypto exchanges
  recorder.py         Writes ticks to data/live/parquet/<symbol>/<date>.parquet
  replayer.py         Replays recorded feeds to backtests at controlled speed

src/quant_research_stack/brokers/
  base.py             BrokerAdapter protocol: place_order, cancel_order, positions, account
  alpaca_paper.py     Alpaca paper trading (free)
  alpaca_live.py      Alpaca live trading (gated)
  ibkr_paper.py       IBKR paper trading
  ibkr_live.py        IBKR live (gated, Pro account required)
  binance_paper.py    Binance Testnet (free)
  binance_live.py     Binance live spot/futures (gated)
  null_broker.py      Dry-run; records intent only
```

**Free-data constraint:**

- Crypto: Binance / Coinbase public WebSocket — real-time, no API key, no rate limit beyond IP throttling.
- Equities (US): Alpaca free tier IEX data — 15-min minimum, no Level 2.
- Backup: yfinance for daily bars (research / backtest only, never live).

**Asymmetric data trust:** every live feed is mirrored to disk in real time so backtests can later replay exact production conditions.

### 3.3 S4 — Execution + Promotion Gates

**Purpose:** The only module that places orders. Hard gates between paper / live_shadow / real-money.

```text
src/quant_research_stack/execution/
  signal_bus.py       Async queue: S1 + S2 verdicts → execution candidates
  risk_engine.py      Pre-trade checks: position, gross/net exposure, daily DD, stale model
  position_book.py    Authoritative state; reconciled with broker every minute
  pnl.py              Mark-to-market, intraday + cumulative
  router.py           Picks broker adapter based on current promotion stage
  kill_switch.py      OS signal + file flag + daily-DD trigger
  audit_log.py        Append-only JSONL of every decision, every veto, every fill
```

**Three promotion stages — single env var `QUANTLAB_STAGE`:**

| Stage | What it does | Broker | Gate to next stage |
|---|---|---|---|
| `paper` | All trades to paper broker | `*_paper.py` | 90 calendar days with Sharpe ≥ 1.0 net of fees, max DD ≤ 15 % |
| `live_shadow` | Real broker connected, every order to `null_broker.py`, parallel paper book | `null_broker.py` + real read-only | 30 days, paper-vs-real quote match within 0.5 % slippage |
| `live` | Real money. Hard caps on position size, gross exposure, daily DD. Kill switch armed. | `*_live.py` | n/a |

**Hard kill conditions (any one halts trading immediately):**

- Daily realized DD > 5 % of account equity
- Cumulative DD > 15 % from peak
- Two consecutive minutes without market data
- Model age > 7 days without retraining ping
- File `KILL_TRADING` present in repo root
- SIGTERM / SIGINT received

`QUANTLAB_STAGE` cannot be promoted from inside the running process. Only a human edits the env file + restarts + signs a `docs/runbooks/stage_change.md` artifact.

---

## 4. Squeeze the Corpus + Models

The training pipeline uses every relevant artifact on disk. Nothing sits idle.

### 4.1 Dataset usage matrix

| Dataset | Role |
|---|---|
| JS data (5 sources, 20 GB) | Primary supervised target (`responder_6`). |
| Optiver preprocessed (1.9 GB) | Transfer-learning pretraining for the MLP; encoder fine-tuned on JS. |
| Optiver baseline models (1.3 GB across 3 datasets) | Additional stack members (pre-trained LGB / XGB fold artifacts). |
| CryptoLOB-2025 (30 GB) | Unsupervised orderbook auto-encoder pretraining; encoder weights frozen for downstream LOB feature extraction. |
| HFT crypto LOB (5.1 GB) + 4 others | Multi-asset orderbook regime tagging. |
| Westland BTC liquidity (9.1 GB) | Auxiliary multi-task regime targets for the MLP. |
| Numerai signals + yfinance | External out-of-distribution validation. |
| Binance futures + BTC OHLCV | Crypto branch of the same architecture; shared encoder, separate heads. |
| SP500 / NASDAQ daily | Cross-asset feature engineering (sector betas, regime indicators). |
| Twitter / FinGPT / FinBank / earnings / 10-K text | Trains S2's LoRA adapter; feeds the FinBERT-based feature extractor (per 4.2). |
| General ML (WikiText, IMDB, CIFAR-10, OpenOrca, GSM8K, MATH-500, monash_tsf, ETT) | Sanity baselines for the training pipeline. |

### 4.2 Model usage matrix

**Tabular base learners (S1):** Ridge / LightGBM / XGBoost / CatBoost / MLP.

**Feature extractors (frozen, cached once over training data):**

| Model | Role | Feature dim |
|---|---|---|
| `ProsusAI/finbert` | Daily sentiment per ticker | 3 logits + 1 score |
| `hasnain43/bert-stock-sentiment-v1` | Second-opinion sentiment | 3 logits |
| `FinLang/finance-embeddings-investopedia` | 768-dim finance-domain embedding for chunks | 768 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dim general embeddings, RAG + fallback features | 384 |
| `distilbert-base-uncased` | Ablation-only embeddings | 768 |

**Time-series foundation models (transfer learning + augmentation):**

| Model | Role |
|---|---|
| `amazon/chronos-2` | Zero-shot forecasts on each `feature_N` column; meta-features for the tabular stack. |
| `autogluon/chronos-bolt-small` | Lower-latency baseline; used in S4 live mode where chronos-2 is too slow. |
| `Maple728/TimeMoE-200M` | Sparse-MoE forecasts; ensemble member, especially for the crypto branch. |
| `ibm-granite/granite-timeseries-ttm-r2` | Short-horizon point forecasts. |
| `NeoQuasar/Kronos-base` | Candlestick / K-line pattern features (crypto + SP500 daily). |

Each foundation model produces 5–20 meta-features per row, concatenated to engineered features in the LightGBM / XGB stack.

**LLM stack (S2):**

| Model | Role |
|---|---|
| `bartowski/Mistral-Small-Instruct-2409-GGUF` Q4_K_M | Primary governor (22B, ~12 GB on disk). |
| `bartowski/Yi-1.5-34B-Chat-GGUF` Q4_K_M | Deeper governor when latency allows (34B, ~19 GB). |
| `bartowski/Qwen2.5-14B-Instruct-GGUF` Q4_K_M | Mid-tier reasoning option. |
| `TheBloke/finance-LLM-13B-GGUF` Q4_K_M / Q5_K_M | Finance-specialist baseline; cross-check vs Mistral. |
| `QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF` | Lightweight RAG path. |
| `Tail-LS/Qwen2.5-3B-dpo-finance` | DPO-aligned 3B for fast veto checks. |
| `Qwen/Qwen2.5-Coder-1.5B` | Code-generation for ad-hoc backtest scripts the governor writes. |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | MLX-native fast inference for high-frequency veto. |
| `Qwen/Qwen2.5-0.5B-Instruct` | LoRA target for fast classification-style veto. |
| `roneneldan/TinyStories-33M` | Smoke-test fallback only. |

### 4.3 Training power budget

The M4 will be used aggressively. Per the operator's instruction: extra hours are acceptable when they improve the model.

| Step | Budget |
|---|---|
| S1 base training per full retrain | up to **24 h** wall-clock. MLP / sequence models can use the full overnight slot. CatBoost depth 12. LightGBM / XGB hyperparameter search: Optuna 200 trials per fold. |
| Orderbook auto-encoder pretraining (30 GB CryptoLOB-2025) | up to **12 h**. Mixed precision; batch size pushed to MPS OOM; gradient accumulation otherwise. |
| LoRA adapter S2 (Qwen2.5-0.5B-Instruct) | up to **8 h**. Plus 4 h on Qwen2.5-Coder-1.5B if quality marginal. |
| Foundation-model feature extraction | parallel batched, ~6 h across 5 models. Cached to Parquet; runs once. |
| Stacking + Optuna meta-search | up to **48 h** combined. |

Total worst case for one full retrain cycle: ~4 days end-to-end. Pipeline checkpoints every step; restarts are cheap. A `make full-retrain` Makefile target chains all of this with progress logging.

### 4.4 Out-of-scope (YAGNI)

- No image / vision models in S1 (CIFAR-10 stays for sanity tests only).
- No general-language fine-tuning (WikiText / TinyStories never enter S1).
- No MLX LM training from scratch — inference only.
- No model larger than 34B GGUF.
- No real-money execution before the three-stage gate is in place.

---

## 5. Risk, Compliance, Commercial Readiness

### 5.1 Risk taxonomy

| Risk class | Where caught | Action on breach |
|---|---|---|
| Model staleness | S4 router | Refuse to trade if last successful S1 retrain > 7 days |
| Data feed gap | S3 → S4 | Refuse to trade if no tick / bar in 2 min (crypto) / 30 min (equity) |
| Spread / liquidity blow-out | S4 risk_engine | Skip order if effective spread > 10× rolling 1-day median |
| Position-size limit | S4 risk_engine | Reject order; per-symbol cap and gross cap |
| Gross / net exposure | S4 risk_engine | Reject if order pushes gross > 100 % equity or net > 50 % |
| Single-name concentration | S4 risk_engine | Reject if one symbol > 10 % gross |
| Daily realized DD | kill_switch | Halt trading for the day; cancel open orders |
| Cumulative DD from peak | kill_switch | Halt trading; require human re-arm |
| Order rate limit | S4 router | Throttle to N orders/min per broker rules |
| Stale clock / NTP drift | S4 audit_log + kill | Halt if local-vs-exchange time > 1 s |
| Broker reconciliation mismatch | position_book | Halt + alert if internal book ≠ broker book at minute reconcile |
| Sanction / restricted-list | risk_engine | Per-symbol blacklist file checked on every order |
| LLM disagreement | governor | If S2 = `veto` or `insufficient_evidence`, S4 skips order regardless of S1 confidence |

These limits live in `configs/risk.yaml`. Two-person rule for editing once `QUANTLAB_STAGE = live_shadow`.

### 5.2 Three promotion stages — formalized

```yaml
# configs/promotion.yaml
stages:
  paper:
    broker_class: "*_paper"
    can_promote_when:
      min_calendar_days: 90
      min_trades: 200
      min_sharpe_net_of_fees: 1.0
      max_observed_drawdown_pct: 15
      out_of_sample_r2_holds: true
    promote_action: "write paper_to_shadow.md report; require human approval"

  live_shadow:
    broker_class: "null_broker + read-only real account"
    can_promote_when:
      min_calendar_days: 30
      paper_book_vs_real_quotes_match_within_pct: 0.5
      no_kill_switch_triggers: true
      min_sharpe_in_shadow_book: 0.8
    promote_action: "human signs shadow_to_live.md; risk_engine caps cut to 50% of design caps for first 30 days of live"

  live:
    broker_class: "*_live"
    invariants:
      kill_switch_armed: true
      max_position_size_pct: 2
      max_gross_exposure_pct: 80
      max_net_exposure_pct: 40
      max_daily_dd_pct: 3
      max_cumulative_dd_pct: 12
    rollback_action: "any kill_switch trip drops back to live_shadow for 7 days"
```

`QUANTLAB_STAGE` is read at process start. The running process cannot promote itself. Only a human edits `.env`, restarts, and signs the runbook artifact.

### 5.3 Audit and immutability

Every decision in S2 / S3 / S4 lands in an append-only JSONL audit log:

```text
logs/audit/YYYY-MM-DD.jsonl
  one record per: signal_received, governor_verdict, risk_check, order_intent,
                  order_sent, fill, reconcile, kill_trigger
  records include: timestamp_utc, stage, model_sha, data_sha, decision,
                   feature_hash, model_output, governor_output, risk_state
```

Files are append-only at the OS level (`chmod a-w` on rotation). Replay-by-audit-log must reproduce the same decision sequence — this is a tested invariant.

### 5.4 Compliance / legal posture

The repo's docs state plainly:

- Real-money trading by a single individual is subject to local regulator rules. US: trading own funds for own account is generally fine; managing others' funds requires registration (RIA / CTA / etc.).
- Crypto trading is jurisdiction-dependent; tax reporting is the operator's responsibility.
- The repository is not a regulated investment advisor and produces no investment advice.
- Every output of the system carries `not_investment_advice: true` in the audit log.
- The operator is solely responsible for funds, taxes, brokerage relationships, and regulatory compliance.

This is text in the docs, not a runtime safeguard. The safeguard is the kill switch.

### 5.5 Disaster recovery and state

- State is durable: SQLite locally + nightly Parquet snapshots to `data/snapshots/`.
- A process crash leaves the position book recoverable from broker reconciliation + last audit replay.
- The kill flag (`KILL_TRADING` file in repo root) survives reboots — the bot will not start trading until the operator deletes it.
- Every stage transition writes a `runbook.md` describing manual rollback steps.

### 5.6 Observability

- Prometheus metrics exported on a local port (the already-installed Docker compose has the slots).
- Grafana dashboards: PnL, exposure, signal rate, veto rate, latency by layer, data-feed gap times.
- CLI command `quantlab status` prints the live snapshot for ssh-only ops.

---

## 6. Doc Rewrites (CLAUDE.md, AGENTS.md, README.md)

The three top-level docs need surgery to align with the commercial scope. This work is **Task 1 of the S1 implementation plan**, not a documentation drive-by, so it is TDD'd against this spec and committed alongside the first round of code.

### 6.1 CLAUDE.md rewrite

- Project goal: rewrite as commercial-grade alpha generation + executable trading platform with stage-gated real-money execution and LLM-governed signal validation.
- Absolute rules: replace the live-trading ban with stage-gated rules; keep the prohibition on full fine-tune of 12B+ models locally; add LoRA-on-≤7B-permitted rule; add "S1 only authoritative numeric forecaster" rule; add "every LLM signal must cite ≥1 chunk_id" rule; add "no edits to configs/promotion.yaml without two-person review" rule.
- Preferred implementation order: S1 → S2 → S3 → S4 with success gates between each; each subsystem gets its own spec under `docs/superpowers/specs/`.
- Repository structure: add `src/quant_research_stack/alpha/`, `governor/`, `feeds/`, `brokers/`, `execution/` subtrees.
- Coding style: keep existing rules; add "any module that places real orders is forbidden from being imported by tests or training code".
- Financial ML methodology: keep §5; add "S1 must beat the Optuna-tuned LightGBM baseline by ≥ 10 % weighted R² on the permanent holdout before being released to S4 in any stage".
- Apple Silicon policy: keep MPS + MLX bullets; add the 4.3 training power budget.
- New section §11 — Risk and execution: reproduces 5.1–5.6 in agent-readable form.
- New section §12 — Observability and audit: append-only JSONL audit log; every decision logged; replay-by-audit must reproduce.
- Done definition: keep, add "audit log written" and "no kill_switch active during the run".

### 6.2 AGENTS.md rewrite

**New roles (add):**

```text
Agent: Tabular Alpha Engineer (S1)
  Owns src/quant_research_stack/alpha/
  Must produce: weighted zero-mean R² improvement, OOF predictions, fold-stable feature importance
  Must NOT: place orders, modify governor schema, edit risk configs

Agent: LLM Governor Engineer (S2)
  Owns src/quant_research_stack/governor/
  Must produce: GBNF grammar, JSON schema, LoRA adapters, RAG index, veto-precision metric
  Must NOT: bypass JSON schema, accept LLM outputs without cited_paper_chunk_ids

Agent: Data Feeds + Broker Adapter Engineer (S3)
  Owns src/quant_research_stack/feeds/, brokers/
  Must produce: typed FeedAdapter/BrokerAdapter implementations, recorder + replayer parity
  Must NOT: skip recording, ship a real-money broker without a paper variant first

Agent: Execution + Risk Engineer (S4)
  Owns src/quant_research_stack/execution/
  Must produce: kill switch tested in CI, three-stage env-var gating, audit log integrity
  Must NOT: in-process stage promotion, weaken risk caps without two-person review
```

**Roles to retire:**

- Paper Trading Engineer — folded into Execution + Risk Engineer (S4).
- Apple Silicon LLM Engineer — folded into LLM Governor Engineer (S2).

**Roles to keep:** Data Engineer, Feature Engineer, Label Engineer, Validation Engineer, Tabular Model Engineer (renamed), Backtesting Engineer, NLP Engineer, Report Engineer.

**Global rule added at top:**

```text
0. Cross-agent invariant: No agent may write or modify code under brokers/*_live.py
   or configs/promotion.yaml unless explicitly assigned by the operator with a signed
   stage_change.md commit. Promotion to a higher stage is human-only.
```

### 6.3 README.md rewrite

**New section ordering:**

1. What QuantLab Alpha is.
2. What it is not (no investment advice; not regulated; user owns risk).
3. Current status table (auto-updated by `scripts/report_artifact_budget.py`).
4. Four-subsystem map (S1 / S2 / S3 / S4 mini-diagrams).
5. Three-stage promotion flow (paper → live_shadow → live).
6. Reproduction commands (corpus rebuild + S1 retrain + S2 RAG index).
7. Safety standard (kill switch, audit log, risk caps).
8. Legal disclaimer.
9. References (papers, datasets, models cited).

`scripts/report_artifact_budget.py` is extended to also emit current R², Sharpe-proxy, audit-log line count, stage env-var value, and days-since-last-retrain into the status table.

### 6.4 README header text

```text
What this is, what this is not

This repository is a single-operator alpha research and execution platform for
mid-frequency equities and tick-frequency crypto. It pairs a tabular predictor
(LightGBM/XGBoost/CatBoost/MLP stack) with an LLM governor that vetoes trades
inconsistent with the cited research corpus. It is designed to operate in three
stages: paper → live_shadow → live, with hard gates between each.

It is not a Jane Street trading desk, a hedge fund stack, or HFT infrastructure
suitable for sub-millisecond equity strategies. Those need paid market data and
colocation. This stack uses free Alpaca / Binance / Coinbase feeds and acknowledges
that limitation in every measurement.

It is not investment advice. The operator (the human running it) is responsible
for funds, taxes, brokerage relationships, and regulatory compliance.
```

---

## 7. Testing, Success Criteria, Spec Doc Transition

### 7.1 Testing strategy

**S1 (tabular alpha):**

- Unit: `cv.PurgedKFold` produces non-overlapping folds with embargo gaps; `metrics.weighted_zero_mean_r2` matches a hand-computed value on a 50-row fixture; every feature function is leakage-tested with synthetic future-injected data.
- Integration: a fast "smoke" pipeline runs end-to-end on a 100k-row JS subset in < 60 s; emits a `metrics.json` whose schema is asserted.
- Stochastic stability: seed-sweep (5 seeds) — std dev of holdout R² ≤ 0.002; otherwise flag as unstable.
- Adversarial: noise-feature regression check — any engineered feature ranked below seeded Gaussian noise across ≥ 3 folds is removed.

**S2 (LLM governor):**

- Unit: every Pydantic schema field validated with positive and negative cases; GBNF grammar verified to reject 50 corrupt outputs.
- Integration: a fixed prompt + fixed model + fixed retrieval set produces a deterministic JSON output (seed-locked decoding).
- Citation invariant: 100 generated outputs; any missing `cited_paper_chunk_ids` is auto-converted to `insufficient_evidence` — assertion never fails.
- LoRA evaluation: held-out Q&A accuracy vs base; ablation on real backtested signals — does LoRA improve veto precision by ≥ 5 pp?

**S3 (feeds + brokers):**

- Unit: each `FeedAdapter` parses recorded fixture WebSocket messages into the canonical `Tick` / `Bar` Pydantic models without loss.
- Integration: record-then-replay parity — a one-hour live recording replayed at 100× speed produces the same downstream S1 + S4 trace as live.
- Broker contract: `null_broker` and every `*_paper.py` pass an identical contract test (place / cancel / positions / account).

**S4 (execution + risk):**

- Unit: every risk check tested with positive (pass) and negative (block) cases; kill switch tested with each trigger (DD, feed gap, file flag, SIGTERM).
- Integration: simulated trading day with synthetic ticks reproduces expected PnL + risk ledger; audit log replays to identical state.
- Promotion gate: integration test attempting in-process self-promotion **must fail** — that is the regression-test invariant.

**Shared CI gates (must pass before any PR merges):**

```bash
PYTHONPATH=src uv run pytest -q
uv run ruff check src scripts tests
uv run mypy src                    # strict mode; add to deps
PYTHONPATH=src uv run python scripts/audit_replay_check.py last-day
```

### 7.2 Success criteria

| # | Criterion | Measurable how | Threshold |
|---|---|---|---|
| 1 | S1 beats existing Ridge baseline | Weighted zero-mean R² on permanent holdout | ≥ 0.012 (current 0.0075 → +60 %) |
| 2 | S1 robust across folds | Std dev of R² across 5 folds | ≤ 0.002 |
| 3 | S1 robust out-of-distribution | R² on Numerai signals data | > 0 (sign-correct) |
| 4 | S2 grammar enforced | LLM outputs failing schema | 0 in 10 000 generations |
| 5 | S2 citation invariant | LLM outputs with missing citations forwarded to S4 | 0 |
| 6 | S2 veto precision | Of S2-vetoed trades, share that would have lost money | ≥ 60 % |
| 7 | S3 parity | Record-then-replay PnL difference vs live | < 0.1 % |
| 8 | S4 kill switch | Time from trigger to all-orders-cancelled | < 2 s |
| 9 | S4 promotion gate | Self-promotion attempts succeeding | 0 |
| 10 | Paper-stage 90-day operation | Sharpe net of fees on the paper account | ≥ 1.0 |
| 11 | Audit replay | Replay of last 24h audit log produces identical state | byte-identical |
| 12 | Documentation | CLAUDE.md, AGENTS.md, README.md, all runbooks, all ADRs | committed and rendered |

Criteria 1–9 gate S1 + S2 + S3 + S4 implementation completion. Criterion 10 gates `paper → live_shadow`. Criteria 11–12 must hold continuously.

### 7.3 Spec doc transition

After approval of this spec:

1. The spec is self-reviewed inline (placeholder scan, internal consistency, scope check, ambiguity check).
2. The spec is committed.
3. The operator reviews the written spec.
4. On approval, `superpowers:writing-plans` is invoked to produce the detailed implementation plan for **S1 only**. S2, S3, S4 each get their own brainstorming + spec + plan cycle later.

CLAUDE.md / AGENTS.md / README.md rewrites land as **Task 1 of the S1 implementation plan**, properly TDD'd and committed alongside the first round of code.

---

## 8. Repository Documentation Layout

```text
docs/
  superpowers/
    specs/
      2026-05-14-quantlab-alpha-platform-design.md   ← this document
      2026-05-12-quant-ml-150gb-corpus-design.md     ← already committed
      <future S1, S2, S3, S4 detailed specs>
    plans/
      <future S1, S2, S3, S4 implementation plans>
      2026-05-12-quant-ml-150gb-corpus-implementation.md ← already committed
  runbooks/
    kill_switch.md
    stage_promotion.md
    incident_response.md
    disaster_recovery.md
  architecture/
    diagrams/
    adrs/
      0001-two-tier-tabular-llm.md
      0002-three-stage-promotion-gate.md
      0003-gbnf-constrained-llm-output.md
      0004-free-data-feed-policy.md
      0005-llm-governor-citation-requirement.md
CLAUDE.md                                            ← rewritten per §6.1
AGENTS.md                                            ← rewritten per §6.2
README.md                                            ← rewritten per §6.3 + §6.4
configs/
  stack.yaml
  risk.yaml                                          ← new
  promotion.yaml                                     ← new
```

Every brainstorm decision lands in the repo as a versioned document.

---

## 9. Risks This Spec Itself Carries

| Risk | Mitigation |
|---|---|
| S1 R² target (0.012) might be unachievable on this data | The plan caps S1 effort at 4 days. If holdout R² < 0.010 after that, we re-brainstorm: either accept a lower bar with a documented reason, or invest in additional feature engineering (likely from the order-flow corpus in 4.1). |
| Free equity data (15-min) is too sparse for a useful equity strategy | README explicitly limits the live equity strategy to mid-frequency; crypto runs at tick frequency where free data is sufficient. |
| LLM governor adds latency that misses the trading window | The fast-path uses Qwen 0.5B + LoRA (< 500 ms). For trades requiring deeper reasoning, the LLM verdict applies to the *next* trade, not the current one. |
| Operator promotes to live too early | The 90-day paper-trade minimum + signed runbook artifacts make a "yolo to live" path require deliberate human action that creates a paper trail. |
| Real-money losses from bugs | Risk caps cut to 50 % for the first 30 days of live; daily kill switch at 5 % realized DD; cumulative kill at 15 %. The platform is designed to lose at most one day of trading to a bug. |
| Regulator action | The README's "What this is not" section + audit log's `not_investment_advice: true` flag + operator-sole-responsibility framing put the legal posture in writing. |
| Spec drift across S1 → S4 | Each subsystem has its own spec brainstormed independently; if reality forces a change to the master architecture, this document is updated with a date-stamped revision section. |
