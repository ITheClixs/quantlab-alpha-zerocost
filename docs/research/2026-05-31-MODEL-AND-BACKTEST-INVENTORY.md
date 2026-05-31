# QuantLab Alpha model and backtest inventory

Date: 2026-05-31

This report inventories the repository as it stands in code, configs, saved
experiment artifacts, and saved research reports. It does not rerun model
training or backtests. Metrics below are therefore documented evidence, not a
fresh benchmark.

The most important finding is that the repository is a broad research and
execution platform, but the saved evidence still supports the README verdict:
there are 0 deployable, 0 paper, and 0 live strategies. Several models have
positive predictive metrics, and some research backtests look attractive before
promotion controls, but every documented candidate is blocked by holdout gates,
costs, data quality, fragility, missing promotion status, or research-only
constraints.

## 1. What the repository is

QuantLab Alpha is a staged quantitative research and execution stack:

| Stage | Main code | Purpose | Production status |
|---|---|---|---|
| S1 tabular alpha | `src/quant_research_stack/alpha/`, `src/quant_research_stack/alpha_eq/` | Train tabular predictors and stackers for Jane-Street-style and equity cross-sectional signals. | Research only; latest Jane Street holdout misses the registered R2 gate. |
| S2 LLM governor | `src/quant_research_stack/governor/` | Veto or pass proposed trades using constrained JSON and cited local research chunks. | Guardrail layer, not an alpha model. Passes without citations are downgraded to insufficient evidence. |
| S3 feeds and brokers | `src/quant_research_stack/feeds/`, `src/quant_research_stack/brokers/` | Typed adapters, recorder/replayer, null/paper/live broker boundaries. | Infrastructure; live use remains promotion-gated. |
| S4 execution and risk | `src/quant_research_stack/execution/`, `src/quant_research_stack/validation/` | Stage-gated order routing, risk caps, audit log, kill switch, paper validation reports. | Infrastructure; no live self-promotion. |
| Research/backtests | `src/quant_research_stack/signal_research/`, `src/quant_research_stack/backtest/`, `src/quant_research_stack/crypto_research/` | Strategy families, validation controls, PBO/DSR/bootstrap checks, reports. | Negative-results research ledger. |
| Local/HF model inventory | `manifests/models.yaml`, `models/`, `configs/governor.yaml` | Candidate local time-series, embedding, sentiment, and LLM models. | Mostly candidates or governor components, not all deployed in backtests. |

The repository enforces financial ML constraints: chronological validation, no
random time-series splits, no future information, no target leakage, no scaler
or imputer fitted on validation/test data, and no backtest without costs.

## 2. Metric vocabulary

The saved artifacts use different metrics for different tasks:

- Weighted zero-mean R2: Jane Street-style numeric target fit, where 0 means
  the zero predictor and positive values are useful.
- Directional accuracy: fraction of predicted signs matching realized signs.
  This can be misleading when labels are imbalanced; the local market zero
  baseline has high directional accuracy for this reason.
- IC / rank IC: correlation between prediction scores and future returns.
- Hit rate / net hit: fraction of profitable periods or trades after the
  report's own rules.
- Sharpe, Calmar, max drawdown, total/net return: strategy backtest metrics.
- PBO, DSR, bootstrap Sharpe CI: overfitting and robustness controls.

When a report labels a result as pseudo-PnL, research-only, diagnostic-only, or
not promotion eligible, that limitation is preserved below.

## 3. Model and strategy inventory

| System | Predicts or decides | How it works | Best documented metric | Status |
|---|---|---|---|---|
| Jane Street S1 stacked alpha | `responder_6` numeric target from Jane Street rows | Ridge, LightGBM, XGBoost, CatBoost, MLP, Conv1D sequence model, then a linear stacker with purged/embargoed CV and permanent holdout | Holdout stacked weighted zero-mean R2 0.005489; best single model CatBoost R2 0.006171; pseudo-backtest stacked hit rate 96.296% | Fails 0.012 holdout R2 gate; not deployable |
| Ridge S1 base model | Same Jane Street target | Linear ridge regression baseline | Holdout R2 0.003915; directional accuracy 52.152% | Kept as baseline anchor |
| LightGBM S1 base model | Same Jane Street target | Gradient boosted trees | Holdout R2 0.002467; directional accuracy 53.110% | Positive CV, weaker holdout |
| XGBoost S1 base model | Same Jane Street target | Gradient boosted trees | Holdout R2 -0.009311; directional accuracy 52.899% | Negative holdout R2 |
| CatBoost S1 base model | Same Jane Street target | Gradient boosted trees | Holdout R2 0.006171; directional accuracy 52.679% | Best single S1 holdout model, still below gate |
| MLP S1 base model | Same Jane Street target | PyTorch MLP with scaler, dropout, weighted loss | Holdout R2 -0.009421; directional accuracy 52.884% | Negative holdout R2 |
| Conv1D sequence S1 base model | Same Jane Street target | PyTorch 1D convolution over feature sequence | Holdout R2 -0.000659; directional accuracy 51.968% | Near-zero/negative holdout R2 |
| S1-EQ equity predictor | Cross-sectional volatility-normalized next-return label `y_xs` | Ridge/LightGBM/XGBoost in fast mode; CatBoost/MLP/sequence in full mode; linear stacker | No canonical saved promotion result found in the inspected docs | Implemented research path; evidence not promotion-grade |
| Local market signal head | Next OHLCV return `future_return_1` | Time-ordered split; Ridge and HistGradientBoosting regressors; zero baseline | Ridge validation zero-mean R2 0.002692; directional accuracy 52.365% | Weak research signal |
| Local orderbook signal head | Next midprice return `future_mid_return_1` | Time-ordered split; Ridge and HistGradientBoosting on LOB features | Ridge validation zero-mean R2 0.016072; directional accuracy 58.553% | Predictive but not sufficient alone |
| Equity daily OHLCV backtest | Cross-sectional daily return rankings | Persisted ridge market head, long/short top/bottom 10%, 5 bps one-way cost | SP500 net return -100%, Sharpe -1.559, hit 38.043%; NASDAQ net -62.154%, Sharpe -1.629; NYSE net -81.453%, Sharpe -5.526 | Fails after costs |
| Equity walk-forward heads | Daily next returns per universe | Ridge, HistGradientBoosting, ensemble_mean retrained by walk-forward | Best shown net result still negative; e.g. NYSE hist_gradient net -12.714%, Sharpe -2.197 | Fails after costs |
| Orderbook microstructure benchmark | `future_mid_return_5` on BTC/ETH/SOL L2 features | Ridge, HistGradientBoosting, ensemble_mean over best bid/ask, spread, depth and imbalance features | HistGradientBoosting R2 0.035083, IC 0.187428; ensemble directional accuracy 72.913% | Grossly predictive but costed sweep best net -0.017%; untradable |
| Trade-flow v1 aggTrades | 20-event BTCUSDT markout direction/return | Ridge, HistGradientBoosting, ensemble_mean on aggressor-signed trade-flow features | Ensemble mean directional accuracy 78.183%, R2 0.22149, IC 0.44912 | Net return negative at every tested threshold; research only |
| Meta-label / triple-barrier RandomForest | Probability a primary momentum trade survives barriers | Primary momentum sign plus technical features; triple-barrier labels; RandomForest classifier | NASDAQ net return 97.301%, daily Sharpe 1.721, net hit 59.565%; SP500 net return 27.028%, Sharpe 0.903, net hit 60.790% | Explicit research-validation only; no isolated permanent holdout; proxy execution |
| Sentiment surrogate model | LLM-generated sentiment class | Supervised surrogate over OHLCV/technical/VIX/turbulence features | Holdout nearest-int accuracy 52.778%, within-1-class 94.967%, quadratic weighted kappa 0.0127 | Predicts DeepSeek-derived sentiment labels, not human labels or returns |
| S2 Tier 1 governor | Trade verdict JSON | Qwen2.5-0.5B-Instruct with optional LoRA adapter | No accuracy metric found in inspected docs | Guardrail; pass without citations downgrades to insufficient evidence |
| S2 Tier 2 governor | Trade verdict JSON with retrieved context | Mistral-Small-Instruct-2409 GGUF plus BM25/dense retrieval/reranking | No accuracy metric found in inspected docs | Guardrail escalation tier |
| S2 Tier 3 governor | Async stricter trade verdict | Yi-1.5-34B-Chat GGUF for larger trade-size review | No accuracy metric found in inspected docs | Guardrail escalation tier |
| HMM single-index regime model | SPY/QQQ risk-on/risk-off exposure timing | Hidden Markov Model variants over single-index features, compared to baselines and delays | Best `hmm_4_full_dev_qqq`: dev Sharpe 2.356, holdout Sharpe 2.621, DSR 1.000, PBO 0.000 | Research pass / exception review required, not paper or production |
| VRP index and VRP x HMM | Volatility risk premium timing/overlay | VRP-only and VRP-sized-by-HMM variants | Best interaction holdout Sharpe 1.768 vs VRP-only 1.203, but HMM-only 1.762 | Useful but not incremental enough over HMM; no promotion |
| Event-conditioned macro/calendar | FOMC risk timing on SPY/QQQ | Scheduled-event risk-off/risk-on windows, vol-target baselines, placebo controls | Best promotable `voltarget_event__QQQ` Sharpe 1.0613; DSR 0.985; bootstrap lower 0.604 | Below 1.5 Sharpe gate and subsumed by vol/regime baselines |
| Zero-cost risk allocator v1 | Weekly long-flat allocation among SPY/QQQ/BTC/ETH | Equal-risk, macro/risk allocation, close t / execute t+1 | Holdout Sharpe 1.237, maxDD -10.93%, DSR 0.972, PBO 0.071 | Pass with caveat only; edge is crisis-dependent; paper_candidate false |
| Crypto-only risk allocator v2 | Weekly long-flat BTC/ETH spot allocation | No leverage, 20 bps one-way cost, weekly rebalance | Sharpe 0.629, maxDD -22.09%, ann return 5.75% | DO_NOT_ADVANCE due ETH concentration and bootstrap lower < 0 |
| Funding carry v1 | Delta-neutral long spot / short perp carry | BTC/ETH funding settlement carry, 8h marking, slippage, liquidation stress | Unlevered Sharpe 8.61, annual return 13.92%; 2020 +23.83%, 2021 +40.18%, 2026 -0.26% YTD | Real carry but DO_NOT_ADVANCE: decaying edge, 2026 failure, leverage/liquidation tail |
| EDGAR 10-K text features | 63-day forward equity return rank from filing text features | 57 classical text features, train <=2017, holdout 2020-2022 | Best text model holdout IC 0.0213, t=1.26, DSR 0.0 | Fails as low-frequency/insufficient sample |

## 4. S1 Jane Street alpha stack

The S1 stack is configured in `configs/alpha.yaml` and implemented under
`src/quant_research_stack/alpha/`. It targets Jane Street's `responder_6`, uses
`weight` as the sample weight, groups by `date_id`, reserves a 20% permanent
holdout, and uses purged/embargoed chronological CV. The configured base models
are Ridge, LightGBM, XGBoost, CatBoost, MLP, and a sequence Conv1D model.

The latest documented full run is `experiments/alpha_s1/20260523-160541`:

| Model | Holdout weighted zero-mean R2 | Directional accuracy | IC |
|---|---:|---:|---:|
| Ridge | 0.003915 | 52.152% | 0.065696 |
| LightGBM | 0.002467 | 53.110% | 0.073550 |
| XGBoost | -0.009311 | 52.899% | 0.060381 |
| CatBoost | 0.006171 | 52.679% | 0.078946 |
| MLP | -0.009421 | 52.884% | 0.045869 |
| Conv1D sequence | -0.000659 | 51.968% | 0.006505 |
| Stacked ensemble | 0.005489 | 53.311% | 0.078674 |

The registered target is weighted zero-mean R2 >= 0.012 on the permanent
holdout. The stacked model reached 0.005489, so it is below gate.

The saved Jane Street pseudo-backtest ranks predictions within each `date_id`
and goes long/short top/bottom 10%. Because the native Jane Street data has no
prices, spreads, or fills, the report measures weighted responder pseudo-PnL,
not dollar PnL. On that proxy the stacked model had total pseudo-PnL 2.5653,
hit rate 96.296%, and a Sharpe-like metric 30.388. This is useful ranking
evidence, but it is not tradable PnL.

## 5. Equity adaptation and local return heads

`src/quant_research_stack/alpha_eq/` adapts S1 to equities. Labels are built as
next-period raw return, volatility-normalized return, and cross-sectional
demeaned return (`y_xs`). Fast mode uses Ridge, LightGBM, and XGBoost; full mode
adds CatBoost, MLP, and sequence models.

The more complete saved evidence is in local/equity reports:

| Backtest/report | Model | Main evidence | Verdict |
|---|---|---|---|
| `reports/local_signal_training.json` market task | Ridge | R2 0.002692, directional accuracy 52.365% on `future_return_1` | Weak |
| `reports/local_signal_training.json` orderbook task | Ridge | R2 0.016072, directional accuracy 58.553% on `future_mid_return_1` | Predictive but not enough |
| `reports/equity_signal_backtest_20260523-202904.md` | Persisted ridge market head | SP500 net -100%, NASDAQ net -62.154%, NYSE net -81.453% after 5 bps costs | Fails |
| `reports/equity_walk_forward_retrain_20260523-215652.md` | Ridge / HistGradient / ensemble | All highlighted universe/model combinations have negative net returns after costs | Fails |

The equity daily OHLCV line is therefore not a deployable strategy. It is a
negative result showing that faint daily prediction does not survive realistic
turnover and costs.

## 6. Microstructure and trade-flow models

The orderbook benchmark and trade-flow runs are the clearest examples of the
repo finding predictive signals that still do not trade.

The orderbook benchmark uses BTCUSDT, ETHUSDT, and SOLUSDT L2 features. It
predicts `future_mid_return_5` with Ridge, HistGradientBoosting, and an
ensemble. HistGradientBoosting reached zero-mean R2 0.035083 and IC 0.187428;
the ensemble reached 72.913% directional accuracy. The best costed sweep,
however, was only two trades and net -0.017%. The report's classification is
that the signal is genuinely predictive but untradable at the modeled spread
and fee.

The trade-flow v1 aggTrades run uses BTCUSDT aggressor-signed spot trade flow
from 2024-04-01 to 2024-04-06 and a 20-event markout. It reports strong
out-of-sample signal metrics: ensemble directional accuracy 78.183%, R2
0.22149, and IC 0.44912. The cost sweep is decisive in the other direction:
with no threshold it loses essentially all capital net of taker costs, and
selective thresholds produce either negative net trades or no trades. It is
research-only and not promotion eligible.

## 7. NLP, sentiment, and filing text

There are two separate NLP lines:

1. The sentiment surrogate in `experiments/sentiment_surrogate/20260520-225356`
   predicts LLM-generated sentiment labels, not future returns and not human
   annotations. On 45,934 holdout rows it reached nearest-integer accuracy
   52.779%, within-one-class accuracy 94.967%, and quadratic weighted kappa
   0.0127. Its own limitations say the upstream LLM remains canonical and
   transfer beyond NASDAQ-100 is not validated.
2. EDGAR 10-K text features are a classical filing-text research line. The
   validation report uses 6,282 label-valid filings, 727 companies, train
   cohorts through 2017, and holdout 2020-2022. The best text model has holdout
   IC 0.0213 with t=1.26, DSR 0.0, and bootstrap spread CI lower 0.0. It is
   classified as low-frequency and insufficient-sample.

The model manifest also lists FinBERT, stock sentiment BERT, finance
embeddings, and local finance LLMs as candidates or support models. The saved
evidence inspected here does not show any of those as independently promoted
alpha models.

## 8. S2 LLM governor

The governor is not a return predictor. It is a safety and evidence layer that
accepts an S1 signal plus research context and emits a constrained JSON verdict:
`pass`, `veto`, or `insufficient_evidence`.

Configured tiers:

| Tier | Model | Role |
|---|---|---|
| Tier 1 | Qwen/Qwen2.5-0.5B-Instruct plus optional LoRA adapter | Fast local first-pass verdict |
| Tier 2 | Mistral-Small-Instruct-2409 GGUF | Retrieval-grounded escalation |
| Tier 3 | Yi-1.5-34B-Chat GGUF | Async review for larger trade sizes |

Retrieval uses BM25, dense embeddings from
`FinLang/finance-embeddings-investopedia`, a MiniLM reranker, and a local
research parquet corpus. The schema requires cited research chunks for `pass`
decisions; missing or invalid citations are converted to `insufficient_evidence`.

No saved governor accuracy, veto precision, or backtest win-rate metric was
found in the inspected artifacts. The correct interpretation is therefore:
implemented guardrail, not validated alpha source.

## 9. Strategy research results

The strategy research tree contains many model-like strategy families. These
are not all ML predictors, but they affect the repo's research conclusions.

| Family | What it does | Key documented result | Current decision |
|---|---|---|---|
| HMM single-index | Times SPY/QQQ exposure by hidden regime states | Best QQQ full-dev HMM: dev Sharpe 2.356, holdout 2.621, DSR 1.000, PBO 0.000 | Research pass; exception review required, not paper/production |
| VRP index / VRP x HMM | Uses volatility risk premium, including HMM interaction | Best interaction holdout Sharpe 1.768, but does not beat HMM-only by required 0.25 | Useful but not incremental; no promotion |
| Event macro/FOMC | Risk timing around scheduled events | Best promotable Sharpe 1.0613, below 1.5 gate; subsumed by vol-targeting/regime | Closed/no promotion |
| Zero-cost risk allocator v1 | Weekly long-flat allocation across SPY/QQQ/BTC/ETH | Holdout Sharpe 1.237, maxDD -10.93%, DSR 0.972 | Pass with caveat; paper_candidate false due crisis dependence |
| Crypto-only risk allocator v2 | Weekly BTC/ETH spot risk allocation | Sharpe 0.629, maxDD -22.09%, ann return 5.75% | DO_NOT_ADVANCE due ETH concentration and bootstrap lower < 0 |
| Funding carry v1 | Long spot / short perp delta-neutral funding carry | Unlevered Sharpe 8.61 and 13.92% annual return, but 2026 net -0.26% and leveraged tails are severe | DO_NOT_ADVANCE |
| Meta-label/triple-barrier | Filters primary momentum trades with RandomForest classifier | NASDAQ net 97.301%, Sharpe 1.721; SP500 net 27.028%, Sharpe 0.903 | Research-validation only |
| EDGAR 10-K | Filing text features predict later equity returns | Best holdout IC 0.0213, DSR 0.0 | Fails |
| Options IV / futures carry / news sentiment audits | Data feasibility and audit paths | Reports focus on coverage, mapping, timestamp, and survivorship constraints | Research/audit only |

## 10. Candidate foundation/local models

`manifests/models.yaml` is an inventory of candidate or support models. Not all
entries are used in saved backtests.

| Group | Models | Intended use |
|---|---|---|
| Time-series forecasting | Chronos Bolt Small, Chronos-2, Granite TTM, TimeMoE-200M | Forecasting baselines and future comparisons |
| Finance/candlestick time series | Kronos-base | Finance/K-line model candidate |
| Retrieval embeddings | FinLang finance embeddings, all-MiniLM-L6-v2 | Research and filing retrieval |
| Sentiment | hasnain43 stock sentiment BERT, ProsusAI FinBERT | News/sentiment feature extraction |
| Finance/instruct LLMs | Qwen finance, Llama-3 finance RAG GGUF, finance-LLM-13B GGUF, stock-themed 12B GGUFs | Explanation, retrieval, hypothesis generation |
| Governor/instruction LLMs | Qwen2.5-0.5B, Mistral Small GGUF, Yi-1.5-34B GGUF | S2 verdict cascade |
| Utility LLMs | TinyStories fallback, DistilBERT, Qwen coder, MLX Qwen 7B, Qwen 14B | Local experiments and tooling |

The manifest should be read as a model catalog, not as proof that every listed
model has an associated validated trading result.

## 11. Bottom line

The repo consists of a serious research platform, a staged guardrailed
execution design, and a large negative-results ledger. The strongest documented
pure prediction evidence is in the Jane Street S1 stack, orderbook
microstructure, trade-flow, and local orderbook heads. The strongest documented
strategy-level Sharpe appears in HMM single-index and unlevered funding carry.

However, the repository's own controls reject or quarantine these results:

- Jane Street S1 misses the registered holdout R2 gate.
- Daily OHLCV/equity signals fail after costs.
- Microstructure and trade-flow signals are predictive but do not survive
  spread/fee/taker-cost economics.
- HMM is research-pass/exception-review only, not paper or production.
- VRP is not incremental over HMM.
- Event macro is below the Sharpe gate and subsumed.
- Zero-cost allocators are fragile, crypto-concentrated, or crisis-dependent.
- Funding carry is real at 1x but capital-inefficient, decaying, and exposed to
  severe liquidation tails if levered.
- NLP/sentiment/filing models have no documented promoted return result.
- S2 is a citation-constrained veto layer, not a source of alpha.

The operational conclusion is unchanged: no model in the inspected code and
docs currently has documented evidence sufficient for paper trading or live
deployment under this repo's own promotion rules.
