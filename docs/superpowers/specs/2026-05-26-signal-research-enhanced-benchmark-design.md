# Signal-Research Enhanced Benchmark — Design Specification

- **Spec date:** 2026-05-26
- **Status:** Brainstormed via `superpowers:brainstorming`; awaiting user spec-review before invoking `superpowers:writing-plans`.
- **Working name:** `signal_research` — a new package at `src/quant_research_stack/signal_research/`.
- **Sibling specs (do not modify):**
  - `2026-05-21-quantlab-alpha-s0-trainer-persistence-design.md` — S1 (JS-trained) trainer persistence.
  - `2026-05-24-quantlab-alpha-s1-eq-equity-adaptation-design.md` — S1-EQ equity adaptation + strict backtest.
  - The existing `strategy_benchmark` package and its 1500-strategy PBO benchmark report (`reports/strategy_benchmark_sp_nasdaq_2yr.md`).
- **Deferred sibling specs (named for cross-reference, not part of this milestone):**
  - `S1-EQ-FUNDAMENTALS` / `S1-EQ-ALT-DATA` — PIT fundamentals, earnings, news sentiment as features (already deferred from S1-EQ).
  - `S5 Microstructure & HFT` — orderbook, tick, intraday execution simulator.
  - `S5 Options & Futures (full)` — separate asset-class pricing + cost models.

> **Not investment advice.** All artifacts produced under this spec carry the
> `not_investment_advice: true` footer. The project is production-intended,
> but research artifacts produced from it are not automatically investment
> advice. If outputs are used to advise external users or to manage capital,
> legal, regulatory, licensing, and compliance review is required before
> deployment. See §7 for the full disclaimer.
>
> **Research-attitude reframe.** The methodology upgrades exist to distinguish
> robust out-of-sample signal from overfit in-sample performance. They may
> reduce false positives and may reject most candidates. **If no candidate
> survives, that is a valid result.** The methodology measures whether success
> is real; it does not manufacture it.

---

## 0. Production-intent framing (non-negotiables)

This is not a toy research project. It is treated as a **production-intended
quant research platform with hedge-fund-grade discipline, constrained by
limited resources**. The goal is not to produce a good-looking backtest. The
goal is to build a system **where bad strategies cannot pass**.

Implementation planning, code review, and runtime behaviour all enforce the
following twelve non-negotiables. They cannot be weakened to produce a
profitable result. If no strategy survives them, that is a valid outcome.

1. **No live or paper promotion from in-sample results.** Promotion decisions
   require out-of-sample validation, holdout, paper trading, and shadow
   deployment in sequence.
2. **No final strategy selection using the permanent holdout.** Holdout is
   read once for the selected finalists, never used for ranking or selection.
3. **No current-constituent cross-sectional universe can be called
   production-grade or institutional-grade.** Such universes are
   research-only labelled.
4. **No strategy can pass without realistic costs, spread, slippage,
   funding/borrow where applicable, 2× cost stress, delay stress, PBO, DSR,
   bootstrap CI, and concentration checks.** All of these are mandatory
   gates, not opt-in.
5. **No sentiment, fundamentals, news, or alternative data can enter promoted
   models** unless timestamp integrity and leakage controls are proven via
   the §3 FinBERT-style audit ladder.
6. **Every dataset must have a manifest, source, hash, timestamp convention,
   data-quality label, and reproducibility record.** Datasets without a
   manifest cannot enter the pipeline.
7. **Every strategy trial must be logged. Failed trials are not hidden.** The
   selection funnel in §6.4 makes the full multiple-testing burden visible.
8. **PBO and DSR must account for the full search process, not only the
   final candidate.** `pbo_raw_global` and the full-pool DSR N are reported
   alongside per-profile and per-family slices.
9. **Any strategy with performance concentrated in one period, one trade,
   one asset, or one regime must be downgraded or rejected.** Concentration
   limits in §6.1 criterion #11 are mandatory.
10. **Any production-intended strategy must pass staged promotion:**
    research validation → permanent holdout → paper trading → monitored
    shadow deployment → limited live capital only after review. Each stage
    is a separate gate; skipping stages is forbidden.
11. **The system must include risk limits, max drawdown controls, exposure
    limits, kill switch, audit logs, reproducibility checks, and failure
    alerts.** These are infrastructure, not strategy features.
12. **The reporting language must distinguish four status tiers:**
    `research_pass`, `promotion_eligible`, `paper_trade_candidate`,
    `production_candidate`. See §6.1 for the precise definitions and
    gating between tiers.

Optimise for institutional-grade falsification, reproducibility, risk
control, and honest strategy discovery. Do not optimise for looking
profitable.

---

## 1. Scope, naming, architecture

### 1.1 What this spec produces

A new package `signal_research` that sits **on top of** the two existing
layers and orchestrates them:

```
        ┌──────────────────────────────────┐
        │       signal_research/           │  ← THIS spec
        │   (paper signals + methodology   │
        │    + cross-sectional bridge)     │
        └──────┬───────────────────┬───────┘
               ▼                   ▼
   ┌────────────────────┐  ┌────────────────────┐
   │ strategy_benchmark │  │      alpha_eq      │
   │  (single-asset     │  │ (M4 cross-sectional│
   │   PBO/DSR runner)  │  │   strict backtest) │
   └────────────────────┘  └────────────────────┘
```

The classical 1500-strategy benchmark in `strategy_benchmark/` and the M4
strict backtest in `alpha_eq/` are **untouched**. Their results stay
reproducible byte-for-byte.

### 1.2 Why a new top-level package

- The existing `strategy_benchmark/` is the *honest baseline*; adding deep
  learning + cross-sectional + new feeds would push it past its module budget
  and conflate "baseline" with "research playground".
- `alpha_eq/` enforces a sha256-locked feature contract that should not be
  contaminated by orchestration code from the paper-derived menu.
- `signal_research` has one clear responsibility — *composition + orchestration*.

### 1.3 Package layout

```
src/quant_research_stack/signal_research/
  __init__.py
  registry.py                  # strategy registry (mandatory schema per §3.6)
  data/                        # extended data feeds (§2)
    long_history.py            # yfinance 2005-2026
    hf_datasets.py             # HuggingFace dataset loaders (sentiment, fundamentals)
    fred.py                    # FRED via fredapi for macro factors
    cboe_proxies.py            # ^VIX, ^VVIX, ^SKEW, ^GVZ, ^VXN, ^OVX
    sp500_components.py        # current S&P 500 universe loader
    nasdaq_components.py       # current Nasdaq 100 universe loader
    crypto_minimal.py          # BTC/ETH minimal v1 (per §6.12)
    manifest.py                # data-quality classifier + sha256 manifest
  papers/                      # paper-derived signals (§3)
    deep_momentum.py           # Lim/Zohren/Roberts 2019 (M5)
    momentum_transformer.py    # Wood/Zohren/Roberts 2022 (M5)
    triple_barrier.py          # López de Prado 2018 — labels + meta-labeling wrapper
    avellaneda_lee.py          # Avellaneda & Lee 2010 — cross-sectional residual MR
    gkx_ohlcv_subset.py        # Gu/Kelly/Xiu 2020 — OHLCV-characteristic subset
    vol_risk_premium.py        # Bondarenko 2014 — feature + tradable variant separately
    hmm_regime.py              # Hamilton 1989 — 2-state regime classifier (feature_generator)
    sentiment_finbert.py       # Araci 2019 FinBERT (research_only_default per §3 FinBERT ladder)
    options_implied.py         # VIX term-structure features + VXN/fallback for Nasdaq
    macro_overlay.py           # FRED-driven macro features (feature_generator)
  methodology/                 # §4 upgrades
    cpcv.py                    # Combinatorial Purged CV (López de Prado 2018 ch. 12)
    meta_labeling.py           # secondary classifier wrapper (survivor-only per §4.2)
    correlation_dedup.py       # cluster + dedup on net OOS returns
    multi_objective.py         # Pareto front (selection + reporting only)
    regime_conditional.py      # by-regime Sharpe/DD with agnostic vs specific declaration
    bootstrap_ci.py            # stationary block bootstrap CIs
    reality_check.py           # White SPA / Reality Check (M6b — Phase 2)
    failure_classifier.py      # taxonomy for failed candidates (§6.3)
  cross_sectional/             # §5 bridge to alpha_eq M4
    signal_to_panel.py         # pure conversion + validation
    panel_to_m4.py             # M4 entry-point wrapper preserving all banners
  runner.py                    # orchestrates full enhanced benchmark
  report.py                    # three-tier reports (family / profile / master)
configs/
  signal_research.yaml         # parameter grids, profile defaults
  signal_research_profiles/
    sp500.yaml
    nasdaq.yaml
    crypto.yaml
    futures_proxy.yaml
scripts/
  fetch_signal_research_data.py
  run_signal_research_benchmark.py
  signal_research_report.py
tests/signal_research/
  ...                          # see §6 milestones for full test list
experiments/signal_research/<run_id>/
  ...                          # see §4 outputs + §6 deliverables
reports/signal_research/
  family/<family_name>.md
  profile/<profile_name>.md
  enhanced_benchmark.md
```

### 1.4 Aggregate sizing (non-binding estimates, see §6.9)

- ~20 source modules + ~15 test modules + 4 CLI scripts + 3 report templates.
- ~3000-3500 LOC.
- Implementation work split by milestone (§6.5); total non-binding.
- Compute split by milestone (§6.5 and §6.9); total non-binding.

---

## 2. Data layer

### 2.1 20-year daily backbone

- `yfinance` for SPY (1993→), QQQ (1999→), ES=F / NQ=F (~2000→), individual
  S&P 500 + Nasdaq 100 component lifetime histories.
- Target window: **2005-01-01 to current** (~5400 trading days, ~21 years).
- Stored at `data/processed/signal_research/long_history/<ticker>.parquet`
  with sha256-locked manifest matching the `alpha_eq` convention.

### 2.2 Data-quality classification (5 tiers)

| Tier | Meaning | Promotion semantics |
|---|---|---|
| `pit_safe` | Full PIT membership + delisting handling | `promotion_eligible` allowed |
| `partial_pit_universe` | Some historical reconstruction (e.g. Wikipedia fallback) | `research_pass` only |
| `public_snapshot_not_pit` | yfinance / public-snapshot data without PIT vendor guarantee | `research_pass` only |
| `survivorship_prototype_only` | Current-only constituents, no exit reconstruction | `research_pass` with explicit prototype banner; gate suspended |

**Note on directly-traded instruments.** SPY, QQQ, BTCUSDT, ETHUSDT and
similar are directly traded — their backtest does **not** depend on a
historical constituent reconstruction. For these, **constituent
survivorship is N/A** and `promotion_eligible` is allowed even though the
underlying data is still `public_snapshot_not_pit` (yfinance / Binance public
snapshot). Vendor / snapshot limitations must still be disclosed in every
report. This is captured operationally by:
- The manifest metadata field `constituent_survivorship_applicable: false`
  on directly-traded artifacts (SPY, QQQ, BTCUSDT, ETHUSDT, etc.).
- The `promotion_eligible` criterion in §6.1 which checks this metadata.

`directly_traded_etf` is **not** a `data_quality_label` value. The data
quality label remains `public_snapshot_not_pit` (or higher if upgraded);
the promotion exception is carried by the separate
`constituent_survivorship_applicable` flag.

**Important:** current-only S&P 500 and current-only Nasdaq 100 constituents
default to **`survivorship_prototype_only` or `public_snapshot_not_pit`**, NOT
`partial_pit_universe`. The latter requires *some* historical reconstruction.

### 2.3 Nasdaq as a first-class track

Per the user-mandated Section 2 amendments, Nasdaq is not a slice of the
S&P 500 path — it has its own model profile, universes, features,
benchmarks, and reports.

#### 2.3.1 Nasdaq universes

- **`nasdaq_index_proxy`** — `QQQ` + `^IXIC` (if usable) + `NQ=F` (if data
  quality sufficient). `TQQQ`/`SQQQ` allowed for diagnostics only, never for
  training (leveraged-ETF path dependency).
- **`nasdaq_100_current`** — current NDX-100 constituents. Labeled
  `survivorship_prototype_only`. **Survivorship warning banner mandatory** in
  every report.
- **`nasdaq_mega_cap_focus`** — AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG,
  TSLA, AVGO, COST, NFLX, AMD, QCOM, ADBE, INTU, CSCO, PEP, AMAT, ARM (if
  available), ASML (ADR), plus selected liquid Nasdaq-heavy names. Reporting
  cohort, not primary statistical universe.
- **`user_focus_tech`** — AAPL, ORCL, PYPL, INTC, META, TSLA, QCOM, PLTR,
  GOOGL, AVGO, ADBE, RKLB. Focused evaluation cohort.

#### 2.3.2 Nasdaq model profile

`signal_research_nasdaq` is a distinct profile from `signal_research_sp500`.
It is allowed to have different feature importances, regime behavior,
volatility scaling, and benchmark comparisons. The Nasdaq model must not be
declared a pass-by-reuse of the S&P 500 model.

#### 2.3.3 Two distinct Nasdaq tasks (kept separate in code + reports)

**A. Single-asset Nasdaq prediction (QQQ / NQ):**
- Time-series forecasting; tradable QQQ strategy PnL; Sharpe / DSR / PBO.
- Labels: QQQ forward return, QQQ direction, QQQ volatility-adjusted return,
  NQ=F forward return (if futures proxy is included).

**B. Cross-sectional Nasdaq stock selection (NDX-100):**
- Rank IC, decile spread, long-short PnL, cross-sectional PBO (on L/S PnL).
- Operates through the §5 cross-sectional bridge to the `alpha_eq` M4 engine.

These two tasks **never** share a metric table.

#### 2.3.4 Nasdaq-specific benchmarks

Every Nasdaq report compares against:
- QQQ buy-and-hold
- SPY buy-and-hold
- Equal-weight Nasdaq-100 current-constituent prototype (explicitly
  survivorship-warned)
- QQQ momentum baseline
- QQQ mean-reversion baseline
- QQQ vol-targeted baseline

#### 2.3.5 Nasdaq-specific features

- QQQ multi-horizon returns (1d, 5d, 20d, 60d, 120d)
- QQQ realized volatility
- QQQ/SPY relative strength, QQQ/SPY rolling beta, QQQ-minus-SPY residual return
- NQ=F returns if available
- VIX, VVIX, **VXN** (Nasdaq-100 implied vol) — and a documented fallback if
  VXN is not available
- FRED rates: **DGS10** (10-year Treasury), **T10Y2Y** (yield-curve slope)
- Dollar index (DTWEXBGS via FRED, or yfinance DXY proxy)
- Mega-cap concentration proxy (top-10 share of QQQ weight or proxy)
- Tech-sector momentum proxy (XLK if available)
- Semiconductor proxy (SMH if available)
- Software/cloud proxy (IGV if available)
- Crypto beta proxy — **diagnostic only**, not default

### 2.4 Data sources roster

| Source | Loader | Coverage | Purpose | Data-quality tier |
|---|---|---|---|---|
| yfinance (extended) | `long_history.py` | 1993-2026 daily | 20-year backbone for SP500 + Nasdaq + ETFs | `public_snapshot_not_pit` (every yfinance series; per-artifact metadata also carries `constituent_survivorship_applicable: false` for SPY/QQQ and similar directly-traded ETFs) |
| HuggingFace datasets | `hf_datasets.py` | varies | Sentiment + fundamentals (research-only by default) | varies — must be classified per dataset |
| FRED (via `fredapi`) | `fred.py` | 1950+ | DGS10, T10Y2Y, DTWEXBGS, GOLD, etc. | `public_snapshot_not_pit` (revision-adjusted via ALFRED is a Phase-2 upgrade) |
| CBOE indices via yfinance | `cboe_proxies.py` | 1990-2026 | `^VIX`, `^VVIX`, `^SKEW`, `^GVZ`, `^OVX`, `^VXN` | `public_snapshot_not_pit` |
| HF FinBERT (`ProsusAI/finbert`) | `sentiment_finbert.py` | pre-trained | News sentiment classifier — research_only_default (§3 ladder) | model artefact only |
| HF sentiment dataset(s) | `hf_datasets.py` | candidates: `Lettria/financial-news-sentiment`, `zeroshot/twitter-financial-news-sentiment`, `oliverwang15/FinNews_Sentiment` | News text + ticker mapping (gated per FinBERT ladder) | varies |
| Crypto (Binance public OHLCV) | `crypto_minimal.py` | BTCUSDT / ETHUSDT daily | Crypto profile v1 (§6.12) | `public_snapshot_not_pit` (vendor only) |

### 2.5 Data-quality contract

Every dataset writes a `_manifest.json` containing: sha256, schema fingerprint,
source URL/identifier, fetch timestamp, row count, symbol count, date range,
data-quality tier, vendor disclosure string, build command line, package
versions, warnings. Reports always show the data-quality banner. The contract
mirrors the manifest contract defined in the S1-EQ spec §2.7.

### 2.6 Sentiment and fundamentals gating (research-only by default)

Per the user's Section 2 §10:
- Sentiment and fundamentals enter the **promoted benchmark** only via the
  FinBERT promotion ladder defined in §3 below.
- News/sentiment carry severe timestamp risk (publication timestamp, duplicate
  articles, ticker mapping, market-hours alignment). v1 default is
  `research_only_default`; the ladder governs whether they can be elevated.

### 2.7 Output artifacts

```
data/processed/signal_research/
  long_history/
    manifest.json
    SPY.parquet
    QQQ.parquet
    ES_F.parquet
    NQ_F.parquet
    <ticker>.parquet           # S&P 500 components, NDX-100 components, etc.
  nasdaq/
    qqq.parquet
    nq_futures_proxy.parquet
    nasdaq_100_current.parquet
    nasdaq_mega_cap_focus.yaml
    user_focus_tech.yaml
    manifest.json
  sp500/
    sp500_current.parquet
    manifest.json
  macro/
    fred_features.parquet
    manifest.json
  cboe/
    cboe_proxies.parquet       # VIX, VVIX, SKEW, GVZ, OVX, VXN
    manifest.json
  hf/
    hf_dataset_manifest.json
  crypto/                       # §6.12 minimal v1
    btcusdt_daily.parquet
    ethusdt_daily.parquet
    manifest.json
```

### 2.8 Cross-sectional reports always show the survivorship banner

If `data_quality_label` is `survivorship_prototype_only` or
`public_snapshot_not_pit`, every cross-sectional report carries:

> ⚠️ **Survivorship warning.** This universe uses current constituents and is
> not a true point-in-time historical universe. Results may be
> survivorship-biased and are research-only. Cannot be described as
> institutional-grade or production-grade.

No strategy passes a real promotion gate on current-only constituents alone.

---

## 3. Paper-derived signal menu

### 3.1 Module-type taxonomy (per amendment §3.1)

Every entry in the menu is one of:
- `standalone_strategy` — directly produces positions.
- `feature_generator` — produces features, not positions.
- `wrapper` — modifies another primary signal (e.g. meta-labeling).
- `model_family` — learns from a feature set and emits predictions.

This distinction matters because PBO and strategy-count accounting must not
pretend every module is an independent strategy.

### 3.2 The 10-family menu

| # | Family | Module type | Primary use | Profiles |
|---|---|---|---|---|
| 1 | **Deep Momentum Network** (Lim/Zohren/Roberts 2019) | `model_family` | Single-asset LSTM time-series forecast | sp500, nasdaq, futures_proxy |
| 2 | **Momentum Transformer** (Wood/Zohren/Roberts 2022) | `model_family` | Single-asset transformer attention forecast | sp500, nasdaq, futures_proxy |
| 3 | **Triple-Barrier + Meta-Labeling** (López de Prado 2018) | `wrapper` | Survivor-only secondary filter | wraps any primary signal |
| 4 | **Avellaneda-Lee Stat-Arb** (2010) | `standalone_strategy` | Cross-sectional residual MR with rolling PCA | sp500 CS, nasdaq CS |
| 5 | **GKX-style OHLCV-characteristic subset** (Gu/Kelly/Xiu 2020) | `model_family` | Cross-sectional ML asset pricing (OHLCV subset only) | sp500 CS, nasdaq CS |
| 6 | **Vol Risk Premium** (Bondarenko 2014) | feature + (optional) `standalone_strategy` *only if a tradable instrument is available* | Single-asset VRP feature + filter | sp500, nasdaq |
| 7 | **HMM Regime Switching** (Hamilton 1989) | `feature_generator` | 2-state regime classifier; filter / interaction term | all profiles |
| 8 | **FinBERT News Sentiment** (Araci 2019) | `feature_generator` (with promotion ladder) | News-derived sentiment features | research_only_default; promotion-eligible after audit |
| 9 | **Options-Implied Term Structure** | `feature_generator` (with optional standalone hypothesis) | VIX/VXN ratios, SKEW, VVIX as features; any standalone rule is a *testable hypothesis* | sp500, nasdaq (with VXN-or-fallback) |
| 10 | **Macro Overlay** (FRED) | `feature_generator` / `position_filter` | Yield-curve, USD, commodities as features/filters | all profiles |

### 3.3 Crucial integration details (consolidated amendments)

**#1, #2 — deep learning (deferred to M5):**
- Trained walk-forward with expanding windows + proper purging.
- PyTorch + MPS on Apple Silicon. Predeclared parameter grids (small).
- Per amendment §3.7: deep models do not block M1-M4 of the enhanced benchmark.

**#3 — Triple-Barrier + Meta-Labeling (wrapper):**
- Strict event definitions: event start = signal flip or rebalance; vertical
  barrier from `{5, 10, 20, 40}` predeclared days; profit/stop barriers `±k·σ_20`
  with `k ∈ {1.0, 1.5, 2.0}` predeclared.
- Vol estimator: realized 20-day daily vol using only data up to event start.
- Side comes from primary; meta-labeler predicts trade-vs-flat (size only).
- Secondary model: RandomForestClassifier, features available only at event start.
- Survivor-only application (per §4.2): pre-filter requires positive validation
  net Sharpe, positive validation hit rate or expectancy after costs, ≥200
  events (single-asset) or ≥500 events (cross-sectional/panel), no inverted-
  signal superiority, not a near-duplicate of a stronger primary.
- Barrier grid logged as predeclared degrees-of-freedom; no post-hoc tuning.

**#4 — Avellaneda-Lee (cross-sectional standalone):**
- Return horizon: predeclared (default 1d forward).
- PCA estimation window: predeclared rolling window (e.g. 252 trading days).
- Number of principal components: predeclared (e.g. top 5) or
  explained-variance threshold (e.g. 60 %).
- Residuals standardized cross-sectionally on each rebalance date.
- Z-score entry threshold and exit rule predeclared in a small grid.
- Rebalance cadence predeclared (e.g. daily).
- **PCA fit using only past data on a rolling basis. Never full-panel fit.**
- Optional sector / market-beta residualization documented if used.
- Cross-sectional bridge: §5.

**#5 — GKX-style OHLCV-characteristic subset (cross-sectional model_family):**
- Name explicitly "GKX-style OHLCV-characteristic subset" — *not* a claim of
  replicating Gu/Kelly/Xiu 2020.
- Feature list (timestamp-disciplined, all available at decision time):
  - Momentum: 1m, 3m, 6m, 12m skip 1m (Jegadeesh-Titman style)
  - Reversal: 1d, 5d, 1m
  - Realized volatility (20d, 60d)
  - Beta to SPY or QQQ (rolling 60d / 252d)
  - Idiosyncratic volatility (residual to market regression)
  - Dollar volume (20d median, lag-1)
  - Amihud illiquidity (20d)
  - Maximum daily return (20d window)
  - Drawdown (60d / 252d)
  - Volume shock (z-score of dollar volume)
  - Close-location (20d high-low position)
- Model: tree ensemble (LightGBM cross-sectional ranking objective).
- Cross-sectional bridge: §5.

**#6 — Vol Risk Premium (feature + tradable-only-if-real-instrument):**
- Per amendment §3.5: distinguish *implied-vol feature* from *tradable volatility
  strategy*.
- v1 feature variant: VIX/VVIX/SKEW/VIX9D as features or regime filters for
  SPY/QQQ (and VXN for Nasdaq).
- v1 tradable variant: only when a tradable VIX-related ETF/ETN with reliable
  data is configured. Otherwise pure VIX-index strategies are labeled
  **diagnostic-only**.

**#7 — HMM Regime (feature_generator):**
- 2-state Gaussian HMM on broad-market returns (SPY for sp500 profile, QQQ for
  nasdaq profile).
- Outputs: per-day `regime_id ∈ {0, 1}` + `regime_prob`.
- Used as feature/filter, not standalone strategy. Per amendment §3.10: filter
  combinations are not counted as independent strategies unless separately
  registered + counted in PBO.

**#8 — FinBERT (promotion ladder):**

Per the user's FinBERT clarification, FinBERT is **not permanently excluded
from promotion**. It is gated through a 4-state ladder:

| State | Meaning |
|---|---|
| `research_only_default` | Timestamp integrity not yet proven. Default for all sentiment work in v1. |
| `shadow_signal` | Timestamp audit passes, but not yet used for model selection. |
| `eligible_for_benchmark` | Passes data-quality audit + standalone validation. |
| `promoted_feature` | Improves out-of-sample net performance and passes PBO/DSR. |

**Promotion gate (10 criteria):**
1. Every news item has a reliable publication timestamp (not just a date).
2. Timestamps normalized to exchange time.
3. Feature obeys the rule `news_timestamp < signal_timestamp`.
4. Duplicate, syndicated, reposted articles deduplicated.
5. Ticker mapping explicit and auditable.
6. Articles with ambiguous ticker mapping excluded or labeled low-confidence.
7. Sentiment aggregation uses only news available before the decision time.
8. The strategy is tested as a standalone sentiment signal first.
9. Survives chronological validation, PBO, DSR, cost-adjusted holdout.
10. Improves net performance versus the no-sentiment baseline.

If the HF dataset has only dates and no reliable intraday timestamps, FinBERT
stays `research_only_default`. The wording in this spec is:

> "FinBERT is research-only by default in v1, but promotion-eligible after
> passing the sentiment timestamp and leakage audit."

**#9 — Options-Implied Term Structure (feature_generator + optional standalone
hypothesis):**
- v1 features: `^VIX9D / ^VIX` ratio (term structure), `^VVIX / ^VIX` ratio,
  `^SKEW`, plus `^VXN` for Nasdaq with documented fallback if VXN is not
  available.
- Standalone rules (e.g. "long QQQ when VIX9D/VIX inverted") are treated as
  **testable hypotheses**, not known bull signals. Report shows whether they
  survive costs + OOS tests.
- For Nasdaq, do not substitute `^VIX` for Nasdaq volatility without labeling
  it as an imperfect market-wide proxy.

**#10 — Macro Overlay (feature_generator / position_filter, not rule library):**
- v1 use: regime features, position gates, model inputs, sensitivity
  diagnostics.
- Macro variables are low frequency with many degrees of freedom — easy to
  overfit. Do **not** become a tuned rule library.

### 3.4 Parameter grids — predeclared, small, robust

Per amendment §3.12: papers' parameters may not transfer. Use small predeclared
grids per signal, e.g.:
- lookback windows: short / medium / long
- vol target levels: low / medium / high
- barrier widths: predeclared {5, 10, 20, 40}
- holding horizons: short / medium / long
- regime thresholds: low / medium / high

Do not tune large grids. Do not use a single arbitrary configuration. Grids are
logged as DoF in `metadata.json` (per §4.7 in S1-EQ spec — DoF disclosure).

### 3.5 Strategy-count accounting

Per amendment §3.2:
- Raw candidate count: classical 1500 (preserved) + ~120 paper-derived
  variants = **~1620 raw candidates**.
- Effective independent strategy count after correlation clustering (§4.3).
- Unique signal-family count.
- Parameter-variant count.
- Model-variant count.

PBO and DSR use the **raw trial count** conservatively. Effective-count
reporting is a secondary diagnostic.

### 3.6 Strategy registry schema

Every signal in the menu registers with the fields listed below. **All
fields listed are mandatory** unless explicitly marked optional. The schema
may carry additional optional fields beyond these.

| Field | Type | Notes |
|---|---|---|
| `strategy_id` | str | Canonical, unique |
| `family` | str | e.g. `AVELLANEDA_LEE` |
| `module_type` | enum | `standalone_strategy` / `feature_generator` / `wrapper` / `model_family` |
| `paper_source` | str | e.g. "Avellaneda & Lee 2010" |
| `asset_class` | str | equity / index / futures / crypto |
| `profile` | str | sp500 / nasdaq / crypto / futures_proxy |
| `single_asset_or_cross_sectional` | enum | `single_asset` / `cross_sectional` |
| `required_data` | list[str] | manifest keys required |
| `timestamp_assumptions` | str | e.g. "after_close_t" |
| `parameter_grid` | dict | predeclared parameters per §3.4 |
| `default_parameters` | dict | within the grid |
| `eligible_for_pbo` | bool | excludes wrappers and feature generators |
| `eligible_for_holdout` | bool | wrappers/features may be excluded |
| `eligible_for_cross_sectional_bridge` | bool | §5 bridge gating |
| `data_quality_requirements` | str | minimum tier acceptable |
| `known_limitations` | list[str] | textual warnings |

(Schema may carry additional optional fields; all fields listed above are mandatory.)

### 3.7 Explicit rejection criteria per family

Per amendment §3.14, every family declares its own rejection criteria. Defaults:
- Validation IC not positive after costs.
- Net Sharpe ≤ 0 after costs.
- High PBO within the family.
- Performance concentrated in one period.
- Does not beat a simpler baseline of the same family.
- Collapses under 2× costs.
- Fails randomization / inverted-signal sanity check.

Family-specific criteria override defaults where specified.

---

## 4. Methodology upgrades

> **Reframe (per amendment §4 opening):** the methodology upgrades exist to
> distinguish robust out-of-sample signal from overfit in-sample performance.
> They may reduce false positives and may reject most candidates. If no
> candidate survives, that is a valid result.

### 4.1 Combinatorial Purged Cross-Validation (CPCV)

`methodology/cpcv.py` (López de Prado 2018 ch. 12).

Constraints (per amendment §4.1):
- CPCV splits are **chronological blocks**, not random row splits.
- Purging removes rows whose label horizon overlaps the test block.
- Embargo removes rows immediately after the test block.
- Time ordering preserved inside each train/test slice.
- **The permanent holdout remains untouched.** CPCV is inside the
  development/validation evaluation process, not a replacement for holdout.

Output: a *purged* OOS returns matrix shape `(T, S)` across all CPCV splits,
which downstream PBO consumes.

### 4.2 Meta-Labeling — survivor-only

Per amendment §4.2: meta-labeling is **not applied to all 1620 strategies
blindly**. Pre-filter eligibility:
- Positive validation net Sharpe.
- Positive validation hit rate after costs OR positive validation expectancy.
- Sufficient event count: **≥ 200 (single-asset) / ≥ 500 (cross-sectional/panel)**.
- No sign-bug / inverted-signal superiority.
- Not a near-duplicate of a stronger primary.

Otherwise, the meta-labeled variant is marked *statistically weak* and excluded
from the promoted candidate set.

10-step recipe per the user (Section 4 §2 — recipe carried verbatim into
implementation).

### 4.3 Correlation-based deduplication

`methodology/correlation_dedup.py`.

Per amendment §4.4:
- Use **net OOS returns**, not gross.
- Compute correlation only over aligned dates where both strategies are active
  (or use a consistent fill policy for inactive days — explicitly documented).
- Report both **signed correlation** and **absolute correlation**.
- If clustering on `|ρ|`, inverse strategies (one is sign-flip of the other) are
  also considered duplicates — explicitly documented.

Representative selection within a cluster:
- Primary rule: `Sharpe / sqrt(turnover)`.
- Also report: highest-DSR representative + lowest-drawdown representative as
  diagnostic alternatives. **No single representative rule hides a better
  robust candidate.**

Reports: PBO on raw pool (primary) + PBO on deduplicated pool (secondary
diagnostic).

### 4.4 Multi-objective Pareto ranking

`methodology/multi_objective.py`.

Per amendment §4.5: Pareto ranking is a **survivor-selection tool and
reporting tool**, not a promotion criterion by itself.

Primary axes (v1):
- maximize Sharpe
- minimize |max drawdown|
- minimize annual turnover
- minimize capacity shrinkage at $10 M (ADV-cap proxy)

Optional secondary axes if cheap:
- maximize DSR
- minimize performance concentration

Output: Pareto-front strategies + per-strategy "dominated-by" count.

### 4.5 Regime-conditional metrics

`methodology/regime_conditional.py`.

Per amendment §4.6: replace the rigid `min(Sharpe_low, Sharpe_high) > 0.3`
with a regime-agnostic / regime-specific declaration.

**Regime-agnostic strategies:** must have both regimes not catastrophically
negative AND at least one regime materially positive.

**Regime-specific strategies:** must declare themselves regime-specific
*before* the result is known, must be **gated to their favorable regime using
only information available at the time**, and must show OOS performance after
applying that gate.

Predeclared gating is mandatory; retroactive declaration after observing
single-regime success is forbidden.

Per-strategy report fields:
- Sharpe by regime
- PnL contribution by regime
- Max drawdown by regime
- Number of active days per regime
- Whether the strategy is regime-agnostic or regime-specific

### 4.6 Block-bootstrap confidence intervals

`methodology/bootstrap_ci.py`.

Stationary block bootstrap (Politis & Romano 1994) with mean block length
`L = T^(1/3)`. 10 000 resamples. Output CIs on Sharpe, Sortino, max-DD, DSR.

Per amendment §4.7, promotion uses point estimates as the primary gate; CIs
are reported with tiered confidence interpretation (see §6.1 success gate).

### 4.7 PBO + DSR — three-tier reporting

Per amendment §6.3:
- `pbo_raw_global` — full candidate pool (~1620). Conservative context.
- `pbo_profile` — within profile (sp500 / nasdaq / crypto / futures_proxy).
  Used in the success gate.
- `pbo_family` — within signal family.

Similarly, DSR reported with full N (raw global) and per-profile contextual
N counts.

### 4.8 Reality Check / SPA — deferred to M6b

Per amendment §4.9: `methodology/reality_check.py` (White 2000 / Hansen 2005
SPA) is added to the roadmap but is **not** an M1-M5 blocker. Deferred to M6b.

### 4.9 Dev-only invariant

Per amendment §4.10: methodology modules **may rank, reject, cluster, and
select candidates only on development and validation data.** The final
holdout is used **once** for the final selected candidates.

The S1-EQ holdout gate (`HoldoutGate` in `alpha_eq/data/holdout.py`) is the
canonical implementation; this spec reuses that pattern.

### 4.10 Failure classification taxonomy

`methodology/failure_classifier.py`. Canonical categories (per amendments
§4.11 and §6.9):

| Category | Meaning |
|---|---|
| `high_pbo` | Overfit by multiple-testing |
| `low_dsr` | Sharpe not significant after deflation |
| `cost_failure` | Positive gross but negative net, or fails 2× cost stress |
| `regime_concentration` | Dominated by one regime, not predeclared regime-specific |
| `insufficient_sample` | Trade count or data window too small |
| `too_few_trades` | < 200 (single-asset) / < 500 (cross-sectional) |
| `delay_stress_fail` | Collapses under signal-delay or fill-shift sensitivity |
| `single_period_dominance` | > 50 % PnL in one quarter / > 25 % in one day / > 20 % in one trade / > 35 % from one asset for multi-asset strategies |
| `over_correlated_with_baseline` | Fails dedup (§4.3) |
| `randomization_fail` | Fails random / inverted-signal sanity check |
| `data_quality_fail` | Non-PIT universe blocks promotion-tier outcomes |
| `holdout_failure` | Validation looked good, permanent holdout failed |
| `capacity_failure` | Passes at tiny notional, fails under realistic capital / ADV |

Each failed candidate maps to ≥ 1 category. **The categorization itself is the
research output.**

### 4.11 Outputs per run

```
experiments/signal_research/<run_id>/
  methodology/
    cpcv_oos_returns_matrix.parquet      # purged OOS returns from CPCV
    correlation_clusters.parquet         # cluster_id per strategy + dedup details
    pareto_front.parquet                 # Pareto-front survivors
    regime_assignments.parquet           # daily regime + per-strategy regime metrics
    bootstrap_ci.parquet                 # 95 % CIs per strategy per metric
  registry.parquet                        # complete strategy registry (§3.6)
  pbo_raw_global.json                     # PBO on full ~1620 pool
  pbo_profile.json                        # PBO per profile
  pbo_family.json                         # PBO per signal family
  pbo_dedup.json                          # PBO on deduplicated pool (secondary)
  dsr_summary.parquet                     # per-strategy DSR + bootstrap CI
  rejected_strategies.parquet             # rejected + failure-class categories
  selection_funnel.parquet                # candidates remaining after each filter (§6.10)
```

---

## 5. Cross-sectional integration with the alpha_eq M4 engine

### 5.1 Single-asset vs cross-sectional matrix

Per amendment §5 and §2.3.3, single-asset and cross-sectional metrics never
share a table. Coverage matrix:

| Signal | SA SP500 | SA Nasdaq (QQQ/NQ) | CS SP500 | CS NDX-100 |
|---|:-:|:-:|:-:|:-:|
| Classical 1500 (preserved) | ✓ existing | covered (`QQQ`/`NQ_F`) | — | — |
| #1 Deep Momentum (M5) | ✓ | ✓ | optional later | optional later |
| #2 Momentum Transformer (M5) | ✓ | ✓ | optional later | optional later |
| #3 Triple-Barrier+Meta (wrapper) | wraps primaries | wraps primaries | wraps primaries | wraps primaries |
| #4 Avellaneda-Lee | — | — | **✓ primary** | **✓ primary** |
| #5 GKX-style OHLCV subset | — | — | **✓ primary** | **✓ primary** |
| #6 Vol Risk Premium | ✓ feature/filter | ✓ feature/filter (`^VXN` or fallback) | filter only | filter only |
| #7 HMM Regime | filter | filter | filter | filter |
| #8 FinBERT (promotion ladder) | feature once promoted | feature once promoted | feature once promoted | feature once promoted |
| #9 Options-Implied | ✓ feature/filter | ✓ feature/filter | filter only | filter only |
| #10 Macro Overlay | feature/filter | feature/filter | feature/filter | feature/filter |

### 5.2 The bridge — two thin modules

#### `signal_research/cross_sectional/signal_to_panel.py`

Per amendment §5.3, this module is **purely a conversion + validation layer**.
It does not train models, tune parameters, or run backtests. Responsibilities:
- Accept per-(date, symbol) predictions.
- Validate schema.
- Validate `feature_as_of_date < execution_date`.
- Validate one prediction per (date, symbol) (no duplicates).
- Drop or flag NaN predictions explicitly.
- Rank only within the date-`t` tradable universe.
- Output an M4-compatible prediction panel.

#### `signal_research/cross_sectional/panel_to_m4.py`

Per amendment §5.2:
- Calls existing **public** M4 interfaces only.
- Never monkey-patches internals, bypasses validators, or injects predictions
  after M4 has already applied universe / timestamp checks.
- If M4 cannot accept an external `y_xs_pred` panel through a clean interface,
  this spec **does not work around it**. A small adapter or public entry point
  in `alpha_eq` may be added only after a **separate explicit amendment**.

Per amendment §5.4:
- Preserves all M4 data-quality banners. Never downgrades or hides warnings.
- If `data_quality_label != pit_safe`, the cross-sectional report cannot use
  "institutional-grade", "production-ready", "fund-grade" language. Only
  "research" or "prototype evidence".

### 5.3 What the bridge adds (M4 discipline)

The M4 engine already provides:
- Dividend-safe PnL (tradable_* close + cash dividends booked once on ex-date).
- ADV-based participation caps (default 1 % dollar-ADV-20d-lag1).
- Per-name borrow proxy with 1× / 2× / 3× stress.
- Min-bucket rules (≥10 long + ≥10 short on full universe; ≥5 + ≥5 on focused
  basket).
- Holder-of-record dividend accounting.
- Hash-locked manifest for byte-reproducible cross-sectional results.

The bridge plugs paper signals into this engine without modifying it.

### 5.4 NDX-100 cross-sectional path

Per amendment §5 + §2.3:
- Universe: current Nasdaq-100 constituents (Wikipedia parse, cached, hashed).
- Data-quality label: `survivorship_prototype_only` (current-only) — with
  visible survivorship warning banner.
- Borrow proxy: **same as SP500 in the headline.** Per amendment §5.7, NDX-100
  borrow defaults are not relaxed without evidence. A Nasdaq-liquidity-adjusted
  borrow proxy may be reported as a **sensitivity case**, never the headline.
- Benchmarks per §2.3.4.

### 5.5 Avellaneda-Lee — precise residualization rules

Per amendment §5.8:
- Return horizon: predeclared (default 1d forward).
- PCA estimation window: predeclared rolling window (e.g. 252 trading days).
- Number of principal components: predeclared (e.g. 5) or
  explained-variance threshold (e.g. 60 %).
- Residuals standardized cross-sectionally on each rebalance date.
- Z-score entry threshold: predeclared grid (e.g. {1.0, 1.5, 2.0}).
- Exit rule: predeclared (e.g. revert to ±0.5 or fixed horizon).
- Rebalance cadence: predeclared (e.g. daily).
- **PCA fit using only past data on a rolling basis. Never fit on the full
  panel and then backtest historically.**
- Sector / market-beta residualization documented if used.

### 5.6 GKX-style feature list (per amendment §5.9)

Per §3.3 #5, the feature list is fixed and explicit. The cross-sectional
output is a tree-ensemble (LightGBM ranking objective) prediction per
(date, symbol) that the bridge converts to `y_xs_pred`.

### 5.7 Cross-sectional metrics (prediction + portfolio, both)

Per amendment §5.5:

**Prediction quality:**
- Daily rank IC
- Mean rank IC
- Newey-West-adjusted t-stat
- IC hit rate
- Decile spread
- Monotonicity across deciles

**Portfolio quality (through M4 engine):**
- Long-short net return
- Net Sharpe
- DSR
- PBO **on L/S PnL** (per amendment §5.6 — primary CS PBO uses PnL, not rank IC)
- Max drawdown
- Annual turnover
- Borrow drag
- Spread + commission drag
- ADV-cap binding frequency
- Capacity at $1 M / $10 M / $100 M AUM

Both blocks reported. Rank IC alone is never sufficient.

### 5.8 Cross-sectional PBO definition

Per amendment §5.6, a "strategy variant" for cross-sectional PBO is the
Cartesian product of:
- signal family
- parameter configuration
- universe
- rebalance cadence
- quantile choice (q ∈ {0.05, 0.10, 0.20})
- cost configuration
- model class (if applicable)

PBO computed on cross-sectional **long-short PnL series**, not raw rank IC.
Rank IC PBO reported as diagnostic.

### 5.9 HMM / Macro / Options / Vol as filters, not standalone CS strategies

Per amendment §5.10: these signals modulate or gate Avellaneda-Lee or
GKX-style, not generate independent cross-sectional strategies by default.
Filter combinations are not counted as independent strategies unless
separately registered + counted in PBO.

### 5.10 10-point bridge test contract

Per amendment §5.11:

1. Bridge schema validation (M4 prediction panel shape + required columns).
2. One prediction per (date, symbol) (no duplicates).
3. `feature_as_of_date < execution_date` (timestamp invariant).
4. No duplicate symbols per date.
5. Rank computed only within tradable universe (out-of-universe ranks NaN).
6. Non-PIT universe survivorship warning preserved end-to-end.
7. NaN predictions handled explicitly (dropped or flagged, never silently
   imputed).
8. M4 result equality when feeding a zero or random prediction panel
   (sanity: bridge does not bias the M4 outcome).
9. Current-constituent universe cannot be promoted as `pit_safe`.
10. PnL-based cross-sectional PBO uses L/S PnL, not rank IC alone.

### 5.11 Cross-sectional outputs

```
experiments/signal_research/<run_id>/cross_sectional/
  sp500/
    avellaneda_lee.parquet           # panel with y_xs_pred
    gkx_ohlcv_subset.parquet
    rank_ic_per_date.parquet
    decile_returns.parquet
    m4_backtest_result.json          # standard M4 BacktestResult
    report.md                        # CS report w/ survivorship banner
  nasdaq_100/
    avellaneda_lee.parquet
    gkx_ohlcv_subset.parquet
    rank_ic_per_date.parquet
    decile_returns.parquet
    m4_backtest_result.json
    report.md                        # NDX-100 CS report w/ survivorship banner
```

### 5.12 The bridge is not a shortcut to promotion

Per amendment §5.13. A signal can pass through M4 and still be rejected if:
- Universe is current-only.
- PBO is high.
- DSR is weak.
- Net PnL fails after costs.
- Performance is concentrated.
- ADV caps bind too often.
- Borrow / spread stress kills the result.

M4 gives stricter evaluation, not automatic credibility.

---

## 6. Success gates, reporting, milestones

### 6.1 Per-strategy status tiers (four levels)

Per the production-intent framing in §0, candidates are scored against
**four** distinct status tiers, each strictly stricter than the previous:

`research_pass` → `promotion_eligible` → `paper_trade_candidate` → `production_candidate`

#### Status: `research_pass`

A candidate is **`research_pass`** iff all of the following hold:

| # | Criterion | Threshold |
|---|---|---|
| 1 | Data quality (research-tier wording) | Data may be `pit_safe`, `partial_pit_universe`, `public_snapshot_not_pit`, or `survivorship_prototype_only`. Non-`pit_safe` results **must be labeled research-only** and **cannot** use institutional-grade or production-grade language. |
| 2 | Point-estimate net Sharpe | ≥ 1.5 |
| 3 | DSR | ≥ 0.5 |
| 4 | PBO — **profile-scoped** | `pbo_profile < 0.25` (preferred `< 0.10`). `pbo_raw_global` is reported as conservative context; high `pbo_raw_global` is flagged but does not automatically fail `research_pass`. `pbo_family` is reported for additional context. |
| 5 | Bootstrap 95 % lower Sharpe bound | `> 0`. "High confidence" tier requires `≥ 1.0`. "Exceptional / production-grade" tier requires `≥ 1.5`. |
| 6 | Cost robustness | Net Sharpe still positive at **2 × baseline costs**. |
| 7 | Max drawdown | `≥ −20 %` over the test window. **Calmar ratio > 1.0 preferred; Calmar ≤ 0.5 → candidate flagged weak even if Sharpe passes.** Annualised return reported alongside Sharpe and drawdown. |
| 8 | Regime gating | Regime-agnostic with both-regimes-not-catastrophic + at-least-one-materially-positive, OR regime-specific with **predeclared** gating using only same-time info + OOS performance post-gate. Retroactive regime declaration forbidden. |
| 9 | Family-specific rejection (§3.7) | None triggered. |
| 10 | Trade count + active-day count + OOS calendar span | ≥ 200 trades single-asset / ≥ 500 cross-sectional. **≥ 252 active trading days** (unless explicitly event-driven). **≥ 3-year OOS calendar span** for any serious success claim. |
| 11 | Performance concentration | No single quarter > 50 % of total PnL. **No single day > 25 %.** **No single trade > 20 %.** **No single asset > 35 %** of total PnL for multi-asset strategies (unless flagged + downgraded). |
| 12 | Required artifacts | Manifest hashes verify; registry entry complete; CI green. |

#### Status: `promotion_eligible`

A candidate is **`promotion_eligible`** iff all of `research_pass` holds AND:

- Data quality is `pit_safe`, **or** the profile uses a directly-traded
  instrument (SPY / QQQ / BTCUSDT / ETHUSDT and similar) where
  `constituent_survivorship_applicable: false` (snapshot/vendor limitations
  still disclosed).
- Bootstrap lower-bound Sharpe `≥ 1.0` (high-confidence tier) is the minimum;
  `≥ 1.5` (exceptional / production-grade) requires data quality also eligible
  AND no other "exceptional" claim is invalidated.
- All §6.3 failure categories are negative for this candidate.

#### Status: `paper_trade_candidate`

A candidate is **`paper_trade_candidate`** iff all of `promotion_eligible`
holds AND:

- **Permanent-holdout evaluation completed** (one-shot, never repeated).
  The holdout result reproduces the development-window-and-validation
  picture within tolerance: net Sharpe ≥ 1.0 on holdout (the holdout bar is
  slightly lower than the dev bar to account for shrinkage), DSR ≥ 0.5 on
  holdout, max drawdown ≥ −25 %.
- **Cost-stress robustness on holdout** (positive net Sharpe at 2× costs
  applied to holdout returns).
- **No `holdout_failure` failure-class entry** in §6.3 taxonomy.
- **Risk limits and kill-switch infrastructure** verified runnable for this
  candidate (see §0 non-negotiable #11).
- **Audit log + reproducibility check** complete: the candidate's full
  artifact set (registry entry, manifests, hashes, run metadata) is
  re-loadable byte-for-byte from cold start.
- Reporting language: this candidate may be called "paper-trade ready";
  cannot be called "live-trade ready" or "production-deployed".

#### Status: `production_candidate`

A candidate is **`production_candidate`** iff all of `paper_trade_candidate`
holds AND:

- **Paper-trade window completed** (separate from the backtest holdout) with
  performance consistent with backtest within tolerance. The duration of
  the paper-trade window is set per profile and asset class; for daily-bar
  equity it is typically ≥ 3 months.
- **Monitored shadow deployment completed**: the candidate ran in live
  shadow mode (no real orders) against live market data for a defined
  window with monitoring, alerting, and reconciliation against the model's
  expected behaviour.
- **Operational reviews passed**: documented review of risk limits, kill
  switch, exposure caps, drawdown control, audit-log completeness, and
  failure-alert routing.
- **Legal / regulatory / licensing / compliance review completed** if the
  candidate will manage external capital or advise external users.
- Reporting language: this candidate may be called "production-ready" but
  must still carry the disclaimer that all live deployment is subject to
  operator approval and the kill-switch protocols.

**Status promotion is sequential.** No candidate moves from `research_pass`
to `paper_trade_candidate` without passing `promotion_eligible`; no candidate
moves from `promotion_eligible` to `production_candidate` without passing
`paper_trade_candidate`. Skipping stages is forbidden per §0 non-negotiable
#10.

**If 0 candidates pass `research_pass`, that is the result.** The reframe
from §4 stands. Higher tiers are even stricter — most runs will end at
`research_pass` or below.

### 6.2 Three-tier reporting

**A. Per-family report** (`reports/signal_research/family/<family>.md`).
Per-strategy line includes (per amendment §4.8): raw Sharpe, net Sharpe, DSR,
`pbo_raw_global`, `pbo_profile`, `pbo_family`, bootstrap Sharpe CI, max DD,
Calmar, annualised return, turnover, trade count, active-day count, cost
sensitivity, cluster ID, family, regime-agnostic-or-specific flag, and the
**status-tier achieved** (one of: `none` / `research_pass` /
`promotion_eligible` / `paper_trade_candidate` / `production_candidate`).

**B. Per-profile report** (`reports/signal_research/profile/<profile>.md`).
For `sp500`, `nasdaq`, `crypto`, `futures_proxy`. The **Nasdaq** profile is
split into two distinct sections per §2.3.3:
- Single-asset QQQ / NQ forecasting.
- NDX-100 cross-sectional selection.

Cross-family ranking, top-K survivors, Pareto front, regime breakdown, family
attribution.

**C. Master enhancement report** (`reports/signal_research/enhanced_benchmark.md`).
Comparison vs. the existing 1500-strategy classical baseline. Headline:
- PBO before (0.90) → PBO after (whatever).
- Median DSR of top-25 before / after.
- Count of strategies passing each criterion of §6.1.
- Selection funnel (§6.4).
- Honest reframe in plain language.

Every report shows the data-quality banner from §2.

### 6.3 Failure classification taxonomy (canonical)

Per §4.10 and amendments §6.9 — 13 categories total:

`high_pbo`, `low_dsr`, `cost_failure`, `regime_concentration`,
`insufficient_sample`, `too_few_trades`, `delay_stress_fail`,
`single_period_dominance`, `over_correlated_with_baseline`,
`randomization_fail`, `data_quality_fail`, **`holdout_failure`**,
**`capacity_failure`**.

Each failed candidate maps to ≥ 1 category. The categorisation itself is the
research output and guides the next research step.

### 6.4 Selection funnel table (master report)

Per amendment §6.10, the master report shows how many candidates remain
after each filter:

| Filter | Candidates remaining |
|---|---:|
| Total raw candidates | ~1620 |
| After data-quality filter | … |
| After cost robustness (2× cost stress) | … |
| After sanity checks (randomization / inverted signal) | … |
| After PBO threshold | … |
| After DSR threshold | … |
| After bootstrap CI lower-bound > 0 | … |
| After regime + concentration checks | … |
| **Final `research_pass` count** | … |
| **Final `promotion_eligible` count** | … |
| **Final `paper_trade_candidate` count** | … (after one-shot holdout) |
| **Final `production_candidate` count** | … (after paper trade + shadow deployment + reviews; typically 0 in any single benchmark run) |

This makes the multiple-testing burden visible at a glance. The lower
status tiers are expected to be sparse; `production_candidate` is typically
zero in any single benchmark run because it requires processes that occur
outside the benchmark itself.

### 6.5 Implementation milestones (dependency-ordered)

Per amendment §6.11, M6 is split into M6a (FinBERT) and M6b (Reality Check)
— they have different dependencies and are independent add-ons.

| # | Milestone | Deliverable | Compute (non-binding) |
|---|---|---|---|
| **M1** | **Data + registry foundation** | Long-history loaders (yfinance 2005-current), HF / FRED / CBOE / crypto adapters with manifests, mandatory strategy registry schema (§3.6), 4 model profiles (`signal_research_{sp500,nasdaq,crypto,futures_proxy}`), 4 Nasdaq universes (§2.3), minimal v1 crypto data (§6.6), 5-tier data-quality classifier with `public_snapshot_not_pit` and the separate `constituent_survivorship_applicable` flag for directly-traded instruments | small |
| **M2** | **Methodology stack** | CPCV (§4.1), meta-labeling with survivor-only pre-filter (§4.2), correlation dedup on net OOS returns (§4.3), multi-objective Pareto (§4.4), regime-conditional with agnostic/specific declaration (§4.5), block-bootstrap CIs (§4.6), three-tier PBO + DSR (§4.7), failure-classification taxonomy (§4.10), dev-only invariant (§4.9) | moderate |
| **M3** | **Classical paper signals** | Triple-Barrier + meta-labeling wrapper (#3), Avellaneda-Lee with rolling PCA (§5.5), GKX-style OHLCV subset (§5.6), Vol Risk Premium as feature + tradable-only-if-real-instrument (#6), HMM regime feature (#7), Options-Implied features (#9) with `^VXN`-or-fallback, Macro Overlay features (#10) — runs entirely on classical models, no LSTM / Transformer | moderate |
| **M4** | **Cross-sectional bridge to alpha_eq M4** | `signal_to_panel.py` (pure conversion + validation per §5.2), `panel_to_m4.py` (banner-preserving per §5.2), 10-point bridge test contract (§5.10), SP500 + NDX-100 cross-sectional runs of Avellaneda-Lee + GKX-style, PnL-based cross-sectional PBO (§5.8) | moderate-to-large |
| **M5** | **Deep models** *(deferred add-on, not blocker)* | Deep Momentum Network (Lim/Zohren) (#1), Momentum Transformer (Wood/Zohren) (#2), trained walk-forward with proper purging on M4 Mac MPS. Single-asset on SP500 + Nasdaq tracks. Deferred until M1-M4 prove the methodology works on cheap classical signals first | large |
| **M6a** | **FinBERT promotion ladder** *(deferred add-on)* | FinBERT sentiment pipeline + 4-state promotion ladder (§3.3 #8) + 10-criterion audit gate. Sentiment features gated behind audit pass | small-to-moderate |
| **M6b** | **Reality Check / SPA** *(deferred add-on)* | `methodology/reality_check.py` — White's Reality Check / Hansen SPA. Statistical bootstrap correction for the best strategy after trying many alternatives | small-to-moderate |

M1–M4 deliverable: a complete enhanced benchmark report.
M5, M6a, M6b: add-on layers that improve the result if they survive their
own gates. Each is independently shippable.

### 6.6 Minimal v1 crypto profile

Per amendment §6.12 — defined now (rather than left empty) so the master
report can make meaningful crypto claims:

**Data:**
- **BTCUSDT** daily OHLCV (Binance public).
- **ETHUSDT** daily OHLCV (Binance public).
- Optional: Binance perpetual futures (if available), funding rates.

**Cost model:**
- Maker / taker fees (Binance public schedule).
- Spread proxy.
- Funding payments (when perpetuals are used).

**Profiles within `signal_research_crypto`:**
- Trend (TS momentum, MA crossover, breakout)
- Mean reversion (Bollinger, z-score, RSI)
- Volatility (vol-targeted, realized-vol breakout)
- Funding / carry (if perpetuals + funding rates are available)

**PBO / DSR** identical to other profiles. Same `research_pass` /
`promotion_eligible` gating. Crypto reports carry the
**directly-traded-instrument** semantics (BTCUSDT and ETHUSDT are spot
crypto pairs, not ETFs — `constituent_survivorship_applicable: false`
applies because there are no constituents; vendor / snapshot limits still
disclosed).

### 6.7 What this spec is NOT

- **Not a promise of Sharpe ≥ 1.5.** The enhancement set may produce zero
  passing strategies. That is a valid research outcome.
- **Not a path to live trading.** Passing strategies still require S2 (LLM
  governor) + S4 (execution + promotion gates) per CLAUDE.md.
- **Not a replacement for the classical 1500-strategy benchmark.** That
  benchmark stays untouched.
- **Not a substitute for PIT data.** Current-constituent universes carry
  survivorship caveats; `pit_safe` requires the deferred
  `S1-EQ-FUNDAMENTALS` PIT work.

### 6.8 Open questions for the writing-plans phase

- Exact HF dataset IDs for FinBERT news (candidates listed in §2.4 —
  pipeline-build-time decision based on availability).
- Specific FRED series IDs for macro overlay (DGS10, T10Y2Y, DTWEXBGS,
  DCOILWTICO, GOLDAMGBD228NLBM).
- Whether `Deep Momentum Network` / `Momentum Transformer` use the official
  reference implementations or from-scratch PyTorch (writing-plans decides).
- Binance public-data adapter selection (multiple HF datasets vs direct
  archive download).

### 6.9 Compute / time estimates (non-binding, per amendment §6.13)

Estimates are **non-binding** and split by milestone. Implementation quality
does not depend on hitting them. Compute reality on an M4 Mac with 24 GB
unified memory:
- M1: minutes-to-an-hour.
- M2: about an hour.
- M3: low single-digit hours.
- M4: low single-digit hours (depends on M4 engine reruns).
- M5: multi-hour to overnight (PyTorch MPS training).
- M6a: small-to-moderate.
- M6b: small-to-moderate.

If any milestone needs more time, it gets more time. Quality is not
sacrificed to estimates.

---

## 7. Disclaimer

This spec is a research-and-engineering plan for a **production-intended**
quant research platform with hedge-fund-grade discipline, constrained by
limited resources.

Nothing produced from this spec, or from artifacts generated under it,
automatically constitutes investment advice. The project is production-
intended; **research artifacts produced from it are not automatically
investment advice**.

**Before any output of this system is used to advise external users or
manage capital, legal, regulatory, licensing, and compliance review is
required.** This applies to:
- Hosting model outputs in any consumer- or client-facing product.
- Sending live orders to a broker on behalf of the operator or any third
  party.
- Publishing strategy performance claims to external audiences.
- Marketing, soliciting, or accepting capital under management.

The strict backtest engine (via the M4 cross-sectional bridge) reduces the
gap between reported and realised performance but does not guarantee it.
Real trading additionally requires the S2 (LLM governor) + S4 (execution,
risk, promotion gates) machinery in the existing specs, plus the
operator-controlled `QUANTLAB_STAGE` and kill-switch protocols. Status
elevation past `paper_trade_candidate` requires operator review under the
staged-promotion protocol (§6.1, §0 non-negotiable #10).

`not_investment_advice: true`
