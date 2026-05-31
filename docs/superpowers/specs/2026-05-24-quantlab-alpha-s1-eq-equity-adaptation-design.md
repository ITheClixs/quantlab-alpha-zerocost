# S1-EQ — US Equity Adaptation of the S1 Stack + Pragmatic-Strict Backtest

- **Spec date:** 2026-05-24
- **Status:** Brainstormed; awaiting user spec-review before invoking `superpowers:writing-plans`.
- **Author:** Brainstormed with user via `superpowers:brainstorming`.
- **Branch context:** `quant-llm-implementation` on top of `main`.
- **Sibling specs (do not modify):**
  - `2026-05-21-quantlab-alpha-s0-trainer-persistence-design.md` — S1 (JS-trained) trainer persistence.
  - `2026-05-20-quantlab-alpha-s4-execution-risk-promotion-design.md` — execution + promotion gates.
- **Deferred sibling specs (named for cross-reference, not part of this milestone):**
  - `S1-EQ-FUNDAMENTALS` / `S1-EQ-ALT-DATA` — PIT fundamentals, earnings, news sentiment as features.
  - `S5 Microstructure & HFT` — orderbook, tick, intraday execution simulator, market impact.
  - `S5 Options & Futures` — separate asset-class pricing + cost models.
  - Phase-2 strict-backtest upgrades — market-impact, MOC auction, sector/factor neutrality, real PIT borrow feed.

> **Not investment advice.** All artifacts produced under this spec carry the
> `not_investment_advice: true` footer. If the data-quality classification is
> `survivorship_prototype_only`, every report carries a visible prototype-only
> warning banner and the success gate is suspended.

---

## 1. Scope, naming, and architectural choice

### 1.1 What this spec produces

A new model family, **S1-EQ**, that is a sibling of the existing JS-trained S1
stack (`experiments/alpha_s1/<run_id>/`). It re-uses the S1 training-pipeline
architecture (6 base learners + linear stacker) but targets real US equity
forward returns using engineered equity-specific features. It is **not** a
modification of the JS-trained S1 stack: the JS feature contract
(`feature_00..feature_78`, sha256-locked) and inference path remain untouched.

The JS-trained S1 stack is preserved on disk and is invoked only as an
**out-of-distribution sanity overlay** during backtest reporting to confirm
that retraining on equity features beats applying the JS-trained weights to
the same equity features.

### 1.2 Package layout

A new top-level package under the existing source tree:

```
src/quant_research_stack/alpha_eq/
```

`alpha_eq` is additive: nothing in `src/quant_research_stack/alpha/` is moved,
renamed, or refactored. The JS inference path keeps its sha256 contract.

### 1.3 Architecture

- **Pooling strategy (v1):** single pooled cross-sectional model. All S&P 500
  names go into one stacker; predictions are interpreted as cross-sectional
  ranks of next-day vol-normalized residual return (`y_xs`).
- **Phase 2 (deferred):** hierarchical pooled-base learners with a sector-
  aware stacker, triggered only if v1 diagnostics show sector-specific
  failure modes.
- **Rejected:** per-sector heads (premature; 11× maintenance for unclear
  payoff).

### 1.4 Universe

- **Primary training universe:** full S&P 500, PIT-filtered, from
  `data/raw/huggingface/jwigginton__timeseries-daily-sp500/` (1980-2024,
  503 historical symbols, 3.82 M rows raw).
- **Reporting cohort:** focused mega-cap basket — `configs/eq_focused_basket.yaml`
  — initially AAPL, ORCL, PYPL, INTC, META, TSLA, QCOM, PLTR, GOOGL, AVGO,
  ADBE, RKLB + curated additional liquid US large-caps (target size 30-50
  names). The focused basket is an **evaluation and deployment cohort, not
  the main training universe**.

### 1.5 Backtest realism: "Pragmatic strict"

Mandatory realism axes (full detail in §5):

- Point-in-time S&P 500 membership.
- Split-adjusted tradable price series for execution and MTM, plus a
  separately maintained dividend feed booked as cash PnL on ex-date
  (no total-return double-count); total-return series reserved for
  labels, diagnostic models, and benchmark analysis (§2.3, §5.4).
- Next-day open as headline fill price using **tradable (not total-return)
  prices**, with HLC3-proxy VWAP and next-day close as sensitivity cases.
- Per-name borrow cost proxy (3-tier static + date-aware upgrades), stress-
  tested at 1×, 2×, 3×.
- Dollar-ADV participation cap (1 % headline, 3 % sensitivity).
- Explicit gross and net leverage targets; financing cost charged or
  stress-tested whenever gross > 1.0.
- Daily, monthly, annual reporting plus PnL decomposition (gross alpha,
  cost drag, spread drag, borrow drag, financing drag, net alpha).
- Two reporting cohorts every run (full universe + focused basket).

Explicitly **deferred** from this spec: market-impact model, queue position,
MOC auction slippage, options/futures, news sentiment, fundamentals,
microstructure/HFT.

---

## 2. Data sources, PIT, corporate actions, manifest

### 2.1 Three-tier data-quality classification (mandatory gate)

A one-shot `scripts/pit_quality_audit.py` classifies the daily-bars dataset
into exactly one of:

| Label | Meaning | Promotion semantics |
|---|---|---|
| `pit_safe` | Date-symbol membership reconstructed; ticker mapping covers known transforms; delistings respected; delisting-return audit (§2.9) passes | **Full technical pass is possible.** Eligible for the full success gate (§6.4). May be described as institutional-grade in reports. |
| `partial_pit_universe` | Some PIT info present (e.g., approximate constituent list, partial delist dates, partial delisting-return capture) but not byte-perfect | **Conditional research pass only, with explicit caveats in every report.** Must **not** be described as institutional-grade. |
| `survivorship_prototype_only` | PIT membership cannot be reconstructed | **Success gate suspended.** Results are published as research-only; every artifact carries a prototype-only banner. |

**Historical symbol coverage is not equivalent to PIT membership.** The
classifier does not award `pit_safe` merely because delisted tickers appear
in the symbol set.

### 2.2 PIT reconstruction

Attempts in order:

1. HuggingFace `andyqin18/sp500-historical-membership` or
   `fmv1992/sp500_historical_index` if available.
2. Kaggle equivalents.
3. Wikipedia historical-revision parse — labelled `wikipedia_fallback` in
   the manifest if used as the sole source.

The result is a **date-symbol membership table** with columns
`(date, symbol, in_index, addition_date, removal_date, removal_reason)` plus
a **ticker-mapping table** covering at least: FB ↔ META, GOOG ↔ GOOGL,
BRK.B variants, CBS ↔ VIAC ↔ PARA, FCAU ↔ STLA, and any other transforms
discovered during the audit.

### 2.3 Corporate actions and the three price series

Three separate price series are stored and used for distinct, non-overlapping
purposes. v1 uses a strict **price-PnL + cash-dividend** accounting
convention to eliminate the dividend double-counting risk that would arise
from using total-return prices for MTM while also booking dividend cash:

| Series | Source / construction | Used for |
|---|---|---|
| `tradable_*` | **Split-adjusted execution-consistent OHLCV** (split-adjusted in v1; raw lot/share accounting is deferred) | **Execution fills, fill-aligned PnL for newly opened/resized positions, and close-to-close MTM of held positions (price PnL only — dividends are booked separately)** |
| `split_adj_*` | Split-adjusted analytical prices (may alias `tradable_*` when the source is already split-adjusted) | Feature engineering: returns, volatility, microstructure proxies, cross-sectional ranks |
| `total_return_*` | Split-adjusted + dividend-reinvested | **Labels** (`y_raw`, `y_vn`, `y_xs`), diagnostic models, **benchmark analysis** (SPY total-return reference, capacity benchmarking) — **not** the MTM series for the portfolio |

**Why `tradable_*` is split-adjusted in v1 rather than raw:** raw prices
across splits would require explicit lot/share accounting (a 2-for-1 split
must double the share count on the holding's effective record). v1 does
not implement raw lot/share accounting, so the safer convention is
split-adjusted execution-consistent prices. Raw lot/share accounting is
deferred to a Phase-2 backtest upgrade.

**Dividend accounting (single source of truth):**
- Dividends are sourced from a separate dividend feed (HF, vendor, or
  yfinance snapshot) and stored in `sp500_dividends.parquet`.
- The dividend feed source quality is recorded in the manifest; if
  yfinance is used, the source is labelled
  `public_snapshot_not_vendor_pit` — never `pit`.
- Dividends are booked **once**, as cash PnL on the ex-date, to the
  holder of record at the prior close. They are **not** also reflected
  via the `total_return_*` series in the portfolio's PnL path.

**If the upstream HF dataset is already total-return adjusted,** that
fact is recorded in the manifest as
`corporate_action_quality: vendor_total_return`, and the `tradable_*`
series is reconstructed by **removing dividend reinvestment** (subtract
the dividend ladder back out) so execution and MTM never silently use
total-return prices.

### 2.4 Borrow proxy (shorts only)

A 3-tier static categorical borrow rate, applied only to short notional:

| Tier | Default annual bps | Date-aware upgrade triggers |
|---|---:|---|
| `easy` | 25 | none |
| `general` | 100 | none |
| `hard` | 500 | any of: recent IPO < 6 mo, low dollar ADV-20d-lag1 (< $5 M), realized vol-20 > 80 %, price < $5, high short-interest if available, manual hard-to-borrow override |

Recent S&P index addition (< 30 days) is a **watchlist flag**, not an
automatic hard-borrow upgrade, unless combined with low liquidity or high
volatility.

Borrow is documented as an approximation, applied only to short notional,
and **stress-tested at 1×, 2×, 3×** in every backtest. Phase 2: replace
with a real PIT borrow feed.

### 2.5 Dollar ADV (for participation cap)

`adv_20d_dollar_lag1 = rolling_median_20( tradable_close × tradable_volume ).shift(1)`

Per-name daily participation cap defaults to **1 %** of `adv_20d_dollar_lag1`,
with **3 %** as a sensitivity case. (Share-volume ADV is **not** used.)

### 2.6 Storage layout

```
data/processed/equities/
  sp500_pit_membership.parquet           # (date, symbol, in_index, add/remove dates, reason) — or absent if prototype-only
  ticker_mapping.parquet                  # historical ticker transforms (FB↔META, etc.)
  sp500_tradable_prices.parquet           # OHLCV, split-adjusted execution-consistent (§2.3)
  sp500_split_adjusted_prices.parquet     # for feature engineering (may alias tradable_*)
  sp500_total_return_prices.parquet       # for labels, diagnostics, benchmark analysis (not MTM)
  sp500_dividends.parquet                 # ex-date, amount, source-quality label
  sp500_delisting_audit.parquet           # exit classification per §2.9
  sp500_adv.parquet                       # adv_20d_dollar_lag1 per (date, symbol)
  sp500_borrow_proxy.parquet              # static (symbol, borrow_tier, annual_bps)
  _manifest.json                          # see §2.7
```

### 2.7 Manifest contents (mandatory)

`data/processed/equities/_manifest.json` records, at minimum:

- `pipeline_version`, `git_sha`
- For every parquet artifact: `path`, `sha256`, `row_count`, `symbol_count`,
  `date_range_start`, `date_range_end`, `schema_fingerprint`, `source_url`
  or `source_dataset_id`, `source_snapshot_date`
- `data_quality_label` (one of the three §2.1 labels)
- `corporate_action_quality` (e.g. `vendor_total_return`,
  `split_adj_plus_external_dividends`, `public_snapshot_not_vendor_pit`)
- `borrow_source_quality` (always `static_proxy_v1` initially)
- `pit_membership_source` (one of: `hf:andyqin18/sp500-historical-membership`,
  `kaggle:...`, `wikipedia_fallback`, `absent_prototype_only`)
- `delisting_audit_quality` (one of: `captured_above_threshold`,
  `partial_capture`, `audit_absent`); audit summary counters
  (`delisted_captured`, `delisted_missing`, `merger_captured`,
  `merger_missing`, `ticker_changed`, `unknown_exit`)
- `build_command_line`, `python_version`, `package_versions` (subset of
  pinned deps relevant to data engineering)
- `warnings` — list of strings (e.g. "fallback dividend source",
  "borrow proxy is static-v1")

### 2.8 Manifest-driven reproducibility

Every training and backtest entry point reads `_manifest.json` first and
**hard-fails** if any required hash, schema fingerprint, row count, or
date range disagrees with what is recorded. The same model `run_id` is
only valid for the manifest hash it was trained against.

### 2.9 Delisting-return audit (mandatory)

PIT membership alone is not sufficient. A symbol may leave the dataset
for several reasons, each with different return implications:

- **Delisting** (going to zero, bankruptcy, regulatory delisting) — the
  terminal return is typically a large negative loss.
- **Acquisition / merger** — the terminal return is the cash + stock
  consideration relative to the prior close.
- **Ticker change** (e.g., FB → META) — no terminal return; the position
  rolls forward to the new ticker.
- **Voluntary deregistration / going private** — usually a deal price.
- **Missing terminal price** without explanation — return-unknown.

For every symbol that exits the dataset between its first and last
observation in the development or holdout window, `pit_quality_audit.py`
classifies the exit into one of: `delisted_captured`, `delisted_missing`,
`merger_acquired_captured`, `merger_acquired_missing`, `ticker_changed`,
`unknown_exit`. The audit produces `sp500_delisting_audit.parquet` with
columns `(symbol, exit_date, exit_reason, terminal_return_captured,
terminal_return_value, classification_source)`.

**Promotion semantics interaction:**
- `pit_safe` requires that at least 95 % of exits in the development +
  holdout windows are classified as `*_captured` or `ticker_changed`,
  and zero `unknown_exit` rows inside the holdout window.
- `partial_pit_universe` is awarded if PIT membership is otherwise
  reconstructable but the delisting-return audit does not meet the
  thresholds above; the report must include a "delisting-return
  limitation" section quantifying the unmeasured loss.
- `survivorship_prototype_only` is awarded if the audit cannot run at
  all (e.g., no exit-reason source available).

**Why this matters:** missing delisting losses systematically inflate
equity strategy backtests, often by 1-3 % annualized for a long-only
S&P 500 strategy and more for long/short books that may have been long
the delisted names. The audit is the difference between a believable
backtest and a misleading one.

---

## 3. Features, labels, and leakage controls

### 3.1 Signal timestamp convention (single global convention)

**Signal is generated after `close_t`.** Features at row date `t` may
include the complete day-`t` OHLCV bar. The label is the forward return
from `t` to `t+1`. Execution is at `t+1` (open is the headline; HLC3 and
close are sensitivities). The hard invariant is:

```
feature_as_of_date < execution_date
```

Not `feature_window_end < t`. This makes the rolling-feature rule
"all rolling stats use only data available by `feature_as_of_date`",
not "all rolling stats use `.shift(1)`".

### 3.2 Labels (three diagnostic targets)

| Target | Definition | Role |
|---|---|---|
| `y_raw` | `total_return_close_{t+1} / total_return_close_t - 1` | Diagnostic-only model + sanity benchmark |
| `y_vn` | `y_raw / realized_vol_20_lag1` | Diagnostic-only model |
| `y_xs` | `y_vn - cross_sectional_mean(y_vn over the date-t PIT tradable universe)` | **Primary stacker target** |

The portfolio trades cross-sectional ranks of `y_xs`. The diagnostic models
(`y_vn`, `y_raw`) are trained per fold but **do not enter the stacker** in
v1; they are reported sidecars used to interpret whether the model is
learning broad market direction, vol-scaled return, or pure cross-sectional
selection.

**Sector-neutral residualization** (subtract within-sector mean after the
cross-sectional mean) is computed and reported as a diagnostic only; not
the default training target in v1.

### 3.3 Feature families

Target final feature count: ~70-90 after pruning.

1. **Returns / momentum** (per symbol, lagged-appropriately):
   `log_return_{1,2,5,10,20,60,120,252}`, `cumulative_return_{60,120,252}_skip5`
   (Jegadeesh-style 12-1 / 6-1), `mean_reversion_5`.

2. **Volatility**: `realized_vol_{5,20,60}`, `parkinson_vol_20`,
   `garman_klass_vol_20`, `vol_of_vol_60`.

3. **Daily-microstructure proxies**:
   - `amihud_illiq_20 = mean(|log_return_1| / dollar_volume)` over 20 d.
   - `roll_spread_20 = 2 × sqrt( max(0, -cov(log_return_t, log_return_{t-1})) )`.
     When the autocovariance is non-negative, the value is **explicitly NaN**
     and falls back to the tiered constant from §5; silent zero-fill is
     forbidden.
   - `kyle_proxy_signed_volume_20`: explicitly labelled as a weak proxy,
     defined as the regression slope of `|log_return|` on
     `sign(log_return) × dollar_volume` over 20 d. **Not** true Kyle lambda.
   - `overnight_gap = log(open_t / close_{t-1})`.
   - `intraday_return = log(close_t / open_t)`.
   - `close_location_20 = (close_t - low_20d) / (high_20d - low_20d)`.

4. **Volume / liquidity**: `dollar_volume`, `log_dollar_volume_20d`,
   `volume_zscore_20d`.
   - `turnover_proxy_20` is **dropped** unless a reliable shares-outstanding
     or market-cap source exists. Synthetic shares-outstanding estimates
     from volume history are forbidden.

5. **Cross-sectional ranks** (per-date, within date-`t` PIT tradable
   universe only):
   `rank-transform of {log_return_1, log_return_5, log_return_20,
   realized_vol_20, dollar_volume, amihud_illiq_20, overnight_gap,
   close_location_20}` mapped to `[-0.5, 0.5]`.

6. **Market / regime context** (single-value features broadcast across
   names on date `t`): `spy_log_return_5`, `spy_realized_vol_20`,
   `vix_close`, `cross_sectional_dispersion = std(log_return_1) across
   the date-t universe`.

   **VIX fallback rule (mandatory):** `vix_close` is used **only** for
   dates where a valid historical value exists in the source feed (VIX
   trading began 1990-01-02; pre-1990 dates have no VIX). For dates
   without a valid VIX value, the feature falls back to
   `cross_sectional_vol_20`, a cross-sectional volatility proxy. The
   absence of VIX **must not** silently truncate the training period
   or restrict the universe, because doing so creates hidden selection
   bias (early decades would be dropped). The feature builder emits a
   single boolean column `vix_is_proxy` recording, per date, whether
   the value came from VIX or the proxy.

7. **Foundation-model meta-features**: **disabled by default in v1**
   behind `enable_meta_features: false`. The decision to enable or
   disable meta-features for a given run is made **using development-
   window validation only** (after timestamp audit, ablation study, and
   baseline comparison). Holdout performance **must not** be used to
   decide whether to enable or disable meta-features for the same run.
   Holdout improvement may be **reported** after the one-shot holdout
   evaluation but it informs only future runs, not the current
   `enable_meta_features` choice.

8. **Sector / industry**: stored if available (GICS L1 + L2) and used
   for diagnostics, reporting, and Phase-2 sector-neutral residualization.
   Not a v1 input feature.

9. **Noise sentinel** (mandatory): `gaussian_noise_seed42` — seeded
   N(0,1) deterministic per `(date, symbol)`.

### 3.4 Feature drop rule (three conditions, all required)

A feature is **dropped** only if **all three** of:

1. Ranks below `gaussian_noise_seed42` on ≥ 3 of 5 folds (per CLAUDE.md §5.6).
2. Univariate rank-IC sign is unstable across folds.
3. Ablation removal does **not** harm fold-validation IC.

Univariate IC, model-based importance, and ablation are treated as
**separate diagnostics**; no single one auto-drops. Foundation-model
meta-features are governed by the §3.3-7 audit gate, not this rule.

### 3.5 Leakage controls (timestamp-contract tests, mandatory CI)

| Rule | Object-level enforcement |
|---|---|
| Every feature carries an explicit `feature_as_of_date` column | Unit test asserts `feature_as_of_date < execution_date` for every training and inference row |
| Label uses only future returns after `feature_as_of_date` | Unit test asserts label is computed from prices strictly after `feature_as_of_date` |
| Train / validation / test splits are chronological | Unit test asserts no fold has earlier-date rows in validation than train |
| Scalers, imputers, encoders fit on training folds only | Unit test asserts scaler objects expose `fitted_on_start_date`, `fitted_on_end_date`, `fold_id`; no val/test rows in fit window |
| Cross-sectional ranks use only date-`t` tradable PIT universe | Unit test asserts rank vector at date `t` is invariant under shuffling future rows or rows outside the universe |
| Rolling features recomputed exactly from the allowed historical window for the chosen timestamp convention | Unit test recomputes rolling features from scratch and asserts equality |
| Future-shifted columns cannot enter `feature_cols.json` | Unit test scans `feature_cols.json`; any name containing `_t+`, `future_`, or sourced from a future-shifted column raises |
| Holdout dates cannot be loaded during training, tuning, stacker fitting, threshold selection, feature pruning, or adversarial validation | Unit test asserts holdout loader is locked during all six lifecycle phases |

Adversarial validation between train and validation folds **inside the
development window** is a **diagnostic only**, not a hard enforcement.
AUC > 0.6 between train and validation flags a feature for ablation
review; it does not auto-drop. Adversarial validation **never** touches
the final holdout.

### 3.6 Permanent holdout

The last 20 % of dates (after PIT filter) is reserved as the permanent
holdout. It is never touched during feature engineering, hyperparameter
search, threshold tuning, stacker fitting, model selection, cost-model
calibration, adversarial validation, or feature pruning. The holdout
date set is written to `experiments/alpha_eq/<run_id>/holdout_dates.json`
and the data loaders refuse to return any row in that set unless the
caller is `inference.evaluate_holdout()` and the run has not yet emitted
`holdout_metrics.json`.

**Minimum holdout length:** the permanent holdout, after PIT and
tradability filters, must contain at least **3 years (≥ 756 trading
days)** of daily observations. If the filtered holdout is shorter, the
M6 success gate **cannot pass** regardless of metric values, because
holdout Sharpe over a sub-3-year window is too unstable to support a
promotion decision. This is enforced by
`tests/alpha_eq/test_holdout_min_length.py` and rechecked at M6.

---

## 4. Training pipeline, walk-forward CV, stacking

### 4.1 Pipeline shape

```
raw panel (manifest-locked)
  → engineered features (timestamp convention §3.1)
  → walk-forward purged + embargoed CV (§4.2)
  → 6 base learners per fold (Ridge, LightGBM, XGBoost, CatBoost, MLP,
    1D-CNN) — full_v1
    OR Ridge, LightGBM, XGBoost — fast_v1
  → OOF predictions per learner per fold
  → linear stacker on OOF, target = y_xs
  → refit-on-full on the entire development window with frozen hyperparams
  → permanent-holdout evaluation (one-shot)
  → portfolio backtest on holdout slice (§5)
```

### 4.2 Walk-forward CV

- Development window = first 80 % of dates. Permanent holdout = last 20 %.
- Inside the development window: **5 expanding-window folds** (anchored
  origin) as the primary v1 layout.
- **Dynamic purge / embargo:**
  - `purge_days = max(5, label_horizon_days + safety_buffer)` where
    `safety_buffer = 2` for the 1-day forward target.
  - `embargo_days = max(5, label_horizon_days)`.
- **Rolling-window robustness diagnostic** (secondary): 10-year train
  / 2-year validation across the development window. Expanding-window
  remains the primary layout; rolling-window is reported alongside as
  a regime-robustness check.
- Fold definitions written to `cv_folds.json` and consumed identically
  by every learner.
- Per-fold scalers/imputers expose `fitted_on_start_date`,
  `fitted_on_end_date`, `fold_id` (per §3.5 contract).

### 4.3 Training modes

Two profiles selectable via CLI:

- `fast_v1` — Ridge, LightGBM, XGBoost, linear stacker. Milestone-
  unblocking; used for iteration cycles.
- `full_v1` — all six base learners + linear stacker. Used for M5/M6
  promotion candidates.

The 1D-CNN is **optional** even in `full_v1` unless its input is a real
temporal tensor (`lookback_window × feature_channels`). A Conv1D over
arbitrary feature columns is not economically well-justified and may be
disabled if the temporal input cannot be constructed cleanly from the
engineered feature pipeline.

### 4.4 Base learners

Each learner targets `y_xs`. Two diagnostic models are trained alongside
each fold (Ridge on `y_vn`, LightGBM on `y_raw`) but **do not enter the
stacker**.

### 4.5 Linear stacker

- `LinearStacker` mirrors the JS S1 stacker surface (sha256-locked
  `feature_order`).
- **Regularized** linear combination of base-learner OOF predictions.
- **Default constraint:** the primary stacker uses L2-regularized weights,
  optionally non-negative-constrained. Signed (unconstrained-sign) weights
  are trained and reported as a diagnostic variant. The primary stacker is
  the signed variant only if it materially improves fold-validation IC
  vs. the L2-constrained variant; otherwise the L2-constrained variant
  is promoted.
- **Large negative weights are flagged** in the report.

### 4.6 Hyperparameter search

Optuna inner-CV on the development window only. Trial budgets are
config-driven (`configs/alpha_eq.yaml`) and capped per CLAUDE.md §8
power budget. Defaults (subject to wall-clock cap):

- LightGBM 50, XGBoost 30, CatBoost 30, MLP 20, Conv1D 20, stacker 30.
- Pruner: median pruner on fold-validation IC.

### 4.7 Research-degrees-of-freedom disclosure

`metadata.json` records, for every run: total Optuna trials, model
classes searched, feature sets evaluated, threshold sweeps, and any
post-hoc decisions made during development. The report's "Configuration"
section discloses these to reduce hidden multiple-testing risk.

### 4.8 Refit-on-full and persistence

- After Optuna selects best hyperparameters from CV, refit every base
  learner on the entire development window with those frozen hyperparameters.
- Each base learner is saved with a config sidecar JSON.
- Stacker saved to `models/stacker.joblib` with `feature_order`.
- `feature_cols.json` sha256-locked.
- `_artifact_sha256.json` covers every model + manifest +
  `cv_folds.json` + `holdout_dates.json` + predictions.

### 4.9 Reproducibility contract

- **Byte-identical:** splits, configs, manifests, `feature_cols.json`,
  hashes.
- **Within strict numerical tolerance:** predictions and metrics (the
  underlying ML libraries are not guaranteed byte-deterministic across
  CPU micro-architectures).
- Same `run_id` re-evaluation is **verification-only**.
- Any change to model class, feature set, config, code, or data
  manifest **requires a new `run_id`**, and the old holdout result is
  preserved untouched.

### 4.10 One-shot holdout evaluation

- Load persisted artifacts via `alpha_eq.inference.load_predictor_from_run`.
- Score every holdout row → `holdout_predictions.parquet` and
  `holdout_metrics.json`.
- Second invocation against the same `run_id` is `holdout_replay`:
  verifies byte-identical predictions and metrics within the tolerance,
  no new metrics emitted.

### 4.11 Outputs (per run)

`experiments/alpha_eq/<run_id>/`:

```
metadata.json                        git_sha, data_manifest_sha256, hyperparams, fold defs, holdout def, DoF disclosure
cv_folds.json
holdout_dates.json
feature_cols.json                    + sha256
predictions.parquet                  OOF dev-window preds (y_xs primary + y_vn, y_raw diagnostics)
holdout_predictions.parquet
metrics.json                         dev-window CV metrics
holdout_metrics.json                 one-shot, immutable
feature_importance.parquet           per-base-learner per-fold
adversarial_validation.json          train-vs-val flags only (never holdout)
ablation.json                        feature ablation results
models/
  ridge.joblib
  lightgbm.txt + lightgbm.config.json
  xgboost.json + xgboost.config.json
  catboost.cbm + catboost.config.json
  mlp.pt
  sequence.pt                        (optional under full_v1)
  stacker.joblib
  diagnostic_ridge_y_vn.joblib
  diagnostic_lgb_raw_y.txt
_artifact_sha256.json
report.md
audit_log_smoke.jsonl
```

---

## 5. Pragmatic-strict backtest engine

### 5.1 Engine location

New package `src/quant_research_stack/alpha_eq/backtest/`. Distinct from
the existing simpler `backtest/` engine because the strict invariants
below would break the simpler API used by JS benchmark scripts.

### 5.2 Temporal contract

```
date t (close):        signal generated using features available through close_t
date t+1 (execution):  fill at next-day OPEN (headline) using tradable prices
date t+1 (close):      MTM at tradable_close_{t+1} (split-adjusted, price-only)
                       + any ex-date dividend booked as separate cash PnL
date t+2 ... :         hold until next rebalance
```

Hard invariant: `feature_as_of_date < execution_date` (asserted per row).

**Single accounting convention (no double counting):**
- Execution and price-MTM use the `tradable_*` series (split-adjusted,
  execution-consistent — §2.3).
- Dividends are booked **once**, as cash PnL on ex-date, to the holder
  of record at the prior close.
- The `total_return_*` series is **not** read by the portfolio MTM path;
  it is reserved for labels, diagnostic models, and benchmark analysis.

### 5.3 Fill model

- **Headline fill:** `fill_price_{t+1} = open_tradable_{t+1}`.
- **Sensitivity case A (VWAP proxy):**
  `fill_price_{t+1} = vwap_proxy_hlc3_{t+1} = (high_tradable + low_tradable + close_tradable) / 3`.
  Always labelled `vwap_proxy_hlc3`; never called real VWAP in any report.
- **Sensitivity case B (close):** `fill_price_{t+1} = close_tradable_{t+1}`.
- No partial fills, no rejection (documented limitation).

### 5.4 Fill-aligned PnL accounting (price PnL + cash dividends)

The v1 convention is **price-PnL on `tradable_*` plus explicit cash
dividend PnL**, eliminating any total-return double-count.

- **Newly opened or resized positions:** price PnL starts from the
  **actual fill price**, not from `close_t`. A position opened at
  `fill_{t+1}` has zero `close_t → fill_{t+1}` PnL by construction.
  First-day PnL on the new lot is
  `(tradable_close_{t+1} − fill_{t+1}) × signed_shares`.
- **Existing held positions:** marked close-to-close using
  `tradable_close` (split-adjusted, price-only).
- **Dividends:** booked once, as a separate cash-PnL line item on
  ex-date, to the holder of record at the prior close.
  Cash dividend PnL = `signed_shares × dividend_per_share`. Long
  positions receive the cash; short positions are debited the cash
  (the canonical short-seller-pays-the-dividend convention).
- **Total-return prices are never used for portfolio MTM.** They are
  used only for labels, diagnostic models, and benchmark analysis
  (e.g., SPY total-return reference).

### 5.5 Portfolio construction (single mode v1)

- **Dollar-neutral long/short, equal-weighted by side**, top/bottom
  q-quantile by predicted `y_xs`.
- Quantile sweep `q ∈ {0.05, 0.10, 0.20}`; headline `q = 0.10`.
- Universe at date `t+1` =
  `pit_universe(t+1) ∩ tradable(t+1) ∩ has_valid_signal(t)`.
- `tradable(t+1)` requires: non-null tradable OHLCV, `adv_20d_dollar_lag1
  ≥ $1 M` (configurable ADV floor), no halt/suspension flag.
- **Minimum bucket sizes:**
  - Full universe: ≥ 10 longs and ≥ 10 shorts.
  - Focused basket: ≥ 5 longs and ≥ 5 shorts.
  - If the quantile rule produces too few names, use the minimum-count
    rule (extend into the next-best-ranked names) or **skip that date**.
    Never silently empty a bucket.
- **Per-name participation cap:**
  `position_notional ≤ min(equal-weight target,
  participation_pct × adv_20d_dollar_lag1)`. Participation default 1 %;
  3 % is a sensitivity case.
- **Per-name weight cap:** `min(2 × equal-weight, 5 % of gross)`.
- Score-weighted construction is a Phase-2 diagnostic, **not** the v1
  headline.

### 5.6 Cost model

- **Commission + exchange fees:** 0.5 bps of notional per side
  (configurable).
- **Bid-ask spread crossing:** `0.5 × spread_proxy`, where
  `spread_proxy = min(roll_spread_20, 50 bps)`. If `roll_spread_20` is
  NaN per §3.3, fall back to tiered constants
  (`easy = 5 bps`, `general = 15 bps`, `hard = 50 bps`).
- **Pre-decimalization adjustment:** for trade dates before
  2001-04-09, spreads are widened (multiplier × 2.5 on the tiered
  fallback, multiplier × 1.5 on the Roll-derived value). The cutoff
  is configurable and documented in the report.

### 5.7 Borrow model

- Apply only to short notional.
- Lookup tier from `sp500_borrow_proxy.parquet` (§2.4).
- Date-aware hard-tier upgrades per §2.4 triggers.
- Daily borrow = `short_notional × annual_bps / 10000 / 252`.
- **Stress sweeps mandatory in every report:** 1×, 2×, 3× borrow.

### 5.8 Financing for leveraged cases

- Gross ≤ 1.0: no financing charged; documented as a limitation.
- Gross > 1.0: financing **charged** at a configurable annual rate, with
  **stress at 0 %, 2 %, 5 %** (SOFR-anchored proxy).
- Financing applied to gross notional above 1.0× equity, daily basis.

### 5.9 Risk constraints (enforced in construction, not post-hoc)

- Per-name weight cap per §5.5.
- Gross-leverage buffer: `target_gross × 1.05`; breaches scale offending
  positions down and log a warning.
- No v1 sector-exposure constraint; sector long / short / net
  timeseries reported as a diagnostic.

### 5.10 Rebalance cadence

- Daily rebalance (default headline).
- Weekly and monthly diagnostic modes reported as turnover sensitivity
  (same signal, fewer turns, lower cost drag).

### 5.11 PnL decomposition (mandatory)

Every report shows the v1 accounting identity:

```
portfolio_pnl  =  price_pnl_split_adjusted
                + cash_dividend_pnl
                − commission_drag
                − spread_drag
                − borrow_drag
                − financing_drag
```

`gross_alpha = price_pnl_split_adjusted + cash_dividend_pnl` and
`net_alpha = portfolio_pnl` are reported separately, alongside every
drag, in both bps/day and annualized %. The invariant
`gross_alpha − (commission_drag + spread_drag + borrow_drag +
financing_drag) ≈ net_alpha` (within float tolerance) is asserted by
`tests/alpha_eq/test_pnl_decomposition.py`.

Cross-check: per-day `cash_dividend_pnl` summed by ex-date equals the
ex-date dividends in `sp500_dividends.parquet` joined to the day's
positions. The test `test_no_dividend_double_count.py` (added to the CI
matrix in §6.2) asserts that the portfolio MTM series does **not**
also embed the dividend signal — i.e., regressing cash_dividend_pnl on
the residual of price_pnl_split_adjusted vs. total_return_pnl is
statistically zero.

### 5.12 Mandatory exposure diagnostics

Every backtest emits time-series of:

1. Daily net exposure
2. Gross exposure
3. Rolling SPY beta (60-day window)
4. Sector long exposure
5. Sector short exposure
6. Sector net exposure
7. Top-10 name notional exposures
8. Largest PnL contributors (by symbol, by month)

### 5.13 Reporting metrics

Per cohort (full universe + focused basket), per sensitivity config:

- Net total return, annualized return, gross return, cost drag, borrow
  drag, financing drag
- Sharpe (daily √252), Sortino (daily √252), Calmar (ann return / max DD)
- Max drawdown, max-DD duration
- Hit rate, avg win / loss
- Average daily turnover, average gross / net exposure
- Per-quantile top-bottom spread
- Rank IC mean + std + Newey-West-adjusted t-stat
- Per-decile spread table
- Capacity estimate at $1 M, $10 M, $100 M AUM (using the 1 % ADV cap)
- Monthly returns table (calendar months)
- Annual returns table (calendar years)

### 5.14 Sensitivity policy (two-tier)

- **Standard report** (per research iteration, default for M3-M4):
  headline config + a small required stress pack:
  - Borrow 1× and 3×
  - Fill open and HLC3 proxy
  - q ∈ {0.05, 0.10}
  - Gross 1.0 only
- **Full audit report** (M5/M6 promotion candidates only): the full
  matrix — borrow {1×, 2×, 3×} × fill {open, HLC3, close} × ADV {1 %, 3 %}
  × gross {0.5, 1.0, 2.0} = 54 runs.

This keeps iteration fast while preserving strictness at the promotion
gate.

### 5.15 Benchmark comparisons (every report)

- SPY buy-and-hold over the same dates (total-return).
- Family B (ridge OHLCV baseline) on the same universe and cost model.
- JS-trained S1 stack overlay on the same engineered equity features
  (sanity comparison only — confirms `S1-EQ > JS-overlay`).

### 5.16 Report structure (`report.md`)

1. Header: `run_id`, `git_sha`, `data_manifest_sha256`,
   `data_quality_label`, wall-clock cost, **prototype-only banner if
   applicable**.
2. Configuration: universe, dates, costs, borrow, ADV cap, gross target,
   fill model, DoF disclosure (§4.7).
3. Holdout metrics — full universe.
4. Holdout metrics — focused basket.
5. Benchmark table (S1-EQ vs SPY vs Family B vs JS-overlay).
6. Sensitivity sweeps (per §5.14).
7. Rolling-window CV diagnostic.
8. PnL decomposition (§5.11).
9. Exposure diagnostics (§5.12), including monthly + annual return
   tables and the largest-contributor lists.
10. Per-decile spread + IC stability.
11. Limitations: every modeled simplification listed with a severity tag.
12. Footer: `not_investment_advice: true`, data-quality reminder.

### 5.17 Backtest determinism and audit log

- All inputs are manifest-locked; hash/schema mismatch is a hard fail.
- `backtest_metrics.json` is byte-identical for the same
  `(model run_id, data manifest, backtest config)` triple (within the
  numerical tolerance from §4.9).
- Every fill and rebalance writes an append-only JSONL row to
  `logs/audit/equity_backtest/<run_id>.jsonl`; the file is `chmod a-w`
  on rotation; `audit_replay_check.py` is extended to verify equity
  backtest replay byte-identically within tolerance.

---

## 6. Project structure, testing, milestones, success criteria

### 6.1 Code layout (additive)

```
src/quant_research_stack/alpha_eq/
  __init__.py
  config.py
  data/
    pit_membership.py
    corporate_actions.py
    borrow_proxy.py
    adv.py
    manifest.py
    loaders.py
  features/
    timestamps.py
    returns_momentum.py
    volatility.py
    microstructure_proxies.py
    volume_liquidity.py
    cross_sectional_ranks.py
    market_regime.py
    noise_sentinel.py
    meta_features.py            # disabled by default
    builder.py
  models/
    ridge.py
    lightgbm_model.py
    xgboost_model.py
    catboost_model.py
    mlp.py
    sequence.py                  # optional under full_v1
  stacking.py
  training.py
  inference.py
  diagnostics/
    adversarial_validation.py    # train-vs-val only
    ablation.py
    feature_selection.py
  backtest/
    contracts.py
    portfolio.py
    fills.py
    costs.py
    borrow.py
    financing.py
    pnl.py
    exposure.py
    metrics.py
    sensitivity.py
    report.py
    runner.py
configs/
  alpha_eq.yaml                  # fast_v1 + full_v1 profiles
  eq_focused_basket.yaml         # versioned (see §6.7)
  backtest_eq.yaml               # standard + audit sensitivity packs
scripts/
  prepare_equity_data.py
  pit_quality_audit.py
  train_s1_eq.py                 # --mode fast_v1|full_v1
  backtest_s1_eq.py              # --mode standard|audit
  s1_eq_overlay_compare.py
Makefile targets:
  prepare-equity-data
  fast-retrain-s1-eq
  full-retrain-s1-eq
  backtest-s1-eq-standard
  backtest-s1-eq-audit
experiments/
  alpha_eq/<run_id>/             # see §4.11 + §5.17
```

### 6.2 Mandatory CI tests

`tests/alpha_eq/`:

| Test file | Asserts |
|---|---|
| `test_timestamp_contract.py` | `feature_as_of_date < execution_date`; rolling-window correctness for the chosen convention; cross-sectional ranks only within date-t tradable PIT universe |
| `test_holdout_isolation.py` | Holdout dates locked during training, tuning, stacker fit, threshold selection, feature pruning, adversarial validation |
| `test_scaler_fit_window.py` | Scalers expose `fitted_on_start_date`, `fitted_on_end_date`, `fold_id`; no val/test rows in fit window |
| `test_manifest.py` | Hash / schema / required-fields enforcement; mismatch is hard fail |
| `test_pit_classifier.py` | Three-tier classifier; reports labelled accordingly; survivorship_prototype_only suspends gate |
| `test_borrow_proxy.py` | Borrow applied only to short notional; 1×/2×/3× monotonic; index-addition alone does NOT auto-upgrade |
| `test_fill_pnl_alignment.py` | New positions: PnL from fill price; existing: close-to-close MTM; dividends on ex-date as cash |
| `test_pre_decimalization_spread.py` | Pre-2001-04-09 spread widening applied |
| `test_min_bucket.py` | Full universe ≥ 10/10; focused basket ≥ 5/5; insufficient → skip, never silently empty |
| `test_exposure_diagnostics.py` | All exposure series present; daily net ≈ 0 ± 1 %; gross within target ± 5 % |
| `test_pnl_decomposition.py` | `gross_alpha − (commission + spread + borrow + financing) ≈ net_alpha` within tolerance |
| `test_no_dividend_double_count.py` | Portfolio MTM uses `tradable_*`, not `total_return_*`; cash dividends booked once on ex-date; regression test against double-count residual is statistically zero |
| `test_delisting_audit.py` | Exits classified per §2.9; `pit_safe` requires ≥ 95 % captured + zero `unknown_exit` in holdout; otherwise downgraded |
| `test_vix_fallback.py` | Pre-1990 dates use `cross_sectional_vol_20`; no silent truncation of training period; `vix_is_proxy` boolean correct per date |
| `test_holdout_min_length.py` | Filtered permanent holdout contains ≥ 3 years of daily observations after PIT + tradability filters; otherwise M6 cannot pass |
| `test_reproducibility.py` | Splits/configs/feature_cols/manifests byte-identical; predictions/metrics within tolerance |
| `test_seeds.py` | All RNGs seeded; metadata records seeds |
| `test_random_signal_sanity.py` | Random predictions → no stable positive Sharpe, no beat SPY, rank IC ≈ 0 |
| `test_rank_direction_sanity.py` | Top-minus-bottom spread, rank IC, and long/short direction are internally consistent |
| `test_backtest_edge_cases.py` | Empty universe / insufficient bucket / all-ADV-capped / all-NaN predictions / missing execution prices → skip or hard-fail, never silent positions |
| `test_e2e_smoke.py` | `fast_v1` training + standard backtest on a 200k-row synthetic-equity slice completes in CI budget; all required artifacts present |

Tooling: `ruff check`, `mypy`, `pytest`, wired into `make check` and CI.

### 6.3 Milestones (gated; each emits a reviewable artifact)

| # | Milestone | Exit criterion (reviewable artifact) |
|---|---|---|
| M1 | Data preparation (PIT, corporate actions, ADV, borrow proxy, manifest) | `data/processed/equities/_manifest.json` present; data-quality classified; ticker-mapping table built; all hashes recorded |
| M2 | Feature pipeline + leakage tests pass | `feature_cols.json` sha256-locked; every CI test in §6.2 categories 1-4 green |
| M3 | `fast_v1` training (Ridge + LightGBM + XGBoost + stacker) | Pipeline artifacts and diagnostics produced; **weak or negative OOF IC does not block M3 itself** because M3 is an engineering milestone, but it does block promotion to M4 unless explicitly justified in the report |
| M4 | Strict backtest engine + standard sensitivity pack | `report.md` produced; exposure diagnostics + PnL decomposition validated; backtest CI tests green |
| M5 | `full_v1` training + audit-level backtest on focused basket and full universe | Permanent holdout evaluated **once**; `holdout_metrics.json` immutable; full 54-case audit matrix produced |
| M6 | Success-gate evaluation + JS-overlay comparison + final report | Go / No-go decision documented; iteration plan if No-go |

If `data_quality_label == survivorship_prototype_only`, **every artifact
across all milestones carries a visible prototype-only warning banner**
and the M6 success gate is suspended.

### 6.4 Success gate (evaluated once at M6 on the permanent holdout)

S1-EQ is promoted iff **all** of the following hold:

1. `data_quality_label == pit_safe` (full technical pass possible).
   `partial_pit_universe` is a **conditional research pass only** with
   explicit caveats and the "not institutional-grade" label in the
   report; `survivorship_prototype_only` → gate suspended.
2. **Permanent holdout length ≥ 3 years (≥ 756 trading days)** after
   PIT + tradability filters (§3.6).
3. **Delisting-return audit passes** per §2.9 thresholds: ≥ 95 %
   captured-or-ticker-changed in dev+holdout, zero `unknown_exit` in
   holdout.
4. S1-EQ headline (`q = 0.10`, gross = 1.0, fill = next-day open,
   borrow ×1.0) **net annualized Sharpe ≥ 0.7** — this is a
   **standalone requirement** that must be satisfied regardless of
   how SPY or Family B perform on the holdout (beating a negative
   SPY Sharpe or a negative Family B Sharpe on its own is never
   sufficient).
5. **Two-branch baseline rule:**
   - If Family B net Sharpe > 0:
     S1-EQ net Sharpe ≥ **1.5 × Family B net Sharpe**.
   - If Family B net Sharpe ≤ 0:
     S1-EQ net Sharpe ≥ **0.7** AND
     `S1-EQ net Sharpe − Family B net Sharpe ≥ 0.5`.
6. S1-EQ net Sharpe **>** SPY buy-and-hold Sharpe over the same holdout
   window. **If SPY Sharpe is negative, criterion (4) — standalone
   Sharpe ≥ 0.7 — is the binding requirement;** the comparison to a
   negative SPY Sharpe is reported but does not lower the bar.
7. **Max drawdown ≥ −25 %** (i.e., not worse than −25 %).
8. **Borrow stress:** at borrow ×2.0 net Sharpe still positive; at
   borrow ×3.0 net total return still positive (annualized).
9. **JS-overlay does NOT beat S1-EQ** on net Sharpe — confirms retraining
   helped.
10. **Rolling-window CV diagnostic** shows S1-EQ alpha is not regime-
    concentrated to a single 2-year window.
11. **Concentration check (all three required):**
    - No single stock contributes > 25 % of total net PnL.
    - No single calendar month contributes > 35 % of total net PnL.
    - No single sector contributes > 50 % of total net PnL **unless
      explicitly flagged and justified** in the report.
12. All mandatory CI tests (§6.2) green.
13. All artifacts present per §4.11 and §5.17; all hashes verify.

If any of (2)-(11) fail, S1-EQ is **not promoted**.

### 6.5 Negative-result handling (mandatory)

If the M6 gate fails, the report's "Iteration plan" section must:

- Identify which gate criterion failed (specific metric and threshold).
- Propose **one** actionable hypothesis class for the next round
  (feature, data, hyperparam, or cost-model change). Multi-direction
  iteration is forbidden inside a single run cycle.
- Be reviewed before any subsequent training run is initiated.

This prevents the "tweak everything and re-run" anti-pattern that
silently destroys statistical validity.

### 6.6 No in-process self-promotion

Even if every criterion in §6.4 passes, promotion to S2 (LLM governor)
or S4 (execution paper stage) requires a two-person review and a signed
commit per CLAUDE.md §11. The success gate in this spec is the
**technical prerequisite**, not the promotion itself.

### 6.7 Focused-basket versioning

`configs/eq_focused_basket.yaml` carries a `version` field and a
hash. The hash is recorded in `metadata.json` for every run. Any change
to the basket constituents requires a version bump and produces a new
`run_id`; otherwise focused-basket results are not reproducible.

### 6.8 Audit-log integration

Every backtest run appends to `logs/audit/equity_backtest/<run_id>.jsonl`.
`audit_replay_check.py` is extended to verify equity backtest replay
byte-identically (within the numerical tolerance from §4.9).

---

## 7. Out of scope (explicit)

- **HFT / orderbook / tick / intraday** — deferred to
  `S5 Microstructure & HFT`. Requires real orderbook reconstruction,
  queue position, latency modeling, partial fills.
- **News sentiment, fundamentals (P/E, P/B, earnings)** — deferred to
  `S1-EQ-FUNDAMENTALS` / `S1-EQ-ALT-DATA`. Each requires point-in-time
  vendor data, revision-history handling, announce-timestamp alignment,
  and leakage-controlled lagging; these are independent data-engineering
  subprojects unrelated to microstructure.
- **Options and futures** — deferred. Separate asset-class pricing and
  cost models.
- **Market-impact model, MOC auction slippage, sector/factor
  neutrality, real PIT borrow feed** — deferred to a Phase-2 strict-
  backtest spec.

---

## 8. Open implementation questions for the writing-plans phase

These are not design decisions but implementation choices the
`superpowers:writing-plans` phase should resolve:

1. Exact PIT-membership dataset to attempt first (HF candidates listed
   in §2.2; the audit script picks one based on availability at
   pipeline-build time).
2. Whether the existing `alpha/meta_features.py` extractor can be reused
   directly behind the §3.3-7 audit gate, or whether a new
   `alpha_eq/features/meta_features.py` wrapper is needed.
3. Exact synthetic-equity fixture for `test_e2e_smoke.py` — a slice of
   the real S&P 500 panel filtered to a small symbol subset, or a
   purely synthetic GBM-driven panel.
4. Whether `prepare_equity_data.py` should be one script with subcommands
   or one script per artifact.

These are documented here so the implementation plan covers them
explicitly rather than picking silently.

---

## 9. Disclaimer

This spec is a research and engineering plan. Nothing in this document
or in the artifacts produced from it constitutes investment advice. The
strict backtest engine and success gate exist to **reduce the gap**
between reported and realised performance, not to guarantee it. Real
trading additionally requires the S4 execution + risk + promotion
machinery defined in `2026-05-20-quantlab-alpha-s4-execution-risk-
promotion-design.md`, including the operator-controlled `QUANTLAB_STAGE`
environment variable and the kill-switch protocol.

`not_investment_advice: true`
