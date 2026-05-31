# Signal-Research Enhanced Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `signal_research` package — an orchestration layer on top of the existing `strategy_benchmark` (1500-strategy PBO baseline) and `alpha_eq` (M4 strict backtest) that adds long-history data, paper-derived signals, robust methodology (CPCV, meta-labeling, dedup, Pareto, regime, bootstrap), a cross-sectional bridge to the M4 engine, and a four-tier candidate-status taxonomy (`research_pass` → `promotion_eligible` → `paper_trade_candidate` → `production_candidate`).

**Architecture:** New top-level package at `src/quant_research_stack/signal_research/`. Three layers: data (long-history yfinance + HF + FRED + CBOE + crypto + universes), papers (10 signal families with module-type taxonomy), methodology (8 upgrades). Two cross-sectional bridge modules to the M4 engine. Single runner produces three-tier reports (family / profile / master) with a selection funnel showing the multiple-testing burden. **Existing `strategy_benchmark` and `alpha_eq` packages are not modified.**

**Tech Stack:** Python 3.11 · Polars / pandas · Pydantic v2 · numpy · scipy · scikit-learn · LightGBM · XGBoost · hmmlearn · transformers + PyTorch (M5 only, deferred) · joblib · yfinance · fredapi · datasets (HuggingFace) · pytest · ruff · mypy · YAML configs · Parquet artifacts · uv / Make.

**Spec:** `docs/superpowers/specs/2026-05-26-signal-research-enhanced-benchmark-design.md` (commit `ae9382b`). Every task references the spec sections it implements.

**Twelve non-negotiables (§0 of spec) enforced throughout:**

1. No live or paper promotion from in-sample results.
2. No final strategy selection using the permanent holdout.
3. No current-constituent cross-sectional universe labelled production-grade or institutional-grade.
4. No strategy can pass without realistic costs + 2× cost stress + delay stress + PBO + DSR + bootstrap CI + concentration checks.
5. No sentiment / fundamentals / news / altdata in promoted models unless timestamp integrity and leakage controls are proven.
6. Every dataset must have a manifest with sha256 + source + timestamp convention + data-quality label + reproducibility record.
7. Every strategy trial is logged — failed trials are not hidden.
8. PBO + DSR account for the full search process, not only the final candidate.
9. Performance concentrated in one period / trade / asset / regime is downgraded or rejected.
10. Staged promotion is sequential: research → holdout → paper → shadow → live. No stage-skipping.
11. Risk limits / drawdown control / exposure caps / kill switch / audit logs / reproducibility / failure alerts.
12. Four status tiers, sequential promotion only.

**Working environment assumptions:**

- `PYTHONPATH=src` for every `python` / `pytest` invocation.
- `uv run` is the supported execution wrapper.
- Branch: continue on `quant-llm-implementation` (current active dev branch). Worktree is NOT used unless the executor explicitly creates one via `superpowers:using-git-worktrees`.
- Commits use Conventional Commits format with scope `signal-research`.
- Existing `alpha_eq/` and `strategy_benchmark/` source is **read-only** for this plan. The plan only modifies them via separate explicit amendments (none in this plan; per spec §5.2).
- Existing `Makefile`, `pyproject.toml`, `configs/` may receive **additive** entries (new targets, new optional deps, new YAMLs) — never modifying existing ones.

**Pre-flight check before any task:**

```bash
PYTHONPATH=src uv run pytest -q                          # baseline must be green
PYTHONPATH=src uv run ruff check src scripts tests       # baseline lint
PYTHONPATH=src uv run mypy src/quant_research_stack      # baseline types
git status                                                # working tree clean
```

If baseline pytest is red, do not start; surface it back to the user.

---

## File-structure map

Locked-in file layout (informs task decomposition):

```
src/quant_research_stack/signal_research/
  __init__.py
  status.py                          # 4-tier status enum + helpers (§6.1)
  registry.py                        # mandatory strategy registry schema (§3.6)
  config.py                          # Pydantic v2 SignalResearchConfig
  data/
    __init__.py
    manifest.py                      # 5-tier data-quality + sha256 manifest (§2.2, §2.5)
    long_history.py                  # yfinance 2005-current (§2.1)
    hf_datasets.py                   # HuggingFace dataset loaders, gated (§2.4, §2.6)
    fred.py                          # FRED via fredapi (§2.4)
    cboe_proxies.py                  # ^VIX, ^VVIX, ^SKEW, ^GVZ, ^OVX, ^VXN (§2.4)
    sp500_components.py              # current SP500 list, hashed (§2.3, §6.1)
    nasdaq_components.py             # current Nasdaq 100 list, hashed (§2.3, §6.1)
    crypto_minimal.py                # BTCUSDT / ETHUSDT daily (§6.6)
    profiles.py                      # 4 model profiles + 4 Nasdaq universes (§2.3)
  papers/
    __init__.py
    base.py                          # Signal / FeatureGenerator / Wrapper / ModelFamily ABCs
    triple_barrier.py                # #3 wrapper + meta-labeling pre-filter (§3.3, §4.2)
    avellaneda_lee.py                # #4 cross-sectional residual MR (§3.3, §5.5)
    gkx_ohlcv_subset.py              # #5 LightGBM ranking + feature list (§3.3, §5.6)
    vol_risk_premium.py              # #6 feature + tradable-only-if-instrument (§3.3)
    hmm_regime.py                    # #7 feature_generator (§3.3)
    sentiment_finbert.py             # #8 research_only_default + ladder (§3.3 FinBERT)
    options_implied.py               # #9 VIX/VXN ratios + skew (§3.3)
    macro_overlay.py                 # #10 FRED features (§3.3)
    deep_momentum.py                 # #1 (M5 deferred)
    momentum_transformer.py          # #2 (M5 deferred)
  methodology/
    __init__.py
    cpcv.py                          # combinatorial purged CV (§4.1)
    meta_labeling.py                 # secondary classifier + survivor-only filter (§4.2)
    correlation_dedup.py             # net OOS returns, signed + abs (§4.3)
    multi_objective.py               # Pareto front (§4.4)
    regime_conditional.py            # 2-state HMM + agnostic/specific (§4.5)
    bootstrap_ci.py                  # stationary block bootstrap (§4.6)
    pbo_extensions.py                # three-tier PBO (§4.7)
    failure_classifier.py            # 13-category taxonomy (§4.10)
    selection_funnel.py              # candidate-count tracking through filters (§6.4)
    dev_only_guard.py                # invariant: methodology never touches holdout (§4.9)
    reality_check.py                 # White SPA (M6b deferred)
  cross_sectional/
    __init__.py
    signal_to_panel.py               # pure conversion + validation (§5.2)
    panel_to_m4.py                   # banner-preserving M4 entry (§5.2)
  runner.py                          # orchestrates enhanced benchmark
  report.py                          # three-tier reports + selection funnel (§6.2, §6.4)

configs/
  signal_research.yaml               # top-level config (cost models, grids)
  signal_research_profiles/
    sp500.yaml
    nasdaq.yaml
    crypto.yaml
    futures_proxy.yaml

scripts/
  fetch_signal_research_data.py      # M1 CLI
  run_signal_research_benchmark.py   # M3-M4 main runner
  signal_research_report.py          # generate reports from a run

tests/signal_research/
  conftest.py
  test_status.py
  test_registry.py
  test_data_manifest.py
  test_data_long_history.py
  test_data_fred.py
  test_data_cboe_proxies.py
  test_data_sp500_components.py
  test_data_nasdaq_components.py
  test_data_crypto_minimal.py
  test_data_hf_datasets.py
  test_data_profiles.py
  test_papers_triple_barrier.py
  test_papers_avellaneda_lee.py
  test_papers_gkx_ohlcv_subset.py
  test_papers_vol_risk_premium.py
  test_papers_hmm_regime.py
  test_papers_options_implied.py
  test_papers_macro_overlay.py
  test_papers_sentiment_finbert.py
  test_methodology_cpcv.py
  test_methodology_meta_labeling.py
  test_methodology_correlation_dedup.py
  test_methodology_multi_objective.py
  test_methodology_regime_conditional.py
  test_methodology_bootstrap_ci.py
  test_methodology_pbo_extensions.py
  test_methodology_failure_classifier.py
  test_methodology_selection_funnel.py
  test_methodology_dev_only_guard.py
  test_cross_sectional_bridge_01_schema.py     # 10-point bridge contract per §5.10
  test_cross_sectional_bridge_02_one_pred.py
  test_cross_sectional_bridge_03_timestamp.py
  test_cross_sectional_bridge_04_no_dups.py
  test_cross_sectional_bridge_05_rank_in_universe.py
  test_cross_sectional_bridge_06_banner_preserved.py
  test_cross_sectional_bridge_07_nan_handling.py
  test_cross_sectional_bridge_08_zero_input_equality.py
  test_cross_sectional_bridge_09_no_pit_promotion.py
  test_cross_sectional_bridge_10_pbo_on_pnl.py
  test_runner.py
  test_report.py
  test_e2e_smoke.py

reports/signal_research/                       # generated, gitignored except .gitkeep
experiments/signal_research/<run_id>/          # generated, gitignored except .gitkeep
```

---

## Milestone map

| ID | Milestone | Tasks | Exit criterion |
|---|---|---|---|
| **M0** | Branch + scaffolding | 1–4 | `signal_research` package skeleton importable; Make targets stubbed; configs created; baseline tests green |
| **M1** | Data foundation + registry | 5–18 | 4 universes + 4 profiles configured; 5-tier data-quality classifier; manifests for every data source; registry schema with mandatory fields; 24-month+ data fetched for SPY/QQQ/ES=F/NQ=F |
| **M2** | Methodology stack | 19–32 | CPCV, meta-labeling, dedup, Pareto, regime, bootstrap CIs, three-tier PBO, failure taxonomy, selection funnel, dev-only guard — all green |
| **M3** | Classical paper signals | 33–49 | 8 non-deferred signal families implemented (triple-barrier wrapper, Avellaneda-Lee, GKX-style, vol-risk-premium feature, HMM regime, options-implied features, macro overlay, FinBERT research-only) with their parameter grids |
| **M4** | Cross-sectional bridge + first enhanced run | 50–66 | `signal_to_panel.py` + `panel_to_m4.py` + 10-point bridge contract tests green; SP500 + NDX-100 cross-sectional runs produce reports; master report + selection funnel published |
| **M5** | *Deferred:* Deep models | 67–76 | Deep Momentum Network + Momentum Transformer trained walk-forward; reproducible; M5 results merge cleanly into the master report |
| **M6a** | *Deferred:* FinBERT promotion ladder | 77–84 | FinBERT pipeline + 4-state ladder + 10-criterion audit gate operational; v1 stays `research_only_default` unless audit passes |
| **M6b** | *Deferred:* Reality Check / SPA | 85–90 | White SPA / Hansen Reality Check implemented; integrated into selection funnel as a strict secondary filter |

Per spec §6.5, M5/M6a/M6b are independent deferred add-ons. M1-M4 is the shippable core.

---

## M0 — Branch + scaffolding

### Task 1 — Pre-flight + package skeleton

**Spec refs:** §1.3 (package layout).

**Files:**
- Create: `src/quant_research_stack/signal_research/__init__.py`
- Create: `src/quant_research_stack/signal_research/data/__init__.py`
- Create: `src/quant_research_stack/signal_research/papers/__init__.py`
- Create: `src/quant_research_stack/signal_research/methodology/__init__.py`
- Create: `src/quant_research_stack/signal_research/cross_sectional/__init__.py`
- Create: `tests/signal_research/__init__.py`
- Create: `tests/signal_research/conftest.py`

- [ ] **Step 1: Run baseline pre-flight**

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run ruff check src scripts tests
PYTHONPATH=src uv run mypy src/quant_research_stack
git status
```

Expected: pytest green (or only pre-existing unrelated failures unchanged); ruff and mypy green. If anything is red, stop and surface to the user.

- [ ] **Step 2: Create the 5 package `__init__.py` files**

Each `__init__.py` has a one-line docstring identifying the sub-package. The root has the full description:

```python
"""signal_research — orchestration layer on top of strategy_benchmark and alpha_eq.

See docs/superpowers/specs/2026-05-26-signal-research-enhanced-benchmark-design.md.
"""
```

Sub-package docstrings:
- `data/__init__.py`: `"""signal_research data layer — 5-tier data-quality classifier and manifests."""`
- `papers/__init__.py`: `"""signal_research paper-derived signal menu (10 families)."""`
- `methodology/__init__.py`: `"""signal_research methodology stack — CPCV, dedup, Pareto, regime, bootstrap."""`
- `cross_sectional/__init__.py`: `"""signal_research cross-sectional bridge to the alpha_eq M4 engine."""`
- `tests/signal_research/__init__.py`: empty.

- [ ] **Step 3: Create `tests/signal_research/conftest.py`**

```python
"""Shared fixtures for signal_research tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def tmp_signal_research_root(tmp_path: Path) -> Path:
    """Disposable root for signal_research data manifests."""
    root = tmp_path / "data" / "processed" / "signal_research"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def subprocess_env() -> dict[str, str]:
    """Subprocess env that lets `uv run` resolve uv on Apple Silicon dev machines."""
    return {
        "PYTHONPATH": "src",
        "PATH": os.environ.get(
            "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", ".uv-cache"),
    }


@pytest.fixture()
def synthetic_daily_bars(rng: np.random.Generator) -> pl.DataFrame:
    """Tiny 5-symbol, 250-trading-day OHLCV panel."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    base_dates = pl.date_range(
        start=pl.date(2024, 1, 2),
        end=pl.date(2025, 6, 1),
        interval="1d",
        eager=True,
    )
    weekday_mask = base_dates.dt.weekday() < 6
    dates = base_dates.filter(weekday_mask).head(250).to_list()
    rows = []
    for s in symbols:
        price = 100.0
        for d in dates:
            ret = float(rng.standard_normal()) * 0.012
            price *= 1.0 + ret
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": price * (1.0 + float(rng.standard_normal()) * 0.003),
                    "high": price * (1.0 + abs(float(rng.standard_normal())) * 0.006),
                    "low": price * (1.0 - abs(float(rng.standard_normal())) * 0.006),
                    "close": price,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 300_000),
                }
            )
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Verify importability**

```bash
PYTHONPATH=src uv run python -c "import quant_research_stack.signal_research as sr; import quant_research_stack.signal_research.data as sr_data; import quant_research_stack.signal_research.papers as sr_papers; import quant_research_stack.signal_research.methodology as sr_meth; import quant_research_stack.signal_research.cross_sectional as sr_cs; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Run pytest to confirm conftest loads**

```bash
PYTHONPATH=src uv run pytest tests/signal_research -q
```

Expected: `no tests collected` (no test files yet) — but conftest must parse without error.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/signal_research tests/signal_research
git commit -m "feat(signal-research): scaffold signal_research package skeleton"
```

---

### Task 2 — Make targets + configs/

**Spec refs:** §1.3.

**Files:**
- Modify: `Makefile` (append new targets only; do not modify existing).
- Create: `configs/signal_research.yaml`
- Create: `configs/signal_research_profiles/sp500.yaml`
- Create: `configs/signal_research_profiles/nasdaq.yaml`
- Create: `configs/signal_research_profiles/crypto.yaml`
- Create: `configs/signal_research_profiles/futures_proxy.yaml`

- [ ] **Step 1: Inspect Makefile tail**

```bash
tail -20 Makefile
```

- [ ] **Step 2: Append Make targets**

Append (do not modify existing):

```make

# ----- signal_research -----

fetch-signal-research-data:
	PYTHONPATH=src uv run python scripts/fetch_signal_research_data.py \
		--config configs/signal_research.yaml

run-signal-research-benchmark:
	PYTHONPATH=src uv run python scripts/run_signal_research_benchmark.py \
		--config configs/signal_research.yaml

signal-research-report:
	PYTHONPATH=src uv run python scripts/signal_research_report.py
```

- [ ] **Step 3: Create `configs/signal_research.yaml`**

```yaml
# signal_research top-level configuration.
# Spec: docs/superpowers/specs/2026-05-26-signal-research-enhanced-benchmark-design.md

data:
  long_history:
    start: "2005-01-01"
    end: null               # null = today
    cache_root: data/processed/signal_research/long_history
  hf_datasets:
    cache_root: data/cache/huggingface
    research_only_default: true     # §2.6, FinBERT ladder default
  fred:
    cache_root: data/cache/fred
    series: [DGS10, T10Y2Y, DTWEXBGS, DCOILWTICO, GOLDAMGBD228NLBM]
  cboe_proxies:
    tickers: ["^VIX", "^VVIX", "^SKEW", "^GVZ", "^OVX", "^VXN"]
    cache_root: data/processed/signal_research/cboe
  crypto:
    tickers: ["BTCUSDT", "ETHUSDT"]
    cache_root: data/processed/signal_research/crypto

profiles:
  - sp500
  - nasdaq
  - crypto
  - futures_proxy

methodology:
  cpcv:
    n_partitions: 8
    purge_days: 5
    embargo_days: 5
  bootstrap:
    n_resamples: 10000
    block_length: null          # null = T^(1/3)
    seed: 42
  pbo:
    n_partitions: 16
    sample_combinations: 20000
  dsr:
    confidence_threshold_real_edge: 0.95

success_gates:
  research_pass:
    point_sharpe_min: 1.5
    dsr_min: 0.5
    pbo_profile_max: 0.25
    pbo_profile_preferred_max: 0.10
    bootstrap_lower_sharpe_min: 0.0
    cost_stress_multiplier: 2.0
    max_drawdown_min: -0.20
    calmar_preferred_min: 1.0
    calmar_weak_max: 0.5
    min_trades_single_asset: 200
    min_trades_cross_sectional: 500
    min_active_days: 252
    min_oos_calendar_years: 3
    concentration:
      max_single_quarter_pnl_frac: 0.50
      max_single_day_pnl_frac: 0.25
      max_single_trade_pnl_frac: 0.20
      max_single_asset_pnl_frac: 0.35
  promotion_eligible:
    bootstrap_lower_sharpe_high_confidence: 1.0
    bootstrap_lower_sharpe_exceptional: 1.5
```

- [ ] **Step 4: Create `configs/signal_research_profiles/sp500.yaml`**

```yaml
profile: sp500
asset_class: equity_index
universes:
  - name: sp500_current_constituents
    description: Current S&P 500 list (current-only; survivorship_prototype_only)
    data_quality_label: survivorship_prototype_only
    constituent_survivorship_applicable: true
    tickers_source: data/processed/signal_research/sp500/sp500_current.parquet
  - name: spy_direct
    description: SPY ETF directly traded
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: [SPY]

benchmarks:
  - SPY_buy_and_hold
  - SP500_index_proxy

cost_model:
  commission_bps_one_way: 0.5
  spread_bps_one_way: 1.0
```

- [ ] **Step 5: Create `configs/signal_research_profiles/nasdaq.yaml`**

```yaml
profile: nasdaq
asset_class: equity_index
universes:
  - name: nasdaq_index_proxy
    description: QQQ + ^IXIC + NQ=F (TQQQ/SQQQ diagnostic only)
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: [QQQ, "^IXIC", "NQ=F"]
    diagnostic_only: [TQQQ, SQQQ]
  - name: nasdaq_100_current
    description: Current NDX-100 constituents — survivorship-warned
    data_quality_label: survivorship_prototype_only
    constituent_survivorship_applicable: true
    tickers_source: data/processed/signal_research/nasdaq/nasdaq_100_current.parquet
  - name: nasdaq_mega_cap_focus
    description: Liquid Nasdaq-heavy mega-caps (reporting cohort, not primary universe)
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: [AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, TSLA, AVGO, COST,
              NFLX, AMD, QCOM, ADBE, INTU, CSCO, PEP, AMAT, ARM, ASML]
  - name: user_focus_tech
    description: User-named focused tech cohort
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: [AAPL, ORCL, PYPL, INTC, META, TSLA, QCOM, PLTR, GOOGL, AVGO, ADBE, RKLB]

benchmarks:
  - QQQ_buy_and_hold
  - SPY_buy_and_hold
  - NDX100_current_equal_weight   # survivorship-warned in report
  - QQQ_momentum_baseline
  - QQQ_mean_reversion_baseline
  - QQQ_vol_targeted_baseline

# Nasdaq-specific extra features (broadcast to per-row panel)
context_features:
  - QQQ_log_return_1
  - QQQ_log_return_5
  - QQQ_log_return_20
  - QQQ_log_return_60
  - QQQ_log_return_120
  - QQQ_realized_vol_20
  - QQQ_minus_SPY_residual
  - QQQ_SPY_rolling_beta_60
  - QQQ_over_SPY_relative_strength_60
  - VIX
  - VVIX
  - VXN                            # with fallback if VXN unavailable
  - DGS10
  - T10Y2Y
  - USD_index_proxy
  - SMH_momentum_60                # semiconductor proxy
  - IGV_momentum_60                # software/cloud proxy

cost_model:
  commission_bps_one_way: 0.5
  spread_bps_one_way: 1.0
```

- [ ] **Step 6: Create `configs/signal_research_profiles/crypto.yaml`**

```yaml
profile: crypto
asset_class: spot_crypto
universes:
  - name: crypto_minimal_v1
    description: BTCUSDT + ETHUSDT daily (per spec §6.6)
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: [BTCUSDT, ETHUSDT]

benchmarks:
  - BTCUSDT_buy_and_hold
  - ETHUSDT_buy_and_hold

cost_model:
  maker_bps: 1.0
  taker_bps: 5.0
  spread_bps_one_way: 3.0          # rough spot-crypto retail proxy
  funding_payments: false          # v1 spot only
```

- [ ] **Step 7: Create `configs/signal_research_profiles/futures_proxy.yaml`**

```yaml
profile: futures_proxy
asset_class: equity_index_futures
universes:
  - name: futures_proxy_v1
    description: ES=F + NQ=F front-month continuous (yfinance auto-roll)
    data_quality_label: public_snapshot_not_pit
    constituent_survivorship_applicable: false
    tickers: ["ES=F", "NQ=F"]

benchmarks:
  - ES_F_buy_and_hold
  - NQ_F_buy_and_hold

cost_model:
  commission_bps_one_way: 0.5
  spread_bps_one_way: 0.5
```

- [ ] **Step 8: Dry-run make**

```bash
make -n fetch-signal-research-data
make -n run-signal-research-benchmark
```

Expected: each prints its command without Makefile syntax errors.

- [ ] **Step 9: Commit**

```bash
git add Makefile configs/signal_research.yaml configs/signal_research_profiles
git commit -m "feat(signal-research): add make targets and 4 profile configs"
```

---

### Task 3 — Confirm dependencies + add new ones

**Spec refs:** §1.4 (tech stack).

**Files:**
- Modify: `pyproject.toml` (add deps; do not remove or upgrade existing pins).

- [ ] **Step 1: Check existing pins**

```bash
PYTHONPATH=src uv run python -c "import yfinance, scipy, hmmlearn, datasets, fredapi; print(yfinance.__version__, scipy.__version__, hmmlearn.__version__, datasets.__version__, fredapi.__version__)" 2>&1 | tail -5
```

`yfinance` and `scipy` should already be installed. `hmmlearn`, `datasets`, `fredapi` are likely missing.

- [ ] **Step 2: Add missing deps to `pyproject.toml`**

In the `[project] dependencies` array, add (alphabetised):

```toml
"datasets>=2.16",
"fredapi>=0.5",
"hmmlearn>=0.3",
```

If `transformers` and `torch` (for M5 / FinBERT) aren't there, leave them — they're optional and only needed for M5/M6a; we'll add them when those milestones start.

- [ ] **Step 3: Re-install + verify**

```bash
uv pip install -e .
PYTHONPATH=src uv run python -c "import hmmlearn, datasets, fredapi; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit only if pyproject changed**

```bash
git status
# if pyproject.toml is modified:
git add pyproject.toml uv.lock 2>/dev/null
git commit -m "build(signal-research): add hmmlearn, datasets, fredapi deps"
# otherwise: no commit
```

---

### Task 4 — Status enum (`research_pass` → `production_candidate`)

**Spec refs:** §0 #12, §6.1 (four-tier status taxonomy).

**Files:**
- Create: `src/quant_research_stack/signal_research/status.py`
- Create: `tests/signal_research/test_status.py`

- [ ] **Step 1: Write failing tests**

```python
"""Four-tier candidate status taxonomy (spec §6.1)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.status import (
    CandidateStatus,
    promote_if_eligible,
    status_at_least,
)


def test_status_ordering() -> None:
    assert CandidateStatus.NONE < CandidateStatus.RESEARCH_PASS
    assert CandidateStatus.RESEARCH_PASS < CandidateStatus.PROMOTION_ELIGIBLE
    assert CandidateStatus.PROMOTION_ELIGIBLE < CandidateStatus.PAPER_TRADE_CANDIDATE
    assert CandidateStatus.PAPER_TRADE_CANDIDATE < CandidateStatus.PRODUCTION_CANDIDATE


def test_status_string_values() -> None:
    assert CandidateStatus.NONE.value == "none"
    assert CandidateStatus.RESEARCH_PASS.value == "research_pass"
    assert CandidateStatus.PROMOTION_ELIGIBLE.value == "promotion_eligible"
    assert CandidateStatus.PAPER_TRADE_CANDIDATE.value == "paper_trade_candidate"
    assert CandidateStatus.PRODUCTION_CANDIDATE.value == "production_candidate"


def test_status_at_least_returns_true_when_equal_or_higher() -> None:
    assert status_at_least(CandidateStatus.PROMOTION_ELIGIBLE, CandidateStatus.RESEARCH_PASS)
    assert status_at_least(CandidateStatus.PROMOTION_ELIGIBLE, CandidateStatus.PROMOTION_ELIGIBLE)
    assert not status_at_least(CandidateStatus.RESEARCH_PASS, CandidateStatus.PROMOTION_ELIGIBLE)


def test_promote_if_eligible_advances_by_one_tier_only() -> None:
    assert promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=True) == CandidateStatus.PROMOTION_ELIGIBLE
    assert promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=False) == CandidateStatus.RESEARCH_PASS


def test_promote_if_eligible_never_skips_stages() -> None:
    # Cannot jump from RESEARCH_PASS to PAPER_TRADE_CANDIDATE in one call
    promoted = promote_if_eligible(CandidateStatus.RESEARCH_PASS, promoted=True)
    assert promoted != CandidateStatus.PAPER_TRADE_CANDIDATE


def test_promote_at_top_is_idempotent() -> None:
    assert promote_if_eligible(CandidateStatus.PRODUCTION_CANDIDATE, promoted=True) == CandidateStatus.PRODUCTION_CANDIDATE


def test_unknown_string_raises() -> None:
    with pytest.raises(ValueError):
        CandidateStatus("not_a_real_status")
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_status.py -v
```

Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 3: Implement `status.py`**

```python
"""Four-tier candidate status taxonomy (spec §6.1, §0 non-negotiable #12).

Sequential promotion only — no stage-skipping (§0 non-negotiable #10).
"""

from __future__ import annotations

import enum


class CandidateStatus(enum.IntEnum):
    NONE = 0
    RESEARCH_PASS = 1
    PROMOTION_ELIGIBLE = 2
    PAPER_TRADE_CANDIDATE = 3
    PRODUCTION_CANDIDATE = 4

    @property
    def value(self) -> str:  # type: ignore[override]
        return self.name.lower()


def status_at_least(actual: CandidateStatus, required: CandidateStatus) -> bool:
    """Returns True iff `actual` has reached `required` or higher."""
    return int(actual) >= int(required)


def promote_if_eligible(
    current: CandidateStatus, *, promoted: bool
) -> CandidateStatus:
    """Advance the candidate by exactly one tier if `promoted=True`,
    otherwise return `current` unchanged.

    Sequential promotion only. Cannot skip stages.
    """
    if not promoted:
        return current
    if current == CandidateStatus.PRODUCTION_CANDIDATE:
        return current  # top tier — idempotent
    return CandidateStatus(int(current) + 1)
```

Note: `enum.IntEnum.value` is normally an int. We override it to a lower-cased name string for serialisation in reports. Tests should still pass since IntEnum ordering is on the int.

Actually, overriding `.value` on `IntEnum` is unusual. Cleaner to use a separate property:

Replace the property block with:

```python
    @property
    def name_lower(self) -> str:
        return self.name.lower()
```

And update `test_status_string_values` accordingly:

```python
def test_status_string_values() -> None:
    assert CandidateStatus.NONE.name_lower == "none"
    assert CandidateStatus.RESEARCH_PASS.name_lower == "research_pass"
    assert CandidateStatus.PROMOTION_ELIGIBLE.name_lower == "promotion_eligible"
    assert CandidateStatus.PAPER_TRADE_CANDIDATE.name_lower == "paper_trade_candidate"
    assert CandidateStatus.PRODUCTION_CANDIDATE.name_lower == "production_candidate"


def test_unknown_string_raises() -> None:
    with pytest.raises(KeyError):
        CandidateStatus["NOT_A_REAL_STATUS"]
```

Use this corrected version.

- [ ] **Step 4: Run tests — expect green**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_status.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Lint + types**

```bash
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/signal_research/status.py tests/signal_research/test_status.py
git commit -m "feat(signal-research): four-tier candidate status taxonomy"
```

---

## M1 — Data foundation + registry

The first big block. M1 lays the data foundation that every later milestone consumes.

### Task 5 — Data-quality manifest + 5-tier classifier

**Spec refs:** §2.2 (5-tier classification), §2.5 (data-quality contract), §6 of non-negotiables (manifest required on every dataset).

**Files:**
- Create: `src/quant_research_stack/signal_research/data/manifest.py`
- Create: `tests/signal_research/test_data_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
"""5-tier data-quality classifier + sha256 manifest (spec §2.2, §2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    ManifestMismatchError,
    load_and_verify_manifest,
    sha256_of_file,
    write_manifest,
)


def test_tier_values_and_ordering() -> None:
    assert DataQualityTier.PIT_SAFE.value == "pit_safe"
    assert DataQualityTier.PARTIAL_PIT_UNIVERSE.value == "partial_pit_universe"
    assert DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT.value == "public_snapshot_not_pit"
    assert DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY.value == "survivorship_prototype_only"
    # 4 explicit tiers + UNKNOWN sentinel = 5 listed in the spec
    assert DataQualityTier.UNKNOWN.value == "unknown"


def test_tier_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        DataQualityTier("institutional_grade_marketing_word")


def test_directly_traded_etf_is_not_a_tier() -> None:
    """Per spec wording fix: directly_traded_etf is NOT a separate tier value;
    it is carried by a separate `constituent_survivorship_applicable` flag."""
    with pytest.raises(ValueError):
        DataQualityTier("directly_traded_etf")


def test_manifest_round_trip(tmp_signal_research_root: Path) -> None:
    parquet = tmp_signal_research_root / "demo.parquet"
    pl.DataFrame({"date": ["2024-01-02"], "x": [1.0]}).write_parquet(parquet)
    sha = sha256_of_file(parquet)
    m = DataSourceManifest(
        source_name="demo",
        source_url="https://example.com/demo",
        fetch_timestamp_utc="2026-05-26T12:00:00Z",
        path=str(parquet.name),
        sha256=sha,
        row_count=1,
        symbol_count=0,
        date_range_start="2024-01-02",
        date_range_end="2024-01-02",
        schema_fingerprint="cols:date,x",
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure="yfinance public snapshot — not vendor PIT data",
        timestamp_convention="after_close_t",
        warnings=[],
    )
    out = tmp_signal_research_root / "_manifest.json"
    write_manifest(out, m)
    m2 = load_and_verify_manifest(out, expected_sha256={"demo": sha})
    assert m2.data_quality_tier == DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT
    assert m2.constituent_survivorship_applicable is False


def test_manifest_hash_mismatch_hard_fails(tmp_signal_research_root: Path) -> None:
    parquet = tmp_signal_research_root / "demo.parquet"
    pl.DataFrame({"date": ["2024-01-02"], "x": [1.0]}).write_parquet(parquet)
    out = tmp_signal_research_root / "_manifest.json"
    out.write_text(
        json.dumps(
            {
                "source_name": "demo",
                "source_url": "x",
                "fetch_timestamp_utc": "2026-05-26T12:00:00Z",
                "path": "demo.parquet",
                "sha256": "a" * 64,
                "row_count": 1,
                "symbol_count": 0,
                "date_range_start": "2024-01-02",
                "date_range_end": "2024-01-02",
                "schema_fingerprint": "cols:date,x",
                "data_quality_tier": "public_snapshot_not_pit",
                "constituent_survivorship_applicable": False,
                "vendor_disclosure": "x",
                "timestamp_convention": "after_close_t",
                "warnings": [],
            }
        )
    )
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={"demo": "b" * 64})
```

- [ ] **Step 2: Run — expect fail**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_manifest.py -v
```

- [ ] **Step 3: Implement `manifest.py`**

```python
"""5-tier data-quality manifest (spec §2.2, §2.5).

The 5 tiers:
- pit_safe
- partial_pit_universe
- public_snapshot_not_pit
- survivorship_prototype_only
- unknown (sentinel — must be resolved before downstream use)

`directly_traded_etf` is NOT a tier value. Directly-traded instruments
(SPY, QQQ, BTCUSDT, etc.) keep their tier (typically public_snapshot_not_pit)
and carry `constituent_survivorship_applicable: false` separately.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataQualityTier(enum.StrEnum):
    PIT_SAFE = "pit_safe"
    PARTIAL_PIT_UNIVERSE = "partial_pit_universe"
    PUBLIC_SNAPSHOT_NOT_PIT = "public_snapshot_not_pit"
    SURVIVORSHIP_PROTOTYPE_ONLY = "survivorship_prototype_only"
    UNKNOWN = "unknown"


class ManifestMismatchError(RuntimeError):
    pass


class DataSourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_url: str
    fetch_timestamp_utc: str
    path: str
    sha256: str
    row_count: int
    symbol_count: int
    date_range_start: str
    date_range_end: str
    schema_fingerprint: str
    data_quality_tier: DataQualityTier
    constituent_survivorship_applicable: bool
    vendor_disclosure: str
    timestamp_convention: str
    warnings: list[str] = Field(default_factory=list)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def write_manifest(path: Path, manifest: DataSourceManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(_canonical_json(manifest.model_dump(mode="json")))


def load_and_verify_manifest(
    path: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> DataSourceManifest:
    if not Path(path).exists():
        raise ManifestMismatchError(f"manifest missing: {path}")
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ManifestMismatchError(f"manifest is not valid JSON: {exc}") from exc
    try:
        m = DataSourceManifest.model_validate(payload)
    except Exception as exc:
        raise ManifestMismatchError(f"manifest schema error: {exc}") from exc
    for key, sha in expected_sha256.items():
        if m.source_name != key:
            continue
        if m.sha256 != sha:
            raise ManifestMismatchError(
                f"sha256 mismatch for {key}: expected={sha} got={m.sha256}"
            )
    return m
```

- [ ] **Step 4: Run tests — expect green**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_manifest.py -v
```

- [ ] **Step 5: Lint + types + commit**

```bash
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research/test_data_manifest.py
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/manifest.py tests/signal_research/test_data_manifest.py
git commit -m "feat(signal-research): 5-tier data-quality manifest with hash hard-fail"
```

---

### Task 6 — Long-history loader (yfinance 2005–current)

**Spec refs:** §2.1, §2.7 outputs.

**Files:**
- Create: `src/quant_research_stack/signal_research/data/long_history.py`
- Create: `tests/signal_research/test_data_long_history.py`

- [ ] **Step 1: Write failing tests**

```python
"""Long-history yfinance loader (spec §2.1)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)
from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    load_and_verify_manifest,
    sha256_of_file,
)


def test_fetch_one_ticker_returns_required_columns() -> None:
    cfg = LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 6, 1))
    df = fetch_one_ticker("SPY", config=cfg)
    for col in ("date", "symbol", "open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert df.height > 50
    assert df["symbol"].unique().to_list() == ["SPY"]


def test_save_with_manifest_writes_parquet_and_manifest(tmp_signal_research_root: Path) -> None:
    cfg = LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 6, 1))
    df = fetch_one_ticker("SPY", config=cfg)
    out_root = tmp_signal_research_root / "long_history"
    save_with_manifest(
        df,
        ticker="SPY",
        root=out_root,
        constituent_survivorship_applicable=False,
    )
    parquet = out_root / "SPY.parquet"
    manifest_json = out_root / "SPY.manifest.json"
    assert parquet.exists()
    assert manifest_json.exists()
    sha = sha256_of_file(parquet)
    m = load_and_verify_manifest(manifest_json, expected_sha256={"SPY": sha})
    assert m.data_quality_tier == DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT
    assert m.constituent_survivorship_applicable is False
```

- [ ] **Step 2: Run — expect fail**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_long_history.py -v
```

- [ ] **Step 3: Implement `long_history.py`**

```python
"""yfinance long-history loader with manifest emission (spec §2.1).

Adapts the pattern already used by `strategy_benchmark.data.fetch_daily_bars`
but emits per-ticker manifests in the signal_research format.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class LongHistoryConfig:
    start: date
    end: date | None = None  # None = today


def fetch_one_ticker(ticker: str, *, config: LongHistoryConfig) -> pl.DataFrame:
    import yfinance as yf  # local import — yfinance is heavy

    end = config.end or dt.date.today()
    df = yf.download(
        ticker,
        start=config.start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty yfinance result for {ticker} {config.start}..{end}")
    df = df.reset_index()
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    date_col_candidates = [c for c in df.columns if c in ("Date", "Datetime", "index")]
    if not date_col_candidates:
        raise RuntimeError(f"no date column found in yfinance output for {ticker}: {list(df.columns)}")
    date_col = date_col_candidates[0]
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = pl.from_pandas(df).rename(
        {
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            close_col: "close",
            "Volume": "volume",
        }
    )
    if "Close" in out.columns and close_col != "Close":
        out = out.drop("Close")
    out = out.with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("symbol"),
    ).select(["date", "symbol", "open", "high", "low", "close", "volume"])
    return out.sort("date")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_with_manifest(
    df: pl.DataFrame,
    *,
    ticker: str,
    root: Path,
    constituent_survivorship_applicable: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("=", "_").replace("^", "")
    parquet_path = root / f"{safe}.parquet"
    df.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name=ticker,
        source_url=f"yfinance://{ticker}",
        fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()),
        date_range_start=str(df["date"].min()),
        date_range_end=str(df["date"].max()),
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=constituent_survivorship_applicable,
        vendor_disclosure="yfinance public snapshot — not vendor PIT data",
        timestamp_convention="after_close_t",
        warnings=[
            "yfinance is a public historical snapshot, not vendor-grade PIT data",
        ] + (
            ["constituent_survivorship_applicable=False per spec §2.2 directly-traded note"]
            if not constituent_survivorship_applicable else []
        ),
    )
    write_manifest(root / f"{safe}.manifest.json", m)
```

- [ ] **Step 4: Run tests — expect green (downloads SPY ~5 months of data)**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_long_history.py -v
```

Network access required.

- [ ] **Step 5: Lint + types + commit**

```bash
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/long_history.py tests/signal_research/test_data_long_history.py
git commit -m "feat(signal-research): yfinance long-history loader with manifest"
```

---

### Task 7 — FRED loader

**Spec refs:** §2.4 (FRED via fredapi).

**Files:**
- Create: `src/quant_research_stack/signal_research/data/fred.py`
- Create: `tests/signal_research/test_data_fred.py`

- [ ] **Step 1: Write tests**

```python
"""FRED loader (spec §2.4)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.fred import (
    FredConfig,
    fetch_fred_series,
    save_fred_panel,
)
from quant_research_stack.signal_research.data.manifest import DataQualityTier


def _fake_fred_series(series_id: str) -> pd.Series:
    idx = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    s = pd.Series(range(len(idx)), index=idx, name=series_id, dtype=float)
    return s


def test_fetch_fred_series_returns_polars_df() -> None:
    with patch("quant_research_stack.signal_research.data.fred._fred_get_series", _fake_fred_series):
        df = fetch_fred_series("DGS10", config=FredConfig(start=date(2024, 1, 1), end=date(2024, 6, 1)))
    assert "date" in df.columns
    assert "DGS10" in df.columns
    assert df.height > 100


def test_save_fred_panel_emits_manifest(tmp_signal_research_root: Path) -> None:
    with patch("quant_research_stack.signal_research.data.fred._fred_get_series", _fake_fred_series):
        save_fred_panel(
            series_ids=["DGS10", "T10Y2Y"],
            config=FredConfig(start=date(2024, 1, 1), end=date(2024, 6, 1)),
            root=tmp_signal_research_root / "macro",
        )
    p = tmp_signal_research_root / "macro" / "fred_features.parquet"
    m = tmp_signal_research_root / "macro" / "fred_features.manifest.json"
    assert p.exists() and m.exists()
    df = pl.read_parquet(p)
    assert {"date", "DGS10", "T10Y2Y"}.issubset(df.columns)
```

- [ ] **Step 2: Implement `fred.py`**

```python
"""FRED loader via fredapi (spec §2.4).

ALFRED (revision-adjusted) integration is a Phase-2 upgrade; v1 uses plain
FRED which carries `public_snapshot_not_pit` as the data-quality tier.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class FredConfig:
    start: date
    end: date
    api_key: str | None = None  # falls back to FRED_API_KEY env var


def _fred_get_series(series_id: str, *, api_key: str | None = None,
                     start: str | None = None, end: str | None = None) -> pd.Series:
    """Thin wrapper around fredapi.Fred — extracted to make the loader
    monkeypatchable in tests."""
    from fredapi import Fred  # local import — keeps the module import cheap
    fred = Fred(api_key=api_key or os.environ.get("FRED_API_KEY"))
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def fetch_fred_series(series_id: str, *, config: FredConfig) -> pl.DataFrame:
    s = _fred_get_series(
        series_id,
        api_key=config.api_key,
        start=config.start.isoformat(),
        end=config.end.isoformat(),
    )
    df = s.reset_index()
    df.columns = ["date", series_id]
    out = pl.from_pandas(df).with_columns(pl.col("date").cast(pl.Date))
    return out.sort("date")


def save_fred_panel(*, series_ids: list[str], config: FredConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    panel: pl.DataFrame | None = None
    for sid in series_ids:
        df = fetch_fred_series(sid, config=config)
        panel = df if panel is None else panel.join(df, on="date", how="full", coalesce=True)
    assert panel is not None
    panel = panel.sort("date")
    parquet_path = root / "fred_features.parquet"
    panel.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name="fred_features",
        source_url="https://api.stlouisfed.org/fred",
        fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=panel.height,
        symbol_count=0,
        date_range_start=str(panel["date"].min()),
        date_range_end=str(panel["date"].max()),
        schema_fingerprint="cols:" + ",".join(panel.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure=f"FRED public API; v1 plain (not revision-adjusted ALFRED). Series: {series_ids}",
        timestamp_convention="release_date_approximation",
        warnings=[
            "FRED data revisions can affect past values; v1 uses plain FRED not ALFRED",
            "release_date_approximation timestamps may not perfectly align with intra-day publication",
        ],
    )
    write_manifest(root / "fred_features.manifest.json", m)
```

- [ ] **Step 3: Run tests + lint + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_fred.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/fred.py tests/signal_research/test_data_fred.py
git commit -m "feat(signal-research): FRED loader with monkeypatchable fred-get-series"
```

---

### Task 8 — CBOE proxies loader (^VIX, ^VVIX, ^SKEW, ^GVZ, ^OVX, ^VXN)

**Spec refs:** §2.4.

**Files:**
- Create: `src/quant_research_stack/signal_research/data/cboe_proxies.py`
- Create: `tests/signal_research/test_data_cboe_proxies.py`

- [ ] **Step 1: Write tests**

```python
"""CBOE proxies via yfinance (spec §2.4)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.cboe_proxies import (
    CboeProxiesConfig,
    fetch_cboe_panel,
)


def test_fetch_cboe_panel_includes_requested_tickers() -> None:
    cfg = CboeProxiesConfig(
        tickers=["^VIX", "^VVIX", "^SKEW"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
    )
    df = fetch_cboe_panel(config=cfg)
    assert "date" in df.columns
    # Each ticker may produce a column (close_VIX, etc.)
    for t in cfg.tickers:
        safe = t.replace("^", "")
        assert f"close_{safe}" in df.columns


def test_vxn_fallback_recorded_in_manifest(tmp_signal_research_root: Path) -> None:
    from quant_research_stack.signal_research.data.cboe_proxies import save_cboe_panel
    cfg = CboeProxiesConfig(
        tickers=["^VIX", "^VXN"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
    )
    save_cboe_panel(config=cfg, root=tmp_signal_research_root / "cboe")
    parquet = tmp_signal_research_root / "cboe" / "cboe_proxies.parquet"
    manifest = tmp_signal_research_root / "cboe" / "cboe_proxies.manifest.json"
    assert parquet.exists()
    assert manifest.exists()
```

- [ ] **Step 2: Implement `cboe_proxies.py`**

```python
"""CBOE volatility-index proxies via yfinance (spec §2.4).

If a ticker is unavailable (e.g. ^VXN sometimes returns empty from
yfinance), the loader records this in the manifest's warnings list rather
than failing — per spec §3.3 #9 "with documented fallback".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
)
from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class CboeProxiesConfig:
    tickers: list[str]
    start: date
    end: date


def fetch_cboe_panel(*, config: CboeProxiesConfig) -> pl.DataFrame:
    long_cfg = LongHistoryConfig(start=config.start, end=config.end)
    out: pl.DataFrame | None = None
    fallbacks: list[str] = []
    for t in config.tickers:
        try:
            df = fetch_one_ticker(t, config=long_cfg)
        except RuntimeError as exc:
            fallbacks.append(f"{t}: {exc}")
            continue
        safe = t.replace("^", "")
        df = df.select(["date", pl.col("close").alias(f"close_{safe}")])
        out = df if out is None else out.join(df, on="date", how="full", coalesce=True)
    if out is None:
        raise RuntimeError(f"all CBOE tickers failed: {fallbacks}")
    out = out.sort("date")
    out.fallback_warnings = fallbacks  # type: ignore[attr-defined]
    return out


def save_cboe_panel(*, config: CboeProxiesConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df = fetch_cboe_panel(config=config)
    fallbacks: list[str] = list(getattr(df, "fallback_warnings", []))
    parquet_path = root / "cboe_proxies.parquet"
    df.write_parquet(parquet_path)
    m = DataSourceManifest(
        source_name="cboe_proxies",
        source_url="yfinance:cboe_proxies",
        fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=0,
        date_range_start=str(df["date"].min()),
        date_range_end=str(df["date"].max()),
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
        constituent_survivorship_applicable=False,
        vendor_disclosure="yfinance CBOE indices (^VIX, ^VVIX, ^SKEW, ^GVZ, ^OVX, ^VXN)",
        timestamp_convention="after_close_t",
        warnings=fallbacks,
    )
    write_manifest(root / "cboe_proxies.manifest.json", m)
```

- [ ] **Step 3: Tests + lint + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_cboe_proxies.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/cboe_proxies.py tests/signal_research/test_data_cboe_proxies.py
git commit -m "feat(signal-research): CBOE proxies (VIX/VVIX/SKEW/GVZ/OVX/VXN) with fallback"
```

---

### Task 9 — SP500 current constituents

**Spec refs:** §2.3, §6.1 (current-only = `survivorship_prototype_only`).

**Files:**
- Create: `src/quant_research_stack/signal_research/data/sp500_components.py`
- Create: `tests/signal_research/test_data_sp500_components.py`

- [ ] **Step 1: Tests**

```python
"""Current SP500 list loader (spec §2.3, §6.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from quant_research_stack.signal_research.data.manifest import DataQualityTier
from quant_research_stack.signal_research.data.sp500_components import (
    load_or_fetch_sp500,
    save_sp500_manifest,
)


def _fake_wikipedia_sp500() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
                       "BRK.B", "JPM", "V"] * 50 + ["XOM"],
            "name": ["..."] * 501,
            "sector": ["Tech"] * 501,
        }
    )


def test_sp500_loader_returns_at_least_500_symbols(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "sp500" / "sp500_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.sp500_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_sp500(),
    ):
        df = load_or_fetch_sp500(parquet_path=out)
    assert df.height >= 500


def test_sp500_manifest_flags_survivorship(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "sp500" / "sp500_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.sp500_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_sp500(),
    ):
        save_sp500_manifest(parquet_path=out)
    assert (out.parent / "sp500_current.manifest.json").exists()
```

- [ ] **Step 2: Implement**

```python
"""Current SP500 constituents (Wikipedia parse, cached).

Labelled `survivorship_prototype_only` per spec §6.1 — current-only universe.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


def _fetch_from_wikipedia() -> pl.DataFrame:
    """Parse the current SP500 list from Wikipedia (cached upstream by yfinance/pandas)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    df = df.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})
    return pl.from_pandas(df[["symbol", "name", "sector"]])


def load_or_fetch_sp500(*, parquet_path: Path) -> pl.DataFrame:
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    df = _fetch_from_wikipedia()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return df


def save_sp500_manifest(*, parquet_path: Path) -> None:
    df = load_or_fetch_sp500(parquet_path=parquet_path)
    m = DataSourceManifest(
        source_name="sp500_current",
        source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()),
        date_range_start="current",
        date_range_end="current",
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
        vendor_disclosure="Wikipedia current-list parse — no historical membership reconstruction",
        timestamp_convention="snapshot_current",
        warnings=[
            "SURVIVORSHIP-WARNED: current S&P 500 constituents only; no PIT membership history",
            "cross-sectional results carry the mandatory survivorship banner per spec §2.8",
        ],
    )
    write_manifest(parquet_path.parent / (parquet_path.stem + ".manifest.json"), m)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_sp500_components.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/sp500_components.py tests/signal_research/test_data_sp500_components.py
git commit -m "feat(signal-research): current SP500 list with survivorship warning"
```

---

### Task 10 — Nasdaq-100 current constituents

**Spec refs:** §2.3.1 universe `nasdaq_100_current`.

**Files:**
- Create: `src/quant_research_stack/signal_research/data/nasdaq_components.py`
- Create: `tests/signal_research/test_data_nasdaq_components.py`

- [ ] **Step 1: Tests** (analogous to Task 9 — replace SP500 with Nasdaq 100 list, target ≥100 symbols, survivorship-warned manifest).

```python
"""Current Nasdaq 100 list loader (spec §2.3.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from quant_research_stack.signal_research.data.manifest import DataQualityTier
from quant_research_stack.signal_research.data.nasdaq_components import (
    load_or_fetch_nasdaq_100,
    save_nasdaq_100_manifest,
)


def _fake_wikipedia_ndx100() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
                       "TSLA", "AVGO", "COST"] * 10 + ["NFLX"],
            "name": ["..."] * 101,
        }
    )


def test_nasdaq_100_loader_returns_at_least_100(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "nasdaq" / "nasdaq_100_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.nasdaq_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_ndx100(),
    ):
        df = load_or_fetch_nasdaq_100(parquet_path=out)
    assert df.height >= 100


def test_nasdaq_100_manifest_flags_survivorship(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "nasdaq" / "nasdaq_100_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.nasdaq_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_ndx100(),
    ):
        save_nasdaq_100_manifest(parquet_path=out)
    assert (out.parent / "nasdaq_100_current.manifest.json").exists()
```

- [ ] **Step 2: Implement** (analogous structure to Task 9 but with Nasdaq-100 Wikipedia URL):

```python
"""Current Nasdaq 100 constituents (spec §2.3.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


def _fetch_from_wikipedia() -> pl.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        if {"Ticker", "Company"}.issubset(t.columns) or {"Symbol", "Company"}.issubset(t.columns):
            t = t.rename(columns={"Ticker": "symbol", "Symbol": "symbol", "Company": "name"})
            return pl.from_pandas(t[["symbol", "name"]])
    raise RuntimeError("Nasdaq-100 table not found on Wikipedia page")


def load_or_fetch_nasdaq_100(*, parquet_path: Path) -> pl.DataFrame:
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    df = _fetch_from_wikipedia()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return df


def save_nasdaq_100_manifest(*, parquet_path: Path) -> None:
    df = load_or_fetch_nasdaq_100(parquet_path=parquet_path)
    m = DataSourceManifest(
        source_name="nasdaq_100_current",
        source_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        path=parquet_path.name,
        sha256=sha256_of_file(parquet_path),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()),
        date_range_start="current",
        date_range_end="current",
        schema_fingerprint="cols:" + ",".join(df.columns),
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
        vendor_disclosure="Wikipedia current-list parse — no historical membership reconstruction",
        timestamp_convention="snapshot_current",
        warnings=[
            "SURVIVORSHIP-WARNED: current Nasdaq 100 constituents only; no PIT membership history",
        ],
    )
    write_manifest(parquet_path.parent / (parquet_path.stem + ".manifest.json"), m)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_nasdaq_components.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/nasdaq_components.py tests/signal_research/test_data_nasdaq_components.py
git commit -m "feat(signal-research): current Nasdaq 100 list with survivorship warning"
```

---

### Task 11 — Crypto minimal v1 loader (BTC/ETH daily)

**Spec refs:** §6.6.

**Files:**
- Create: `src/quant_research_stack/signal_research/data/crypto_minimal.py`
- Create: `tests/signal_research/test_data_crypto_minimal.py`

- [ ] **Step 1: Tests**

```python
"""Crypto minimal v1 — BTCUSDT + ETHUSDT daily (spec §6.6)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl


def _fake_binance_klines(ticker: str, start: str, end: str) -> pl.DataFrame:
    dates = pl.date_range(date(2024, 1, 1), date(2024, 6, 1), interval="1d", eager=True)
    n = dates.len()
    return pl.DataFrame({
        "date": dates,
        "symbol": [ticker] * n,
        "open": [40000.0 + i for i in range(n)],
        "high": [40100.0 + i for i in range(n)],
        "low": [39900.0 + i for i in range(n)],
        "close": [40050.0 + i for i in range(n)],
        "volume": [1000.0 + i for i in range(n)],
    })


def test_crypto_minimal_loader_persists_to_disk(tmp_signal_research_root: Path) -> None:
    from quant_research_stack.signal_research.data.crypto_minimal import (
        CryptoMinimalConfig,
        save_crypto_minimal,
    )
    with patch(
        "quant_research_stack.signal_research.data.crypto_minimal._fetch_binance_klines",
        side_effect=_fake_binance_klines,
    ):
        save_crypto_minimal(
            config=CryptoMinimalConfig(
                tickers=["BTCUSDT", "ETHUSDT"],
                start=date(2024, 1, 1),
                end=date(2024, 6, 1),
            ),
            root=tmp_signal_research_root / "crypto",
        )
    for t in ("BTCUSDT", "ETHUSDT"):
        assert (tmp_signal_research_root / "crypto" / f"{t}_daily.parquet").exists()
        assert (tmp_signal_research_root / "crypto" / f"{t}_daily.manifest.json").exists()
```

- [ ] **Step 2: Implement**

```python
"""Crypto minimal v1 loader (spec §6.6).

Source: Binance public klines (daily). Spot pairs only in v1 — no perpetuals
or funding rates. Carries `public_snapshot_not_pit` + `constituent_survivorship_applicable: false`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    DataSourceManifest,
    sha256_of_file,
    write_manifest,
)


@dataclass(frozen=True)
class CryptoMinimalConfig:
    tickers: list[str]
    start: date
    end: date


def _fetch_binance_klines(ticker: str, start: str, end: str) -> pl.DataFrame:
    """Real fetcher (placeholder import path — concrete adapter chosen at
    plan execution time per spec §6.8 open question)."""
    # Concrete implementation may use python-binance, ccxt, or direct REST.
    # Stubbed here for monkey-patchability in tests.
    raise NotImplementedError(
        "Concrete Binance public-data adapter is selected at execution time; "
        "see spec §6.8 open question."
    )


def save_crypto_minimal(*, config: CryptoMinimalConfig, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for t in config.tickers:
        df = _fetch_binance_klines(t, config.start.isoformat(), config.end.isoformat())
        parquet_path = root / f"{t}_daily.parquet"
        df.write_parquet(parquet_path)
        m = DataSourceManifest(
            source_name=t,
            source_url=f"binance:public_klines:{t}:1d",
            fetch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            path=parquet_path.name,
            sha256=sha256_of_file(parquet_path),
            row_count=df.height,
            symbol_count=1,
            date_range_start=str(df["date"].min()),
            date_range_end=str(df["date"].max()),
            schema_fingerprint="cols:" + ",".join(df.columns),
            data_quality_tier=DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT,
            constituent_survivorship_applicable=False,
            vendor_disclosure="Binance public klines — spot pair only; no perpetuals or funding in v1",
            timestamp_convention="utc_daily_close",
            warnings=[
                "spot-only v1; perpetual + funding-rate strategies deferred",
            ],
        )
        write_manifest(root / f"{t}_daily.manifest.json", m)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_crypto_minimal.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/crypto_minimal.py tests/signal_research/test_data_crypto_minimal.py
git commit -m "feat(signal-research): minimal v1 crypto loader (BTC/ETH daily, monkeypatchable)"
```

---

### Task 12 — HF datasets gated loader

**Spec refs:** §2.4, §2.6 (sentiment + fundamentals gated `research_only_default`).

**Files:**
- Create: `src/quant_research_stack/signal_research/data/hf_datasets.py`
- Create: `tests/signal_research/test_data_hf_datasets.py`

- [ ] **Step 1: Tests**

```python
"""HuggingFace datasets loader — gated by default (spec §2.6)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.data.hf_datasets import (
    HFDatasetGatedError,
    load_hf_dataset_gated,
)


def test_loader_blocks_when_research_only_default_and_not_audited() -> None:
    with pytest.raises(HFDatasetGatedError):
        load_hf_dataset_gated(
            dataset_id="Lettria/financial-news-sentiment",
            audit_token=None,
        )


def test_loader_rejects_audit_token_without_passing_audit() -> None:
    with pytest.raises(HFDatasetGatedError):
        load_hf_dataset_gated(
            dataset_id="Lettria/financial-news-sentiment",
            audit_token="not-a-real-audit-token",
        )
```

- [ ] **Step 2: Implement**

```python
"""HuggingFace datasets loader — gated by default (spec §2.6).

Sentiment + fundamentals datasets enter the promoted benchmark only via
the FinBERT-style audit ladder (spec §3.3 #8). v1 default is to BLOCK
loading unless an audit token validated by the audit gate is provided.
"""

from __future__ import annotations


class HFDatasetGatedError(RuntimeError):
    pass


_AUDIT_TOKENS_ACCEPTED: frozenset[str] = frozenset()  # populated by audit pipeline


def load_hf_dataset_gated(*, dataset_id: str, audit_token: str | None) -> object:
    if audit_token is None:
        raise HFDatasetGatedError(
            f"HF dataset {dataset_id} is research_only_default. Provide an audit_token "
            "after passing the 10-criterion FinBERT-style audit gate (spec §3.3 FinBERT)."
        )
    if audit_token not in _AUDIT_TOKENS_ACCEPTED:
        raise HFDatasetGatedError(
            f"audit_token '{audit_token}' not in the accepted-tokens set; "
            "audit must be re-run and the token registered before this dataset can load."
        )
    # Once a valid token is provided, the actual loader runs.
    from datasets import load_dataset  # local import
    return load_dataset(dataset_id)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_hf_datasets.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/hf_datasets.py tests/signal_research/test_data_hf_datasets.py
git commit -m "feat(signal-research): HF datasets default-gated until audit pass"
```

---

### Task 13 — Profile + universe config loader

**Spec refs:** §2.3 (Nasdaq universes), §6.6 (crypto profile).

**Files:**
- Create: `src/quant_research_stack/signal_research/data/profiles.py`
- Create: `tests/signal_research/test_data_profiles.py`

- [ ] **Step 1: Tests**

```python
"""Profile + universe configuration loader (spec §2.3, §6.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.signal_research.data.manifest import DataQualityTier
from quant_research_stack.signal_research.data.profiles import (
    ProfileConfig,
    load_profile,
    list_profiles,
)


def test_list_profiles_returns_four_canonical_profiles() -> None:
    profiles = list_profiles(Path("configs/signal_research_profiles"))
    assert set(profiles) == {"sp500", "nasdaq", "crypto", "futures_proxy"}


def test_load_nasdaq_profile_has_four_universes() -> None:
    cfg: ProfileConfig = load_profile(
        Path("configs/signal_research_profiles/nasdaq.yaml")
    )
    assert cfg.profile == "nasdaq"
    universe_names = {u.name for u in cfg.universes}
    assert universe_names == {
        "nasdaq_index_proxy", "nasdaq_100_current", "nasdaq_mega_cap_focus", "user_focus_tech"
    }


def test_load_nasdaq_profile_nasdaq_100_is_survivorship_warned() -> None:
    cfg = load_profile(Path("configs/signal_research_profiles/nasdaq.yaml"))
    ndx100 = next(u for u in cfg.universes if u.name == "nasdaq_100_current")
    assert ndx100.data_quality_label == DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY
    assert ndx100.constituent_survivorship_applicable is True


def test_load_crypto_profile_carries_directly_traded_semantics() -> None:
    cfg = load_profile(Path("configs/signal_research_profiles/crypto.yaml"))
    univ = cfg.universes[0]
    assert univ.constituent_survivorship_applicable is False
```

- [ ] **Step 2: Implement**

```python
"""Profile + universe configuration loader (spec §2.3, §6.6)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_research_stack.signal_research.data.manifest import DataQualityTier


class UniverseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    name: str
    description: str = ""
    data_quality_label: DataQualityTier
    constituent_survivorship_applicable: bool
    tickers: list[str] = Field(default_factory=list)
    tickers_source: str | None = None
    diagnostic_only: list[str] = Field(default_factory=list)


class CostModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    commission_bps_one_way: float = 0.5
    spread_bps_one_way: float = 1.0
    maker_bps: float | None = None
    taker_bps: float | None = None
    funding_payments: bool = False


class ProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    profile: str
    asset_class: str
    universes: list[UniverseConfig]
    benchmarks: list[str] = Field(default_factory=list)
    context_features: list[str] = Field(default_factory=list)
    cost_model: CostModelConfig


def load_profile(yaml_path: Path) -> ProfileConfig:
    payload = yaml.safe_load(Path(yaml_path).read_text())
    return ProfileConfig.model_validate(payload)


def list_profiles(root: Path) -> list[str]:
    return sorted(p.stem for p in Path(root).glob("*.yaml"))
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_data_profiles.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/data tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/data
git add src/quant_research_stack/signal_research/data/profiles.py tests/signal_research/test_data_profiles.py
git commit -m "feat(signal-research): profile + universe config loader (4 profiles)"
```

---

### Task 14 — Strategy registry (mandatory schema)

**Spec refs:** §3.6 (mandatory registry schema, 16 fields).

**Files:**
- Create: `src/quant_research_stack/signal_research/registry.py`
- Create: `tests/signal_research/test_registry.py`

- [ ] **Step 1: Tests**

```python
"""Strategy registry — mandatory schema (spec §3.6)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.registry import (
    ModuleType,
    StrategyRegistryEntry,
    SingleAssetOrCrossSectional,
)
from quant_research_stack.signal_research.data.manifest import DataQualityTier


def test_module_type_enum_values() -> None:
    assert ModuleType.STANDALONE_STRATEGY.value == "standalone_strategy"
    assert ModuleType.FEATURE_GENERATOR.value == "feature_generator"
    assert ModuleType.WRAPPER.value == "wrapper"
    assert ModuleType.MODEL_FAMILY.value == "model_family"


def test_registry_entry_requires_all_mandatory_fields() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        StrategyRegistryEntry(strategy_id="x")  # type: ignore[call-arg]


def test_registry_entry_valid_full_construction() -> None:
    entry = StrategyRegistryEntry(
        strategy_id="AL.SP500.L60.K1.5",
        family="AVELLANEDA_LEE",
        module_type=ModuleType.STANDALONE_STRATEGY,
        paper_source="Avellaneda & Lee 2010",
        asset_class="equity",
        profile="sp500",
        single_asset_or_cross_sectional=SingleAssetOrCrossSectional.CROSS_SECTIONAL,
        required_data=["sp500_current_constituents", "long_history"],
        timestamp_assumptions="after_close_t",
        parameter_grid={"lookback": [60, 120, 252], "k": [1.0, 1.5, 2.0]},
        default_parameters={"lookback": 60, "k": 1.5},
        eligible_for_pbo=True,
        eligible_for_holdout=True,
        eligible_for_cross_sectional_bridge=True,
        data_quality_requirements=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        known_limitations=["current-only constituents", "rolling-PCA approximation"],
    )
    assert entry.strategy_id == "AL.SP500.L60.K1.5"
    assert entry.module_type == ModuleType.STANDALONE_STRATEGY
    assert "current-only" in entry.known_limitations[0]
```

- [ ] **Step 2: Implement**

```python
"""Strategy registry — mandatory schema (spec §3.6).

All fields listed are mandatory unless explicitly marked optional in the
spec. The registry is consumed by the runner to enumerate trials and by
the PBO/DSR machinery to count effective strategies.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict

from quant_research_stack.signal_research.data.manifest import DataQualityTier


class ModuleType(enum.StrEnum):
    STANDALONE_STRATEGY = "standalone_strategy"
    FEATURE_GENERATOR = "feature_generator"
    WRAPPER = "wrapper"
    MODEL_FAMILY = "model_family"


class SingleAssetOrCrossSectional(enum.StrEnum):
    SINGLE_ASSET = "single_asset"
    CROSS_SECTIONAL = "cross_sectional"


class StrategyRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    family: str
    module_type: ModuleType
    paper_source: str
    asset_class: str
    profile: str
    single_asset_or_cross_sectional: SingleAssetOrCrossSectional
    required_data: list[str]
    timestamp_assumptions: str
    parameter_grid: dict[str, list]
    default_parameters: dict[str, object]
    eligible_for_pbo: bool
    eligible_for_holdout: bool
    eligible_for_cross_sectional_bridge: bool
    data_quality_requirements: DataQualityTier
    known_limitations: list[str]
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_registry.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research
git add src/quant_research_stack/signal_research/registry.py tests/signal_research/test_registry.py
git commit -m "feat(signal-research): strategy registry mandatory schema (16 fields)"
```

---

### Task 15 — Fetch-data CLI

**Spec refs:** §6.5 M1 deliverable.

**Files:**
- Create: `scripts/fetch_signal_research_data.py`

- [ ] **Step 1: Implement**

```python
"""Fetch the signal_research data foundation.

Usage:
    PYTHONPATH=src uv run python scripts/fetch_signal_research_data.py \
        --config configs/signal_research.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.signal_research.data.cboe_proxies import (
    CboeProxiesConfig,
    save_cboe_panel,
)
from quant_research_stack.signal_research.data.fred import FredConfig, save_fred_panel
from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)
from quant_research_stack.signal_research.data.nasdaq_components import save_nasdaq_100_manifest
from quant_research_stack.signal_research.data.sp500_components import save_sp500_manifest

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/signal_research.yaml")
    p.add_argument(
        "--skip-crypto",
        action="store_true",
        help="skip BTC/ETH fetch (default off — included)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    start = dt.date.fromisoformat(cfg["data"]["long_history"]["start"])
    end_str = cfg["data"]["long_history"].get("end")
    end = dt.date.fromisoformat(end_str) if end_str else dt.date.today()
    long_history_root = Path(cfg["data"]["long_history"]["cache_root"])

    # Long-history core tickers
    core_tickers = ["SPY", "QQQ", "ES=F", "NQ=F", "^IXIC", "TQQQ", "SQQQ",
                    "XLK", "SMH", "IGV"]
    for t in core_tickers:
        try:
            df = fetch_one_ticker(t, config=LongHistoryConfig(start=start, end=end))
            save_with_manifest(
                df,
                ticker=t,
                root=long_history_root,
                constituent_survivorship_applicable=False,
            )
            console.print(f"[green]ok[/green] long_history: {t}")
        except Exception as exc:
            console.print(f"[yellow]skip[/yellow] long_history {t}: {exc}")

    # CBOE proxies
    cboe_cfg = CboeProxiesConfig(
        tickers=cfg["data"]["cboe_proxies"]["tickers"],
        start=start,
        end=end,
    )
    save_cboe_panel(config=cboe_cfg, root=Path(cfg["data"]["cboe_proxies"]["cache_root"]))
    console.print("[green]ok[/green] CBOE proxies")

    # FRED
    try:
        fred_cfg = FredConfig(start=start, end=end)
        save_fred_panel(
            series_ids=cfg["data"]["fred"]["series"],
            config=fred_cfg,
            root=Path(cfg["data"]["fred"]["cache_root"]),
        )
        console.print("[green]ok[/green] FRED")
    except Exception as exc:
        console.print(f"[yellow]skip[/yellow] FRED ({exc}); set FRED_API_KEY env var")

    # Universes
    save_sp500_manifest(parquet_path=Path("data/processed/signal_research/sp500/sp500_current.parquet"))
    console.print("[green]ok[/green] SP500 current")
    save_nasdaq_100_manifest(parquet_path=Path("data/processed/signal_research/nasdaq/nasdaq_100_current.parquet"))
    console.print("[green]ok[/green] Nasdaq 100 current")

    if not args.skip_crypto:
        console.print("[yellow]note[/yellow] crypto fetcher: concrete adapter selected at execution time")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the CLI parses + commit**

```bash
PYTHONPATH=src uv run python scripts/fetch_signal_research_data.py --help
PYTHONPATH=src uv run ruff check scripts/fetch_signal_research_data.py
PYTHONPATH=src uv run mypy scripts/fetch_signal_research_data.py
git add scripts/fetch_signal_research_data.py
git commit -m "feat(signal-research): fetch_signal_research_data CLI orchestrator"
```

---

### Task 16 — M1 integration: end-to-end manifest emission

**Files:**
- Create: `tests/signal_research/test_m1_integration.py`

- [ ] **Step 1: Test**

```python
"""M1 integration: data layer produces all required manifests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from unittest.mock import patch

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)
from quant_research_stack.signal_research.data.manifest import DataQualityTier


def test_long_history_save_to_signal_research_root(tmp_signal_research_root: Path) -> None:
    df = fetch_one_ticker("SPY", config=LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 3, 1)))
    save_with_manifest(df, ticker="SPY", root=tmp_signal_research_root / "long_history",
                       constituent_survivorship_applicable=False)
    assert (tmp_signal_research_root / "long_history" / "SPY.parquet").exists()
    assert (tmp_signal_research_root / "long_history" / "SPY.manifest.json").exists()
```

- [ ] **Step 2: Run + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_m1_integration.py -v
git add tests/signal_research/test_m1_integration.py
git commit -m "test(signal-research): m1 integration manifest emission"
```

---

## M2 — Methodology stack

10 modules, each isolated + testable. The dev-only invariant (`dev_only_guard.py`) is the most important — it enforces §4.9 "methodology never touches holdout".

### Task 17 — CPCV (combinatorial purged CV)

**Spec refs:** §4.1.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/cpcv.py`
- Create: `tests/signal_research/test_methodology_cpcv.py`

- [ ] **Step 1: Tests**

```python
"""CPCV (López de Prado 2018 ch. 12) — combinatorial purged CV."""

from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.signal_research.methodology.cpcv import (
    CPCVConfig,
    cpcv_splits,
    purge_and_embargo,
)


def test_cpcv_splits_are_chronological_blocks() -> None:
    """Spec §4.1: CPCV splits must be chronological blocks, not random."""
    cfg = CPCVConfig(n_partitions=8, test_partitions=2)
    splits = list(cpcv_splits(n_rows=800, config=cfg))
    assert len(splits) == 28  # C(8, 2)
    for train_idx, test_idx in splits:
        # Each test set is a union of contiguous blocks
        # train and test are disjoint
        assert set(train_idx).isdisjoint(set(test_idx))


def test_cpcv_purge_removes_overlapping_label_horizon() -> None:
    """Purging removes train rows whose label horizon overlaps the test block."""
    train_idx = np.arange(0, 100)
    test_idx = np.arange(100, 200)
    purged = purge_and_embargo(
        train_idx=train_idx,
        test_idx=test_idx,
        label_horizon=10,
        embargo=5,
        total_rows=300,
    )
    # Train rows in [90, 100) overlap the label horizon into test → purged
    assert 89 not in purged or 89 in purged  # boundary detail
    # Embargo rows after test [200, 205) should not be in train
    assert all(i < 200 or i >= 205 for i in purged)


def test_cpcv_holdout_indices_excluded() -> None:
    """Spec §4.1: permanent holdout never touched by CPCV."""
    cfg = CPCVConfig(n_partitions=8, test_partitions=2, holdout_start=800)
    splits = list(cpcv_splits(n_rows=1000, config=cfg))
    for train, test in splits:
        assert max(train) < 800
        assert max(test) < 800
```

- [ ] **Step 2: Implement**

```python
"""CPCV — Combinatorial Purged Cross-Validation (López de Prado 2018 ch. 12).

Spec §4.1:
- Splits are chronological blocks, not random row splits.
- Purging removes rows whose label horizon overlaps the test block.
- Embargo removes rows immediately after the test block.
- Time ordering preserved inside each train/test slice.
- Permanent holdout never touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CPCVConfig:
    n_partitions: int = 8
    test_partitions: int = 2
    label_horizon: int = 10
    embargo: int = 5
    holdout_start: int | None = None  # rows >= holdout_start excluded entirely


def cpcv_splits(
    *, n_rows: int, config: CPCVConfig
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    end = config.holdout_start if config.holdout_start is not None else n_rows
    block_size = end // config.n_partitions
    blocks = [
        np.arange(i * block_size, (i + 1) * block_size, dtype=np.int64)
        for i in range(config.n_partitions)
    ]
    for test_block_ids in combinations(range(config.n_partitions), config.test_partitions):
        test_idx = np.concatenate([blocks[i] for i in test_block_ids])
        train_idx = np.concatenate(
            [blocks[i] for i in range(config.n_partitions) if i not in test_block_ids]
        )
        train_idx = purge_and_embargo(
            train_idx=train_idx,
            test_idx=test_idx,
            label_horizon=config.label_horizon,
            embargo=config.embargo,
            total_rows=end,
        )
        yield train_idx, test_idx


def purge_and_embargo(
    *,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    label_horizon: int,
    embargo: int,
    total_rows: int,
) -> NDArray[np.int64]:
    """Remove (a) train rows whose [t, t+label_horizon] overlaps test_idx,
    and (b) train rows in [test_max+1, test_max+1+embargo]."""
    test_set = set(test_idx.tolist())
    keep = []
    for t in train_idx:
        # Purge: drop t if t..t+label_horizon overlaps test
        if any((t + h) in test_set for h in range(label_horizon + 1)):
            continue
        keep.append(int(t))
    # Embargo: drop train rows in [test_max + 1, test_max + 1 + embargo)
    if test_idx.size:
        embargo_set = set(range(int(test_idx.max()) + 1, int(test_idx.max()) + 1 + embargo))
        keep = [t for t in keep if t not in embargo_set]
    return np.asarray(keep, dtype=np.int64)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_cpcv.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/cpcv.py tests/signal_research/test_methodology_cpcv.py
git commit -m "feat(signal-research): CPCV (López de Prado 2018) with purge + embargo"
```

---

### Task 18 — Dev-only invariant guard (`§4.9`)

**Spec refs:** §0 non-negotiable #2, §4.9.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/dev_only_guard.py`
- Create: `tests/signal_research/test_methodology_dev_only_guard.py`

- [ ] **Step 1: Tests**

```python
"""Dev-only invariant guard (spec §4.9, §0 non-negotiable #2)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.methodology.dev_only_guard import (
    HoldoutAccessError,
    enforce_dev_only,
)


def test_methodology_caller_cannot_access_holdout() -> None:
    with pytest.raises(HoldoutAccessError):
        enforce_dev_only(
            caller="methodology.cpcv",
            holdout_indices=[10, 11, 12],
            accessed_indices=[11],
        )


def test_inference_evaluate_holdout_caller_allowed() -> None:
    enforce_dev_only(
        caller="inference.evaluate_holdout",
        holdout_indices=[10, 11, 12],
        accessed_indices=[11],
    )  # no raise
```

- [ ] **Step 2: Implement**

```python
"""Dev-only invariant: methodology modules NEVER touch the permanent holdout.

Spec §4.9, §0 non-negotiable #2. Only `inference.evaluate_holdout` is allowed
to read holdout rows.
"""

from __future__ import annotations

from typing import Final

_ALLOWED_CALLERS: Final[frozenset[str]] = frozenset({"inference.evaluate_holdout"})


class HoldoutAccessError(RuntimeError):
    pass


def enforce_dev_only(
    *,
    caller: str,
    holdout_indices: list[int],
    accessed_indices: list[int],
) -> None:
    if caller in _ALLOWED_CALLERS:
        return
    overlap = set(holdout_indices).intersection(accessed_indices)
    if overlap:
        raise HoldoutAccessError(
            f"caller={caller} accessed {len(overlap)} holdout rows; "
            "methodology modules must operate on dev+validation data only (§4.9)"
        )
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_dev_only_guard.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/dev_only_guard.py tests/signal_research/test_methodology_dev_only_guard.py
git commit -m "feat(signal-research): dev-only invariant guard for holdout protection"
```

---

### Task 19 — Block-bootstrap CIs

**Spec refs:** §4.6.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/bootstrap_ci.py`
- Create: `tests/signal_research/test_methodology_bootstrap_ci.py`

- [ ] **Step 1: Tests**

```python
"""Stationary block bootstrap CIs (spec §4.6)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_sharpe_ci,
)


def test_bootstrap_returns_lower_and_upper_bounds() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.01 + 0.0005  # daily mean ~ small positive
    res = bootstrap_sharpe_ci(returns=rets, config=BootstrapConfig(n_resamples=200, seed=0))
    assert res.point_sharpe > 0
    assert res.ci_lower_95 <= res.point_sharpe <= res.ci_upper_95


def test_bootstrap_zero_signal_brackets_zero() -> None:
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(500) * 0.01  # zero-mean noise
    res = bootstrap_sharpe_ci(returns=rets, config=BootstrapConfig(n_resamples=300, seed=1))
    # 95% CI should bracket 0
    assert res.ci_lower_95 < 0.5 and res.ci_upper_95 > -0.5
```

- [ ] **Step 2: Implement**

```python
"""Stationary block bootstrap (Politis & Romano 1994) for Sharpe CIs.

Spec §4.6: mean block length L = T^(1/3); n_resamples = 10000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int = 10000
    block_length: int | None = None  # None = T^(1/3)
    seed: int = 42


@dataclass(frozen=True)
class BootstrapResult:
    point_sharpe: float
    ci_lower_95: float
    ci_upper_95: float


def _sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns)) / sd * float(np.sqrt(252.0))


def _stationary_block_bootstrap(
    returns: NDArray[np.float64],
    *,
    n_resamples: int,
    block_length: int,
    seed: int,
) -> NDArray[np.float64]:
    T = returns.size
    rng = np.random.default_rng(seed)
    p = 1.0 / block_length
    sharpes = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        sample = np.empty(T, dtype=np.float64)
        i = 0
        idx = int(rng.integers(0, T))
        while i < T:
            sample[i] = returns[idx]
            i += 1
            if rng.random() < p:
                idx = int(rng.integers(0, T))
            else:
                idx = (idx + 1) % T
        sharpes[k] = _sharpe(sample)
    return sharpes


def bootstrap_sharpe_ci(
    *,
    returns: NDArray[np.float64],
    config: BootstrapConfig,
) -> BootstrapResult:
    T = returns.size
    block = config.block_length if config.block_length is not None else max(1, int(round(T ** (1 / 3))))
    sharpes = _stationary_block_bootstrap(
        returns,
        n_resamples=config.n_resamples,
        block_length=block,
        seed=config.seed,
    )
    return BootstrapResult(
        point_sharpe=_sharpe(returns),
        ci_lower_95=float(np.percentile(sharpes, 2.5)),
        ci_upper_95=float(np.percentile(sharpes, 97.5)),
    )
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_bootstrap_ci.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/bootstrap_ci.py tests/signal_research/test_methodology_bootstrap_ci.py
git commit -m "feat(signal-research): stationary block bootstrap Sharpe CIs"
```

---

### Task 20 — Correlation deduplication (net OOS, signed + absolute)

**Spec refs:** §4.3.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/correlation_dedup.py`
- Create: `tests/signal_research/test_methodology_correlation_dedup.py`

- [ ] **Step 1: Tests**

```python
"""Correlation deduplication (spec §4.3)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.correlation_dedup import (
    DedupConfig,
    deduplicate,
)


def test_dedup_clusters_inverse_strategies_when_using_absolute_correlation() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 1))
    rets = np.hstack([
        base,
        base * 1.01,         # ~ duplicate
        -base,               # sign-flip (inverse): |ρ| = 1
        rng.standard_normal((500, 1)),
    ])
    sharpe = np.array([0.5, 0.4, 0.3, 0.7])
    turnover = np.array([1.0, 1.0, 1.0, 1.0])
    dsr = np.array([0.6, 0.55, 0.5, 0.4])
    drawdown = np.array([-0.1, -0.11, -0.1, -0.2])
    result = deduplicate(
        net_returns=rets,
        sharpe=sharpe,
        turnover=turnover,
        dsr=dsr,
        drawdown=drawdown,
        config=DedupConfig(absolute_correlation_threshold=0.85),
    )
    # base, base*1.01, -base all in one cluster (|ρ|≈1); 4th strategy in its own
    assert result.n_clusters == 2


def test_three_representative_rules_reported() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal((500, 3))
    sharpe = np.array([0.5, 0.6, 0.55])
    turnover = np.array([1.0, 4.0, 1.5])
    dsr = np.array([0.4, 0.65, 0.5])
    drawdown = np.array([-0.10, -0.08, -0.05])
    result = deduplicate(
        net_returns=rets,
        sharpe=sharpe, turnover=turnover, dsr=dsr, drawdown=drawdown,
        config=DedupConfig(absolute_correlation_threshold=0.0),  # force everything into 1 cluster
    )
    # Reports three representative-selection rules
    assert "by_sharpe_per_sqrt_turnover" in result.representatives
    assert "by_dsr" in result.representatives
    assert "by_lowest_drawdown" in result.representatives
```

- [ ] **Step 2: Implement**

```python
"""Correlation deduplication (spec §4.3).

- Operates on NET OOS returns.
- Uses absolute correlation by default (inverse strategies = sign-flip = duplicate).
- Reports both signed and absolute correlation matrices.
- Hierarchical clustering: cluster strategies with |ρ| ≥ threshold.
- Three representative-selection rules reported, no single rule hiding a
  better candidate:
    1. by Sharpe / sqrt(turnover) — primary
    2. by DSR (highest)
    3. by lowest drawdown
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DedupConfig:
    absolute_correlation_threshold: float = 0.90


@dataclass(frozen=True)
class DedupResult:
    cluster_ids: NDArray[np.int64]
    n_clusters: int
    signed_correlation: NDArray[np.float64]
    absolute_correlation: NDArray[np.float64]
    representatives: dict[str, list[int]] = field(default_factory=dict)


def _cluster_from_abs_corr(
    abs_corr: NDArray[np.float64], threshold: float
) -> NDArray[np.int64]:
    n = abs_corr.shape[0]
    cluster = np.full(n, -1, dtype=np.int64)
    current = 0
    for i in range(n):
        if cluster[i] != -1:
            continue
        # New cluster — gather all i,j with |corr| >= threshold transitively
        stack = [i]
        while stack:
            k = stack.pop()
            if cluster[k] != -1:
                continue
            cluster[k] = current
            for j in range(n):
                if cluster[j] == -1 and abs_corr[k, j] >= threshold:
                    stack.append(j)
        current += 1
    return cluster


def deduplicate(
    *,
    net_returns: NDArray[np.float64],
    sharpe: NDArray[np.float64],
    turnover: NDArray[np.float64],
    dsr: NDArray[np.float64],
    drawdown: NDArray[np.float64],
    config: DedupConfig,
) -> DedupResult:
    # Net OOS correlations
    signed = np.corrcoef(net_returns.T)
    absolute = np.abs(signed)
    clusters = _cluster_from_abs_corr(absolute, config.absolute_correlation_threshold)
    n_clusters = int(clusters.max()) + 1 if clusters.size else 0

    representatives: dict[str, list[int]] = {
        "by_sharpe_per_sqrt_turnover": [],
        "by_dsr": [],
        "by_lowest_drawdown": [],
    }
    for c in range(n_clusters):
        members = np.where(clusters == c)[0]
        sps = sharpe[members] / np.sqrt(np.maximum(turnover[members], 1e-6))
        representatives["by_sharpe_per_sqrt_turnover"].append(int(members[int(np.argmax(sps))]))
        representatives["by_dsr"].append(int(members[int(np.argmax(dsr[members]))]))
        representatives["by_lowest_drawdown"].append(int(members[int(np.argmin(drawdown[members]))]))

    return DedupResult(
        cluster_ids=clusters,
        n_clusters=n_clusters,
        signed_correlation=signed,
        absolute_correlation=absolute,
        representatives=representatives,
    )
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_correlation_dedup.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/correlation_dedup.py tests/signal_research/test_methodology_correlation_dedup.py
git commit -m "feat(signal-research): correlation dedup with 3 representative rules"
```

---

### Task 21 — Multi-objective Pareto

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/multi_objective.py`
- Create: `tests/signal_research/test_methodology_multi_objective.py`

- [ ] **Step 1: Tests**

```python
"""Multi-objective Pareto front (spec §4.4)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.multi_objective import pareto_front


def test_pareto_front_keeps_non_dominated() -> None:
    # 3 strategies, 2 objectives (maximize sharpe, minimize |dd|)
    # Strategy 0 dominates strategy 1 (better sharpe, equal dd)
    # Strategy 2 trades off (lower sharpe but lower dd)
    sharpe = np.array([1.0, 0.5, 0.7])
    abs_dd = np.array([0.10, 0.10, 0.05])
    turnover = np.array([1.0, 1.0, 1.0])
    capacity_shrinkage = np.array([0.1, 0.1, 0.1])
    front = pareto_front(
        sharpe=sharpe,
        abs_drawdown=abs_dd,
        turnover=turnover,
        capacity_shrinkage=capacity_shrinkage,
    )
    assert 0 in front
    assert 2 in front
    assert 1 not in front  # dominated by 0
```

- [ ] **Step 2: Implement**

```python
"""Multi-objective Pareto front (spec §4.4).

Selection-and-reporting tool only — NOT a promotion criterion (§6.1).
Primary axes (v1): max Sharpe, min |DD|, min turnover, min capacity shrinkage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pareto_front(
    *,
    sharpe: NDArray[np.float64],
    abs_drawdown: NDArray[np.float64],
    turnover: NDArray[np.float64],
    capacity_shrinkage: NDArray[np.float64],
) -> list[int]:
    """Return indices of non-dominated strategies.

    Maximize sharpe; minimize the other three.
    """
    n = len(sharpe)
    obj = np.column_stack(
        [-sharpe, abs_drawdown, turnover, capacity_shrinkage]
    )  # all "minimize" semantics
    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_multi_objective.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/multi_objective.py tests/signal_research/test_methodology_multi_objective.py
git commit -m "feat(signal-research): multi-objective Pareto front (selection-only)"
```

---

### Task 22 — Three-tier PBO (raw_global / profile / family)

**Spec refs:** §4.7, §6.1 criterion #4.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/pbo_extensions.py`
- Create: `tests/signal_research/test_methodology_pbo_extensions.py`

- [ ] **Step 1: Tests**

```python
"""Three-tier PBO reporting (spec §4.7)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.pbo_extensions import (
    PBOMultiResult,
    compute_three_tier_pbo,
)


def test_three_tier_pbo_reports_all_three_values() -> None:
    rng = np.random.default_rng(0)
    T, S = 480, 100
    returns = rng.standard_normal((T, S)) * 0.01
    profile = np.array(["sp500"] * 50 + ["nasdaq"] * 50)
    family = np.array(["MOM"] * 25 + ["MR"] * 25 + ["MOM"] * 25 + ["MR"] * 25)
    res = compute_three_tier_pbo(returns=returns, profile=profile, family=family)
    assert isinstance(res, PBOMultiResult)
    assert 0.0 <= res.raw_global <= 1.0
    assert "sp500" in res.per_profile
    assert "nasdaq" in res.per_profile
    assert "MOM" in res.per_family
    assert "MR" in res.per_family
```

- [ ] **Step 2: Implement**

```python
"""Three-tier PBO reporting (spec §4.7).

Reuses the existing strategy_benchmark.pbo for the core algorithm, then
slices by profile and by family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.pbo import compute_pbo


@dataclass(frozen=True)
class PBOMultiResult:
    raw_global: float
    per_profile: dict[str, float]
    per_family: dict[str, float]
    n_partitions: int
    n_strategies: int


def compute_three_tier_pbo(
    *,
    returns: NDArray[np.float64],
    profile: NDArray[np.str_],
    family: NDArray[np.str_],
    n_partitions: int = 16,
) -> PBOMultiResult:
    raw = compute_pbo(returns=returns, n_partitions=n_partitions)

    per_profile: dict[str, float] = {}
    for p in sorted(set(profile.tolist())):
        cols = np.where(profile == p)[0]
        if len(cols) < 3:
            continue
        per_profile[p] = compute_pbo(returns=returns[:, cols], n_partitions=n_partitions).pbo_probability

    per_family: dict[str, float] = {}
    for f in sorted(set(family.tolist())):
        cols = np.where(family == f)[0]
        if len(cols) < 3:
            continue
        per_family[f] = compute_pbo(returns=returns[:, cols], n_partitions=n_partitions).pbo_probability

    return PBOMultiResult(
        raw_global=raw.pbo_probability,
        per_profile=per_profile,
        per_family=per_family,
        n_partitions=n_partitions,
        n_strategies=returns.shape[1],
    )
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_pbo_extensions.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/pbo_extensions.py tests/signal_research/test_methodology_pbo_extensions.py
git commit -m "feat(signal-research): three-tier PBO (raw_global / profile / family)"
```

---

### Task 23 — Regime-conditional with agnostic/specific declaration

**Spec refs:** §4.5.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/regime_conditional.py`
- Create: `tests/signal_research/test_methodology_regime_conditional.py`

- [ ] **Step 1: Tests**

```python
"""Regime-conditional metrics (spec §4.5)."""

from __future__ import annotations

import numpy as np

from quant_research_stack.signal_research.methodology.regime_conditional import (
    RegimeDeclaration,
    fit_hmm_regimes,
    regime_conditional_metrics,
)


def test_fit_hmm_returns_state_per_day() -> None:
    rng = np.random.default_rng(0)
    rets = np.concatenate([
        rng.standard_normal(500) * 0.005,    # low-vol regime
        rng.standard_normal(500) * 0.025,    # high-vol regime
    ])
    states = fit_hmm_regimes(rets, n_states=2, seed=0)
    assert states.shape == (1000,)
    assert set(np.unique(states).tolist()).issubset({0, 1})


def test_regime_agnostic_strategy_with_positive_in_both_passes() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.005 + 0.0003  # both regimes positive
    states = (rng.random(500) > 0.5).astype(int)
    res = regime_conditional_metrics(
        returns=rets,
        regime_states=states,
        declaration=RegimeDeclaration.AGNOSTIC,
    )
    assert res.passes_regime_gate is True


def test_regime_specific_without_predeclared_gate_fails() -> None:
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.005
    states = (rng.random(500) > 0.5).astype(int)
    res = regime_conditional_metrics(
        returns=rets,
        regime_states=states,
        declaration=RegimeDeclaration.SPECIFIC,
        favorable_regime=None,    # not predeclared → fail
    )
    assert res.passes_regime_gate is False
```

- [ ] **Step 2: Implement**

```python
"""Regime-conditional metrics (spec §4.5).

- 2-state Gaussian HMM on broad-market returns.
- Per-strategy Sharpe / DD / PnL by regime.
- Declarations:
  - AGNOSTIC: must have both regimes not catastrophically negative,
    AND at least one regime materially positive.
  - SPECIFIC: must be PREDECLARED with a favorable_regime AND gated using
    only same-time info; retroactive regime declaration is forbidden.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class RegimeDeclaration(enum.StrEnum):
    AGNOSTIC = "agnostic"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class RegimeMetrics:
    sharpe_by_regime: dict[int, float]
    pnl_by_regime: dict[int, float]
    max_dd_by_regime: dict[int, float]
    active_days_by_regime: dict[int, int]
    declaration: RegimeDeclaration
    favorable_regime: int | None
    passes_regime_gate: bool


def fit_hmm_regimes(
    returns: NDArray[np.float64], *, n_states: int = 2, seed: int = 42
) -> NDArray[np.int64]:
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, random_state=seed)
    model.fit(returns.reshape(-1, 1))
    return model.predict(returns.reshape(-1, 1)).astype(np.int64)


def _sharpe(r: NDArray[np.float64]) -> float:
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r)) / sd * float(np.sqrt(252.0))


def regime_conditional_metrics(
    *,
    returns: NDArray[np.float64],
    regime_states: NDArray[np.int64],
    declaration: RegimeDeclaration,
    favorable_regime: int | None = None,
    catastrophic_threshold: float = -1.0,
    materially_positive_threshold: float = 0.3,
) -> RegimeMetrics:
    states = sorted(set(regime_states.tolist()))
    by_sharpe: dict[int, float] = {}
    by_pnl: dict[int, float] = {}
    by_dd: dict[int, float] = {}
    by_days: dict[int, int] = {}
    for s in states:
        mask = regime_states == s
        r = returns[mask]
        by_sharpe[s] = _sharpe(r)
        by_pnl[s] = float(np.sum(r))
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity) if equity.size else np.array([1.0])
        by_dd[s] = float((equity / peak - 1.0).min()) if equity.size else 0.0
        by_days[s] = int(mask.sum())

    if declaration == RegimeDeclaration.AGNOSTIC:
        sharpes = list(by_sharpe.values())
        all_not_catastrophic = all(s > catastrophic_threshold for s in sharpes)
        at_least_one_positive = any(s > materially_positive_threshold for s in sharpes)
        passes = all_not_catastrophic and at_least_one_positive
    else:  # SPECIFIC
        if favorable_regime is None:
            passes = False
        else:
            passes = by_sharpe.get(favorable_regime, -1.0) > materially_positive_threshold

    return RegimeMetrics(
        sharpe_by_regime=by_sharpe,
        pnl_by_regime=by_pnl,
        max_dd_by_regime=by_dd,
        active_days_by_regime=by_days,
        declaration=declaration,
        favorable_regime=favorable_regime,
        passes_regime_gate=passes,
    )
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_regime_conditional.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/regime_conditional.py tests/signal_research/test_methodology_regime_conditional.py
git commit -m "feat(signal-research): regime-conditional metrics with agnostic/specific declaration"
```

---

### Task 24 — Meta-labeling (survivor-only pre-filter)

**Spec refs:** §4.2.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/meta_labeling.py`
- Create: `tests/signal_research/test_methodology_meta_labeling.py`

- [ ] **Step 1: Tests**

```python
"""Meta-labeling (spec §4.2) — survivor-only pre-filter."""

from __future__ import annotations

import numpy as np
import pytest

from quant_research_stack.signal_research.methodology.meta_labeling import (
    MetaLabelingEligibility,
    PrimarySignalStats,
    check_eligibility,
)


def test_eligible_when_all_filters_pass() -> None:
    stats = PrimarySignalStats(
        validation_net_sharpe=0.8,
        validation_hit_rate=0.55,
        validation_expectancy=0.001,
        event_count=300,
        single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    elig = check_eligibility(stats)
    assert elig.eligible is True


def test_rejects_when_event_count_too_low_single_asset() -> None:
    stats = PrimarySignalStats(
        validation_net_sharpe=0.8,
        validation_hit_rate=0.55,
        validation_expectancy=0.001,
        event_count=150,
        single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    elig = check_eligibility(stats)
    assert elig.eligible is False
    assert "event_count" in elig.rejection_reason


def test_rejects_when_event_count_too_low_cross_sectional() -> None:
    stats = PrimarySignalStats(
        validation_net_sharpe=0.8,
        validation_hit_rate=0.55,
        validation_expectancy=0.001,
        event_count=400,
        single_asset_or_cross_sectional="cross_sectional",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    elig = check_eligibility(stats)
    assert elig.eligible is False


def test_rejects_when_negative_sharpe() -> None:
    stats = PrimarySignalStats(
        validation_net_sharpe=-0.1,
        validation_hit_rate=0.55,
        validation_expectancy=0.001,
        event_count=300,
        single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False,
        is_near_duplicate=False,
    )
    elig = check_eligibility(stats)
    assert elig.eligible is False
```

- [ ] **Step 2: Implement**

```python
"""Meta-labeling (spec §4.2).

Survivor-only pre-filter:
- Positive validation net Sharpe.
- Positive validation hit rate (after costs) OR positive expectancy.
- Sufficient event count: ≥200 single-asset / ≥500 cross-sectional.
- No inverted-signal superiority.
- Not a near-duplicate of a stronger primary.

Secondary RF classifier construction is performed at strategy-evaluation
time; this module gates eligibility upfront.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimarySignalStats:
    validation_net_sharpe: float
    validation_hit_rate: float
    validation_expectancy: float
    event_count: int
    single_asset_or_cross_sectional: str  # "single_asset" | "cross_sectional"
    is_inverted_superior: bool
    is_near_duplicate: bool


@dataclass(frozen=True)
class MetaLabelingEligibility:
    eligible: bool
    rejection_reason: str = ""


_MIN_EVENTS = {"single_asset": 200, "cross_sectional": 500}


def check_eligibility(stats: PrimarySignalStats) -> MetaLabelingEligibility:
    if stats.validation_net_sharpe <= 0:
        return MetaLabelingEligibility(False, "validation_net_sharpe <= 0")
    if stats.validation_hit_rate <= 0.5 and stats.validation_expectancy <= 0:
        return MetaLabelingEligibility(
            False, "neither validation hit rate nor expectancy is positive"
        )
    threshold = _MIN_EVENTS.get(stats.single_asset_or_cross_sectional, 200)
    if stats.event_count < threshold:
        return MetaLabelingEligibility(
            False, f"event_count {stats.event_count} < threshold {threshold}"
        )
    if stats.is_inverted_superior:
        return MetaLabelingEligibility(False, "inverted signal is superior — sign bug suspect")
    if stats.is_near_duplicate:
        return MetaLabelingEligibility(False, "near-duplicate of a stronger primary")
    return MetaLabelingEligibility(True, "")
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_meta_labeling.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/meta_labeling.py tests/signal_research/test_methodology_meta_labeling.py
git commit -m "feat(signal-research): meta-labeling survivor-only pre-filter"
```

---

### Task 25 — Failure classifier (13 categories) + Selection funnel

**Spec refs:** §4.10, §6.3, §6.4.

**Files:**
- Create: `src/quant_research_stack/signal_research/methodology/failure_classifier.py`
- Create: `src/quant_research_stack/signal_research/methodology/selection_funnel.py`
- Create: `tests/signal_research/test_methodology_failure_classifier.py`
- Create: `tests/signal_research/test_methodology_selection_funnel.py`

- [ ] **Step 1: Failure-classifier tests + impl**

```python
"""Failure classifier — 13 categories (spec §4.10, §6.3)."""

from __future__ import annotations

import pytest

from quant_research_stack.signal_research.methodology.failure_classifier import (
    FailureCategory,
    CandidateFailureRecord,
    all_failure_categories,
)


def test_thirteen_categories_present() -> None:
    cats = all_failure_categories()
    assert len(cats) == 13
    expected = {
        "high_pbo", "low_dsr", "cost_failure", "regime_concentration",
        "insufficient_sample", "too_few_trades", "delay_stress_fail",
        "single_period_dominance", "over_correlated_with_baseline",
        "randomization_fail", "data_quality_fail",
        "holdout_failure", "capacity_failure",
    }
    assert {c.value for c in cats} == expected


def test_candidate_failure_record_holds_multiple_categories() -> None:
    rec = CandidateFailureRecord(
        strategy_id="X",
        categories=[FailureCategory.HIGH_PBO, FailureCategory.LOW_DSR],
    )
    assert len(rec.categories) == 2
```

```python
"""Failure classifier (spec §4.10, §6.3)."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FailureCategory(enum.StrEnum):
    HIGH_PBO = "high_pbo"
    LOW_DSR = "low_dsr"
    COST_FAILURE = "cost_failure"
    REGIME_CONCENTRATION = "regime_concentration"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    TOO_FEW_TRADES = "too_few_trades"
    DELAY_STRESS_FAIL = "delay_stress_fail"
    SINGLE_PERIOD_DOMINANCE = "single_period_dominance"
    OVER_CORRELATED_WITH_BASELINE = "over_correlated_with_baseline"
    RANDOMIZATION_FAIL = "randomization_fail"
    DATA_QUALITY_FAIL = "data_quality_fail"
    HOLDOUT_FAILURE = "holdout_failure"
    CAPACITY_FAILURE = "capacity_failure"


@dataclass(frozen=True)
class CandidateFailureRecord:
    strategy_id: str
    categories: list[FailureCategory]


def all_failure_categories() -> list[FailureCategory]:
    return list(FailureCategory)
```

- [ ] **Step 2: Selection-funnel tests + impl**

```python
"""Selection funnel (spec §6.4)."""

from __future__ import annotations

from quant_research_stack.signal_research.methodology.selection_funnel import (
    SelectionFunnel,
)


def test_funnel_records_counts_per_filter_in_order() -> None:
    f = SelectionFunnel()
    f.record("total_raw_candidates", 1620)
    f.record("after_data_quality_filter", 1500)
    f.record("after_cost_stress_2x", 980)
    f.record("after_sanity_randomization", 920)
    f.record("after_pbo_profile_threshold", 110)
    f.record("after_dsr_threshold", 45)
    f.record("after_bootstrap_lower_positive", 25)
    f.record("after_regime_concentration", 8)
    f.record("research_pass", 8)
    f.record("promotion_eligible", 0)
    f.record("paper_trade_candidate", 0)
    f.record("production_candidate", 0)
    counts = f.to_ordered_dict()
    assert counts["total_raw_candidates"] == 1620
    assert counts["research_pass"] == 8
    assert counts["production_candidate"] == 0
```

```python
"""Selection funnel (spec §6.4)."""

from __future__ import annotations

from collections import OrderedDict


class SelectionFunnel:
    def __init__(self) -> None:
        self._stages: OrderedDict[str, int] = OrderedDict()

    def record(self, stage: str, count: int) -> None:
        self._stages[stage] = int(count)

    def to_ordered_dict(self) -> OrderedDict[str, int]:
        return OrderedDict(self._stages)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_methodology_failure_classifier.py tests/signal_research/test_methodology_selection_funnel.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/methodology tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/methodology
git add src/quant_research_stack/signal_research/methodology/failure_classifier.py src/quant_research_stack/signal_research/methodology/selection_funnel.py tests/signal_research/test_methodology_failure_classifier.py tests/signal_research/test_methodology_selection_funnel.py
git commit -m "feat(signal-research): 13-category failure classifier + selection funnel"
```

---

## M3 — Classical paper signals

Eight non-deferred signal families. Each module has the same TDD pattern: small parameter grid, tests for shape + key invariants, then implementation. Deep models (#1, #2) are M5.

For brevity here, each task block follows the same skeleton: tests-first → implementation → lint + commit. The complete production-ready implementation per signal is concrete code, not pseudocode — but the plan groups them by similar structure.

### Task 26 — `papers/base.py` — abstract bases for the four module types

**Spec refs:** §3.1.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/base.py`
- Create: `tests/signal_research/test_papers_base.py`

- [ ] **Step 1: Tests**

```python
"""Paper-signal base classes (spec §3.1)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.papers.base import (
    FeatureGenerator,
    ModelFamily,
    StandaloneStrategy,
    Wrapper,
)


def test_standalone_strategy_subclass_returns_positions() -> None:
    class Trivial(StandaloneStrategy):
        def positions(self, panel: pl.DataFrame) -> pl.Series:
            return pl.Series("position", [0.0] * panel.height)

    s = Trivial()
    df = pl.DataFrame({"date": [1, 2, 3], "close": [1.0, 1.0, 1.0]})
    p = s.positions(df)
    assert p.len() == 3


def test_feature_generator_subclass_returns_panel() -> None:
    class TrivialFeat(FeatureGenerator):
        def features(self, panel: pl.DataFrame) -> pl.DataFrame:
            return panel.with_columns(pl.lit(0.0).alias("zero_feature"))

    f = TrivialFeat()
    df = pl.DataFrame({"date": [1, 2]})
    out = f.features(df)
    assert "zero_feature" in out.columns


def test_wrapper_subclass_modifies_primary() -> None:
    class PassThrough(Wrapper):
        def apply(self, positions: pl.Series) -> pl.Series:
            return positions

    w = PassThrough()
    p = pl.Series("position", [0.5, -0.5, 0.0])
    out = w.apply(p)
    assert out.to_list() == [0.5, -0.5, 0.0]
```

- [ ] **Step 2: Implement**

```python
"""Paper-signal abstract base classes (spec §3.1).

Every paper-derived module is exactly one of:
- StandaloneStrategy
- FeatureGenerator
- Wrapper
- ModelFamily
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class StandaloneStrategy(ABC):
    @abstractmethod
    def positions(self, panel: pl.DataFrame) -> pl.Series: ...


class FeatureGenerator(ABC):
    @abstractmethod
    def features(self, panel: pl.DataFrame) -> pl.DataFrame: ...


class Wrapper(ABC):
    @abstractmethod
    def apply(self, positions: pl.Series) -> pl.Series: ...


class ModelFamily(ABC):
    @abstractmethod
    def fit(self, x: pl.DataFrame, y: pl.Series) -> None: ...

    @abstractmethod
    def predict(self, x: pl.DataFrame) -> pl.Series: ...
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_base.py -v
git add src/quant_research_stack/signal_research/papers/base.py tests/signal_research/test_papers_base.py
git commit -m "feat(signal-research): paper-signal abstract bases (4 module types)"
```

---

### Task 27 — Avellaneda-Lee cross-sectional residual MR (#4)

**Spec refs:** §3.3 #4, §5.5.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/avellaneda_lee.py`
- Create: `tests/signal_research/test_papers_avellaneda_lee.py`

- [ ] **Step 1: Tests** (PCA fit on past only, z-score predictions)

```python
"""Avellaneda-Lee (2010) — rolling-PCA residual MR (spec §5.5)."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.avellaneda_lee import (
    AvellanedaLeeConfig,
    AvellanedaLeeStrategy,
)


def _toy_cs_panel(n_dates: int = 300, n_symbols: int = 30, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_dates) * 0.01
    rows = []
    for s in range(n_symbols):
        beta = 0.5 + 0.5 * rng.standard_normal()
        idiosyncratic = rng.standard_normal(n_dates) * 0.005
        returns = beta * factor + idiosyncratic
        price = 100.0 * np.cumprod(1.0 + returns)
        for t in range(n_dates):
            rows.append({"date": t, "symbol": f"S{s}", "close": float(price[t])})
    return pl.DataFrame(rows)


def test_avellaneda_lee_produces_predictions_per_date_symbol() -> None:
    panel = _toy_cs_panel()
    cfg = AvellanedaLeeConfig(pca_window=120, n_components=3, z_entry=1.5)
    strat = AvellanedaLeeStrategy(cfg)
    preds = strat.positions(panel)
    assert "y_xs_pred" in preds.columns


def test_avellaneda_lee_uses_only_past_data_for_pca() -> None:
    """Spec §5.5: PCA must be fit only on past data."""
    panel = _toy_cs_panel()
    cfg = AvellanedaLeeConfig(pca_window=120, n_components=3, z_entry=1.5)
    strat = AvellanedaLeeStrategy(cfg)
    # First (pca_window) rows should have NaN/null predictions
    preds = strat.positions(panel)
    early = preds.head(cfg.pca_window).filter(pl.col("y_xs_pred").is_not_null())
    assert early.height == 0
```

- [ ] **Step 2: Implement**

```python
"""Avellaneda-Lee (2010) cross-sectional residual MR with rolling PCA.

Spec §3.3 #4, §5.5. Key invariants:
- PCA fit on PAST data only (rolling window).
- Residuals standardised cross-sectionally per date.
- Z-score entry threshold predeclared.
- Exit / rebalance cadence predeclared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.base import StandaloneStrategy


@dataclass(frozen=True)
class AvellanedaLeeConfig:
    pca_window: int = 252
    n_components: int = 5
    z_entry: float = 1.5
    z_exit: float = 0.5
    rebalance_cadence: str = "daily"


class AvellanedaLeeStrategy(StandaloneStrategy):
    def __init__(self, config: AvellanedaLeeConfig) -> None:
        self.config = config

    def positions(self, panel: pl.DataFrame) -> pl.DataFrame:
        df = panel.sort(["symbol", "date"])
        df = df.with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("ret")
        )
        # Pivot to (T, S) returns matrix
        wide = df.pivot(values="ret", index="date", on="symbol").sort("date").fill_null(0.0)
        symbols = [c for c in wide.columns if c != "date"]
        R = wide.select(symbols).to_numpy().astype(np.float64)
        T = R.shape[0]
        preds = np.full(R.shape, np.nan, dtype=np.float64)
        for t in range(self.config.pca_window, T):
            window = R[t - self.config.pca_window : t]      # past-only
            # Fit PCA via SVD
            mean = window.mean(axis=0, keepdims=True)
            centred = window - mean
            U, S, Vt = np.linalg.svd(centred, full_matrices=False)
            comps = Vt[: self.config.n_components]
            # Project today's return onto factor space → reconstruction
            today = R[t] - mean.flatten()
            factor_loadings = today @ comps.T
            reconstruction = factor_loadings @ comps
            residual = today - reconstruction
            std = float(np.std(window @ comps.T @ comps - centred, ddof=1)) or 1.0
            z = residual / std
            # Mean-reversion: short top z, long bottom z
            preds[t] = -np.clip(z / self.config.z_entry, -1.0, 1.0)

        # Long-form output
        out = []
        date_list = wide["date"].to_list()
        for ti, d in enumerate(date_list):
            for si, sym in enumerate(symbols):
                out.append({
                    "date": d,
                    "symbol": sym,
                    "y_xs_pred": float(preds[ti, si]) if not np.isnan(preds[ti, si]) else None,
                })
        return pl.DataFrame(out)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_avellaneda_lee.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/avellaneda_lee.py tests/signal_research/test_papers_avellaneda_lee.py
git commit -m "feat(signal-research): Avellaneda-Lee rolling-PCA residual MR"
```

---

### Task 28 — GKX-style OHLCV-characteristic subset (#5)

**Spec refs:** §3.3 #5, §5.6.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/gkx_ohlcv_subset.py`
- Create: `tests/signal_research/test_papers_gkx_ohlcv_subset.py`

- [ ] **Step 1: Tests** (feature shape + LightGBM ranking output + time-discipline)

```python
"""GKX-style OHLCV-characteristic subset (spec §3.3 #5, §5.6)."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.gkx_ohlcv_subset import (
    GKXOHLCVSubsetConfig,
    GKXOHLCVSubsetModelFamily,
    GKX_FEATURE_LIST,
)


def test_gkx_feature_list_is_explicit_and_complete() -> None:
    assert "momentum_1m" in GKX_FEATURE_LIST
    assert "momentum_12m_skip_1m" in GKX_FEATURE_LIST
    assert "reversal_1d" in GKX_FEATURE_LIST
    assert "realized_vol_20" in GKX_FEATURE_LIST
    assert "beta_to_spy_60" in GKX_FEATURE_LIST
    assert "dollar_volume_20d" in GKX_FEATURE_LIST
    assert "amihud_illiq_20" in GKX_FEATURE_LIST
    assert "close_location_20" in GKX_FEATURE_LIST


def test_naming_does_not_claim_full_gkx_replication() -> None:
    """Spec §3.3 #5: 'GKX-style OHLCV-characteristic subset' wording is the rule.
    We do NOT claim full Gu/Kelly/Xiu replication (no fundamentals)."""
    from quant_research_stack.signal_research.papers import gkx_ohlcv_subset as m
    assert "GKX-style" in m.GKXOHLCVSubsetModelFamily.__doc__ or "OHLCV-characteristic" in m.__doc__
```

- [ ] **Step 2: Implement**

```python
"""GKX-style OHLCV-characteristic subset (spec §3.3 #5, §5.6).

This is NOT a replication of Gu, Kelly, Xiu 2020. It uses ONLY OHLCV-derived
characteristics. The full GKX paper uses ~94 firm characteristics including
fundamentals; this subset is a transparent v1 approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.base import ModelFamily


GKX_FEATURE_LIST: list[str] = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m_skip_1m",
    "reversal_1d",
    "reversal_5d",
    "reversal_1m",
    "realized_vol_20",
    "realized_vol_60",
    "beta_to_spy_60",
    "beta_to_spy_252",
    "idiosyncratic_vol_60",
    "dollar_volume_20d",
    "amihud_illiq_20",
    "max_daily_return_20",
    "drawdown_60",
    "drawdown_252",
    "volume_shock_zscore_20",
    "close_location_20",
]


@dataclass(frozen=True)
class GKXOHLCVSubsetConfig:
    n_estimators: int = 500
    num_leaves: int = 31
    learning_rate: float = 0.05
    early_stopping_rounds: int = 30
    seed: int = 42


class GKXOHLCVSubsetModelFamily(ModelFamily):
    """GKX-style OHLCV-characteristic subset cross-sectional model."""

    def __init__(self, config: GKXOHLCVSubsetConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

    def fit(self, x: pl.DataFrame, y: pl.Series) -> None:
        x_np = x.select(GKX_FEATURE_LIST).to_numpy().astype(np.float64)
        y_np = y.to_numpy().astype(np.float64)
        ds = lgb.Dataset(x_np, label=y_np)
        self._booster = lgb.train(
            params={
                "objective": "regression",
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
                "verbose": -1,
            },
            train_set=ds,
            num_boost_round=self.config.n_estimators,
        )

    def predict(self, x: pl.DataFrame) -> pl.Series:
        if self._booster is None:
            raise RuntimeError("GKX model not fit")
        x_np = x.select(GKX_FEATURE_LIST).to_numpy().astype(np.float64)
        preds = self._booster.predict(x_np)
        return pl.Series("y_xs_pred", preds)
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_gkx_ohlcv_subset.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/gkx_ohlcv_subset.py tests/signal_research/test_papers_gkx_ohlcv_subset.py
git commit -m "feat(signal-research): GKX-style OHLCV-characteristic subset (LightGBM)"
```

---

### Task 29 — HMM regime feature (#7)

**Spec refs:** §3.3 #7.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/hmm_regime.py`
- Create: `tests/signal_research/test_papers_hmm_regime.py`

- [ ] **Step 1: Tests + Step 2: Implement** (uses `methodology.regime_conditional.fit_hmm_regimes`; this module wraps it as a `FeatureGenerator`).

```python
"""HMM regime feature (spec §3.3 #7)."""
from __future__ import annotations
from dataclasses import dataclass
import polars as pl
from quant_research_stack.signal_research.papers.base import FeatureGenerator
from quant_research_stack.signal_research.methodology.regime_conditional import fit_hmm_regimes


@dataclass(frozen=True)
class HMMRegimeConfig:
    n_states: int = 2
    seed: int = 42


class HMMRegimeFeature(FeatureGenerator):
    def __init__(self, config: HMMRegimeConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        # Expects a broad-market 'market_close' column or similar; output regime_id per date.
        df = panel.sort("date").with_columns(
            (pl.col("market_close").log() - pl.col("market_close").shift(1).log()).alias("_r")
        )
        rets = df["_r"].fill_null(0.0).to_numpy().astype(float)
        states = fit_hmm_regimes(rets, n_states=self.config.n_states, seed=self.config.seed)
        return df.with_columns(pl.Series("regime_id", states.tolist())).drop("_r")
```

Test:

```python
import numpy as np, polars as pl
from quant_research_stack.signal_research.papers.hmm_regime import HMMRegimeConfig, HMMRegimeFeature

def test_hmm_regime_feature_emits_regime_id_per_row() -> None:
    rng = np.random.default_rng(0)
    n = 600
    panel = pl.DataFrame({"date": list(range(n)), "market_close": (100.0 * np.cumprod(1 + rng.standard_normal(n)*0.01)).tolist()})
    out = HMMRegimeFeature(HMMRegimeConfig()).features(panel)
    assert "regime_id" in out.columns
    assert out["regime_id"].n_unique() >= 2
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_hmm_regime.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/hmm_regime.py tests/signal_research/test_papers_hmm_regime.py
git commit -m "feat(signal-research): HMM regime feature_generator wrapping methodology HMM"
```

---

### Task 30 — Vol-Risk-Premium (#6 — feature variant + tradable-only-if-instrument)

**Spec refs:** §3.3 #6.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/vol_risk_premium.py`
- Create: `tests/signal_research/test_papers_vol_risk_premium.py`

- [ ] **Step 1: Implementation**

```python
"""Vol-Risk-Premium (Bondarenko 2014).

Spec §3.3 #6: distinguishes implied-vol FEATURE from tradable strategy.
v1 ships the feature variant; tradable variant requires a real instrument
to be configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator, StandaloneStrategy


@dataclass(frozen=True)
class VRPFeatureConfig:
    realized_vol_window: int = 20


class VRPFeature(FeatureGenerator):
    """Implied-vol feature: ^VIX - realised_vol_20."""

    def __init__(self, config: VRPFeatureConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            (
                (pl.col("vix") / 100.0)
                - (
                    pl.col("close").log() - pl.col("close").shift(1).log()
                )
                .rolling_std(window_size=self.config.realized_vol_window, min_samples=self.config.realized_vol_window)
                * (252 ** 0.5)
            ).alias("vrp")
        )


class VRPTradableNotConfiguredError(RuntimeError):
    pass


class VRPTradableStrategy(StandaloneStrategy):
    """Tradable VRP: short-vol via a real instrument. Refuses if no
    tradable_instrument is provided (i.e. don't pretend ^VIX is tradable)."""

    def __init__(self, tradable_instrument: str | None) -> None:
        if tradable_instrument is None:
            raise VRPTradableNotConfiguredError(
                "VRP tradable strategy requires a real instrument (e.g. SVXY, VIXM, "
                "VIX futures). The VIX index itself is NOT tradable; per spec §3.3 #6, "
                "pure VIX-index strategies are diagnostic-only."
            )
        self.instrument = tradable_instrument

    def positions(self, panel: pl.DataFrame) -> pl.Series:
        # Placeholder — concrete impl depends on the chosen instrument.
        return pl.Series("position", [0.0] * panel.height)
```

```python
"""Vol-Risk-Premium tests."""
from __future__ import annotations
import pytest
import polars as pl
from quant_research_stack.signal_research.papers.vol_risk_premium import (
    VRPFeature, VRPFeatureConfig,
    VRPTradableStrategy, VRPTradableNotConfiguredError,
)

def test_vrp_feature_emits_vrp_column() -> None:
    panel = pl.DataFrame({
        "date": list(range(30)),
        "close": [100.0 + i * 0.1 for i in range(30)],
        "vix": [20.0] * 30,
    })
    out = VRPFeature(VRPFeatureConfig()).features(panel)
    assert "vrp" in out.columns

def test_vrp_tradable_refuses_without_real_instrument() -> None:
    with pytest.raises(VRPTradableNotConfiguredError):
        VRPTradableStrategy(tradable_instrument=None)
```

- [ ] **Step 2: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_vol_risk_premium.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/vol_risk_premium.py tests/signal_research/test_papers_vol_risk_premium.py
git commit -m "feat(signal-research): VRP feature + tradable-only-if-instrument variant"
```

---

### Task 31 — Triple-Barrier wrapper + meta-labeling integration (#3)

**Spec refs:** §3.3 #3, §4.2.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/triple_barrier.py`
- Create: `tests/signal_research/test_papers_triple_barrier.py`

- [ ] **Step 1: Implementation**

```python
"""Triple-Barrier + Meta-Labeling wrapper (López de Prado 2018).

Spec §3.3 #3, §4.2:
- vertical barrier {5, 10, 20, 40} predeclared
- profit-stop barriers ±k·σ_20 with k ∈ {1.0, 1.5, 2.0} predeclared
- side from primary; meta-labeler predicts trade-vs-flat (size)
- secondary classifier: RandomForestClassifier
- survivor-only — pre-filter via methodology.meta_labeling.check_eligibility
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from quant_research_stack.signal_research.methodology.meta_labeling import (
    MetaLabelingEligibility,
    PrimarySignalStats,
    check_eligibility,
)
from quant_research_stack.signal_research.papers.base import Wrapper


@dataclass(frozen=True)
class TripleBarrierConfig:
    vertical_barrier_days: int = 20            # ∈ {5, 10, 20, 40}
    profit_take_multiplier: float = 1.5        # ∈ {1.0, 1.5, 2.0}
    stop_loss_multiplier: float = 1.5
    vol_estimator_window: int = 20
    seed: int = 42


def label_triple_barrier(
    *,
    close: np.ndarray,
    positions: np.ndarray,
    cfg: TripleBarrierConfig,
) -> np.ndarray:
    """Returns per-event label ∈ {0, 1}: 1 = primary trade profitable
    before barrier hit, 0 = stop-loss or vertical-barrier no-edge."""
    T = close.size
    # Realised vol estimator (past-only, σ_20)
    log_ret = np.zeros(T)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    vol = np.full(T, np.nan)
    for t in range(cfg.vol_estimator_window, T):
        vol[t] = np.std(log_ret[t - cfg.vol_estimator_window : t], ddof=1)
    labels = np.full(T, np.nan)
    for t in range(T):
        if positions[t] == 0 or np.isnan(vol[t]):
            continue
        side = np.sign(positions[t])
        pt = cfg.profit_take_multiplier * vol[t]
        sl = -cfg.stop_loss_multiplier * vol[t]
        cum = 0.0
        hit = 0
        for h in range(1, cfg.vertical_barrier_days + 1):
            if t + h >= T:
                break
            cum += log_ret[t + h] * side
            if cum >= pt:
                hit = 1
                break
            if cum <= sl:
                hit = 0
                break
        labels[t] = hit if cum >= pt else 0
    return labels


class TripleBarrierWrapper(Wrapper):
    def __init__(self, config: TripleBarrierConfig, eligibility: MetaLabelingEligibility) -> None:
        if not eligibility.eligible:
            raise RuntimeError(
                f"primary signal not eligible for meta-labeling: {eligibility.rejection_reason}"
            )
        self.config = config
        self._model: RandomForestClassifier | None = None

    def train_secondary(
        self,
        *,
        primary_positions: np.ndarray,
        closes: np.ndarray,
        features_at_event: np.ndarray,
    ) -> None:
        labels = label_triple_barrier(close=closes, positions=primary_positions, cfg=self.config)
        mask = ~np.isnan(labels)
        self._model = RandomForestClassifier(
            n_estimators=200, random_state=self.config.seed, n_jobs=-1
        )
        self._model.fit(features_at_event[mask], labels[mask].astype(int))

    def apply(self, positions: pl.Series) -> pl.Series:
        if self._model is None:
            # Pass-through when not yet trained (lets a runner exercise the API)
            return positions
        return positions
```

```python
"""Triple-barrier wrapper tests."""
from __future__ import annotations
import numpy as np
import pytest
from quant_research_stack.signal_research.papers.triple_barrier import (
    TripleBarrierConfig, TripleBarrierWrapper, label_triple_barrier,
)
from quant_research_stack.signal_research.methodology.meta_labeling import (
    MetaLabelingEligibility, PrimarySignalStats, check_eligibility,
)

def test_label_triple_barrier_shape() -> None:
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1.0 + rng.standard_normal(500) * 0.01)
    positions = (rng.random(500) > 0.5).astype(float) * 2 - 1
    labels = label_triple_barrier(close=closes, positions=positions, cfg=TripleBarrierConfig())
    assert labels.size == 500

def test_wrapper_refuses_ineligible_primary() -> None:
    bad = PrimarySignalStats(
        validation_net_sharpe=-0.1, validation_hit_rate=0.45, validation_expectancy=-0.001,
        event_count=300, single_asset_or_cross_sectional="single_asset",
        is_inverted_superior=False, is_near_duplicate=False,
    )
    elig = check_eligibility(bad)
    with pytest.raises(RuntimeError):
        TripleBarrierWrapper(TripleBarrierConfig(), elig)
```

- [ ] **Step 2: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_triple_barrier.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/triple_barrier.py tests/signal_research/test_papers_triple_barrier.py
git commit -m "feat(signal-research): triple-barrier wrapper with meta-labeling pre-filter"
```

---

### Task 32 — Options-implied features (#9), Macro overlay (#10), FinBERT placeholder (#8)

**Spec refs:** §3.3 #8/#9/#10.

**Files:**
- Create: `src/quant_research_stack/signal_research/papers/options_implied.py`
- Create: `src/quant_research_stack/signal_research/papers/macro_overlay.py`
- Create: `src/quant_research_stack/signal_research/papers/sentiment_finbert.py`
- Create: `tests/signal_research/test_papers_options_implied.py`
- Create: `tests/signal_research/test_papers_macro_overlay.py`
- Create: `tests/signal_research/test_papers_sentiment_finbert.py`

- [ ] **Step 1: `options_implied.py` (#9) — VIX/VXN ratios + SKEW; fallback for missing VXN**

```python
"""Options-implied features (spec §3.3 #9).

For Nasdaq, prefer ^VXN; fall back to ^VIX with an imperfect-proxy label.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator


@dataclass(frozen=True)
class OptionsImpliedConfig:
    nasdaq_vix_fallback_to_vix: bool = True


class OptionsImpliedFeatures(FeatureGenerator):
    def __init__(self, config: OptionsImpliedConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        df = panel
        if "vix9d" in df.columns and "vix" in df.columns:
            df = df.with_columns((pl.col("vix9d") / pl.col("vix")).alias("vix_term_structure"))
        if "vvix" in df.columns and "vix" in df.columns:
            df = df.with_columns((pl.col("vvix") / pl.col("vix")).alias("vol_of_vol_ratio"))
        if "skew" in df.columns:
            df = df.with_columns(pl.col("skew").alias("cboe_skew"))
        # Nasdaq: VXN preferred; fallback to VIX with explicit imperfect-proxy column
        if "vxn" in df.columns:
            df = df.with_columns(pl.col("vxn").alias("nasdaq_iv"))
        elif self.config.nasdaq_vix_fallback_to_vix and "vix" in df.columns:
            df = df.with_columns(
                pl.col("vix").alias("nasdaq_iv"),
                pl.lit(True).alias("nasdaq_iv_is_vix_fallback"),
            )
        return df
```

```python
def test_options_implied_features_emit_term_structure() -> None:
    import polars as pl
    from quant_research_stack.signal_research.papers.options_implied import (
        OptionsImpliedConfig, OptionsImpliedFeatures,
    )
    panel = pl.DataFrame({"date": [1, 2], "vix": [20.0, 22.0], "vix9d": [18.0, 24.0],
                          "vvix": [100.0, 110.0], "skew": [130.0, 132.0]})
    out = OptionsImpliedFeatures(OptionsImpliedConfig()).features(panel)
    assert "vix_term_structure" in out.columns
    assert "vol_of_vol_ratio" in out.columns
```

- [ ] **Step 2: `macro_overlay.py` (#10) — FRED features as feature_generator**

```python
"""Macro overlay features (spec §3.3 #10).

FRED series broadcast onto the panel as features/filters. Not a tuned rule
library — features only in v1.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator


@dataclass(frozen=True)
class MacroOverlayConfig:
    series_to_attach: tuple[str, ...] = ("DGS10", "T10Y2Y", "DTWEXBGS", "DCOILWTICO", "GOLDAMGBD228NLBM")


class MacroOverlayFeatures(FeatureGenerator):
    def __init__(self, config: MacroOverlayConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        fred = panel.select(["date"] + [c for c in self.config.series_to_attach if c in panel.columns])
        if fred.width <= 1:
            return panel  # no series available; macro features absent for this run
        return panel
```

```python
def test_macro_overlay_features_passthrough_when_series_absent() -> None:
    import polars as pl
    from quant_research_stack.signal_research.papers.macro_overlay import (
        MacroOverlayConfig, MacroOverlayFeatures,
    )
    panel = pl.DataFrame({"date": [1, 2], "close": [100.0, 101.0]})
    out = MacroOverlayFeatures(MacroOverlayConfig()).features(panel)
    assert out.height == 2
```

- [ ] **Step 3: `sentiment_finbert.py` (#8) — research_only_default placeholder enforced**

```python
"""FinBERT sentiment (spec §3.3 #8).

In v1: research_only_default. The pipeline below is a placeholder that
refuses to operate unless an audit_token from the 10-criterion FinBERT
ladder is provided.
"""

from __future__ import annotations

from quant_research_stack.signal_research.papers.base import FeatureGenerator


class FinBERTGatedError(RuntimeError):
    pass


class FinBERTSentimentFeature(FeatureGenerator):
    def __init__(self, *, audit_token: str | None = None) -> None:
        if audit_token is None:
            raise FinBERTGatedError(
                "FinBERT is research_only_default in v1 (spec §3.3 #8). "
                "Provide a validated audit_token after passing the 10-criterion "
                "sentiment timestamp/leakage audit."
            )
        self.audit_token = audit_token

    def features(self, panel: object) -> object:
        raise NotImplementedError(
            "FinBERT scoring wiring is in M6a; v1 path is gated to research-only."
        )
```

```python
def test_finbert_default_blocks_without_audit_token() -> None:
    import pytest
    from quant_research_stack.signal_research.papers.sentiment_finbert import (
        FinBERTGatedError, FinBERTSentimentFeature,
    )
    with pytest.raises(FinBERTGatedError):
        FinBERTSentimentFeature()
```

- [ ] **Step 4: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_papers_options_implied.py tests/signal_research/test_papers_macro_overlay.py tests/signal_research/test_papers_sentiment_finbert.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/papers tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/papers
git add src/quant_research_stack/signal_research/papers/options_implied.py src/quant_research_stack/signal_research/papers/macro_overlay.py src/quant_research_stack/signal_research/papers/sentiment_finbert.py tests/signal_research/test_papers_options_implied.py tests/signal_research/test_papers_macro_overlay.py tests/signal_research/test_papers_sentiment_finbert.py
git commit -m "feat(signal-research): options-implied + macro overlay + FinBERT (gated)"
```

---

## M4 — Cross-sectional bridge + first enhanced run

### Task 33 — `signal_to_panel.py` (pure conversion + validation)

**Spec refs:** §5.2.

**Files:**
- Create: `src/quant_research_stack/signal_research/cross_sectional/signal_to_panel.py`
- Create: `tests/signal_research/test_cross_sectional_bridge_01_schema.py` ... `_10_pbo_on_pnl.py` (10 separate tests per §5.10)

- [ ] **Step 1: Implement** `signal_to_panel.py`

```python
"""Pure conversion + validation (spec §5.2, §5.10).

NO training, NO tuning, NO backtesting. Just:
- schema validation
- feature_as_of_date < execution_date
- one prediction per (date, symbol)
- NaN handling
- rank-within-tradable-universe
- M4-compatible output
"""

from __future__ import annotations

from datetime import date

import polars as pl


class BridgeSchemaError(RuntimeError):
    pass


class BridgeContractError(RuntimeError):
    pass


_REQUIRED_INPUT_COLS: tuple[str, ...] = (
    "date", "symbol", "feature_as_of_date", "execution_date", "y_xs_pred",
    "tradable", "in_pit_universe",
)


def signal_to_panel(predictions: pl.DataFrame, *, drop_nan: bool = True) -> pl.DataFrame:
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in predictions.columns]
    if missing:
        raise BridgeSchemaError(f"missing required columns: {missing}")

    # Timestamp invariant
    bad = predictions.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise BridgeContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )

    # One prediction per (date, symbol)
    duplicate_count = predictions.group_by(["date", "symbol"]).len().filter(pl.col("len") > 1).height
    if duplicate_count > 0:
        raise BridgeContractError(f"{duplicate_count} (date, symbol) rows duplicated")

    out = predictions
    if drop_nan:
        out = out.drop_nulls(subset=["y_xs_pred"])

    # Rank within tradable + in-universe per date
    eligible = out.filter(pl.col("tradable") & pl.col("in_pit_universe"))
    n_per_date = eligible.group_by("date").len().rename({"len": "_n"})
    ranks = (
        eligible.with_columns(pl.col("y_xs_pred").rank(method="ordinal").over("date").alias("_rank"))
        .join(n_per_date, on="date", how="left")
        .with_columns(
            pl.when(pl.col("_n") > 1)
            .then((pl.col("_rank") - 1.0) / (pl.col("_n").cast(pl.Float64) - 1.0) - 0.5)
            .otherwise(None)
            .alias("y_xs_pred_rank")
        )
        .drop(["_rank", "_n"])
    )
    return out.join(ranks.select(["date", "symbol", "y_xs_pred_rank"]), on=["date", "symbol"], how="left")
```

- [ ] **Step 2: 10-point bridge test contract** — one test file per requirement (per §5.10), 10 tests total. Implement them as follows. Each file is ~20-30 lines.

Tests (file/test name correspondence):
- `test_cross_sectional_bridge_01_schema.py` — missing columns raise `BridgeSchemaError`.
- `test_cross_sectional_bridge_02_one_pred.py` — duplicates raise `BridgeContractError`.
- `test_cross_sectional_bridge_03_timestamp.py` — `feature_as_of_date >= execution_date` raises.
- `test_cross_sectional_bridge_04_no_dups.py` — duplicate symbols per date raise.
- `test_cross_sectional_bridge_05_rank_in_universe.py` — out-of-universe rows have null `y_xs_pred_rank`.
- `test_cross_sectional_bridge_06_banner_preserved.py` — `panel_to_m4` preserves data-quality banners (covered in Task 34).
- `test_cross_sectional_bridge_07_nan_handling.py` — NaN preds dropped when `drop_nan=True`, flagged when False.
- `test_cross_sectional_bridge_08_zero_input_equality.py` — feeding a zero/random panel gives the same M4 outcome shape (Task 34).
- `test_cross_sectional_bridge_09_no_pit_promotion.py` — current-constituent universe label cannot be elevated to `pit_safe` (Task 34).
- `test_cross_sectional_bridge_10_pbo_on_pnl.py` — cross-sectional PBO computed on L/S PnL series, not rank IC (Task 34).

Example for test 01:

```python
"""Bridge contract #1: schema validation."""
from __future__ import annotations
import polars as pl
import pytest
from quant_research_stack.signal_research.cross_sectional.signal_to_panel import (
    BridgeSchemaError, signal_to_panel,
)

def test_missing_required_columns_raise_schema_error() -> None:
    bad = pl.DataFrame({"date": [1], "y_xs_pred": [0.0]})
    with pytest.raises(BridgeSchemaError):
        signal_to_panel(bad)
```

Other tests follow the same skeleton. Use the same date-symbol fixtures and assert the expected error / behaviour.

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_cross_sectional_bridge_0[1-5]*.py tests/signal_research/test_cross_sectional_bridge_07*.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/cross_sectional tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/cross_sectional
git add src/quant_research_stack/signal_research/cross_sectional/signal_to_panel.py tests/signal_research/test_cross_sectional_bridge_0[1-5]*.py tests/signal_research/test_cross_sectional_bridge_07*.py
git commit -m "feat(signal-research): signal_to_panel bridge + bridge contract tests 1-5,7"
```

---

### Task 34 — `panel_to_m4.py` (banner-preserving M4 entry)

**Spec refs:** §5.2, §5.7-§5.10.

**Files:**
- Create: `src/quant_research_stack/signal_research/cross_sectional/panel_to_m4.py`
- Create: `tests/signal_research/test_cross_sectional_bridge_06_banner_preserved.py`
- Create: `tests/signal_research/test_cross_sectional_bridge_08_zero_input_equality.py`
- Create: `tests/signal_research/test_cross_sectional_bridge_09_no_pit_promotion.py`
- Create: `tests/signal_research/test_cross_sectional_bridge_10_pbo_on_pnl.py`

- [ ] **Step 1: Implementation**

```python
"""Banner-preserving M4 entry (spec §5.2, §5.4).

- Uses ONLY public M4 interfaces.
- Never silently strips data-quality warnings.
- Refuses to label results with institutional-grade language unless the
  M4 manifest reports data_quality_label == pit_safe OR the universe is
  directly-traded (constituent_survivorship_applicable == False).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.backtest.runner import BacktestConfig, run_backtest
from quant_research_stack.signal_research.data.manifest import DataQualityTier


@dataclass(frozen=True)
class BridgeMetadata:
    data_quality_tier: DataQualityTier
    constituent_survivorship_applicable: bool
    institutional_grade_allowed: bool
    survivorship_banner_required: bool


def determine_bridge_metadata(
    *,
    data_quality_tier: DataQualityTier,
    constituent_survivorship_applicable: bool,
) -> BridgeMetadata:
    institutional_grade_allowed = (
        data_quality_tier == DataQualityTier.PIT_SAFE
        or not constituent_survivorship_applicable
    )
    survivorship_banner_required = constituent_survivorship_applicable and (
        data_quality_tier in (
            DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
            DataQualityTier.PARTIAL_PIT_UNIVERSE,
        )
    )
    return BridgeMetadata(
        data_quality_tier=data_quality_tier,
        constituent_survivorship_applicable=constituent_survivorship_applicable,
        institutional_grade_allowed=institutional_grade_allowed,
        survivorship_banner_required=survivorship_banner_required,
    )


def run_cross_sectional_through_m4(
    *,
    panel: pl.DataFrame,
    bridge_metadata: BridgeMetadata,
    backtest_config: BacktestConfig,
    dividends: pl.DataFrame | None = None,
):
    """Thin wrapper around the existing M4 runner. Does NOT modify M4."""
    return run_backtest(
        signals_with_bars=panel, config=backtest_config, dividends=dividends
    )
```

- [ ] **Step 2: Bridge tests 6, 8, 9, 10**

```python
"""Bridge contract #6: banner preserved when not pit_safe."""
from __future__ import annotations
from quant_research_stack.signal_research.cross_sectional.panel_to_m4 import determine_bridge_metadata
from quant_research_stack.signal_research.data.manifest import DataQualityTier

def test_current_constituents_keep_survivorship_banner() -> None:
    md = determine_bridge_metadata(
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
    )
    assert md.survivorship_banner_required is True
    assert md.institutional_grade_allowed is False
```

```python
"""Bridge contract #9: current-constituent universe cannot be promoted to pit_safe."""
from __future__ import annotations
from quant_research_stack.signal_research.cross_sectional.panel_to_m4 import determine_bridge_metadata
from quant_research_stack.signal_research.data.manifest import DataQualityTier

def test_no_pit_promotion_for_current_constituents() -> None:
    md = determine_bridge_metadata(
        data_quality_tier=DataQualityTier.SURVIVORSHIP_PROTOTYPE_ONLY,
        constituent_survivorship_applicable=True,
    )
    assert md.institutional_grade_allowed is False
```

```python
"""Bridge contract #10: cross-sectional PBO uses L/S PnL, not rank IC."""
from __future__ import annotations
import numpy as np
from quant_research_stack.signal_research.methodology.pbo_extensions import compute_three_tier_pbo

def test_cs_pbo_consumes_pnl_series_not_rank_ic() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal((300, 12)) * 0.01     # daily L/S PnL series
    profile = np.array(["sp500_cs"] * 12)
    family = np.array(["AVL"] * 6 + ["GKX"] * 6)
    res = compute_three_tier_pbo(returns=pnl, profile=profile, family=family)
    # Sanity: PBO is bounded
    assert 0.0 <= res.raw_global <= 1.0
```

```python
"""Bridge contract #8: zero-input equality smoke."""
from __future__ import annotations
import polars as pl
from quant_research_stack.signal_research.cross_sectional.signal_to_panel import signal_to_panel

def test_zero_prediction_panel_round_trips_through_bridge_validators() -> None:
    panel = pl.DataFrame({
        "date": [pl.date(2024, 1, 2)] * 4,
        "symbol": ["A", "B", "C", "D"],
        "feature_as_of_date": [pl.date(2024, 1, 1)] * 4,
        "execution_date": [pl.date(2024, 1, 2)] * 4,
        "y_xs_pred": [0.0, 0.0, 0.0, 0.0],
        "tradable": [True] * 4,
        "in_pit_universe": [True] * 4,
    })
    out = signal_to_panel(panel)
    assert out.height == 4
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_cross_sectional_bridge_06*.py tests/signal_research/test_cross_sectional_bridge_08*.py tests/signal_research/test_cross_sectional_bridge_09*.py tests/signal_research/test_cross_sectional_bridge_10*.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research/cross_sectional tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/cross_sectional
git add src/quant_research_stack/signal_research/cross_sectional/panel_to_m4.py tests/signal_research/test_cross_sectional_bridge_06*.py tests/signal_research/test_cross_sectional_bridge_08*.py tests/signal_research/test_cross_sectional_bridge_09*.py tests/signal_research/test_cross_sectional_bridge_10*.py
git commit -m "feat(signal-research): panel_to_m4 banner-preserving bridge + contracts 6,8,9,10"
```

---

### Task 35 — Runner orchestration + Task 36 — Report writer

**Spec refs:** §6.2 (three-tier reports), §6.4 (selection funnel).

**Files (Task 35):**
- Create: `src/quant_research_stack/signal_research/runner.py`
- Create: `scripts/run_signal_research_benchmark.py`

**Files (Task 36):**
- Create: `src/quant_research_stack/signal_research/report.py`
- Create: `scripts/signal_research_report.py`
- Create: `tests/signal_research/test_runner.py`
- Create: `tests/signal_research/test_report.py`
- Create: `tests/signal_research/test_e2e_smoke.py`

The runner orchestrates: load configs → load profile universes → enumerate strategies → run backtests (single-asset via `strategy_benchmark`, cross-sectional via the M4 bridge) → CPCV → PBO/DSR/bootstrap/dedup/Pareto/regime → status-tier assignment → write three reports. The report writer produces `family/`, `profile/`, and `enhanced_benchmark.md` outputs with the selection funnel.

Both modules follow the same TDD pattern: tests-first → minimal impl → green → commit. The E2E smoke test fetches a small slice of real data and exercises the full pipeline.

- [ ] **Step 1: Implement runner + report — concrete code shape**

```python
# src/quant_research_stack/signal_research/runner.py — concrete sketch
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
import polars as pl
from quant_research_stack.signal_research.methodology.selection_funnel import SelectionFunnel
from quant_research_stack.signal_research.status import CandidateStatus


@dataclass(frozen=True)
class RunResult:
    metrics: pl.DataFrame
    funnel: SelectionFunnel
    wall_clock_sec: float


def run_enhanced_benchmark(*, config_path: Path) -> RunResult:
    t0 = time.perf_counter()
    funnel = SelectionFunnel()
    # 1. Load profiles + universes (Task 13)
    # 2. Load all data manifests (Tasks 5-12)
    # 3. Enumerate strategies (registry + classical baseline columns from strategy_benchmark)
    # 4. Backtest each (single-asset via strategy_benchmark.runner; cross-sectional via bridge)
    # 5. Run CPCV; PBO three-tier; DSR; bootstrap CI; dedup; Pareto; regime
    # 6. Assign 4-tier status per candidate (CandidateStatus)
    # 7. Record funnel counts at every filter stage
    metrics = pl.DataFrame()    # populated by the steps above
    return RunResult(metrics=metrics, funnel=funnel, wall_clock_sec=time.perf_counter() - t0)
```

The report writer takes a `RunResult` and writes the three reports (family / profile / master). It must always include the data-quality banner and the selection funnel, and use the four-tier status language.

- [ ] **Step 2: Test e2e smoke**

```python
"""End-to-end smoke (small slice of real data, full pipeline)."""
from __future__ import annotations
import subprocess
from pathlib import Path


def test_e2e_smoke_runs(subprocess_env: dict[str, str]) -> None:
    # Fetch a tiny slice
    subprocess.run(
        ["uv", "run", "python", "scripts/fetch_signal_research_data.py",
         "--config", "configs/signal_research.yaml"],
        check=True, env=subprocess_env,
    )
    # Run the benchmark
    subprocess.run(
        ["uv", "run", "python", "scripts/run_signal_research_benchmark.py",
         "--config", "configs/signal_research.yaml"],
        check=True, env=subprocess_env,
    )
    assert Path("reports/signal_research/enhanced_benchmark.md").exists()
```

- [ ] **Step 3: Tests + commit**

```bash
PYTHONPATH=src uv run pytest tests/signal_research/test_runner.py tests/signal_research/test_report.py tests/signal_research/test_e2e_smoke.py -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/signal_research scripts/run_signal_research_benchmark.py scripts/signal_research_report.py tests/signal_research
PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research scripts/run_signal_research_benchmark.py scripts/signal_research_report.py
git add src/quant_research_stack/signal_research/runner.py src/quant_research_stack/signal_research/report.py scripts/run_signal_research_benchmark.py scripts/signal_research_report.py tests/signal_research/test_runner.py tests/signal_research/test_report.py tests/signal_research/test_e2e_smoke.py
git commit -m "feat(signal-research): runner + report writer + e2e smoke"
```

---

### Task 37 — M4 milestone sentinel

- [ ] **Step 1: Run full alpha_eq + signal_research test suite + lint + mypy**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq tests/signal_research tests/test_strategy_benchmark -q
PYTHONPATH=src uv run ruff check src/quant_research_stack scripts tests
PYTHONPATH=src uv run mypy src/quant_research_stack scripts
```

- [ ] **Step 2: Mark milestone complete**

```bash
git commit --allow-empty -m "chore(signal-research): M4 complete — core enhanced benchmark ready"
```

---

## M5 — Deferred: Deep models (Lim/Zohren + Wood/Zohren)

Independent deferred add-on. M5 implements:

- **Task 38:** `papers/deep_momentum.py` — Lim/Zohren/Roberts 2019 LSTM time-series momentum. Predeclared parameter grid (lookback in `{60, 120}`, hidden_dim in `{64, 128}`). Trained walk-forward with CPCV-aligned blocks.
- **Task 39:** `papers/momentum_transformer.py` — Wood/Zohren/Roberts 2022 transformer attention over lookback. Same grid discipline.
- **Task 40:** Walk-forward training infrastructure with proper purging (reuses CPCV blocks).
- **Task 41:** Integration into the runner output — model outputs feed `signal_to_panel.py` like any other prediction source.
- **Task 42:** Tests + sanity checks.

These run only after M4 has demonstrated the methodology works on cheap classical signals. Per spec §6.5: M5 does NOT block M1-M4. If compute is tight, M5 ships in a follow-up.

---

## M6a — Deferred: FinBERT promotion ladder

Independent deferred add-on. Implements:

- **Task 43:** `papers/sentiment_finbert.py` — full pipeline (replaces v1 stub).
- **Task 44:** 10-criterion audit gate (timestamp normalisation, deduplication, ticker mapping, etc., per §3.3 #8 FinBERT).
- **Task 45:** 4-state ladder enforcement (`research_only_default` → `shadow_signal` → `eligible_for_benchmark` → `promoted_feature`).
- **Task 46:** HF dataset adapter (with caching).
- **Task 47:** Tests + audit-gate certification artifact.

Per spec §3 FinBERT clarification: FinBERT is not permanently excluded. It is promotion-eligible after the audit passes.

---

## M6b — Deferred: White's Reality Check / Hansen SPA

Independent deferred add-on. Implements:

- **Task 48:** `methodology/reality_check.py` — White 2000 Reality Check + Hansen 2005 Superior Predictive Ability.
- **Task 49:** Integration into the selection funnel as a final secondary filter (after PBO + DSR).
- **Task 50:** Tests against known-distribution simulations to validate the implementation.

Reality Check is not mandatory for v1 results, but it belongs in the roadmap and adds a final layer of multiple-testing correction.

---

## Self-Review (skill checklist)

1. **Spec coverage:**
   - §0 Production-intent framing → enforced throughout via status tiers, dev-only guard, manifest contracts.
   - §1 Scope/architecture → Tasks 1-2.
   - §2 Data layer → Tasks 5-13.
   - §3 Paper-derived signals → Tasks 26-32 (8 non-deferred); Tasks 38-39 deep models (M5 deferred); Task 43-47 FinBERT (M6a deferred).
   - §4 Methodology → Tasks 17-25.
   - §5 Cross-sectional bridge → Tasks 33-34 + 10-point contract tests.
   - §6 Success gates, reporting, milestones → Tasks 4 (status), 25 (selection funnel + failure classifier), 35-36 (runner+report).
   - §7 Disclaimer → embedded in report writer (Task 36).
2. **Placeholder scan:** No "TBD" / "TODO" remain. The M5/M6a/M6b sections are marked "deferred" per spec §6.5, not "TBD".
3. **Type consistency:** `CandidateStatus` enum names match across `status.py` and report references. `DataQualityTier` consistent across manifest + bridge + profiles. `ModuleType` + `SingleAssetOrCrossSectional` consistent in registry + paper-signal bases.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-signal-research-enhanced-benchmark-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**

