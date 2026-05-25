# Crypto Strategy Research Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a reproducible crypto strategy search loop that logs every candidate, applies costed chronological validation, estimates PBO, and promotes only robust out-of-sample strategies.

**Architecture:** Add a focused `quant_research_stack.crypto_research` package for dataset manifests, strategy registry, costed backtests, PBO diagnostics, and markdown/parquet outputs. Add a CLI script that runs the first 18-month BTCUSDT minute-data iteration from local data, with no holdout tuning and explicit failure reporting.

**Tech Stack:** Python, Polars, NumPy, pytest, ruff, mypy.

---

### Task 1: Dataset Manifest And Period Splits

**Files:**
- Create: `src/quant_research_stack/crypto_research/data.py`
- Test: `tests/test_crypto_research_loop.py`

- [x] **Step 1: Write failing tests** for dataset manifest fields, 18-month period split ordering, and timestamp coverage rejection when data is too short.
- [x] **Step 2: Implement** `DatasetManifest`, `load_btcusdt_1m_panel`, `write_dataset_manifest`, and `build_chronological_periods`.
- [x] **Step 3: Verify** with `PYTHONPATH=src uv run pytest tests/test_crypto_research_loop.py -q`.

### Task 2: Strategy Registry

**Files:**
- Create: `src/quant_research_stack/crypto_research/strategies.py`
- Test: `tests/test_crypto_research_loop.py`

- [x] **Step 1: Write failing tests** requiring unique `strategy_id` values and at least the configured number of variants across momentum, mean reversion, breakout, volatility, liquidity, and paper-derived families.
- [x] **Step 2: Implement** `StrategyVariant` and deterministic candidate generation with every parameterized variant logged.
- [x] **Step 3: Verify** focused tests.

### Task 3: Costed Backtester

**Files:**
- Create: `src/quant_research_stack/crypto_research/backtest.py`
- Test: `tests/test_crypto_research_loop.py`

- [x] **Step 1: Write failing tests** for one-bar execution delay, turnover costs, no-cost/spread-only/fee-only regimes, inverted signals, and per-trade audit columns.
- [x] **Step 2: Implement** vectorized strategy signal construction, position shifting, costed PnL, drawdown, monthly return, regime grouping, and audit rows.
- [x] **Step 3: Verify** focused tests.

### Task 4: PBO And Multiple Testing Diagnostics

**Files:**
- Create: `src/quant_research_stack/crypto_research/pbo.py`
- Test: `tests/test_crypto_research_loop.py`

- [x] **Step 1: Write failing tests** for CSCV-style block ranking and high-PBO rejection.
- [x] **Step 2: Implement** PBO, OOS rank/logit rank distribution, Bonferroni-style multiple-testing correction, and approximate deflated Sharpe fields.
- [x] **Step 3: Verify** focused tests.

### Task 5: CLI And Reports

**Files:**
- Create: `scripts/crypto_strategy_research_loop.py`
- Create: `src/quant_research_stack/crypto_research/reports.py`
- Test: `tests/test_crypto_research_loop.py`

- [x] **Step 1: Write failing smoke test** that a synthetic run writes `strategy_registry.parquet`, `all_backtests.parquet`, `pbo_report.json`, `pbo_report.md`, `best_candidates_report.md`, `cost_sensitivity_report.md`, `holdout_report.md`, and `failure_report.md`.
- [x] **Step 2: Implement** CLI with reproducible run configuration and permanent holdout used only once for finalists.
- [x] **Step 3: Run** a bounded local BTCUSDT iteration and inspect reports before any claim.

### Task 6: Verification And Commit

**Files:**
- Modify: all files above.

- [x] **Step 1: Run** `PYTHONPATH=src uv run ruff check src scripts tests`.
- [x] **Step 2: Run** `PYTHONPATH=src uv run mypy src`.
- [x] **Step 3: Run** `PYTHONPATH=src uv run pytest -q`.
- [x] **Step 4: Commit and push** only after verification passes.

### Notes

- Research seed papers used for strategy families include PBO/CSCV by Bailey et al., cryptocurrency momentum/reversal papers, and recent crypto microstructure papers. The first implementation does not assert paper replication unless the paper logic is encoded exactly.
- The first run may be a rigorous negative result. The promotion gate is not relaxed after seeing results.
- Final benchmark artifact: `experiments/crypto_strategy_loop/20260525-090328/`.
