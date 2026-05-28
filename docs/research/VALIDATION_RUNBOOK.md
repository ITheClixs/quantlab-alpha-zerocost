# Validation Runbook

How to run the productionized validation pipeline on a new strategy.

## Prerequisites

1. Intake document committed to `docs/research/intake/`.
2. Signal function implemented as a `SignalFn`:
   ```python
   def my_strategy_signal(bars: pl.DataFrame, spec: ValidationSpec) -> pl.DataFrame:
       # return long-form (date, symbol, y_xs_pred)
       ...
   ```
3. `ValidationSpec` constructed with:
   - `strategy_name`, `hypothesis_statement`
   - `information_sources` (must include at least one non-OHLCV value
     for the strategy to be promotion-eligible)
   - `intake_doc_ref` pointing at the intake markdown
   - dev/holdout split, costs, gates

## Standard invocation

```python
from pathlib import Path
from quant_research_stack.signal_research.validation import (
    InformationSource, ValidationSpec, validate_strategy, render_pipeline_report,
)

spec = ValidationSpec(
    strategy_name="vrp_index_v1",
    hypothesis_statement="Implied minus realized vol prices a risk premium...",
    information_sources=(InformationSource.OHLCV, InformationSource.OPTIONS_IMPLIED_VOL),
    universe_tickers=["SPY", "QQQ"],
    start=..., end=..., dev_end=..., holdout_start=...,
    intake_doc_ref="docs/research/intake/2026-06-01-vrp-index-v1.md",
    proposer="...",
)
report = validate_strategy(spec=spec, signal_fn=my_strategy_signal, bars=bars)
render_pipeline_report(report, output_path=Path("reports/.../vrp_v1.md"))
```

## What gets reported

- Headline dev/holdout Sharpe + drawdown + bootstrap CI
- Cost decomposition (no-cost / fee-only / spread-only / full / 2× stress)
- Delay stress (1-bar shift)
- Sanity baselines (random_signal, inverted_signal)
- Concentration diagnostics (monthly/yearly PnL share)
- Cross-strategy PBO/DSR (requires a strategy pool)
- 8-criteria promotion-gate scorecard
- Status assignment (NONE / RESEARCH_PASS / PROMOTION_ELIGIBLE)
- "No promotion without new information source" rule check

## Strategy-pool mode

For PBO/DSR to be meaningful, the validation needs a pool of strategies
sharing the same data. Two patterns:

**Pattern A: predeclared variant grid (e.g. VRP iteration).**
Build a small predeclared grid (e.g. 3 lookbacks × 2 risk-budgets =
6 variants). Run `validate_strategy` for each, then pass the
`dev_net_returns` of the other 5 as `pool_dev_returns` when validating
the chosen variant.

**Pattern B: cross-iteration comparison.**
Compare a new strategy against the recorded `dev_net_returns` of
strategies from prior iterations (e.g. `mom_12_1`, `simple_reversal`).
This anchors the new strategy against the empirical noise floor.

## Failure classes

The pipeline emits zero or more failure classes from
`methodology.failure_classifier.FailureCategory`:

- `high_pbo` — dev/IS winner doesn't generalize
- `low_dsr` — fails multi-test deflation
- `cost_failure` — 2× cost stress flips negative
- `insufficient_sample` — bootstrap CI lower-bound < 0
- `delay_stress_fail` — 1-bar delay tanks the Sharpe by ≥ 0.5
- `single_period_dominance` — one month carries > 50% of |PnL|
- `randomization_fail` — beats random by less than 0.1 Sharpe
- `over_correlated_with_baseline` — loses to its own sign-flip
- `holdout_failure` — holdout Sharpe < gate threshold

Each class maps to a specific structural problem with the proposal.

## Promotion gates (8-criteria, all must pass)

1. `dev Sharpe ≥ spec.gate_dev_sharpe_min` (default 1.5)
2. `holdout Sharpe ≥ spec.gate_holdout_sharpe_min` (default 0.5)
3. `cost-stress 2× Sharpe > 0`
4. `bootstrap 95% lower-CI Sharpe > 0`
5. `PBO ≤ 0.25` across the strategy pool
6. `DSR ≥ 0.50` for the selected best variant
7. Beats `random_signal` by ≥ 0.1 dev Sharpe
8. Beats its own `inverted_signal`

PLUS the non-OHLCV information source declaration.

## Post-pass workflow

If a strategy passes all 8 gates AND declares a non-OHLCV source, the
status becomes `PROMOTION_ELIGIBLE`. Note this is **not** a promotion
itself. Capital deployment requires:

- a signed `docs/runbooks/stage_change.md` commit
- updated `.env` and `QUANTLAB_STAGE`
- two-person review
- process restart (per CLAUDE.md §11)

The validation pipeline does **not** auto-promote. The 4-tier ladder
(`research_pass → promotion_eligible → paper_trade_candidate →
production_candidate`) requires operator action at every step.

## Reproducibility

Every validation report records:
- git SHA at run time
- ValidationSpec (frozen Pydantic-style dataclass)
- intake doc reference
- proposer
- seeds for bootstrap and walk-forward training
- raw daily-returns arrays (dev + holdout) for downstream PBO/DSR

The same intake + same spec + same code should reproduce the same
report byte-for-byte modulo timestamps.
