# VRP Index Validation — `SPY`

## Hypothesis
> Implied option-market variance prices a risk premium relative to 
> subsequently-realized variance. Long-vol hedgers pay a structural 
> premium during normal regimes; the short-vol seller earns it but bears 
> crash risk during volatility spikes (Bondarenko 2014).

## Information sources declared
- `ohlcv` (SPY underlying)
- `options_implied_vol` (^VIX, ^VIX9D, ^VIX3M, ^VVIX, ^SKEW, ^VXN)
- **non-OHLCV source declared: YES** → eligible for promotion if gates pass

## Fixture
- target symbol: SPY
- history: 2010-01-01 → 2026-05-26
- dev:     2010-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- realized-variance window: 21 days
- costs: 0.5 bps commission + 0.5 bps spread one-way
- cost stress multipliers: [2.0, 3.0]
- delay stress: [1] bars

## All strategies side-by-side

| Strategy | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cs-2x | cs-3x | delay-1d | max month share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vrp_long_only` | +0.898 | -20.33% | +0.454 | +1.419 | +1.203 | -18.30% | +0.880 | +0.861 | +0.787 | 3.1% |
| `vrp_long_short` | +0.809 | -26.04% | +0.367 | +1.294 | +0.779 | -18.89% | +0.778 | +0.747 | +0.608 | 3.9% |
| `vrp_with_term_structure` | +0.571 | -21.34% | +0.048 | +1.096 | +0.900 | -10.82% | +0.498 | +0.424 | +0.561 | 2.7% |
| `vrp_with_vvix` | +0.416 | -24.03% | +0.012 | +0.895 | +0.433 | -17.05% | +0.371 | +0.326 | +0.254 | 4.4% |
| `vrp_with_skew` | +0.471 | -20.56% | +0.061 | +0.947 | +0.072 | -17.30% | +0.412 | +0.353 | +0.137 | 4.3% |
| `vrp_combined` | +0.459 | -27.09% | +0.026 | +0.990 | +0.381 | -14.31% | +0.416 | +0.372 | +0.073 | 4.7% |
| _(baseline)_ `spy_buy_and_hold` | +0.728 | -33.72% | +0.251 | +1.251 | +1.465 | -18.76% | +0.728 | +0.728 | +0.728 | 2.3% |
| _(baseline)_ `hmm_only_gate` | +1.742 | -8.53% | +1.263 | +2.235 | +1.762 | -9.97% | +1.736 | +1.730 | +1.447 | 2.1% |
| _(baseline)_ `mom_12_1_single_asset` | +0.557 | -33.72% | +0.080 | +1.074 | +1.366 | -18.76% | +0.550 | +0.542 | +0.575 | 3.2% |

## Cross-strategy multiple-testing controls

- **PBO raw_global**: 0.000  (gate ≤ 0.25)
- **Best strategy**: `hmm_only_gate`
- **DSR for best**: 1.000  (gate ≥ 0.5)
- **PSR_zero for best**: 1.000
- **n_strategies in DSR deflation**: 9

## Pre-registered failure modes (from intake §8)

- Mode 1 (single-event concentration, any variant max month > 50%): not triggered
- Mode 2 (variant grid is duplicates, PBO > 0.20): not triggered
- Mode 3 (combined variant inflates grid, DSR < 0.5 when combined is best): not triggered

## Decision rule outcome

**FAIL — best VRP dev Sharpe +0.898 < 1.5.**

failure_class: `no_alpha_at_threshold`

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5 and the QuantLab promotion runbook).
