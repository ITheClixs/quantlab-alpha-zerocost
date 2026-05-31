# GKX-Style LightGBM Scale-Up — Predeclared Variant Matrix

## Fixture
- universes: top-100 and top-200 SP500 by ADV
- history: 2006-01-01 → 2026-05-26
- dev:     2006-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- label horizons: [5, 21, 63]
- LightGBM: n_estimators=500, num_leaves=31, lr=0.05
- walk-forward: 5 folds, embargo=label_horizon + 5
- features: 19 OHLCV characteristics
- costs: 0.5 bps commission + 10.0 bps spread
- cost-stress: 2.0× multiplier

## Data quality banner

DATA QUALITY: data_quality_label=survivorship_prototype_only, constituent_survivorship_applicable=True. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## All strategies side-by-side

| Strategy | Universe | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `gkx_lgb_h5d_top100` | top100 | -0.264 | -51.29% | -0.693 | +0.158 | -1.765 | -41.02% | -1.252 | no |
| `gkx_lgb_h21d_top100` | top100 | -0.249 | -42.26% | -0.684 | +0.179 | -1.260 | -32.05% | -1.026 | no |
| `gkx_lgb_h63d_top100` | top100 | -0.279 | -42.11% | -0.753 | +0.181 | -1.417 | -32.76% | -0.923 | no |
| `gkx_lgb_h5d_top200` | top200 | -0.337 | -42.70% | -0.802 | +0.115 | -1.713 | -33.96% | -1.708 | no |
| `gkx_lgb_h21d_top200` | top200 | -0.432 | -38.90% | -0.848 | -0.011 | -0.806 | -16.96% | -1.486 | no |
| `gkx_lgb_h63d_top200` | top200 | -0.697 | -48.39% | -1.138 | -0.286 | -0.369 | -11.36% | -1.543 | no |
| `random_signal_top100` | top100 | -4.900 | -98.64% | -5.601 | -4.256 | -5.302 | -59.23% | -9.300 | no |
| `simple_reversal_5d_top100` | top100 | -0.532 | -71.95% | -1.028 | -0.072 | -1.422 | -40.71% | -1.454 | no |
| `mom_12_1_top100` | top100 | +0.152 | -45.24% | -0.387 | +0.671 | +0.164 | -20.36% | -0.005 | no |
| `random_signal_top200` | top200 | -6.370 | -97.93% | -7.162 | -5.690 | -8.320 | -60.56% | -12.695 | no |
| `simple_reversal_5d_top200` | top200 | -0.184 | -62.05% | -0.672 | +0.295 | -1.060 | -29.52% | -1.219 | no |
| `mom_12_1_top200` | top200 | -0.149 | -54.86% | -0.613 | +0.338 | +0.592 | -13.76% | -0.324 | no |

## Cross-strategy multiple-testing controls

- **PBO raw_global**: 0.266  (gate: ≤ 0.25)
- **Best variant index**: 8 (`mom_12_1_top100`)
- **DSR for best**: 0.000  (gate: ≥ 0.50)
- **PSR_zero for best**: 0.733
- **n_strategies in DSR deflation**: 12

## Decision rule outcome

**FAIL — PBO=0.266 > 0.25 (overfit variant grid).**

failure_class: `overfit_parameter_grid`

## Promotion gates (per-variant)
- dev Sharpe ≥ 1.0
- holdout Sharpe ≥ 0.5
- cost-stress 2× Sharpe > 0
- bootstrap 95% lower-CI Sharpe > 0
- DSR ≥ 0.50 (after multi-test deflation)
- beats best non-GKX baseline

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5).
