# Momentum Scale-Up — Predeclared Variant Matrix

## Fixture
- universes: top-100 and top-200 SP500 by ADV (100 / 200 tickers)
- history: 2006-01-01 → 2026-05-26
- dev:     2006-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- costs: 0.5 bps commission + 10.0 bps spread
- cost-stress: 2.0× multiplier

## Data quality banner

DATA QUALITY: data_quality_label=survivorship_prototype_only, constituent_survivorship_applicable=True. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## Side-by-side results (10 variants)

| Variant | Universe | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `mom_12_1` | top100 | +0.152 | -45.24% | -0.374 | +0.712 | +0.164 | -20.36% | -0.005 | no |
| `mom_12_3` | top100 | +0.095 | -44.43% | -0.427 | +0.597 | +0.208 | -19.62% | -0.087 | no |
| `mom_24_1` | top100 | -0.168 | -56.29% | -0.638 | +0.306 | -0.389 | -24.66% | -0.301 | no |
| `mom_12_1_vol_scaled` | top100 | +0.121 | -39.39% | -0.373 | +0.685 | -0.152 | -20.25% | -0.072 | no |
| `mom_12_1_hmm_gated` | top100 | +0.504 | -17.49% | +0.065 | +0.954 | +0.140 | -20.36% | +0.284 | no |
| `mom_12_1` | top200 | -0.149 | -54.86% | -0.623 | +0.330 | +0.592 | -13.76% | -0.324 | no |
| `mom_12_3` | top200 | -0.214 | -51.38% | -0.662 | +0.278 | +0.513 | -12.52% | -0.420 | no |
| `mom_24_1` | top200 | -0.465 | -62.47% | -0.907 | -0.036 | +0.181 | -17.00% | -0.616 | no |
| `mom_12_1_vol_scaled` | top200 | -0.178 | -49.98% | -0.630 | +0.326 | +0.327 | -14.11% | -0.408 | no |
| `mom_12_1_hmm_gated` | top200 | +0.221 | -25.23% | -0.210 | +0.655 | +0.585 | -13.96% | -0.052 | no |

## Cross-strategy multiple-testing controls

- **PBO raw_global**: 0.004  (gate: ≤ 0.5)
- **PBO per profile**:
- top100: 0.059
- top200: 0.057
- **PBO per family**:

- **Best variant index**: 4 (`mom_12_1_hmm_gated` on top100)
- **DSR for best (P(true SR>0 after multi-test penalty))**: 0.608  (gate: ≥ 0.95)
- **PSR_zero for best (P(true SR>0 ignoring multi-test))**: 0.980
- **n_strategies in DSR deflation**: 10

## Decision rule outcome

**NO VARIANT SURVIVES — classify +0.58 baseline holdout Sharpe from prior iteration as noise. Move on to sector-conditional AvL after documenting this failure.**

## Promotion gates (applied per-variant)
- dev Sharpe ≥ 1.0
- holdout Sharpe ≥ 0.5
- cost-stress 2× Sharpe > 0
- bootstrap 95% lower-CI Sharpe > 0

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5).
