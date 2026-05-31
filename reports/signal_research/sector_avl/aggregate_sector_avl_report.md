# Sector-Conditional Avellaneda-Lee — Aggregate Report

## Fixture
- sectors evaluated: 18 variants × 6 sectors
- history: 2006-01-01 → 2026-05-26
- dev:     2006-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- PCA: rolling 252d window, components ∈ [1, 2, 3]
- z-entry grid: [1.0, 1.5, 2.0]
- HMM gate: ['none', 'risk_on']
- exit: |z| ≤ 0.5 OR held > 10d

## Data quality banner

DATA QUALITY: data_quality_label=survivorship_prototype_only, constituent_survivorship_applicable=True. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## All strategies side-by-side

| Strategy | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x | pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `avl_pca1_z1.0_hmmnone` | -1.920 | -63.73% | -2.448 | -1.434 | -1.075 | -12.80% | -4.305 | no |
| `avl_pca1_z1.0_hmmrisk_on` | -1.693 | -54.60% | -2.264 | -1.210 | -1.104 | -12.99% | -3.769 | no |
| `avl_pca1_z1.5_hmmnone` | -1.287 | -49.25% | -1.780 | -0.798 | -0.982 | -9.78% | -2.773 | no |
| `avl_pca1_z1.5_hmmrisk_on` | -1.176 | -42.02% | -1.639 | -0.709 | -1.011 | -9.98% | -2.473 | no |
| `avl_pca1_z2.0_hmmnone` | -1.009 | -39.83% | -1.481 | -0.575 | -0.938 | -9.35% | -2.029 | no |
| `avl_pca1_z2.0_hmmrisk_on` | -0.938 | -34.96% | -1.398 | -0.472 | -0.881 | -8.73% | -1.822 | no |
| `avl_pca2_z1.0_hmmnone` | -1.869 | -59.86% | -2.457 | -1.307 | -1.433 | -15.43% | -4.508 | no |
| `avl_pca2_z1.0_hmmrisk_on` | -1.805 | -55.65% | -2.377 | -1.290 | -1.274 | -13.77% | -4.011 | no |
| `avl_pca2_z1.5_hmmnone` | -1.251 | -45.55% | -1.722 | -0.770 | -1.205 | -12.07% | -2.924 | no |
| `avl_pca2_z1.5_hmmrisk_on` | -1.187 | -40.77% | -1.686 | -0.683 | -1.112 | -11.21% | -2.570 | no |
| `avl_pca2_z2.0_hmmnone` | -0.940 | -35.32% | -1.418 | -0.508 | -0.907 | -8.39% | -2.091 | no |
| `avl_pca2_z2.0_hmmrisk_on` | -0.905 | -32.82% | -1.375 | -0.459 | -0.859 | -7.96% | -1.836 | no |
| `avl_pca3_z1.0_hmmnone` | -2.003 | -61.03% | -2.522 | -1.512 | -2.491 | -22.97% | -4.791 | no |
| `avl_pca3_z1.0_hmmrisk_on` | -1.898 | -56.51% | -2.416 | -1.436 | -2.427 | -21.75% | -4.210 | no |
| `avl_pca3_z1.5_hmmnone` | -1.056 | -40.46% | -1.577 | -0.547 | -1.425 | -13.42% | -2.826 | no |
| `avl_pca3_z1.5_hmmrisk_on` | -1.077 | -38.78% | -1.590 | -0.611 | -1.500 | -13.59% | -2.530 | no |
| `avl_pca3_z2.0_hmmnone` | -0.883 | -33.72% | -1.375 | -0.409 | -1.279 | -11.26% | -2.086 | no |
| `avl_pca3_z2.0_hmmrisk_on` | -0.778 | -29.93% | -1.232 | -0.339 | -1.224 | -10.72% | -1.747 | no |
| `random_signal` | -7.335 | -94.87% | -8.059 | -6.686 | -7.171 | -47.27% | -14.840 | no |
| `inverted_signal_mom` | -0.332 | -37.74% | -0.818 | +0.132 | -0.892 | -27.40% | -0.636 | no |
| `simple_reversal_5d` | -0.452 | -48.27% | -0.943 | +0.024 | -0.998 | -20.17% | -2.091 | no |

## Cross-strategy multiple-testing controls

- **PBO raw_global**: 0.014  (gate: ≤ 0.25)
- **Best variant index**: 19 (`inverted_signal_mom`)
- **DSR for best**: 0.000  (gate: ≥ 0.50)
- **PSR_zero for best**: 0.089
- **n_strategies in DSR deflation**: 21

## Decision rule outcome

**FAIL — best strategy is a sanity baseline, not AvL.**

failure_class: `no_residual_meanreversion_edge`

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5).
