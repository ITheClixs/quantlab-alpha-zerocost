# VRP × HMM Interaction — Validation Report

## Pre-registered variant grid
1. `hmm_only_baseline` (anchor)
2. `vrp_only_baseline` (anchor)
3. `vrp_when_hmm_risk_on` — intersection: long if VRP > 0 AND HMM = risk_on
4. `vrp_when_hmm_risk_off` — intersection: long if VRP > 0 AND HMM = risk_off
5. `hmm_sized_by_vrp` — HMM gate × clip(vrp_z60, 0, 1)
6. `vrp_sized_by_hmm_prob` — VRP gate × p_risk_on (continuous)
7. `additive_50_50` — 0.5 × HMM + 0.5 × VRP
8. `additive_70_30` — 0.7 × HMM + 0.3 × VRP
9. `orthogonalized_vrp` — sign(residual of VRP regressed on HMM, dev-only fit)

Plus sanity baselines: `sanity_random`, `sanity_buy_and_hold`.

## Results table

| Strategy | category | dev SR | dev CI_lo | dev CI_hi | holdout SR | cs-2x SR | cs-3x SR | delay-1d SR | max DD dev | turnover | exposure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmm_only_baseline` | anchor | +1.742 | +1.263 | +2.235 | +1.762 | +1.736 | +1.730 | +1.447 | -8.53% | 38.00 | 76.2% |
| `vrp_only_baseline` | anchor | +0.898 | +0.454 | +1.419 | +1.203 | +0.880 | +0.861 | +0.787 | -20.33% | 179.00 | 86.1% |
| `vrp_when_hmm_risk_on` | interaction | +1.523 | +1.025 | +2.041 | +1.484 | +1.504 | +1.485 | +1.208 | -8.53% | 112.00 | 69.0% |
| `vrp_when_hmm_risk_off` | interaction | -0.004 | -0.388 | +0.405 | +0.158 | -0.019 | -0.034 | +0.065 | -30.64% | 119.00 | 17.1% |
| `hmm_sized_by_vrp` | interaction | +1.216 | +0.740 | +1.714 | +0.304 | +1.128 | +1.039 | +0.748 | -5.09% | 311.26 | 37.5% |
| `vrp_sized_by_hmm_prob` | interaction | +1.773 | +1.303 | +2.282 | +1.768 | +1.746 | +1.720 | +1.423 | -8.01% | 137.16 | 85.7% |
| `additive_50_50` | interaction | +1.369 | +0.887 | +1.873 | +1.563 | +1.353 | +1.338 | +1.167 | -11.08% | 106.50 | 93.2% |
| `additive_70_30` | interaction | +1.572 | +1.096 | +2.074 | +1.679 | +1.559 | +1.547 | +1.323 | -8.53% | 79.10 | 93.2% |
| `orthogonalized_vrp` | interaction | +0.898 | +0.454 | +1.419 | +1.203 | +0.880 | +0.861 | +0.787 | -20.33% | 179.00 | 86.1% |
| `sanity_random` | sanity | +0.479 | +0.010 | +1.014 | +0.700 | +0.268 | +0.056 | -0.152 | -23.21% | 1703.00 | 49.7% |
| `sanity_buy_and_hold` | sanity | +0.728 | +0.251 | +1.251 | +1.465 | +0.728 | +0.728 | +0.728 | -33.72% | 1.00 | 100.0% |

## Cross-strategy controls

- PBO raw_global: 0.010  (gate ≤ 0.25)
- Best strategy: `vrp_sized_by_hmm_prob`
- DSR for best: 0.999  (gate ≥ 0.5)
- PSR_zero for best: 1.000
- n_strategies: 11

## Decision

**Branch: B** — PASS-B(no-promo) — best interaction `vrp_sized_by_hmm_prob` holdout +1.768 improves on VRP-only (+1.203) but does not exceed HMM-only (+1.762) by 0.25. VRP is useful but NOT incremental over HMM. Keep VRP as a research note.

failure_class: `vrp_useful_but_not_incremental_over_hmm`
