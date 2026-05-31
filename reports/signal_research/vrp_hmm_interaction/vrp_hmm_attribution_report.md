# VRP × HMM Interaction — Attribution Report

## Per-strategy attribution

| Strategy | dev SR | ρ(HMM) | ρ(VRP) | incr. over HMM | incr. over VRP | residual SR vs HMM | residual SR vs VRP | excl-2020 SR | excl-2022 SR | excl-holdout SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmm_only_baseline` | +1.742 | +1.000 | +0.563 | +0.000 | +0.844 | +0.116 | +0.000 | +1.645 | +1.839 | +1.742 |
| `vrp_only_baseline` | +0.898 | +0.563 | +1.000 | -0.844 | +0.000 | +0.000 | +0.565 | +0.890 | +1.038 | +0.898 |
| `vrp_when_hmm_risk_on` | +1.523 | +0.957 | +0.589 | -0.219 | +0.625 | +0.000 | +0.000 | +1.405 | +1.611 | +1.523 |
| `vrp_when_hmm_risk_off` | -0.004 | +0.000 | +0.809 | -1.747 | -0.902 | -0.000 | +0.000 | -0.023 | +0.022 | -0.004 |
| `hmm_sized_by_vrp` | +1.216 | +0.694 | +0.426 | -0.526 | +0.318 | +0.000 | +0.000 | +1.070 | +1.266 | +1.216 |
| `vrp_sized_by_hmm_prob` | +1.773 | +0.924 | +0.611 | +0.030 | +0.875 | +0.000 | +0.000 | +1.677 | +1.867 | +1.773 |
| `additive_50_50` | +1.369 | +0.818 | +0.936 | -0.374 | +0.471 | +0.000 | +0.000 | +1.322 | +1.507 | +1.369 |
| `additive_70_30` | +1.572 | +0.924 | +0.836 | -0.171 | +0.674 | +0.000 | +0.000 | +1.495 | +1.692 | +1.572 |
| `orthogonalized_vrp` | +0.898 | +0.563 | +1.000 | -0.844 | +0.000 | +0.000 | +0.565 | +0.890 | +1.038 | +0.898 |
| `sanity_random` | +0.479 | +0.372 | +0.598 | -1.264 | -0.419 | +0.000 | +0.000 | +0.435 | +0.690 | +0.479 |
| `sanity_buy_and_hold` | +0.728 | +0.531 | +0.863 | -1.015 | -0.170 | +0.000 | +0.000 | +0.771 | +0.904 | +0.728 |

## PnL by year (dev only)

| Strategy | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmm_only_baseline` | +25.28% | +6.25% | +12.82% | +28.62% | +13.79% | +5.97% | +14.47% | +19.88% | +10.35% | +29.06% | +24.91% | +21.69% | -2.21% |
| `vrp_only_baseline` | +22.00% | +13.65% | +19.31% | +17.99% | +5.27% | +5.87% | +12.65% | +19.88% | +1.41% | +13.56% | +26.99% | +24.26% | -5.80% |
| `vrp_when_hmm_risk_on` | +25.29% | +6.18% | +15.26% | +17.99% | +5.80% | +7.91% | +12.80% | +19.88% | +7.03% | +16.33% | +24.51% | +19.89% | -2.21% |
| `vrp_when_hmm_risk_off` | -3.41% | +7.43% | +4.00% | +0.00% | -0.68% | -2.12% | -0.19% | +0.00% | -5.78% | -2.81% | +2.36% | +4.17% | -3.63% |
| `hmm_sized_by_vrp` | +8.65% | +12.11% | +10.71% | +10.31% | -0.05% | +7.51% | +5.80% | +5.03% | -0.19% | +5.83% | +15.05% | +4.87% | +0.00% |
| `vrp_sized_by_hmm_prob` | +24.92% | +9.23% | +15.19% | +18.96% | +7.63% | +8.67% | +12.60% | +19.86% | +10.48% | +15.29% | +22.06% | +21.54% | -1.31% |
| `additive_50_50` | +23.64% | +9.95% | +16.06% | +23.31% | +9.53% | +5.92% | +13.56% | +19.88% | +5.88% | +21.31% | +25.97% | +23.00% | -4.00% |
| `additive_70_30` | +24.30% | +8.47% | +14.76% | +25.43% | +11.23% | +5.94% | +13.92% | +19.88% | +7.67% | +24.41% | +25.54% | +22.48% | -3.28% |
| `orthogonalized_vrp` | +22.00% | +13.65% | +19.31% | +17.99% | +5.27% | +5.87% | +12.65% | +19.88% | +1.41% | +13.56% | +26.99% | +24.26% | -5.80% |

## PnL by HMM regime (dev only)

| Strategy | risk_on PnL | risk_off PnL |
|---|---:|---:|
| `hmm_only_baseline` | +237.52% | -26.64% |
| `vrp_only_baseline` | +205.19% | -28.15% |
| `vrp_when_hmm_risk_on` | +202.15% | -25.50% |
| `vrp_when_hmm_risk_off` | +2.72% | -3.37% |
| `hmm_sized_by_vrp` | +96.79% | -11.16% |
| `vrp_sized_by_hmm_prob` | +201.34% | -16.23% |
| `additive_50_50` | +221.39% | -27.39% |
| `additive_70_30` | +227.84% | -27.09% |
| `orthogonalized_vrp` | +205.19% | -28.15% |

## Interpretation guide

- `incr. over HMM` is the raw Sharpe difference. Positive means the strategy *appears* to add value, but does not account for shared regime exposure.
- `residual SR vs HMM` is the Sharpe of the orthogonal component after regressing the strategy's daily returns on HMM-only's. Positive means the strategy has real incremental information.
- `excl-X SR` shows how much of the dev edge depends on a specific year. 
A strategy whose Sharpe materially collapses when one crisis is removed is regime-concentrated.
