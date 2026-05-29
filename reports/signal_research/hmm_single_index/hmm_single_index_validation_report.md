# HMM Single-Index v1 — Validation Report

Intake: `docs/research/intake/2026-05-28-hmm-single-index-v1.md`
Exception policy: `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`

## Fixture
- instruments: ['SPY', 'QQQ']
- history: 2010-01-01 → 2026-05-26
- dev:     2010-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- costs: 0.5 bps commission + 0.5 bps spread one-way
- evaluated against CONSERVATIVE after-fee cash leg (T-bill DTB3 minus 25 bps prime-broker default)

## Side-by-side results (conservative cash leg)

| Variant | instrument | dev SR | dev DD | dev CI_lo | holdout SR | cs-2x SR | cs-3x SR | delay-1d SR | delay-2d SR | year share | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hmm_2_full_dev_spy` | SPY | +1.772 | -5.89% | +1.341 | +1.960 | +1.764 | +1.755 | +1.139 | +0.896 | 15.0% | research_pass |
| `hmm_2_expanding_spy` | SPY | +0.857 | -24.50% | +0.390 | +1.465 | +0.856 | +0.855 | +0.719 | +0.703 | 14.3% | none |
| `hmm_2_rolling_5y_spy` | SPY | +1.090 | -17.35% | +0.629 | +1.465 | +1.089 | +1.087 | +0.935 | +0.935 | 16.1% | none |
| `hmm_3_full_dev_spy` | SPY | +1.878 | -4.60% | +1.439 | +2.384 | +1.861 | +1.845 | +0.875 | +0.573 | 15.0% | research_pass |
| `hmm_3_expanding_spy` | SPY | +0.536 | -33.72% | +0.038 | +1.465 | +0.534 | +0.532 | +0.344 | +0.307 | 26.7% | none |
| `hmm_3_rolling_5y_spy` | SPY | +0.641 | -19.71% | +0.099 | +1.465 | +0.638 | +0.635 | +0.371 | +0.430 | 41.0% | none |
| `hmm_4_full_dev_spy` | SPY | +1.776 | -3.76% | +1.347 | +2.367 | +1.757 | +1.737 | +0.645 | +0.382 | 14.8% | research_pass |
| `hmm_4_expanding_spy` | SPY | +0.666 | -24.50% | +0.235 | +1.776 | +0.663 | +0.661 | +0.424 | +0.476 | 17.9% | none |
| `hmm_4_rolling_5y_spy` | SPY | +0.718 | -18.61% | +0.244 | +1.465 | +0.718 | +0.718 | +0.718 | +0.715 | 29.5% | none |
| `hmm_2_full_dev_qqq` | QQQ | +1.810 | -7.69% | +1.329 | +1.939 | +1.802 | +1.795 | +1.052 | +0.897 | 12.3% | research_pass |
| `hmm_2_expanding_qqq` | QQQ | +0.905 | -26.26% | +0.434 | +1.560 | +0.903 | +0.902 | +0.630 | +0.631 | 13.5% | none |
| `hmm_2_rolling_5y_qqq` | QQQ | +1.269 | -13.94% | +0.805 | +1.614 | +1.266 | +1.263 | +0.838 | +0.950 | 26.9% | none |
| `hmm_3_full_dev_qqq` | QQQ | +2.326 | -3.10% | +1.892 | +2.569 | +2.311 | +2.295 | +1.291 | +0.985 | 14.8% | research_pass |
| `hmm_3_expanding_qqq` | QQQ | +0.765 | -35.12% | +0.267 | +2.002 | +0.763 | +0.761 | +0.512 | +0.542 | 17.4% | none |
| `hmm_3_rolling_5y_qqq` | QQQ | +0.708 | -25.18% | +0.236 | +1.614 | +0.703 | +0.697 | +0.418 | +0.402 | 57.0% | none |
| `hmm_4_full_dev_qqq` | QQQ | +2.356 | -3.09% | +1.905 | +2.621 | +2.340 | +2.323 | +1.259 | +0.983 | 14.9% | research_pass |
| `hmm_4_expanding_qqq` | QQQ | +1.091 | -24.85% | +0.598 | +1.614 | +1.087 | +1.083 | +0.639 | +0.546 | 22.3% | none |
| `hmm_4_rolling_5y_qqq` | QQQ | +1.052 | -16.10% | +0.585 | +1.758 | +1.049 | +1.047 | +0.824 | +0.792 | 27.0% | none |
| `buy_and_hold_spy` | SPY | +0.729 | -33.72% | +0.251 | +1.465 | +0.729 | +0.729 | +0.728 | +0.727 | 13.8% | none |
| `vol_targeted_buy_and_hold_spy` | SPY | +0.828 | -14.70% | +0.332 | +1.365 | +0.824 | +0.820 | +0.726 | +0.721 | 18.6% | none |
| `sma_50_200_gate_spy` | SPY | +0.621 | -33.72% | +0.113 | +1.164 | +0.619 | +0.618 | +0.605 | +0.569 | 19.6% | none |
| `mom_12_1_spy` | SPY | +0.543 | -33.72% | +0.061 | +1.267 | +0.535 | +0.528 | +0.557 | +0.575 | 18.9% | none |
| `random_spy` | SPY | +0.233 | -25.66% | -0.220 | +1.783 | +0.032 | -0.169 | +0.452 | +0.629 | 15.7% | none |
| `buy_and_hold_qqq` | QQQ | +0.797 | -35.12% | +0.346 | +1.614 | +0.797 | +0.797 | +0.797 | +0.799 | 16.2% | none |
| `vol_targeted_buy_and_hold_qqq` | QQQ | +0.934 | -15.05% | +0.455 | +1.532 | +0.930 | +0.926 | +0.828 | +0.814 | 17.2% | none |
| `sma_50_200_gate_qqq` | QQQ | +0.798 | -28.56% | +0.316 | +1.327 | +0.797 | +0.795 | +0.772 | +0.783 | 19.0% | none |
| `mom_12_1_qqq` | QQQ | +0.753 | -28.56% | +0.284 | +1.379 | +0.747 | +0.742 | +0.667 | +0.765 | 19.5% | none |
| `random_qqq` | QQQ | +0.375 | -30.38% | -0.134 | +1.412 | +0.203 | +0.031 | +0.489 | +0.544 | 15.2% | none |
| `inverted_of_best_hmm_spy` | SPY | +0.027 | -38.43% | -0.408 | +0.239 | +0.020 | +0.014 | +0.383 | +0.509 | 21.7% | none |
| `inverted_of_best_hmm_qqq` | QQQ | +0.021 | -39.88% | -0.432 | +0.762 | +0.015 | +0.009 | +0.355 | +0.450 | 27.9% | none |

## Cross-strategy controls

- PBO raw_global: 0.000  (gate ≤ 0.25)
- Best strategy: `hmm_4_full_dev_qqq`
- DSR for best: 1.000  (gate ≥ 0.5)
- PSR_zero for best: 1.000
- n_strategies: 30

## Status outcomes

Per intake §11, the maximum status reachable from this validation is
`exception_review_required`. No `paper_trade_candidate` or
`production_candidate` status is emitted by this run.
