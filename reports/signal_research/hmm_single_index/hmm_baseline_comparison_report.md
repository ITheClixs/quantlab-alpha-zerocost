# HMM Single-Index v1 — Baseline Comparison Report

Per intake §9.16-§9.21. Each HMM variant must beat ALL non-sanity
baselines on Sharpe AND max drawdown, AND the random / inverted
sanity baselines must fail the §9.3 gate.

| Variant | dev SR | beats BAH | beats VT | beats SMA | beats MOM | random fails | inverted fails |
|---|---:|---|---|---|---|---|---|
| `hmm_2_full_dev_spy` | +1.772 | P | P | P | P | P | P |
| `hmm_2_expanding_spy` | +0.857 | P | F | P | P | P | P |
| `hmm_2_rolling_5y_spy` | +1.090 | P | F | P | P | P | P |
| `hmm_3_full_dev_spy` | +1.878 | P | P | P | P | P | P |
| `hmm_3_expanding_spy` | +0.536 | F | F | F | F | P | P |
| `hmm_3_rolling_5y_spy` | +0.641 | F | F | P | P | P | P |
| `hmm_4_full_dev_spy` | +1.776 | P | P | P | P | P | P |
| `hmm_4_expanding_spy` | +0.666 | F | F | P | P | P | P |
| `hmm_4_rolling_5y_spy` | +0.718 | F | F | P | P | P | P |
| `hmm_2_full_dev_qqq` | +1.810 | P | P | P | P | P | P |
| `hmm_2_expanding_qqq` | +0.905 | P | F | P | P | P | P |
| `hmm_2_rolling_5y_qqq` | +1.269 | P | P | P | P | P | P |
| `hmm_3_full_dev_qqq` | +2.326 | P | P | P | P | P | P |
| `hmm_3_expanding_qqq` | +0.765 | F | F | F | F | P | P |
| `hmm_3_rolling_5y_qqq` | +0.708 | F | F | F | F | P | P |
| `hmm_4_full_dev_qqq` | +2.356 | P | P | P | P | P | P |
| `hmm_4_expanding_qqq` | +1.091 | P | F | P | P | P | P |
| `hmm_4_rolling_5y_qqq` | +1.052 | P | F | P | P | P | P |
