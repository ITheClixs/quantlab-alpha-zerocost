# HMM Single-Index v1 — State Stability Report

Per intake §5.4 / exception policy §4.5: economic-identity flips
drive demotion; raw HMM label permutations do NOT.

Gate: economic-identity flip rate ≤ 20% across refits.

| Variant | n_refits | n_economic_flips | flip_rate | raw_label_flips | gate |
|---|---:|---:|---:|---:|---|
| `hmm_2_full_dev_spy` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_2_expanding_spy` | 13 | 2 | 16.7% | 5 | PASS |
| `hmm_2_rolling_5y_spy` | 13 | 2 | 16.7% | 5 | PASS |
| `hmm_3_full_dev_spy` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_3_expanding_spy` | 13 | 10 | 83.3% | 9 | FAIL |
| `hmm_3_rolling_5y_spy` | 13 | 6 | 50.0% | 6 | FAIL |
| `hmm_4_full_dev_spy` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_4_expanding_spy` | 9 | 3 | 37.5% | 6 | FAIL |
| `hmm_4_rolling_5y_spy` | 11 | 2 | 20.0% | 5 | PASS |
| `hmm_2_full_dev_qqq` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_2_expanding_qqq` | 13 | 2 | 16.7% | 5 | PASS |
| `hmm_2_rolling_5y_qqq` | 13 | 3 | 25.0% | 7 | FAIL |
| `hmm_3_full_dev_qqq` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_3_expanding_qqq` | 13 | 4 | 33.3% | 5 | FAIL |
| `hmm_3_rolling_5y_qqq` | 13 | 6 | 50.0% | 8 | FAIL |
| `hmm_4_full_dev_qqq` | 1 | 0 | 0.0% | 0 | PASS |
| `hmm_4_expanding_qqq` | 12 | 3 | 27.3% | 7 | FAIL |
| `hmm_4_rolling_5y_qqq` | 12 | 5 | 45.5% | 3 | FAIL |
