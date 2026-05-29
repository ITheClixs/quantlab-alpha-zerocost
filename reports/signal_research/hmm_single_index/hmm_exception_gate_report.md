# HMM Single-Index v1 — Exception-Gate Scorecard

Per accepted exception policy §3 and intake §9 (24 gates total).

| Variant | passed | all pass? | failed gates |
|---|---:|---|---|
| `hmm_2_full_dev_spy` | 22/24 | no | delay1d, delay2d |
| `hmm_2_expanding_spy` | 19/24 | no | dev_sharpe, holdout_sharpe, dd/calmar, beats_vt, boot_lo |
| `hmm_2_rolling_5y_spy` | 21/24 | no | dev_sharpe, holdout_sharpe, beats_vt |
| `hmm_3_full_dev_spy` | 22/24 | no | delay1d, delay2d |
| `hmm_3_expanding_spy` | 12/24 | no | dev_sharpe, holdout_sharpe, dd/calmar, excl2020, excl2022, pre2020, beats_bah, beats_vt, beats_sma, beats_mom, boot_lo, stability |
| `hmm_3_rolling_5y_spy` | 16/24 | no | dev_sharpe, holdout_sharpe, excl2020, excl2022, beats_bah, beats_vt, boot_lo, stability |
| `hmm_4_full_dev_spy` | 22/24 | no | delay1d, delay2d |
| `hmm_4_expanding_spy` | 17/24 | no | dev_sharpe, dd/calmar, excl2020, beats_bah, beats_vt, boot_lo, stability |
| `hmm_4_rolling_5y_spy` | 17/24 | no | dev_sharpe, holdout_sharpe, excl2020, excl2022, beats_bah, beats_vt, boot_lo |
| `hmm_2_full_dev_qqq` | 22/24 | no | delay1d, delay2d |
| `hmm_2_expanding_qqq` | 20/24 | no | dev_sharpe, dd/calmar, beats_vt, boot_lo |
| `hmm_2_rolling_5y_qqq` | 22/24 | no | dev_sharpe, stability |
| `hmm_3_full_dev_qqq` | 22/24 | no | delay1d, delay2d |
| `hmm_3_expanding_qqq` | 15/24 | no | dev_sharpe, dd/calmar, excl2020, beats_bah, beats_vt, beats_sma, beats_mom, boot_lo, stability |
| `hmm_3_rolling_5y_qqq` | 12/24 | no | dev_sharpe, dd/calmar, year_share, excl2020, excl2022, pre2020, beats_bah, beats_vt, beats_sma, beats_mom, boot_lo, stability |
| `hmm_4_full_dev_qqq` | 22/24 | no | delay1d, delay2d |
| `hmm_4_expanding_qqq` | 19/24 | no | dev_sharpe, delay2d, dd/calmar, beats_vt, stability |
| `hmm_4_rolling_5y_qqq` | 21/24 | no | dev_sharpe, beats_vt, stability |
