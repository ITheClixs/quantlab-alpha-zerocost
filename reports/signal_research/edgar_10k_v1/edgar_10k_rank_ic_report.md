# EDGAR 10-K v1 — Rank IC Report

**Built:** 2026-05-30T09:20:04.393908+00:00 | label `fwd_ret_63` (~63d fwd) | cross-section = filing year | quintile buckets.
Splits: train ≤2017, val 2018-2019, holdout 2020-2022. Univariate sign fixed on train.

| signal | train IC | val IC | holdout IC | holdout IC t | holdout spread |
|---|---:|---:|---:|---:|---:|
| `feat_rf_negative_ratio` | -0.0003 | 0.0056 | 0.0342 | 0.50 | 0.0221 |
| `model_lgbm_text` | 0.0254 | 0.0506 | 0.0213 | 1.26 | 0.0147 |
| `placebo_shuffled_text` | -0.0019 | 0.0023 | 0.0153 | 1.22 | 0.0045 |
| `baseline_event_ret` | 0.0008 | 0.0462 | 0.0080 | 0.33 | 0.0074 |
| `sanity_inverted_text` | -0.0158 | 0.0214 | 0.0047 | 0.29 | 0.0063 |
| `feat_rf_uncertainty_ratio` | 0.0388 | 0.0131 | 0.0036 | 0.07 | 0.0032 |
| `model_elasticnet_text` | 0.0151 | -0.0188 | 0.0031 | 0.20 | -0.0006 |
| `baseline_size` | 0.0104 | 0.0255 | 0.0005 | 0.01 | 0.0045 |
| `model_ridge_text` | 0.0158 | -0.0214 | -0.0047 | -0.29 | -0.0063 |
| `placebo_random_rank` | 0.0172 | 0.0497 | -0.0145 | -0.54 | -0.0178 |
| `feat_mda_net_tone` | 0.0030 | 0.0390 | -0.0198 | -0.54 | -0.0085 |
| `feat_rf_yoy_change` | 0.0017 | 0.0054 | -0.1162 | -35.51 | -0.0702 |
