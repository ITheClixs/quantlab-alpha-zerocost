# crypto_only_riskalloc_v2 — Multiple Holdouts (vs vol-targeted 50/50 BTC/ETH)

**Built:** 2026-05-30T15:29:57.515484+00:00
| window | strat Sharpe | bench Sharpe | strat maxDD | bench maxDD | strat Calmar | bench Calmar | beats | DD better |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 2020+ | 0.762 | 0.816 | -0.221 | -0.276 | 0.342 | 0.392 | False | True |
| 2021+ | 0.52 | 0.521 | -0.221 | -0.276 | 0.22 | 0.23 | False | True |
| 2022+ | 0.52 | 0.34 | -0.163 | -0.229 | 0.306 | 0.167 | True | True |
| 2023+ | 0.776 | 0.763 | -0.161 | -0.187 | 0.541 | 0.541 | True | True |
| 2024+ | 0.963 | 0.452 | -0.161 | -0.187 | 0.644 | 0.285 | True | True |

- Beats (Sharpe or Calmar) on **3/5**; improves maxDD on **5/5**.