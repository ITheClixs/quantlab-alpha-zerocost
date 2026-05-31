# crypto_only_riskalloc_v2 — Regime Attribution

**Built:** 2026-05-30T15:29:57.515484+00:00
| regime | strat return | bench return | strat maxDD | bench maxDD | avoided DD |
|---|---:|---:|---:|---:|---:|
| 2018_bear | 0.0 | -0.3461 | 0.0 | -0.4005 | 0.4005 |
| 2020_covid | -0.0691 | -0.0643 | -0.0623 | -0.127 | 0.0648 |
| 2021_bull | 0.063 | 0.2744 | -0.0763 | -0.0902 | 0.014 |
| 2022_crash | -0.0969 | -0.2097 | -0.103 | -0.2287 | 0.1257 |
| 2023_26_recovery | 0.5115 | 0.609 | -0.1611 | -0.1867 | 0.0256 |

- `avoided DD` positive = shallower drawdown than vol-targeted 50/50. Negative bull-regime `strat return` minus `bench return` = missed upside / re-entry cost of the trend+regime gate.