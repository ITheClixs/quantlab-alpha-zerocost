# Strict Review (3) — Multiple Anchored Holdouts

**Built:** 2026-05-30T15:13:52.779560+00:00
| window | strat Sharpe | bench Sharpe | strat maxDD | bench maxDD | strat Calmar | bench Calmar | beats(S/C) | DD improved |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 2020+ | 1.262 | 1.111 | -0.141 | -0.181 | 0.935 | 0.685 | True | True |
| 2021+ | 0.875 | 0.897 | -0.141 | -0.181 | 0.601 | 0.529 | True | True |
| 2022+ | 0.842 | 0.702 | -0.109 | -0.159 | 0.713 | 0.466 | True | True |
| 2023+ | 1.071 | 1.36 | -0.109 | -0.132 | 1.05 | 1.146 | False | True |
| 2024+ | 1.237 | 1.032 | -0.109 | -0.132 | 1.262 | 0.866 | True | True |

- Beats vol-targeted BAH (Sharpe OR Calmar) on **4/5** windows.
- Improves max drawdown on **5/5** windows.