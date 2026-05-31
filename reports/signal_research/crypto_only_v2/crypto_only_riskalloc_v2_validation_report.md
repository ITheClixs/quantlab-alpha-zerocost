# crypto_only_riskalloc_v2 — Validation Report

**Built:** 2026-05-30T15:29:57.515484+00:00 | basket 2017-11-09..2026-05-30 (3125 days) | BTCUSDT+ETHUSDT, long-flat, no leverage, weekly rebalance, close t / execute t+1.
**Pre-registration:** operator message 2026-05-30 (reframe after zero_cost_riskalloc_v1 strict review).
**Cost:** SPOT, 20 bps one-way (taker ~10 + spread ~5 + slippage ~5); funding N/A (spot data). research_only.

| variant | Sharpe | maxDD | Calmar | ann ret | total ret |
|---|---:|---:|---:|---:|---:|
| `crypto_only_riskalloc_v2` | 0.6287 | -0.2209 | 0.2605 | 0.0575 | 1.001 |
| `voltarget_5050` | 0.5343 | -0.4005 | 0.1715 | 0.0687 | 1.2791 |
| `bah_5050` | 0.6242 | -0.8794 | 0.2399 | 0.2109 | 9.7337 |
| `trend_only` | 0.5634 | -0.7433 | 0.2125 | 0.158 | 5.1652 |
| `bah_btc` | 0.6188 | -0.834 | 0.2478 | 0.2066 | 9.2717 |
| `bah_eth` | 0.5669 | -0.9396 | 0.1699 | 0.1596 | 5.2736 |
| `voltarget_btc` | 0.5237 | -0.39 | 0.1872 | 0.073 | 1.3959 |
| `voltarget_eth` | 0.488 | -0.4278 | 0.1447 | 0.0619 | 1.1062 |
| `random_alloc` | 0.4228 | -0.7727 | 0.1146 | 0.0886 | 1.8644 |
| `inverted` | 0.4208 | -0.4005 | 0.1153 | 0.0462 | 0.7501 |

## Stress + robustness
- Sharpe @2x cost: 0.570 | @3x cost: 0.511 | @delay-2: 0.712
- bootstrap Sharpe CI lower: -0.01874813007331706 | DSR: 0.739097296325705

## Instrument & year attribution
- BTC PnL share 0.3 | ETH 0.7 (flag if one >65%: max 0.7)
- max single-year PnL share: 0.417 (flag if >50%)
- year shares: {2017: 0.0, 2018: 0.0, 2019: 0.034, 2020: 0.417, 2021: 0.089, 2022: -0.134, 2023: 0.107, 2024: 0.368, 2025: 0.119, 2026: 0.0}

## Gate scorecard
- ✅ windows_beaten_ge_3of5
- ✅ dd_improved_ge_4of5
- ✅ sharpe_pos_2x
- ✅ sharpe_pos_3x
- ✅ delay_ok
- ✅ no_year_gt_50pct
- ❌ no_asset_gt_65pct
- ❌ bootstrap_lower_pos
- ✅ dsr_ok
- ✅ random_inverted_fail
- ✅ paper_feasible_spot_weekly

## Decision: **DO_NOT_ADVANCE** | failure_class: **single_asset_concentration_eth** | promotion_eligible: False (paper at most)