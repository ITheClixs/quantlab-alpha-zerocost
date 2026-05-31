# Zero-Cost Risk-Allocator v1 — Strict Second-Stage Review

**Built:** 2026-05-30T15:13:52.779560+00:00 | basket 2017-11-09..2026-05-22 | research_only, no paper/live.

## (2) Instrument PnL attribution
| instrument | PnL share | crisis-period PnL |
|---|---:|---:|
| SPY | 22.8% | 0.0606 |
| QQQ | 32.3% | 0.0338 |
| BTCUSDT | 22.7% | 0.141 |
| ETHUSDT | 22.2% | 0.0856 |

- Crypto (BTC+ETH) total PnL share: **44.9%** | max single-instrument share: **32.3%** (flag if >35%) | max crisis-period share: **43.9%** (flag if >50%).

## (5) Ex-crisis diagnostic (binding)
- Strategy ex-crisis Sharpe **0.977** vs vol-targeted BAH **1.586** → CRISIS-DEPENDENT (insurance, not calm-market alpha).
- Product label: **crisis_insurance_allocator**.

## (6) Exception-style stress gate
- ✅ beats_bench_full_sharpe
- ✅ improves_full_maxdd
- ✅ windows_beaten_ge_3of5
- ✅ dd_improved_ge_4of5
- ✅ delay_degradation_le_0.5
- ✅ survives_2x_cost
- ✅ survives_3x_cost
- ✅ not_crypto_dependent
- ❌ crypto_out_still_beats_bench_dd
- ✅ no_single_instrument_gt_35pct
- ✅ excrisis_nonneg
- ✅ inverted_worse_than_strat
- ✅ random_worse_than_strat
- delay-1→delay-2 Sharpe degradation: -0.111 (gate ≤ 0.5).
- inverted-sanity Sharpe 0.726; random-alloc Sharpe 0.587 (both must be < strategy 1.099).

## (7) Decision
- **DO_NOT_ADVANCE** | failure_class: **crypto_regime_concentration** | label: **crisis_insurance_allocator** | promotion_eligible: False
- See `zero_cost_riskalloc_paper_decision.md`.