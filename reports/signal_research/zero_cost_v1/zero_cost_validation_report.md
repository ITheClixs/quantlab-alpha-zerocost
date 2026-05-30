# Zero-Cost Risk-Allocator v1 — Validation Report

**Built:** 2026-05-30T10:50:40.743212+00:00 | basket 2017-11-09..2026-05-22 (2144 days) | instruments ['SPY', 'QQQ', 'BTCUSDT', 'ETHUSDT']
**Intake:** `docs/research/intake/2026-05-30-zero-cost-deployable-v1.md` | paper_trade_after_pass.
Long-flat, weekly rebalance, equal-risk, decision close t / execution t+1. Cost: SPY/QQQ 1bp, BTC/ETH 8bp.

| variant | full Sharpe | holdout Sharpe | full maxDD | holdout maxDD | full Calmar | ann ret | Sharpe@2xcost | Sharpe@delay2 | ex-crisis Sharpe | ann turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_cost_riskalloc_v1` | 1.0986 | 1.2372 | -0.1428 | -0.1093 | 0.7466 | 0.1066 | 1.0746 | 1.2096 | 0.9771 | 2.43 |
| `voltarget_bah` | 0.8756 | 1.0323 | -0.234 | -0.1318 | 0.4009 | 0.0938 | 0.8674 | 0.8677 | 1.5862 | 0.82 |
| `buy_and_hold` | 0.9046 | 0.7766 | -0.6211 | -0.3494 | 0.5259 | 0.3266 | 0.9045 | 0.9154 | 1.5371 | 0.03 |
| `trend_only` | 0.7962 | 1.1834 | -0.5883 | -0.2202 | 0.3857 | 0.2269 | 0.7884 | 0.8085 | 1.2672 | 1.67 |
| `regime_only` | 0.7637 | 1.369 | -0.5935 | -0.275 | 0.4161 | 0.247 | 0.7411 | 0.7606 | 1.2985 | 4.24 |
| `random_alloc` | 0.5866 | 0.3351 | -0.5024 | -0.2493 | 0.2111 | 0.106 | 0.5484 | 0.6208 | 1.2916 | 4.41 |

## Binding gate (vs vol-targeted buy-and-hold, holdout)
- beats voltarget_bah on holdout Sharpe: **True**
- beats voltarget_bah on holdout max drawdown: **True**
- bootstrap Sharpe CI lower (95%): **0.4409793601446298** | PBO **0.07142857142857142** | DSR **0.9717240166858837**

## Decision
- **PASS_WITH_CAVEAT_NEEDS_STRICTER_REVIEW** | paper_candidate: False | promotion_eligible: False
- hard blockers (§6 kill): `none`
- caveats (pass-but-fragile): `edge_is_crisis_dependent_vs_benchmark`
- **Ex-crisis check: strategy Sharpe 0.9771 vs vol-targeted BAH 1.5862** — if the strategy is lower, its edge over the benchmark is crisis-driven (better drawdowns in 2018/2020/2022), NOT calm-market alpha.

## Notes / honest caveats
- Basket window is ETH-bound (~2017+); crypto history is short and crisis-heavy → wide CIs.
- Crypto daily returns aligned to the equity trading calendar (weekend moves fold into Monday).
- Macro features (VIX term structure, credit, yield slope) market-priced, used t+1. research_only until paper.
