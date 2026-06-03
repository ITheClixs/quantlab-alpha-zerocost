# Funding-Carry Paper Sim — Live-vs-Model Reconciliation

**Observation-only.** Strategy verdict: **DO_NOT_ADVANCE**. Not validation, not a step toward live (CLAUDE.md §7, §11).

- cycles: 1  |  rebalances: 4
- funding P&L collected: 0.29 USD
- equity: 100000.00 -> 100000.29 (delta +0.29)
- live basis mean: -0.0354%  |  max |basis|: 0.0365%  (backtest daily-close model: ~0% mean, <0.1% p95)

Funding/basis are REAL (public mainnet); fills are simulated (FillModel). Compare live funding/basis above against the backtest cost-and-tail models in `reports/signal_research/funding_carry_v1/funding_carry_realism_results.md`.
