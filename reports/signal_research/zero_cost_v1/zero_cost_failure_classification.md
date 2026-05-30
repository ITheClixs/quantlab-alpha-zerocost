# Zero-Cost Risk-Allocator v1 — Classification

**Decision:** PASS_WITH_CAVEAT_NEEDS_STRICTER_REVIEW | primary: `edge_is_crisis_dependent_vs_benchmark`
**Hard blockers (§6 kill):** `none`
**Caveats:** `edge_is_crisis_dependent_vs_benchmark`

## Evidence
- Strategy holdout Sharpe 1.237 vs voltarget_bah 1.032; maxDD -0.109 vs -0.132.
- **Ex-crisis Sharpe 0.9771 vs benchmark 1.5862.**

## Decision rationale
- The strategy clears the literal §6 gate (beats vol-targeted BAH on holdout Sharpe AND max drawdown, PBO<0.25, DSR high) BUT its outperformance over the benchmark is **crisis-driven** — ex-crisis it UNDERPERFORMS vol-targeted BAH. It is a drawdown-control / crisis-insurance overlay, not calm-market alpha. **Do NOT auto-advance to paper.** Stricter review required first: the single-index exception policy's full 24-criterion gate, a crypto-out (SPY/QQQ-only) test, and multiple holdout windows.

_research_only; no paper/live; no promotion until stricter review passes._
