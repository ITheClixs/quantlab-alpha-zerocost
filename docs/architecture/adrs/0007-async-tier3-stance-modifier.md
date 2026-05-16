# ADR 0007: Tier 3 verdicts apply to NEXT trade, not current

## Status
Accepted, 2026-05-16.

## Context
Tier 3 (Yi 34B Q4) takes 20–30 s per call. We want its deep reasoning without blocking
the trading loop on signals that need to fire within seconds.

## Decision
When Tier 3 is triggered (trade_size_pct > 1 %), it is scheduled async. The current
verdict for S4 is the Tier 2 verdict. Tier 3's eventual verdict is written to a
separate file `experiments/s2_verdicts_tier3/<date>.jsonl`. S4's risk engine reads
Tier 3 verdicts as a stance modifier for the NEXT trade in the same symbol — never
the current one. A Tier 3 veto on the previous trade in this symbol widens the next
signal's confidence threshold by 0.20 (a tightening, not a hard block).

## Consequences
+ Trading loop never waits on Yi 34B.
+ Deep reasoning still influences the system, just with a one-trade lag.
+ Crashes in the async worker do not block trades.
- Operators must understand the lag semantics; runbook covers this.
- A single Tier 3 verdict from a stale signal could over-tighten the next trade;
  the modifier is intentionally small (20 %) to limit overreaction.
