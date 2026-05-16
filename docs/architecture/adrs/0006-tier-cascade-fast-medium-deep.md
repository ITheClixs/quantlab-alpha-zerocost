# ADR 0006: Three-tier governor cascade — fast / medium / deep

## Status
Accepted, 2026-05-16.

## Context
S2 governs every S1 signal but on the M4 a single Mistral 22B Q4 call costs 5–10 s,
and Yi 34B Q4 costs 20–30 s. Calling either on every signal misses the trading window
for tick-frequency crypto and is overkill for low-confidence S1 signals that won't
trade anyway.

## Decision
Three tiers with explicit gates:

- Tier 1: Qwen 0.5B-Instruct + LoRA, runs on every signal, < 500 ms, decision space
  reduced to {pass, veto}.
- Tier 2: Mistral 22B Q4_K_M, runs only when Tier 1 passes AND |signal.confidence| > 0.6,
  ~5–10 s, RAG top-5 evidence required, citations mandatory.
- Tier 3: Yi 34B Q4_K_M, runs only when trade_size_pct > 1 %, async; verdict applies
  to NEXT trade in the same symbol (stance modifier).

A pass requires unanimity across every tier that ran.

## Consequences
+ Latency budget honored on tick-frequency crypto.
+ Every model on disk has a role; nothing wasted.
+ Deep reasoning on big trades without blocking the loop.
- Three runtime classes to maintain.
- LoRA training adds 8 hours to the full-retrain budget.
