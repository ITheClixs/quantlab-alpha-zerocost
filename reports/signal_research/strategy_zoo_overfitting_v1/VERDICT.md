# Strategy-Zoo Backtest-Overfitting Demonstration — VERDICT

**Date:** 2026-06-04
**Status:** research_only. **DEMONSTRATION PASS — thesis empirically confirmed.** Nothing
transfers to Prevalence. No paper, no live, no promotion.

## Verdict

The demonstration succeeded: a large grid of systematically-enumerated strategies, run
through PBO + Deflated Sharpe + a price-permutation MCPT on real 2015–2026 data, shows that
**the best in-sample performer is a selection artifact, not skill.** At the N = 1,000 headline
tier:

- best in-sample Sharpe **0.96 < expected-max-under-null 1.02** (the winner is below the
  chance bar),
- **PBO 0.69**, **Deflated Sharpe of the best 0.44**, **0 / 1000 strategies survive** the
  deflated bar,
- permutation MCPT **p-value 1.0** (real best 0.83 < permuted mean 1.35).

This turns README §6's *theory* (`E[max] ≈ √(2 ln N)`, McLean–Pontiff, Bailey–López de Prado)
into a *measured* result, and reproduces the program's standing thesis: **with enough trials,
impressive backtests are guaranteed by chance, and the gates correctly reject them.**

## On the "surprise" branch

No family cleared PBO **and** Deflated Sharpe **and** the permutation control. Had one done so,
the spec's rule applies: do **not** celebrate — escalate it to its own intake with a full
survivorship/PIT/capacity audit. One lucky cell in a million is the null hypothesis until
proven otherwise. That did not occur here.

## Follow-up (optional, same conclusion)

The 10k and 100k headline tiers are documented in `report.md §4`; they only sharpen the
result (the deflated bar rises as √(2 ln N)). The reusable deliverable is the **zoo harness**
(`strategy_benchmark/zoo/*`) and this demonstration — a methodology artifact for the
research-paper codebase, referenced from `README.md`.
