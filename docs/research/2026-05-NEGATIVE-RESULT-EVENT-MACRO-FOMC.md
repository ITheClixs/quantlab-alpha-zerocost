# Negative-Result Note — Event-Conditioned Macro/Calendar v1 (FOMC) (closed)

**Date:** 2026-05-30
**Status:** CLOSED at `research_only`. No v2. Do not continue FOMC variants.
**Failure class:** `already_subsumed_by_vol_or_regime`
**Author:** QuantLab research

## 0. The finding

> Scheduled FOMC windows contain risk-timing information, but the economically
> useful part is already captured by simpler volatility targeting.

The FOMC effect is **real, robust, and placebo-distinguishable** — it is not noise.
It is nonetheless **not promotable**, because a plain vol-targeted buy-and-hold
already captures the deployable part, and it does not clear the 1.5 single-index
Sharpe gate. This is the cleanest "real signal, not deployable" result in the
program, distinct from the OHLCV noise-floor and the microstructure
predictive-but-untradable closes.

## 1. Evidence (run `ca2d716`, FOMC-only, SPY/QQQ daily 2010-2026, 131 events)

Reports: `reports/signal_research/event_macro_v1/`
(`event_conditioned_validation_report.md`, `event_placebo_report.md`,
`event_failure_classification.md`, `event_strategy_registry.parquet`).

- **Real effect:** FOMC risk-off (day-after / ±2-day window) Sharpe ~1.02-1.05 on
  QQQ vs buy-and-hold ~0.86-0.96.
- **Passes placebos:** real ~1.05 vs random-calendar 0.70 and shifted-calendar
  (+10d) 0.77-0.86 — the edge is event-driven, not a calendar artifact.
- **Statistically clean:** PBO = 0.0, DSR = 0.985, bootstrap-lower = 0.60.
- **Survives stress:** 2× cost ~1.05, 1-bar delay ~0.98, crisis removal
  (ex-2020 1.02, ex-2022 1.20, ex-2023-26 1.03).
- **But subsumed:** best event variant 1.06 < vol-targeted BAH 1.10; below the
  1.5 single-index gate. The FOMC window largely coincides with a volatility
  regime that simple vol-targeting already exploits — the same subsumption
  pattern as the earlier VRP × HMM result.

## 2. Explicit non-actions (operator decision, 2026-05-30)

- **No more FOMC windows.** The predeclared windows (t-1, t, t+1, t-2..t+2,
  t-5..t+5 diag) are exhausted; no further tuning.
- **Do not lower the 1.5 Sharpe gate.**
- **Do not add HMM or VRP overlays to rescue it.** Rescue-by-overlay would be
  post-hoc and is forbidden.
- **No v2.** Per the intake decision rule, a second iteration runs only on a
  surviving event family; none survived.
- **CPI/NFP remain deferred** (no timestamp-clean source; same event-window /
  risk-timing channel — not pursued now).

## 3. Program meta-finding

Across VRP, HMM single-index, and now FOMC event-timing, **volatility-targeting /
regime exposure subsumes most single-index risk-timing edges**. New directions
should seek a *structurally different* information source, not another risk-timing
overlay on the same index. Next: futures carry / term-structure v1
(`docs/research/intake/2026-05-30-futures-carry-term-structure-v1.md`). Program
`/goal`: find **taker-tradable** alpha.
