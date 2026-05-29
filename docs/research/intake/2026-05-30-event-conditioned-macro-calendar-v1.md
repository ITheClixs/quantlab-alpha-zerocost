# Intake — Event-Conditioned Macro/Calendar v1

**Date:** 2026-05-30
**Status:** CLOSED `research_only` — `already_subsumed_by_vol_or_regime`, no v2.
See `docs/research/2026-05-NEGATIVE-RESULT-EVENT-MACRO-FOMC.md` and run `ca2d716`.
(CPI/NFP families never run — deferred on data grounds.)
**Strategy name:** `event_conditioned_macro_v1`
**Proposer:** QuantLab research
**Promotion intent:** `research_only` for v1 (see §9)
**Program `/goal`:** find **taker-tradable** alpha for QuantLab.

## 0. Context — why this channel now

The free-data crypto microstructure arc is closed
(`docs/research/2026-05-NEGATIVE-RESULT-MICROSTRUCTURE.md`): L2 depth, L1 quotes,
and tick trade-flow were all predictive but untradable after realistic taker
cost. The binding constraint there was execution economics. This intake pivots
to the cheapest untested channel — **scheduled-event timing** — where the
information channel is the *event calendar itself* (a known-time schedule),
conditioning the existing SPY/QQQ daily-bar series. Costs on SPY/QQQ are tiny, so
if there is a regime-conditional edge it has a real chance of being taker-tradable.

## 1. Strategy name and one-line description

`event_conditioned_macro_v1` — test whether scheduled macro events (FOMC, CPI,
NFP) and calendar regimes (earnings season, month/quarter-end) create tradable
regime-conditional return/volatility behavior in SPY and QQQ.

## 2. Hypothesis statement

Scheduled macro releases are pre-announced points of concentrated information
arrival. Around them, dealers and systematic participants de-risk into the
uncertainty and re-risk after it resolves, and realized volatility is
structurally elevated in a narrow window. This creates regime-conditional
behavior that a fixed calendar (known fully in advance, hence leak-free) can
exploit — primarily as **risk timing** (reduce/lift exposure around the window)
rather than directional alpha. The mechanism is documented: the pre-FOMC
announcement drift (Lucca & Moench 2015), elevated event-window implied/realized
vol and the variance risk premium concentration around FOMC/CPI (Bondarenko 2014
for the vol side; Savor & Wilson 2013 for higher average returns on scheduled
announcement days compensating macro risk). The edge, if any, is expected to come
from *avoiding/sizing* the volatility regime, not from forecasting the surprise.

## 3. Information source declaration (machine-readable)

- `event_window` — FOMC / CPI / NFP / earnings-season / period-end calendars
  (the primary, driving channel).
- `ohlcv` — SPY/QQQ daily bars (the conditioned series).

The driving channel is `event_window`. Per `STRATEGY_INTAKE.md` §3, this is a
declared non-OHLCV channel, so the no-OHLCV-only-promotion rule is satisfied: the
hypothesis is driven by event timing, with OHLCV as the instrument acted upon.

## 4. Provenance and timestamp integrity (the binding constraint)

Event-conditioning is only valid if the calendar is **ex-ante and timestamp-clean**.
All event dates used are **scheduled and known in advance**, so conditioning on
distance-to-event introduces no look-ahead provided we use the *announced
schedule*, never revised/realized data, and execute with a documented lag.

| Family | Source | Release time (ET) | Look-ahead control |
|---|---|---|---|
| FOMC decision | Federal Reserve FOMC calendar | ~14:00 announcement | scheduled years ahead; daily position set at **prior close** |
| CPI | BLS release calendar | 08:30 (pre-open) | scheduled; CPI hits before the 09:30 open, so any day-of position is set at **prior close** (no same-day open entry on the release) |
| NFP / employment | BLS Employment Situation calendar (if accessible) | 08:30 (pre-open) | same as CPI |
| Earnings season | public season-window proxy (deterministic mid-Jan/Apr/Jul/Oct windows), **broad regime only, not single-stock** | n/a | calendar regime, fully deterministic |
| Month/quarter-end | exchange calendar (deterministic) | n/a | deterministic; separate calendar feature |

**`event_timestamp_audit.md` will verify:** every event date is from the ex-ante
published schedule; no revised dates; pre-open releases (CPI/NFP) never trigger a
same-day-open entry; all conditioning features (`days_to_next_event`,
`days_since_last_event`, `in_window` flags) are computable using only information
available at the prior close; and the SPY/QQQ bar timestamps align (UTC/ET) with
the event dates. License/cost: all sources public, $0.

## 5. Expected gross Sharpe and capacity

- **Ex-ante expected gross daily Sharpe:** modest, **0.3–0.8**. The honest prior
  is that the edge (if real) is a **drawdown/vol-reduction** effect, not a large
  return uplift — so the headline win, if any, shows up in **Calmar / max
  drawdown** more than raw Sharpe.
- **Capacity:** high (SPY/QQQ are among the most liquid instruments); not a
  binding constraint at v1.

## 6. Cost assumptions

- SPY/QQQ: **1.0 bps one-way** commission+spread (liquid ETF), plus the pipeline's
  **2× cost stress** and **1-bar (close→next-open) delay stress**. A risk-timing
  gate that only wins at zero cost is not a strategy.

## 7. Universe and history

- **Universe:** SPY and QQQ. **Daily bars first.** Intraday only if clean
  timestamped data is *already* on disk (not pursued otherwise in v1).
- **History:** existing SPY/QQQ OHLCV (target 2006–2026, subject to on-disk
  coverage).
- **Split:** chronological **train / validation / holdout**; holdout extends
  ≥18 months after validation_end; dev-only-guard enforced. No random splits
  except placebo event-date generation.
- **Crisis-removal robustness:** re-evaluate with **2020 removed**, **2022
  removed**, and **2023–2026 removed** to test crisis dependence.

## 8. What would make this fail? (pre-registered failure classes)

Per the decision rule, a failed v1 is classified into exactly one of:
1. **no event edge** — event windows indistinguishable from baseline.
2. **event edge too small after costs** — gross effect real, dies on 1 bps + 2×.
3. **regime concentration** — edge concentrated in one crisis year; fails
   crisis-removal.
4. **placebo indistinguishable** — random-calendar and/or shifted-calendar
   placebos match the real-calendar result.
5. **insufficient event count** — too few events (esp. FOMC ~8/yr) for stable
   inference; bootstrap CI too wide.
6. **already subsumed by HMM / vol-targeting** — event gate adds nothing over the
   HMM-only and vol-targeted-BAH baselines (the VRP×HMM subsumption pattern).

Three specific expected failure modes (falsification): (a) the CPI/FOMC vol
effect is real but the day-of position must be set at prior close, and by then
it is already priced — net edge ≈ 0 after costs; (b) FOMC event count (~8/yr) is
too small, so any apparent edge has a wide bootstrap CI and fails DSR; (c) the
event gate is subsumed by simple vol-targeting (the vol regime *is* the signal).

## 9. Promotion intent

`research_only` for v1. If a variant clears the full gate **and** beats the
HMM-only / vol-targeted baselines **and** survives both placebos and
crisis-removal, it may reach `exception_review_required` under the single-index
risk-timing exception policy (`docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`),
since this is single-index risk timing. v1 itself does not self-promote; any
paper/live step requires the §11 promotion gate. No OHLCV-only promotion.

## 10. Sign-off

Proposer: QuantLab research. Intake date: 2026-05-30. Subjected to the full
validation gate with **no post-hoc tuning after holdout**. This is **not a broad
parameter search**: event windows and variants are predeclared below and frozen.

---

## 11. Pre-declared experiment design (frozen)

### 11.1 Event families
1. FOMC decision dates
2. CPI release dates
3. NFP / employment release dates (if accessible)
4. Earnings-season windows — broad calendar regime, **not** single-stock earnings
5. Month-end / quarter-end rebalancing windows — separate calendar feature

### 11.2 Pre-declared event windows
- `t-1`
- `t`
- `t+1`
- `t-2 .. t+2`
- `t-5 .. t+5` — **diagnostic only** (not a promotable variant)

### 11.3 Strategy variants (frozen pool)
1. FOMC risk-off gate
2. CPI risk-off gate
3. FOMC risk-on / post-event re-entry
4. CPI risk-on / post-event re-entry
5. Combined FOMC + CPI calendar gate
6. Event-conditioned volatility-targeted SPY
7. Event-conditioned volatility-targeted QQQ
8. Event-conditioned HMM overlay — **diagnostic only**, not the primary strategy

### 11.4 Baselines (first-class, in the PBO pool)
- Buy-and-hold SPY / QQQ
- Volatility-targeted buy-and-hold
- HMM-only risk timing
- Simple SMA 50/200 gate
- **Random event dates** matched by frequency (placebo)
- **Shifted event dates** (e.g. event date + 10 trading days) (placebo)

### 11.5 Validation battery
- Chronological train / validation / holdout (no random splits except placebo
  date generation)
- PBO and DSR over the full frozen pool
- Stationary block-bootstrap CI on Sharpe
- Cost stress (declared, 2×)
- 1-bar (close→next-open) delay stress
- Concentration diagnostics (monthly/yearly PnL share)
- Crisis removal: remove 2020, remove 2022, remove 2023–2026
- Event-family attribution (which family carries the effect)
- Random-calendar placebo test
- Shifted-calendar placebo test

### 11.6 Required deliverables
- `docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md` (this file)
- `event_calendar_manifest.json`
- `event_timestamp_audit.md`
- `event_strategy_registry.parquet`
- `event_conditioned_validation_report.md`
- `event_placebo_report.md`
- `event_failure_classification.md` (if no variant survives)

### 11.7 Decision rule
- **If it fails:** classify into one of the §8 failure classes.
- **If it succeeds:** run a second iteration **only on the surviving event
  family**, with stricter tests and intraday timestamp refinement.

## 12. What happens after this intake

1. This document is committed to `docs/research/intake/`.
2. Build the event calendar layer (manifest + timestamp audit) **before** any
   backtest; the audit must pass (§4) or the channel is rejected on data grounds.
3. Implement the frozen variant pool + baselines as signal generators.
4. Run `ValidationPipeline` with the §11.5 battery; emit the §11.6 deliverables.
5. Commit the status classification alongside the reports. No re-tuning after
   holdout numbers are seen; no expanding the frozen pool in response to a near-miss.
