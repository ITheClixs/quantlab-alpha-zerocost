# Intake — Regime-Fingerprint-Conditioned VWAP-Entry Meta-Labeling v1

**Date:** 2026-06-03
**Status:** OPEN — research_only. **No paper. No live.** No promotion language.
**Branch / artifacts root:** `reports/signal_research/fingerprint_vwap_meta_v1/`
**Provenance:** approach adapted from an external project (`stockML`) — per-stock
multi-window "fingerprints" + an `entry_vwap` success model. Re-expressed here in this
repo's leakage-safe, cost-aware, gate-disciplined harness. It is a **meta-labeling /
entry-quality** design (López de Prado), not a new alpha source.
**Data audit:** PENDING — must PASS before any model is trained
(`reports/signal_research/fingerprint_vwap_meta_v1/data_audit.md`).

## 1. Thesis

A **stock fingerprint** is a compact multi-window descriptor of a name's behavior —
trend direction, trend strength, trend linearity (R²), spikiness, volatility, and
liquidity computed over `{20, 60, 120, 252, full}`-day lookbacks. The hypothesis: a
**VWAP-anchored entry** (buy when price is at/below the session VWAP) has *conditional*
quality — it works in some regimes and not others — and a stock's fingerprint predicts
when. We therefore **meta-label** VWAP entries: a secondary model maps the fingerprint to
`P(entry succeeds)`, and we only take (or up-size) entries with high predicted success.

"Success" is a triple-barrier outcome measured **net of execution cost** over a short
horizon (e.g. `3d_vwap_success`: from the VWAP fill, the upper barrier is hit before the
lower/vertical barrier, after costs). This is an *entry-timing/sizing overlay on top of a
primary rule*, not an independent return forecast.

## 2. Relation to prior work in this repo

This is the same family as `signal_research/papers/triple_barrier.py` and
`signal_research/backtests/triple_barrier_av_lee.py`, reusing
`signal_research/methodology/meta_labeling.py` and
`signal_research/training/meta_label_walk_forward.py`. **What is new:** the *conditioning
set* is a multi-window **regime fingerprint** (assembled from existing `alpha_eq/features/`
builders), and the primary signal is a **VWAP entry** rather than a momentum/MA primary.
No new engine is required — this is a composition of existing modules.

## 3. Honest priors (stated before any backtest, so they cannot be rationalized away)

1. **Cost wall is the dominant threat.** "Enter at VWAP, exit within ~3 days" is an
   execution-timing overlay. This program has repeatedly shown timing/microstructure
   edges die on spread + slippage + fees. The success label **must** be defined net of
   realistic round-trip cost; the gate is **net-of-cost** forward return, never the raw
   `vwap_success` rate.
2. **Meta-labeling cannot manufacture a primary edge.** It improves the precision/sizing
   of an *already-positive* primary. If "enter at VWAP" has no gross edge, a perfect
   `P(success)` classifier still nets ~0. We therefore run
   `meta_labeling.check_eligibility(PrimarySignalStats(...))` **first** and stop if the
   primary is ineligible (this is a built-in gate, not optional).
3. **Overfitting / multiple testing.** ~16 base features × 5 windows ≈ 84 fingerprint
   columns over a small basket is a large search relative to information content
   (Bailey–López de Prado: ~45 variants ⇒ P(overfit) > 50%). PBO + Deflated Sharpe are
   mandatory; feature count must be justified against the noise sentinel.
4. **Survivorship.** A basket of *today's* large caps (AAPL/NVDA/META/…) is survivorship-
   biased. The universe must use point-in-time membership (`alpha_eq/data/pit_quality.py`)
   or the result is inflated and disqualified.
5. **Look-ahead in `full`-window features.** A full-history fingerprint feature computed
   over the entire series leaks the future into past decisions. All fingerprint features
   must be **as-of** (rolling, right-anchored); the `full` window is allowed only as a
   *static descriptor known before the test period*, never recomputed with future data.
6. **Data frequency.** True intraday VWAP needs intraday bars. On daily panels, VWAP is a
   daily proxy (e.g. volume-weighted typical price) and the entry-timing edge is muted;
   an intraday version exists only where `feeds`/recorded intraday data is available, and
   that variant must clear the cost wall even harder.

## 4. Two-wall test (the program's standing filter)

- **Cost wall:** NOT obviously escaped — this is the central risk. Mitigation: cost-baked
  labels + net-of-cost gate + a low-turnover (filter, don't churn) variant.
- **Subsumption wall:** plausibly escaped — an entry-quality/sizing overlay is not single-
  index vol-timing, so vol-targeting does not trivially restate it. But the overlay is
  only meaningful if the primary VWAP entry has measurable gross edge (prior #2).

## 5. Data

- **Universe:** liquid US equities (NYSE/NASDAQ/S&P/DOW) with **point-in-time** membership;
  source from `data/processed/` panels first; reference `data/raw/` via manifests only if
  needed. Start from the existing processed equity panel used by `alpha_eq`.
- **VWAP:** daily VWAP proxy from OHLCV (volume-weighted typical price) for v1; flag an
  intraday variant as out-of-scope unless recorded intraday bars are available via `feeds`.
- **Data audit (must PASS first):** no missing/duplicated bars on the trading calendar;
  features are right-anchored (no look-ahead); labels are timestamped after the entry;
  PIT membership verified; corporate-action adjustment consistent. Write
  `reports/signal_research/fingerprint_vwap_meta_v1/data_audit.md` and gate on it.

## 6. Method / pipeline (composition of existing modules)

1. **Fingerprint features** — assemble multi-window descriptors via `alpha_eq/features/`
   (`market_regime.build_market_regime`, `volatility`, `volume_liquidity`,
   `returns_momentum`, `microstructure_proxies`, `cross_sectional_ranks`) over
   `{20,60,120,252,full}`, via `features/builder.py`. All as-of.
2. **Primary signal** — VWAP entry rule: candidate long when `close ≤ VWAP·(1−band)`
   (mean-reversion to VWAP) on an eligible liquid name. Deterministic, parameterised by
   `band`.
3. **Labels** — `signal_research/papers/triple_barrier.label_triple_barrier(...)` from the
   VWAP fill (`alpha_eq/backtest/fills.py`), with **cost-baked** upper/lower/vertical
   barriers; horizon = 3 trading days (`3d_vwap_success`) plus a `range_success` variant.
4. **Eligibility gate** — compute `PrimarySignalStats` over the primary entries and run
   `meta_labeling.check_eligibility(...)`; **stop here** if ineligible.
5. **Meta-model** — `meta_label_walk_forward` (`MetaLabelWalkForwardConfig`): purged +
   embargoed walk-forward classifier `fingerprint → P(success)`; predictions size/filter
   entries.
6. **Backtest** — net-of-cost equity curve via `backtest/runner.py` (+ `report.py`);
   compare against the **baseline of taking every eligible VWAP entry** (the lift test).

## 7. Gates (pass/fail bar — all must hold)

- Leakage-safe purged + embargoed walk-forward (no future data; no fit-on-test).
- Adversarial validation (train-vs-holdout AUC ≤ 0.6) + noise-sentinel (no fingerprint
  feature ranks below the seeded noise feature in ≥3/5 folds).
- **PBO** acceptable (combinatorially-symmetric CV) and **Deflated Sharpe** positive at
  the trial count actually searched.
- **Net-of-cost forward Sharpe** (after 2× cost + entry delay) materially positive.
- **Lift:** meta-filtered net Sharpe > the take-every-VWAP-entry baseline by a
  pre-registered margin (otherwise the fingerprint adds nothing).

## 8. Pre-committed kill criteria (bound the work)

Close the branch and write a negative-result note if **any** holds:
- primary VWAP entry is **ineligible** for meta-labeling (`check_eligibility` fails); or
- net-of-cost OOS Sharpe < **0.7**, or no statistically-meaningful lift over the
  take-every-entry baseline; or
- PBO high / Deflated Sharpe ≤ 0 at the searched trial count; or
- the result depends on survivorship or any look-ahead the audit catches.

## 9. Deliverables & transfer trigger

Artifacts under `reports/signal_research/fingerprint_vwap_meta_v1/`: `data_audit.md`,
`fingerprint_features.md`, `meta_label_walkforward_results.md`, `realism_results.md`
(net-of-cost), and a final verdict note (`PASS` or `NEGATIVE-RESULT … DO_NOT_ADVANCE`).
**Transfer to Prevalence only if all §7 gates pass** — then it becomes a selectable,
gated strategy there (same engine), with its disclosure (algo + datasets + cited papers).

## 10. Non-actions / scope

- research_only; **no paper, no live, no promotion language**; kill-switch discipline.
- Do not present `vwap_success` rate, gross Sharpe, in-sample, or cost-free numbers as edge.
- Do not expand the feature/window grid mid-study to chase a number (multiple-testing).

## 11. References

- Internal: `signal_research/methodology/meta_labeling.py`,
  `signal_research/training/meta_label_walk_forward.py`,
  `signal_research/papers/triple_barrier.py`,
  `signal_research/backtests/triple_barrier_av_lee.py`, `alpha_eq/features/*`,
  `alpha_eq/backtest/fills.py`, `validation/*`, `crypto_research/perps/validation.py` (PBO/DSR).
- External: López de Prado, *Advances in Financial Machine Learning* (2018) —
  triple-barrier labeling, meta-labeling, purged/embargoed CV, PBO/DSR.
