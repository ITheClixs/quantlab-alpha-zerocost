# Program Review — The Data-Entitlement Constraint (May 2026)

**Date:** 2026-05-30
**Status:** Program-level review; **decision point — no new channel until direction chosen.**
**Author:** QuantLab research
**Scope:** consolidates the post-HMM arc (microstructure ×3, event-macro FOMC,
futures carry) and frames the central question now blocking progress: **is the
binding constraint methodology, or data access?**
**Program `/goal`:** find **taker-tradable** alpha for QuantLab.

## 0. Executive summary

Since the prior program review (`2026-05-PROGRAM-REVIEW-SIGNAL-RESEARCH.md`), five
more channels were opened and closed under the same disciplined gate. The program
still has **0 deployable / taker-tradable strategies**. Two findings now dominate:

1. **Vol-targeting / regime exposure subsumes single-index risk-timing.** VRP, HMM,
   and FOMC event-timing all produced real, statistically-clean signals that a
   simple vol-targeted buy-and-hold already captures.
2. **The binding constraint has shifted from methodology to DATA ENTITLEMENT.**
   Every *structurally new* channel reached for is blocked by the same wall — the
   Massive.com 403 download paywall or the absence of any free clean source. The
   freely-accessible channels are all OHLCV / index-risk-timing, which (1) subsumes.

This review does not propose action. It lays out the channel × data matrix, the
honest cost/benefit of a paid data tier, the two lessons that persist *even with*
data, and a decision framework.

## 1. Closed channels (full ledger)

| Channel | Verdict | Binding reason | Reference |
|---|---|---|---|
| OHLCV cross-sectional (6 iters) | closed | noise floor; PSR/DSR kill it | `2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` |
| VRP (index) | closed | real but subsumed by HMM regime | program review §2 |
| HMM single-index | closed | delay-sensitive + refit-unstable | program review §3 |
| Microstructure L2 depth | closed | predictive but untradable (cost) | `2026-05-NEGATIVE-RESULT-MICROSTRUCTURE.md` |
| Microstructure L1 quotes (Mode B) | closed | net<0 at every threshold | same |
| Microstructure tick trade-flow (Mode A) | closed | IC 0.45 yet untradable (bounce) | same |
| Event-macro FOMC | closed | real, placebo-clean, but subsumed by vol-targeting | `2026-05-NEGATIVE-RESULT-EVENT-MACRO-FOMC.md` |
| Futures carry / term-structure | closed | **no curve data** (Massive 403; yfinance continuous-only) | `intake/2026-05-30-futures-carry-term-structure-v1.md` |

The durable deliverable remains the **validation infrastructure** (ValidationPipeline,
PBO/DSR, bootstrap, cost/delay stress, placebo tests, leakage-safe feature layers,
the data-audit-first gate) — it has correctly rejected every strategy and caught
real bugs (in-sample leakage, the `mid_direction_up` future-label leak, a
falsy-zero PBO classifier bug).

## 2. The data-entitlement constraint (the core issue)

| Data the program can freely access | What it supports | Result |
|---|---|---|
| OHLCV daily bars (yfinance, on-disk) | cross-sectional + index timing | noise / subsumed |
| Free crypto microstructure (Binance public) | tick/quote/depth backtest | untradable after taker cost |
| Continuous front-month futures (yfinance `=F`) | trend baselines | not carry (no curve) |
| Ex-ante FOMC calendar (public) | event risk-timing | subsumed by vol-targeting |

| Data the program CANNOT access (free) | Would unlock | Wall |
|---|---|---|
| SIP equity trades/quotes at scale | equity microstructure | Massive `GetObject` 403 (paid) |
| Futures per-contract curve (front/next/expiry) | carry/term-structure | Massive 403; no free clean source |
| OPRA options trades/quotes | options risk premia | Massive 403 (paid) |
| Clean CPI/NFP release calendar | event timing | ALFRED/BLS 403 to automation |
| PIT fundamentals (CRSP/Compustat) | fundamental cross-section | paid vendor |

**The pattern is unambiguous:** the cheap channels are exhausted; the structurally
different channels require entitled data.

## 3. Channels still untested, by data requirement

- **On-disk, heavy audit (no new purchase):**
  - **Options-chain IV** — `gauss314__options-IV-SP500` is on disk. Per-stock/index
    IV surface, skew, term structure, put-call. Genuinely non-OHLCV. Gate: a hard
    timestamp/PIT audit (was each IV observable when used; no look-ahead).
  - **News/sentiment** — `FinGPT-sentiment`, `sp500-earnings-transcripts`,
    `sp500-edgar-10k` on disk. Gate: publication-vs-event timestamp separation —
    historically the hostile, binding constraint.
- **Free + feasible now (no purchase):**
  - **Cross-asset macro** — equity vs bond/FX/commodity regime, using free
    continuous series. Structurally different from single-index timing. Risk: it
    is still a risk-timing overlay that vol-targeting may subsume (the recurring
    failure mode).
- **Paid feed required:** futures curve, SIP equity microstructure, OPRA options,
  clean CPI/NFP, PIT fundamentals.

## 4. Would a paid data tier unlock alpha? (honest cost/benefit)

A paid Massive/Polygon-class flat-file tier would remove the 403 wall and unlock,
in one purchase, **SIP equity tick/quote history, OPRA options, and futures
per-contract curve**. A separate vendor (Sharadar/Compustat) would unlock PIT
fundamentals. Indicative cost: tens to low-hundreds of USD/month for the flat-file
bundle; more for institutional fundamentals.

What each paid unlock would let us test, and the honest prior:

| Paid unlock | New channel it enables | Prior from evidence so far |
|---|---|---|
| SIP equity tick/quote | equity microstructure at scale | **Low.** Crypto microstructure (3 channels) was predictive-but-untradable; equity taker costs are not lower. Likely repeats. |
| Futures per-contract curve | cross-market carry / term-structure | **Medium-high.** Carry is a documented, *diversifiable, cross-sectional* premium — NOT a single-index risk-timing overlay, so it is the most likely to escape the subsumption failure. Untested here only because of data. |
| OPRA options | per-stock VRP, skew, surface | **Medium.** Index VRP was subsumed by HMM, but *per-stock* skew/VRP cross-section is a different, richer object that was never tested. |
| PIT fundamentals | fundamental cross-section | **Medium.** OHLCV cross-section was noise; fundamentals are the missing input the GKX literature actually uses. |
| Clean CPI/NFP calendar | event timing | **Low.** Same event-window/risk-timing channel that FOMC showed is subsumed by vol-targeting. |

**Conclusion: a paid tier is necessary but not sufficient.** It removes the data
wall, but the program's two hard lessons still apply — (a) the **cost wall**
(signals must clear taker cost) and (b) the **subsumption wall** (single-index
risk-timing is captured by vol-targeting). The paid investment is only justified
for channels that plausibly beat *both* walls. By that filter, the ranked targets
are: **futures-curve carry** (cross-sectional, diversifiable, not a timing overlay)
> **per-stock options skew/VRP** > **PIT fundamentals** ≫ equity microstructure ≈
CPI/NFP (both expected to repeat prior failures).

## 5. The two walls that persist even with data

1. **Cost wall.** Every microstructure channel had a real signal that died on
   realistic taker cost. Any new channel must show a *net* edge after 2× cost +
   delay, not a gross one.
2. **Subsumption wall.** Any single-index risk-timing overlay (VRP, HMM, FOMC,
   plausibly cross-asset macro) must beat vol-targeted buy-and-hold, or it is
   redundant. The escape is a **cross-sectional / diversifiable** premium (carry,
   skew, fundamentals), not another index-timing gate.

## 6. Options for the operator

- **A — Targeted paid-data investment.** Buy one tier aimed at the single
  highest-EV channel (recommend: **futures-curve carry**, then options skew/VRP),
  with a *pre-committed kill criterion* (e.g., "if cross-market carry net-of-cost
  Sharpe < 0.7 OOS and is subsumed by trend, close"). Highest expected payoff;
  costs money; bounded downside via the kill criterion.
- **B — Exhaust the free/on-disk channels first.** Run cross-asset macro (free)
  and the on-disk options-IV / sentiment audits before spending. Cheapest; but the
  prior says cross-asset macro likely subsumes and the on-disk audits are
  timestamp-hostile. Information value mainly in *confirming* the data wall.
- **C — Reframe the program goal.** Accept that *taker-tradable* alpha is not
  reachable with current free data, and redirect to the durable asset: harden and
  document the validation infrastructure as the product (it is the thing that has
  consistently worked), and gate any future alpha work behind a data purchase.

## 7. Recommendation

If the goal remains taker-tradable alpha, **Option A targeted at futures-curve
carry** is the highest-expected-value next step, because carry is the one remaining
candidate that is both structurally non-OHLCV *and* a cross-sectional/diversifiable
premium (not a single-index timing overlay subsumption is known to kill). It is the
cleanest test of whether *data*, not methodology, was the constraint. Pair it with
a written kill criterion so the spend is bounded.

If no data spend is acceptable, **Option B's cross-asset macro** is the only free
structurally-new test left — run it data-audit-first, with the explicit
expectation that it must beat vol-targeted buy-and-hold to not be subsumed.

**Option C** is the honest fallback and is not a failure: the validation stack is a
real, reusable asset, and "we proved free-data retail alpha is not taker-tradable
here, and scoped exactly what data would change that" is a legitimate result.

## 8. Non-actions

- Do not open another free-data single-index risk-timing channel expecting a
  different outcome (subsumption is established).
- Do not buy data without a pre-committed channel + kill criterion.
- Do not reopen closed channels without a materially new information source.
- CPI/NFP, per-stock options-IV, and FinBERT remain gated on their respective
  data/timestamp prerequisites; none is opened by this review.
