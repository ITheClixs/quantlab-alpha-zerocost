# Paid-Data Acquisition Recommendation (May 2026)

**Date:** 2026-05-30
**Status:** Decision document — triggered by the 10-Q audit `REJECT_ON_DATA`.
**Author:** QuantLab research
**Builds on:** `reports/2026-05-30-PROGRAM-REVIEW-DATA-CONSTRAINT.md`
**Program `/goal`:** find taker-tradable alpha for QuantLab.

## 0. Why this document exists

The 10-Q data audit (`reports/signal_research/edgar_10q_v1/edgar_10q_data_audit.md`)
rejected on data grounds: 10-Q **text** is free from SEC EDGAR, but the **labels**
(survivorship-safe returns) and **point-in-time constituent membership** are not
available for free — the same wall that closed the options-IV cross-section. Per
the operator's standing instruction, when a free-data SEC source fails we **pause
the free-data search and produce a paid-data acquisition recommendation** rather
than open another free-data model hunt.

## 1. The binding constraint, precisely

The program has now hit four walls (data-constraint review §1). Three are
data-acquisition walls that a purchase could remove; one (vol-targeting
subsumption) is methodological and a purchase cannot fix:

| Wall | Removable by data purchase? |
|---|---|
| Survivorship-safe **return/fundamentals panel** (labels + PIT constituents) | **Yes** — this is the single most reused missing piece (blocks 10-Q labeling, options-IV cross-section, any equity cross-section) |
| Futures-curve (front/next/expiry) | Yes |
| Equity tick/quote + OPRA options | Yes (the Massive 403 paywall) |
| Vol-targeting subsumption of single-index risk timing | **No** — methodology, not data |

**The highest-reuse missing piece is a survivorship-safe daily-price + delisting +
CIK↔ticker-mapping panel.** It alone re-opens *two* already-built branches (10-Q
filing-drift labeling, and the options-IV cross-section) and is a prerequisite for
any future equity cross-sectional work.

## 2. Candidate acquisitions (indicative — verify current pricing/terms)

| Option | Unlocks | Survivorship | Indicative cost | Notes |
|---|---|---|---|---|
| **A. Sharadar SEP + TICKERS + ACTIONS + SF1** (Nasdaq Data Link) | survivorship-safe daily prices, delisting, CIK/ticker map, PIT fundamentals | **bias-free** (incl. delisted) | low (≈ $X00s/yr retail) | Directly labels 10-Q text + re-enables options-IV cross-section + enables fundamentals cross-section. Highest reuse. |
| **B. Upgrade Massive/Polygon to a downloadable tier** | lifts the 403 on equity daily/tick aggs + options OPRA + futures per-contract | daily aggs incl delisted (survivorship-safe) | low–med (≈ $X00s/yr) | Reuses the key already in `.env`; one upgrade lifts several walls at once (review §4). |
| **C. Futures-curve data** (front/next/expiry; a dedicated futures vendor) | carry / term-structure | n/a | med | Review ranked carry the **highest structural-EV** channel (cross-sectional/diversifiable, not a timing overlay) — but a separate build. |
| **D. CRSP/Compustat (WRDS)** | gold-standard PIT prices + fundamentals | bias-free + PIT | high / institutional | Best quality; cost and licensing are the barrier. |

## 3. Recommendation

**Primary: acquire a survivorship-safe price + delisting + mapping panel
(Option A or B), as a single targeted purchase, with a pre-committed kill
criterion.** Rationale:

- It is the **cheapest, highest-reuse** unlock: it labels the already-built 10-Q
  text pipeline (addressing the frequency wall that closed 10-K v1) **and**
  re-enables the options-IV cross-section, with no new modeling risk.
- Option **B** (upgrading the existing Massive/Polygon entitlement) is the most
  operationally efficient first move — one upgrade lifts the equity-download 403
  *and* moves toward options/futures, reusing the key already configured.
- Option **A** (Sharadar) is the cleanest if a separate, explicitly
  survivorship-bias-free vendor is preferred and to also get PIT fundamentals.

**Secondary (highest structural EV, separate track): Option C, futures-curve
carry** — per the data-constraint review the one remaining channel that is both
non-OHLCV *and* not a single-index risk-timing overlay (so not auto-subsumed by
vol-targeting). Pursue after, or in parallel with, the price-panel acquisition.

**Pre-committed kill criterion (mandatory for any spend):** before purchasing,
write the gate. Example for the price-panel + 10-Q relabel: *"If 10-Q classical
text features, with a survivorship-safe return panel and ~40+ quarterly
cross-sections, do not beat the size/value/momentum factor baselines on rank IC
AND decile spread AND net long-short PnL on a chronological holdout, close the SEC
filing branch and do not renew the data subscription."* This bounds the downside
of the spend.

## 4. The honest caveat (necessary but not sufficient)

A purchase removes the *data* wall; it does **not** remove the two lessons the
program has proven repeatedly:

1. **Cost wall** — any signal must clear realistic transaction cost net, not gross.
2. **Subsumption wall** — any single-index risk-timing overlay (and possibly a
   cross-sectional text signal) must beat the standard factor baselines or it is
   redundant. The price panel finally makes the *full* factor-subsumption test
   possible (it was only partial in 10-K v1 — size-only).

So the recommended spend is justified **only** with the kill criterion attached.
"We bought the data, ran the pre-registered gate, and it failed cleanly" is a
legitimate, bounded outcome — consistent with the program's discipline.

## 5. What NOT to do

- Do not assemble a survivor-only 10-Q backtest from free data (invalid — repeats
  the options-IV survivorship mistake).
- Do not scrape EDGAR 10-Q text without first securing a survivorship-safe return
  panel to label it.
- Do not add embeddings/LLMs to rescue 10-K v1 (separate intake; not a data fix).
- Do not buy data without a written, pre-committed kill criterion.
- Do not open another free-data model search expecting a different outcome; the
  free-data channels are exhausted (OHLCV, microstructure, event-macro, options-IV
  cross-section, futures-curve, 10-K, 10-Q).

## 6. Decision requested

Choose one: **(A)** Sharadar-class survivorship-safe price/fundamentals panel;
**(B)** upgrade the existing Massive/Polygon plan to a downloadable tier; **(C)**
futures-curve vendor for the carry track; or **(defer)** no spend now — in which
case the program's deliverable is the validation infrastructure + the documented
map of exactly which paid data would change the answer.
