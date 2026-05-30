# Data-Purchase Feasibility & Integration Checklist — Sharadar (Option A)

**Date:** 2026-05-30
**Status:** Feasibility checklist — **no purchase, no code, no strategy run** until this passes.
**Author:** QuantLab research
**Builds on:** `2026-05-30-PAID-DATA-ACQUISITION-RECOMMENDATION.md`,
`reports/signal_research/edgar_10q_v1/edgar_10q_data_audit.md`,
`reports/signal_research/options_iv_v1/equity_return_data_audit.md` (commit `3f9a658`).
**Vendor:** Sharadar Equity Bundle via Nasdaq Data Link (NDL).
**Convention:** items I'm confident about are stated; uncertain items are marked
**[VERIFY]** — these must be confirmed against the vendor's published schema / a
free sample **before** any spend. All pricing is indicative — **[VERIFY pricing]**.

## 0. Why Sharadar (recap)

The binding program constraint is a **survivorship-safe equity daily-price +
corporate-action + ticker/CIK-mapping layer**. It is the single highest-reuse
missing piece: it supplies leak-safe **labels** for the already-built EDGAR text
pipeline and re-opens the options-IV cross-section (rejected only because HexQuant
was survivorship-biased). This is reusable infrastructure, not a one-off dataset.

## 1. Required tables

| Table | Purpose | Required? |
|---|---|---|
| **SEP** (Sharadar Equity Prices) | daily OHLCV + dividends + adjusted/unadjusted close for US **stocks**, incl. delisted | **required** (labels for the stock cross-section) |
| **TICKERS** | symbol metadata: `permaticker` (stable ID across ticker changes), ticker, name, exchange, `siccode`, sector, `isdelisted`, `firstpricedate`/`lastpricedate`, `cusips` | **required** (mapping + delisting flags + sector) |
| **ACTIONS** | splits, dividends, delistings, mergers, ticker changes (dated) | **required** (corporate actions + delisting events) |
| **SF1** (fundamentals) | income/balance/cashflow with `datekey` (SEC filing/availability date = PIT) and `dimension` (ARQ/MRQ/...) | optional v1 (enables PIT factor baselines + fundamentals features) |
| **CIK mapping** | EDGAR CIK ↔ Sharadar `permaticker`/ticker | **required for the EDGAR branches** — **[VERIFY]** whether a CIK field is in TICKERS/SF1 or must be bridged via CUSIP/ticker |
| (SFP — Sharadar Fund Prices) | ETF/fund prices (SPY/QQQ) | not required (SPY/QQQ bars already on disk; the options-IV cross-section is single stocks) |

## 2. Vendor-capability checklist (confirm before buying)

| Capability | Needed for | Confidence |
|---|---|---|
| Delisted / acquired / bankrupt names included | survivorship safety | high (Sharadar advertises survivorship-bias-free) — **[VERIFY]** TWTR/CELG/XLNX/CERN/ATVI/SIVB/FRC/AABA present |
| Historical ticker changes (stable `permaticker`) | mapping through renames | high — **[VERIFY]** permaticker semantics |
| Splits | adjusted returns | high (ACTIONS) |
| Dividends | total returns | high (SEP `dividends` + ACTIONS) |
| Adjusted **and** unadjusted prices | returns + leakage control | high (SEP `closeadj` + `close`/`closeunadj`) |
| Delisting returns or fields to compute them | tail-correct returns (bankruptcies/mergers) | **medium — [VERIFY]**: Sharadar gives price path to `lastpricedate` + delisting ACTION, but a CRSP-style explicit *delisting return* may need manual construction; confirm acquisition price / final-return handling |
| CIK / SEC mapping | join to EDGAR (CIK-keyed) | **[VERIFY]** — the linchpin for the 10-K/10-Q branches |
| Point-in-time fundamentals (`datekey`) | factor baselines / fundamentals features | high (SF1 `datekey`) — only if SF1 purchased |
| License permits local storage + research use | legal | **[VERIFY]** the specific NDL/Sharadar tier license |

## 3. Window coverage

| Branch | Required window | Sharadar coverage |
|---|---|---|
| EDGAR 10-K relabel | 2010–2022 | SEP typically covers 1998→present → **covers** **[VERIFY]** start date for the universe |
| EDGAR 10-Q (if scraped) | ~2010–2025 | covered **[VERIFY]** |
| Options-IV cross-section | 2019–2023 | covered |
| Future factor baselines | as needed | covered |

## 4. Branches this purchase reopens

1. **EDGAR 10-Q text features** — Sharadar supplies survivorship-safe labels for
   scraped 10-Q text (fixes the 10-K low-frequency wall with ~4x cross-sections).
2. **EDGAR 10-K rerun** — replace/validate the dataset's bundled returns with
   independent survivorship-safe labels; enables the **full** factor-subsumption
   test (10-K v1 could only test size).
3. **Options-IV cross-sectional v1** — the rejected branch; reopened with a
   survivorship-safe stock return panel (no more survivor-only bias).
4. **General PIT equity factor baselines** — size/value/momentum/low-vol from
   SEP + SF1, reusable across all future equity research.

## 5. Kill criterion (do NOT buy unless ALL hold)

1. Includes **delisted names** (verified via the 8-name probe in §2).
2. Provides **survivorship-safe daily returns** or enough fields (`close`,
   `closeadj`, `dividends`, ACTIONS) to compute total returns including the
   delisting tail.
3. Supports **ticker changes** (stable `permaticker`) and **corporate actions**.
4. **Maps to EDGAR CIKs** directly, or via CUSIP/ticker with **acceptable loss**
   (target: ≥ 90% of the EDGAR 10-K 727-company universe mappable; measure on a
   free sample before buying).
5. **Covers** 2010–2022 (10-K), the 10-Q window, and 2019–2023 (options-IV).
6. **License permits local research use** at the intended tier.

If any fail → **do not buy**; revert to the data-acquisition decision tree
(Option B Massive entitlement check, or Option C futures-curve, or defer).

## 6. Purchase-feasibility check (pre-spend, free/low-cost steps)

1. Read the Sharadar SEP/TICKERS/ACTIONS/SF1 published schemas; confirm fields in §2.
2. Pull a **free sample** (NDL sample endpoints / docs); run the **8-name delisted
   probe** and a **CIK-mapping loss estimate** against the EDGAR 727-company set.
3. Confirm license tier + cost **[VERIFY pricing]**; confirm local-storage rights.
4. Only if §5 all-pass: purchase, then write `manifests/sharadar/` + a Sharadar
   data audit (its own manifest + audit) **before** any strategy code.

## 7. First experiment after purchase (one branch only)

- **If 10-Q text scrapes cleanly** (EDGAR `edgar-crawler`): assemble + audit the
  10-Q panel (Sharadar labels), then 10-Q text-features v1 (same discipline as 10-K v1).
- **Otherwise:** options-IV cross-sectional v1 using the reopened survivorship-safe
  return panel (the IV features are already audited as next-day EOD features).

Exactly one branch at a time. The purchased data must pass **its own manifest +
audit** before any strategy run.

## 8. Explicitly NOT allowed

- Do **not** weaken any gate because the data was paid for.
- Do **not** open multiple branches in parallel.
- Do **not** treat paid data as alpha — it is infrastructure.
- Do **not** run strategy code until the purchased data passes its manifest + audit.
- Do **not** make promotion claims from the purchase alone.
- Do **not** auto-buy: §6 feasibility check + §5 kill criterion gate the spend.

## 9. Decision

Option A is the preferred acquisition path. **Next action is the §6 purchase-
feasibility check (free sample + schema review + CIK-mapping loss estimate), not a
purchase and not code.** Proceed to buy only if §5 holds on the sample.
