# Options-IV Data Audit — Summary & Decision

**Built:** 2026-05-30T08:17:17.435367+00:00  **Dataset:** `gauss314__options-IV-SP500/data_IV_USA.csv`
**Rows:** 3,161,661  **Range:** 2019-10-14 → 2023-07-28  **Symbols:** 3,893
**Binding question:** can options-chain IV support a leakage-safe, tradable research program?

## Findings (see per-section audits)
- **§1 Timestamp:** daily EOD only; safe **only** as next-day (t→t+1) features → `timestamp_uncertain`.
- **§2 Universe:** broad ~3,900 US optionable names incl. SPY/QQQ/DIA/IWM; delisted names retained (AABA drops 2019-11-25) → **not current-constituent survivorship-biased**.
- **§3 Chain structure:** aggregate only — **no strikes / expiries / bid-ask / per-contract prices** →
  `options_features_only`; direct option trading impossible.
- **§4 Data quality:** clean (0 dups, 0 bad ATM_IV, no key nulls); liquidity filter feasible/needed.
- **§5 Tradability:** options are **features only**; tradable instruments = SPY/QQQ/DIA/IWM or equities.
- **§6 Features:** ATM IV, skew proxy, IV rank, RV−IV (VRP) proxy, vol/OI imbalance, cross-sectional
  dispersion all feasible; **IV term structure / slope NOT feasible**.
- **§7 Leakage:** IV market-implied (not future-RV), `hv_*` trailing, no underlying-price column to leak,
  universe not current-only → no hard-fail; the binding caveat is daily observability (→ next-day only).

## Data-quality labels: `options_features_only, timestamp_uncertain`

## VERDICT: **RESEARCH_ONLY_FEATURES**

Per the operator decision rule:
- Chain structure does **not** pass (no strikes/expiries/bid-ask) → **no promotion-grade options-IV intake**.
- Timestamps are daily/EOD but the data **is** usable as end-of-day features for **next-day** trading →
  **`research_only` is permitted**.
- The dataset lacks bid/ask/expiry/strike → **do NOT run a promotion-style strategy**; **no direct option
  trading**.

### Recommended next step (operator decision)
Open a **`research_only` options-IV-features v1 intake**: EOD IV features (ATM IV, skew proxy, IV rank,
RV−IV / VRP proxy, put/call vol-OI imbalance, cross-sectional IV dispersion) → **next-day** timing/
selection on SPY/QQQ (and/or a liquid equity cross-section), validated under the standard gate with the
explicit research_only ceiling and the vol-targeting-subsumption baseline. **No direct option trades,
no promotion language.** If declined, close options-IV and move to the news/sentiment timestamp audit.

## Constraints honored
- No strategy code. No promotion-style claims. No direct option-trading assumptions. No term-structure
  features fabricated. Survivorship/timestamp caveats recorded.
