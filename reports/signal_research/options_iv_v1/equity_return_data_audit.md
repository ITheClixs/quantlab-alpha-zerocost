# Equity-Return Data Audit (§4.2 gate) — Summary & Decision

**Built:** 2026-05-30T08:33:26.840328+00:00  **Source:** HexQuant__Stocks-Daily-Price
**IV window:** 2019-10-14 → 2023-07-28  **IV symbols:** 3893
**Binding question:** survivorship-aware, CA-adjusted next-day return panel for the IV universe?

## Findings
- **Coverage:** 2224/3893 (57.1%) mapped; 1669 missing (0 due to ticker format → none). See coverage report.
- **Survivorship:** `survivorship_biased` — every probed delisted/merged/failed 2019-2023 name (TWTR, FB,
  RTN, CELG, XLNX, MXIM, NLSN, CERN, ZNGA, SIVB, FRC, AABA) is ABSENT from HexQuant in-window. See
  survivorship report.
- **Corporate actions:** `adj_close` is dividend+split adjusted (total-return proxy on covered names).
- **ETFs:** HexQuant has no SPY/QQQ/DIA/IWM (stocks-only). Secondary SPY/QQQ track uses the clean bars.
- **Liquidity/quality:** clean on the survivor set; not the binding constraint.
- **Other on-disk sources:** benstaf (narrow curated RL universe), jwigginton (S&P500 current-
  constituent → also survivorship-biased), mospira (2025 only) — none survivorship-safe for the broad
  IV universe. yfinance is disallowed as primary (survivorship-biased for delisted names).

## Labels: `survivorship_biased_research_only, price_return_only_unless_adjclose_total, mapping_incomplete, liquidity_limited`

## VERDICT: **REJECT_CROSS_SECTIONAL_ON_DATA — run SPY/QQQ secondary diagnostic only**

Per the §4.2 / operator decision rule: the only available return source is **clearly
survivorship-biased** for the 2019-2023 cross-section, so the **cross-sectional options-IV track is
REJECTED on data grounds** (a cross-sectional backtest on survivors only would be invalid — it would
exclude the bankruptcies and merger targets that dominate the tails). Mapping is not the issue (0
format mismatches); the defect is genuine missing delisted names.

### Consequence
- **Run only the SPY/QQQ secondary diagnostic track** (per the options-IV intake §4.2 / §11.3): index
  IV features (VRP proxy, cross-sectional IV dispersion, vol-OI imbalance) from gauss314 → next-day
  SPY/QQQ timing, using the already-clean SPY/QQQ bars. research_only; must beat vol-targeted BAH or it
  is subsumed.
- If the SPY/QQQ diagnostic is also weak, close options-IV on data grounds and move to the next branch.

## Constraints honored
- No strategy backtest. No silent symbol dropping (full mapping table written). No future-availability
  filtering. No yfinance promotion claims. No direct options. research_only throughout.
