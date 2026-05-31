# Zero-Cost Macro-Feature Timestamp Audit (P0)

**Built:** 2026-05-30T10:43:16.571480+00:00  **Rule:** every feature observed at close t, used only at **t+1**.
**Verdict:** **PASS_TIMESTAMP_SAFE**

## Allowed macro features (timestamp-safe)

| feature | source | ref | classification | available | coverage | rationale |
|---|---|---|---|:---:|---|---|
| vix | yfinance | ^VIX | **market_price_clean** | True | 2010-01-04..2026-05-29 | CBOE VIX daily index close; market-priced, not revised |
| vix3m | yfinance | ^VIX3M | **market_price_clean** | True | 2010-01-04..2026-05-29 | CBOE 3-month VIX daily close; with VIX gives the VIX term structure |
| bonds_tlt | yfinance | TLT | **market_price_clean** | True | 2010-01-04..2026-05-29 | 20y+ Treasury ETF daily close; market-priced rates proxy |
| gold_gld | yfinance | GLD | **market_price_clean** | True | 2010-01-04..2026-05-29 | gold ETF daily close; cross-asset risk proxy |
| credit_hyg | yfinance | HYG | **market_price_clean** | True | 2010-01-04..2026-05-29 | HY credit ETF daily close; credit-stress proxy |
| usd_uup | yfinance | UUP | **market_price_clean** | True | 2010-01-04..2026-05-29 | USD bull ETF daily close; dollar trend proxy |
| ust10y | fred | DGS10 | **daily_next_day_only** | True | 2010-01-04..2026-05-28 | 10y Treasury CMT; published EOD t by Treasury, not revised; use at t+1 |
| ust2y | fred | DGS2 | **daily_next_day_only** | True | 2010-01-04..2026-05-28 | 2y Treasury CMT; published EOD t, not revised; use at t+1 |

## Forbidden (revised aggregates — NOT fetched, NOT usable without PIT vintages)
- `GDP`: revised aggregate; needs ALFRED vintage
- `CPIAUCSL`: revised; release/revision lag; needs PIT vintage
- `PAYEMS`: nonfarm payrolls; heavily revised; needs PIT vintage
- `UNRATE`: unemployment; revised; needs PIT vintage
- `PCE`: revised aggregate; needs PIT vintage

## Classification legend
- `market_price_clean`: daily market close, not revised → safe at t+1.
- `daily_next_day_only`: published EOD t (e.g. Treasury CMT), not revised → safe at t+1.
- `revision_risk` / `reject`: revised aggregates without PIT vintages → forbidden.

## Decision
- **PASS** — all instruments covered and all available macro features are timestamp-safe; no forbidden series used. Proceed to P1 (zero_cost_riskalloc_v1).
