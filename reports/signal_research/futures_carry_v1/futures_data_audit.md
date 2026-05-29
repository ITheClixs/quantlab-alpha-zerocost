# Futures Carry / Term-Structure v1 — Data Audit

**Built:** 2026-05-29T22:52:06.087922+00:00  **Intake:** `docs/research/intake/2026-05-30-futures-carry-term-structure-v1.md`
**Binding question:** Can we compute futures carry / roll yield from clean, timestamp-safe **front AND
deferred** contract data, without relying on a single back-adjusted continuous series?
**Live re-verification:** `{'massive_futures_get': '403', 'yfinance_proxy_VXZ_rows': 5}`

## 1-3. Data availability + curve classification (liquid markets first)

| market | class | label | front | next | expiry | note |
|---|---|---|---|---|---|---|
| ES (e-mini S&P futures) | equity_index | **continuous_only** | yes (yfinance ES=F) | no | no | continuous front only; usable for trend baselines, NOT carry |
| NQ (e-mini Nasdaq futures) | equity_index | **continuous_only** | yes (yfinance NQ=F) | no | no | continuous front only |
| CL (WTI crude futures) | commodity | **continuous_only** | yes (yfinance CL=F) | no | no | continuous front only; dated NYMEX contracts not served by yfinance |
| GC (gold futures) | commodity | **continuous_only** | yes (yfinance GC=F) | no | no | continuous front only |
| SI/NG/ZC/ZS/ZW (commodity futures) | commodity | **continuous_only** | partial (yfinance =F) | no | no | continuous front at best; no curve |
| Rates futures (ZN/ZB/ZF) | rates | **reject** | weak/none | no | no | no clean free source for the curve in this environment |
| FX futures (6E/6J/6B) | fx | **reject** | weak/none | no | no | no clean free source for the curve |
| Massive us_futures_* flat files | all | **reject** | catalog-only | catalog-only | catalog-only | session/minute/trades/quotes per-contract data EXISTS in catalogue (2017-2024) but GetObject=403 (paid entitlement); not downloadable |
| VXX/VXZ (VIX term-structure ETNs) | vol_proxy | **proxy_not_native_futures** | VXX (short-term VIX futures) | VXZ (mid-term) | embedded (not exposed) | ETN price embeds VIX-futures roll yield; clean free history; term-structure SLOPE proxy, NOT native front/next prices |
| USO/USL (WTI oil curve ETFs) | commodity_proxy | **proxy_not_native_futures** | USO (front WTI) | USL (12-month ladder) | embedded (not exposed) | USO/USL relationship reflects WTI curve shape; clean free history; proxy, NOT native curve |
| UNG (natgas ETF) | commodity_proxy | **proxy_not_native_futures** | UNG (front natgas) | none | embedded | single front proxy; contango bleed only |

**Curve requirement:** carry needs front + next contract prices + expiries. A single back-adjusted
continuous series is NOT enough (usable for trend baselines only).

## 4. Roll & back-adjustment audit
- No native front/next contract data is downloadable here, so a leakage-safe roll cannot be constructed.
- yfinance `=F` series are continuous/stitched front-month — exactly the back-adjusted single series the
  intake forbids as a carry signal. Their roll/back-adjustment convention is undocumented (cannot verify
  whether future roll info leaks).
- Massive per-contract flat files would carry raw prices + a contract ticker encoding expiry (curve_clean
  if downloadable), but `GetObject` = 403 (paid). Not usable.
- Conclusion: the §4 leakage checks cannot be satisfied for any native market in this environment.

## 5. Carry / roll-yield formula (pre-registered)
- **native_intended**: carry_t = ln(F_front_t / F_next_t) * (365 / (expiry_next - expiry_front in days))
- **sign_convention**: positive => backwardation (front > next) => positive roll yield to a long holder
- **annualization**: 365 / (calendar days between front and next expiry)
- **contracts_used**: front and first deferred (next) contract; raw (unadjusted) contract prices
- **missing_next_handling**: no carry signal on dates lacking a clean next-contract price
- **near_expiry_handling**: enforce min_days_to_expiry on the front (e.g. 5 trading days); roll before front expiry; never compute carry from an expiring front inside the threshold
- **leakage_rule**: signal at date t uses only contracts and expiries known at date t; carry computed from RAW prices BEFORE any back-adjustment
- **proxy_intended**: term-structure SLOPE proxy = ln(P_mid / P_short) e.g. ln(VXZ/VXX) or ln(USL/USO); labeled proxy_not_native_futures; NOT a native roll yield

## 6. Cost model
- **native_futures**: NOT applicable in v1 (no native data). Reference: ES ~0.25-tick spread + ~$2-4/contract commission; CL ~1-tick; roll cost ~1 spread/roll.
- **etf_proxies**: VXX/VXZ/USO ~1-2 bps spread, ~0 commission (liquid); USL/UNG slightly wider. 2x cost stress + 1-bar delay stress feasible.
- **verdict**: native cost model cannot be exercised (no native data); proxy cost model is feasible -> proxy markets are research-only at best.

## 7. ETF proxy fallback
- Proxies available + downloadable (clean free history): VXX/VXZ (VIX term-structure ETNs), USO/USL (WTI oil curve ETFs), UNG (natgas ETF).
- These EMBED roll yield (VIX-futures roll for VXX/VXZ; WTI curve for USO/USL) but do NOT expose
  front/next contract prices + expiries. A research-only term-structure-SLOPE test is possible
  (labeled `proxy_not_native_futures`); it must NOT claim native futures carry.

## 8. Data-quality labels (summary)
- `curve_clean`: NONE
- `continuous_only`: 5 markets (yfinance =F) — trend baselines only, NOT carry
- `proxy_not_native_futures`: 3 (VIX term ETNs, oil curve ETFs)
- `reject`: rates/FX curve, Massive flat files (403)

## 9. Audit decision

### VERDICT: **PARTIAL_PASS**

- **No `curve_clean` native futures market exists** in this environment: Massive per-contract data is
  403-paywalled, and yfinance gives only continuous front-month (no next contract, no expiries).
- **Proxy-only research test IS possible** (VIX term structure via VXX/VXZ; oil curve via USO/USL),
  which is exactly the `PARTIAL_PASS` condition: only proxy/limited markets exist, research-only,
  **no promotion language**.

### Decision-rule consequence
- Per the intake §10 and the operator decision rule: **PARTIAL_PASS → ASK before implementing a
  proxy-only research run.** No strategy code is written. Native futures-carry v1 (promotion-grade) is
  rejected on data grounds; only a clearly-labeled `proxy_not_native_futures` research probe remains
  on the table, pending operator approval.
- If the operator declines the proxy run, futures carry closes on data grounds and we return to the
  program-review decision tree (CPI/NFP only if a timestamp-clean calendar source is supplied).

## Constraints honored
- No strategy backtest run. No single back-adjusted series used as a carry signal. No post-hoc roll
  convention selection. No universe expansion. No options-IV / FinBERT. No CPI/NFP.
