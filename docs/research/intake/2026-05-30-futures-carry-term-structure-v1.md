# Intake — Futures Carry / Term-Structure v1

**Date:** 2026-05-30
**Status:** PRE-REGISTRATION (intake submitted; **data audit first, no strategy code yet**)
**Strategy name:** `futures_carry_term_structure_v1`
**Proposer:** QuantLab research
**Promotion intent:** `research_only` for v1 (see §9)
**Program `/goal`:** find **taker-tradable** alpha for QuantLab.

## 0. Context — why this channel now

Three single-index risk-timing channels are closed: OHLCV cross-sectional (noise
floor), free-data microstructure (predictive-but-untradable), and FOMC
event-timing (real but subsumed by vol-targeting —
`docs/research/2026-05-NEGATIVE-RESULT-EVENT-MACRO-FOMC.md`). The recurring
meta-finding is that **vol-targeting/regime subsumes most index risk-timing
edges**. CPI/NFP would be the *same* event-window/risk-timing channel and are
blocked by a clean-date data bottleneck. Per-stock options-chain IV and FinBERT
are heavier data/timestamp projects and are **explicitly not opened here**.

**Futures carry / term-structure is the next cleanest structurally-different
information channel:** the signal lives in the *shape of the futures curve*
(front vs deferred contract, roll yield), not in a single price series — so it is
genuinely non-OHLCV, with strong economic rationale (hedging pressure, storage,
funding, risk premia).

## 1. Strategy name and one-line description

`futures_carry_term_structure_v1` — test whether futures-curve carry / roll yield
(and its interaction with trend) carries return-predictive information on liquid
markets that is not reducible to OHLCV-only price timing.

## 2. Hypothesis statement

In futures, expected return is linked to the **term structure**: a market in
backwardation (front > deferred) pays a positive roll yield to a long holder,
while contango pays it to a short. Carry (the annualized roll yield) proxies the
risk premium that hedgers pay speculators — driven by hedging pressure, storage
cost, and funding — and is documented to predict returns across asset classes
(Gorton & Rouwenhorst 2006 on commodities; Koijen, Moskowitz, Pedersen & Vrugt
2018, "Carry"; with a trend interaction per Moskowitz, Ooi & Pedersen 2012,
time-series momentum). The hypothesis: **a carry / term-structure signal earns a
premium that survives realistic roll and transaction costs and is not subsumed by
trend-following or vol-targeting** — the failure mode that closed the prior
channels.

## 3. Information source declaration (machine-readable)

- Driving channel: **futures term-structure / carry** (front vs deferred contract
  prices + expiry → roll yield). Genuinely non-OHLCV: it requires the curve, not a
  single continuous price.
- Asset-class channels (per instrument): `macro_rates`, `macro_commodity`,
  `macro_fx` (and equity-index futures under `cross_asset`).

**Note:** the `InformationSource` enum has no `futures_term_structure` value yet;
v1 may warrant adding one. The no-OHLCV-only-promotion rule is satisfied because
carry/roll-yield is computed from the **front/next spread + expiries**, which is
not derivable from a single OHLCV series.

## 4. DATA AUDIT FIRST (binding constraint — hard gate before any strategy)

This intake authorizes a **data-feasibility audit only**. Strategy code is gated
on the audit passing, exactly as for the microstructure and event-macro channels.
The audit (`futures_data_audit.md` + `futures_data_manifest.json`) must establish:

- **Available continuous futures data** (which markets, what history).
- **Roll convention** (calendar / volume / open-interest roll; roll dates).
- **Front / next contract prices** (required for carry — a single back-adjusted
  series is insufficient).
- **Contract expiry dates** (ex-ante, timestamp-clean).
- **Volume / open interest** if available (for roll timing + liquidity screen).
- **Roll-yield calculation** (front vs deferred, annualized; documented formula).
- **Survivorship and back-adjustment treatment** (back-adjustment must NOT leak
  future roll information into past prices — a known leakage trap).
- **Timestamp and source manifest** (provenance + hashes, reproducible).
- **Cost and slippage assumptions** (per-contract commission, bid/ask, roll cost).

### 4.1 Scope (v1)

- **Liquid markets only.** Possible instruments: equity index futures, rates
  futures, commodities, FX futures — **or public ETF proxies if clean futures
  curve data is unavailable**.
- **No cross-sectional stock selection. No hidden constituent universe.**
- ETF-proxy fallback (if no clean curve data): futures-based / term-structure
  ETFs where the proxy's roll mechanics are documented and the carry is
  observable; clearly labeled as a proxy, not native futures.

## 5. Expected gross Sharpe and capacity

- **Ex-ante expected gross Sharpe:** 0.4–1.0 for a diversified carry sleeve
  (consistent with the published carry literature net of costs). Honest prior:
  single-market carry is noisier and more crisis-concentrated; the diversified
  cross-market sleeve is where the premium has historically been cleaner.
- **Capacity:** high for liquid futures; not binding at v1.

## 6. Cost assumptions

- Per-instrument commission + bid/ask in bps one-way, **plus an explicit roll
  cost** each roll, plus the pipeline's 2× cost stress and 1-bar delay stress.
  Roll cost is the carry-specific failure mode (the edge can be eaten by rolling).

## 7. Universe and history

- Liquid markets / proxies per §4.1; start date driven by on-disk coverage
  (audit reports it). Chronological train / validation / holdout; dev-only-guard;
  holdout ≥ 18 months after validation_end.
- Crisis-removal robustness (e.g., 2008, 2020, 2022) per the audit's coverage.

## 8. Pre-registered failure modes

1. **No carry edge** — curve shape does not predict returns net of costs.
2. **Roll construction artifact** — apparent edge is an artifact of the roll/
   stitching method.
3. **Back-adjustment leakage** — back-adjusted series leaks future roll info.
4. **Cost failure** — gross edge real, dies on commission + roll + slippage.
5. **Subsumed by trend / vol-targeting** — carry adds nothing over trend-following
   or vol-targeted trend (the prior-channel failure mode).
6. **Insufficient history** — too few independent roll cycles for stable inference.
7. **Data-quality fail** — curve/expiry/OI data not clean or not sourceable.
8. **Concentration** — edge concentrated in one commodity or one crisis window.
9. **Placebo indistinguishable** — random/inverted sanity baselines match the
   carry signal.

## 9. Validation

- Same `ValidationPipeline`; PBO / DSR; stationary block-bootstrap CI; cost
  stress (2×); delay stress (1-bar); crisis exclusion; **random + inverted sanity
  baselines**; concentration diagnostics.
- Baselines (first-class, in the PBO pool): **buy-and-hold, trend-following,
  vol-targeted trend, and OHLCV-only** baselines — the carry signal must beat
  these or it is subsumed.
- Promotion intent `research_only` for v1; the 1.5 Sharpe gate and no-OHLCV-only
  rule apply; no self-promotion.

## 10. Decision rule

- **If clean futures curve data is NOT available → reject futures-carry v1 on
  data grounds** (document in `futures_data_audit.md`), and only then consider
  CPI/NFP — and only if a timestamp-clean calendar source is supplied.
- **Do NOT open per-stock options-chain IV or FinBERT** — heavier data/timestamp
  projects, deferred.
- If the audit passes: implement the carry/term-structure signal + baselines and
  run the full validation; classify into a §8 failure mode if it does not clear.

## 11. Sign-off

Proposer: QuantLab research. Intake date: 2026-05-30. Data audit precedes any
strategy code. Full validation gate, no post-hoc tuning after holdout. This is a
data-feasibility-gated pre-registration, not a parameter search.

## What happens after this intake

1. This document is committed to `docs/research/intake/`.
2. Run the **data audit** (§4) → `futures_data_manifest.json` +
   `futures_data_audit.md` with a PASS / REJECT-ON-DATA verdict. No strategy code
   until the audit passes.
3. If PASS: implement carry/term-structure signal + baselines; run validation;
   emit registry + validation + placebo/sanity + failure-classification reports.
4. Commit the status classification alongside the reports.
