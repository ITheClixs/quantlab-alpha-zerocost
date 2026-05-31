# Close-Out — Zero-Cost Alpha Search (May 2026)

**Date:** 2026-05-30
**Status:** **CLOSED — zero-cost alpha discovery exhausted.** Program enters
validation-infrastructure mode. No new strategy branch without a genuinely new
information source + a clean path to tradable implementation (operator-authorized).
**Authorizations:** **No paper trading. No live trading.** Neither is authorized by
this note or by any result in the program.

## 0. Decision

The zero-cost strategy search is closed as exhausted under current (free-data-only)
constraints. **No deployable alpha was found.** The binding bottleneck is **data
acquisition / the information set — not model architecture or validation method.**
The validation infrastructure is the durable asset and remains in place.

## 1. Branches tested (full ledger)

| # | Branch | Verdict | Why it failed / was rejected | Reference |
|---|---|---|---|---|
| 1 | OHLCV cross-sectional (6 iterations, top-50/100/200 SP500, crypto top-30) | closed | noise floor; PSR/DSR kill the +0.15–0.6 holdout flicker | `2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` |
| 2 | VRP (index) | closed | real but **subsumed by HMM regime** (orthogonal residual Sharpe ≈ 0) | program review §2 |
| 3 | HMM single-index | closed | strong static fit but **delay-sensitive + refit-unstable** | program review §3 |
| 4 | Microstructure L2 depth (Binance futures) | closed | **predictive but untradable** (markout < spread+fee) | `2026-05-NEGATIVE-RESULT-MICROSTRUCTURE.md` |
| 5 | Microstructure L1 quotes (trade-flow Mode B) | closed | net < 0 at every cost threshold | same |
| 6 | Microstructure tick trade-flow (Mode A) | closed | IC 0.45 yet untradable (bid-ask-bounce, markout < cost) | same |
| 7 | Event-macro FOMC | closed | real + placebo-clean but **subsumed by vol-targeting**; below gate | `2026-05-NEGATIVE-RESULT-EVENT-MACRO-FOMC.md` |
| 8 | Futures carry / term-structure | rejected on data | **no native curve** (Massive 403; yfinance continuous-only) | `intake/2026-05-30-futures-carry-term-structure-v1.md` |
| 9 | Options-IV cross-sectional | rejected on data | only return source (HexQuant) **survivorship-biased** | `reports/signal_research/options_iv_v1/equity_return_data_audit.md` |
| 10 | EDGAR 10-K text features | closed | clean (PIT + survivorship) but **annual = too low-frequency**; placebo-indistinguishable | `reports/signal_research/edgar_10k_v1/` |
| 11 | EDGAR 10-Q text features | rejected on data | text free, but **no survivorship-safe labels / PIT constituents** | `reports/signal_research/edgar_10q_v1/edgar_10q_data_audit.md` |
| 12 | Zero-cost mixed allocator (SPY/QQQ/BTC/ETH) v1 | DO_NOT_ADVANCE | cleared literal gate but **crypto-regime-carried + crisis-dependent** (crypto-out loses to vol-targeted BAH) | `reports/signal_research/zero_cost_v1/` |
| 13 | Zero-cost crypto-only allocator v2 (BTC/ETH) | DO_NOT_ADVANCE | beats benchmark marginally but **ETH-concentrated (70%) + bootstrap-lower Sharpe < 0** (statistically fragile) | `reports/signal_research/crypto_only_v2/` |

## 2. The four blocker categories

Every failure reduces to one (or more) of four walls. Only the last is a
methodology issue; the first three are data-access issues.

1. **Cost wall** — signal real but markout < realistic transaction cost
   (all microstructure channels).
2. **Subsumption wall** — single-index risk-timing is captured by vol-targeting /
   regime exposure (VRP, HMM, FOMC, the zero-cost allocators' equity sleeves).
3. **Data-access / survivorship wall** — the structurally-new channels need
   entitled or survivorship-safe data we do not have for free (futures curve,
   options-IV cross-section, 10-Q labels, PIT fundamentals).
4. **Frequency / sample wall** — clean free data exists but at too low a frequency
   for robust inference (EDGAR 10-K annual: 3 holdout cross-sections).

## 3. Explicit findings

- **No deployable alpha was found under zero-cost constraints.**
- **0 production candidates, 0 paper candidates, 0 live candidates.**
- The blocker map (operator-stated, confirmed by the audits):
  - free OHLCV signals are weak or subsumed;
  - free microstructure signals are predictive but below executable cost;
  - event/regime signals are mostly crisis-insurance or vol-targeting restatements;
  - cross-sectional equity ideas are blocked by survivorship-safe return data;
  - futures carry is blocked by native curve data;
  - SEC 10-K is clean but too low-frequency;
  - 10-Q is blocked by lack of survivorship-safe labels;
  - options-IV cross-section is blocked by survivorship-biased returns;
  - zero-cost crypto allocation is ETH-concentrated and statistically fragile.
- **The next bottleneck is data acquisition, not model architecture.** Methodology
  and validation are not the constraint; the information set is.

## 4. Durable asset

The validation infrastructure is the program's deliverable: `ValidationPipeline`,
three-tier PBO / DSR, stationary bootstrap CI, CPCV / walk-forward, dev-only guard,
cost decomposition + delay stress, placebo (shuffled/random/inverted) baselines,
concentration + crisis-removal diagnostics, the single-index risk-timing exception
policy, the failure-class taxonomy, and the **data-audit-first gate** that rejected
every fragile result before it could mislead (it caught in-sample leakage, the
`mid_direction_up` future-label leak, a falsy-zero PBO bug, the survivorship gaps,
the crypto-out reversal, and the ETH concentration). The **Sharadar ingestion+audit
scaffold** (`data/sharadar/`, commit `88cf14d`) is dormant and ready to run the
moment a survivorship-safe equity dataset is supplied.

## 5. Reopen conditions

Alpha discovery reopens ONLY if one of these holds (operator-authorized):

1. **Survivorship-safe equity return/actions/mapping data** (Sharadar SEP/TICKERS/
   ACTIONS/SF1 or equivalent) → reopens 10-Q/10-K relabel, options-IV cross-section,
   PIT factor baselines. (Scaffold ready; kill criterion in
   `2026-05-DATA-PURCHASE-FEASIBILITY-SHARADAR.md`.)
2. **True options-chain data** (strikes, expiries, bid/ask, greeks, timestamps,
   underlying mapping) → reopens options risk-premia / surface strategies.
3. **Native futures curve data** (front/next contracts, expiries, roll convention,
   volume/OI) → reopens carry / term-structure.
4. **Paid/clean microstructure data** sufficient for maker/queue execution modeling
   → could flip the microstructure "predictive-but-untradable" verdict.
5. A **genuinely new free information source** with a clean, documented path to
   tradable implementation (not a re-slice of an exhausted channel; no post-hoc
   narrowing after seeing attribution).

A new free-data *model* branch on an already-mapped channel does **not** qualify.

## 6. Operating mode until reopen

- Repo stays in **validation-infrastructure mode**.
- **No paper trading. No live trading.** No promotion language.
- No new strategy branch is opened without satisfying §5 and explicit
  operator authorization.
- The next productive step, if deployable alpha remains the goal, is the
  **data-acquisition decision** (`2026-05-30-PAID-DATA-ACQUISITION-RECOMMENDATION.md`),
  not another model.
