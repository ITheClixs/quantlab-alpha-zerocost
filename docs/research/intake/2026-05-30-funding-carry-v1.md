# Intake — Crypto Perp Funding-Rate Carry v1

**Date:** 2026-05-30
**Status:** OPEN — research_only. **No paper. No live.** No promotion language.
**Reopen basis:** close-out §5 condition #5 (genuinely new free information source
with a clean path to tradable implementation). Funding is a NEW channel — the
microstructure work was spot/trade-flow only and explicitly excluded funding.
**Data audit:** PASS (commit `f169ac0`, `reports/signal_research/funding_carry_v1/funding_carry_data_audit.md`).

## 1. Thesis

Perpetual futures have no expiry, so an 8-hourly **funding** payment tethers the perp
to spot. When perp trades above spot (leverage-long demand) longs pay shorts; the sign
flips when perps trade below. Empirically (2020–2026) funding is **persistently
positive** on BTC/ETH: longs overpay to be long. A **delta-neutral** book that is long
spot and short the perp collects this funding while carrying ~no price exposure. This
is a *carry* return source, structurally distinct from the price-timing / vol-timing
signals that the program has already exhausted.

## 2. Why this is allowed to reopen the search

It is the only candidate that escapes **both** walls that closed every prior branch:

- **Cost wall** (killed all microstructure): carry is *held* across 8h intervals, not a
  per-trade taker bet. Funding accrues for holding; turnover is low.
- **Subsumption wall** (killed VRP/HMM/FOMC/allocators): carry ≠ single-index
  vol-timing, so vol-targeting cannot restate it. Delta-neutral removes the market beta
  that vol-targeting captures.
- **Data-access/survivorship wall**: funding + spot + perp + basis are all free
  (Binance Vision), settlement-timestamped (leak-safe), and BTC/ETH perps were never
  delisted (survivorship-clean).

## 3. Honest priors (stated before any backtest, so they cannot be rationalized away)

1. **This is the most crowded trade in crypto.** The edge is widely known and has
   compressed since 2021. A clean v1 must show it survives *recent* regimes, not just
   the 2020–2021 leverage bull.
2. **Gross ≠ net.** The audit's +12.2% (BTC) / +14.5% (ETH) annualized is **gross
   funding only**. The v1 backtest must net: spot taker (~10bps one-way), perp taker
   (~5bps one-way), the spot–perp **basis** paid to establish/unwind the hedge, and
   margin/borrow cost on the short leg.
3. **Regime dependence is real.** Per-year gross funding: strong 2020 (+17/+27%) and
   2021 (+31/+38%); thin 2022 (+4/+1%) and 2026 (+0.4/−0.4%); modest 2023–2025.
4. **Tail risk on the short perp.** A violent rally pressures short-leg margin even when
   delta-neutral on price. Must be modeled (margin buffer, liquidation distance), not
   assumed away.

## 4. Scope

### Instruments
BTCUSDT, ETHUSDT perpetuals (USDT-margined) + matching spot.

### Data (all free, on disk or free-fetchable)
- **Funding:** `crypto_research/funding/load_funding` (Binance Vision monthly archives).
- **Perp OHLCV:** `data/raw/huggingface/123olp__binance-futures-ohlcv-2018-2026` (verify
  BTC/ETH coverage + schema in P1; fall back to Vision futures klines if gapped).
- **Spot:** BTC on disk (`vaquum__binance_btcusdt_1m_klines`); ETH spot fetched free
  from Vision spot klines (or yfinance ETH-USD as a cross-check only, not primary).
- **Basis:** mark vs index from Vision `premiumIndex`/`markPrice` if needed for cost.

### Strategies
- **A — delta-neutral carry** (operator pre-approved): long spot, short perp, per asset;
  collect funding each 8h held; rebalance to neutral on a fixed cadence; fully costed.
- **B — directional carry-timing** (carries price risk): scale *perp-only* exposure by
  the sign/magnitude of trailing funding. Held to a **stricter** test (see gate v).

### Leakage convention
Funding realized at settlement *t* is known at *t*. The position for interval
[t, t+8h) is decided from funding ≤ t and earns the settlement at *t+8h*. Single
explicit shift, mirroring `apply_execution_shift` (decision *t*, earn *t+1*). No
contemporaneous funding in the position that earns it.

## 5. Binding gate (NO weakening, NO post-hoc tuning)

A variant may be written up as a candidate only if **all** hold on the permanent
holdout, net of all costs in §3.2:

- **(i) Regime robustness:** net carry is positive in the **majority of individual
  calendar years**, and explicitly **not negative in 2022 and 2026** (the thin regimes).
  A result that is positive full-sample but driven by 2020–2021 alone is killed.
- **(ii) Cost + delay survival:** survives realistic cost (§3.2) AND a 1-interval
  execution delay without the Sharpe collapsing.
- **(iii) Beats benchmark + placebo:** beats delta-neutral buy-and-hold of the same
  funding stream held passively, and beats random/shuffled/inverted placebos.
- **(iv) Statistical:** PBO < 0.5, deflated Sharpe payload positive, stationary-bootstrap
  lower-CI Sharpe > 0 (reuse `crypto_research/perps/validation.py`).
- **(v) Strategy B beta test:** residual Sharpe must be > 0 *after* regressing out
  BTC (and ETH) spot beta. If B's return is just long-crypto beta in disguise, B is
  killed regardless of headline Sharpe.
- **(vi) Concentration:** not >60% of PnL from a single asset or a single year.

Failing any one → DO_NOT_ADVANCE + negative-result note. Passing all → candidate is
documented; **still no paper/live** without separate operator authorization.

## 6. Plan

- **P1 — data alignment audit (no strategy code):** load funding + perp + spot for
  BTC/ETH, align to a common 8h (or daily) grid, verify perp/spot coverage + gaps,
  confirm the leak-safe join. Write alignment report. *Gate before P2.*
- **P2 — delta-neutral carry backtest (Strategy A):** fully-costed, per-year + pooled,
  against the §5 gate.
- **P3 — directional variant (Strategy B):** only if A is informative; held to gate (v).
- **P4 — strict review + verdict:** run §5 (iv) statistical battery; write candidate or
  negative-result note.

## 7. What this is not

- Not a price predictor. The S1 tabular predictor remains the only authoritative numeric
  forecaster (CLAUDE.md §11). This is a carry harvest, not a directional model.
- Not a promotion. research_only. No paper, no live, no `QUANTLAB_STAGE` change.
- Not a re-slice of an exhausted channel (close-out §5 disqualifies those); funding is
  new information.
