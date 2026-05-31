# Intake — Trade-Flow v1 (aggressor-signed order-flow imbalance)

**Date:** 2026-05-29
**Status:** PRE-REGISTRATION (intake submitted; no backtest run yet)
**Strategy name:** `trade_flow_v1`
**Proposer:** QuantLab research
**Promotion intent:** `research_only` (free data; hard-capped — see §9)

## 1. Strategy name and one-line description

`trade_flow_v1` — predict short-horizon BTCUSDT midprice direction from
**aggressor-signed trade-flow imbalance** in the tick (aggTrades) stream, and
test whether the predicted markout can clear realistic execution cost.

## 2. Hypothesis statement

Aggressive market orders carry information: a burst of buyer-initiated
(taker-buy) volume reflects impatient informed/liquidity demand that pushes the
midprice up over the next seconds before the book refills, and symmetrically for
seller-initiated flow. Net aggressor-signed volume over a short lookback should
therefore predict the sign and magnitude of the next-interval midprice move.
This is a **different information channel** from the L2-depth benchmark
(`orderbook_microstructure_benchmark`), which already showed a genuinely
predictive depth signal (~72% directional accuracy, hist_gradient IC 0.187) that
was **"predictive but untradable"** as a spread-crossing taker. Trade-flow
imbalance is hypothesized to (a) carry incremental information over static depth
because it is the realized aggression, not the resting intent, and (b) admit a
markout large enough, at a longer horizon, to survive cost — which is the
specific failure this intake attacks.

## 3. Information source declaration (machine-readable)

- `microstructure_tick` — aggressor-signed trade prints (Binance aggTrades).

Secondary, derived within the same stream (not a separate paid feed):
top-of-book midprice is reconstructed from the trade prints' price path; no
resting L2 book is used (that is the separate `microstructure_book` channel
already covered by the order-book benchmark).

## 4. Provenance and data-quality (non-OHLCV path)

- **Provenance:** Binance public archive `data.binance.vision`
  (`spot/daily/aggTrades/BTCUSDT`). Free, downloadable (HTTP GET, no
  entitlement wall — unlike the Massive.com flat files, which 403 on download;
  see `reports/signal_research/microstructure/massive_data_feed_audit.md`).
- **Timestamp integrity:** audited `trade_only_clean`
  (`reports/signal_research/microstructure/data_audit_report.md`): Unix-ms UTC,
  100% monotonic, 0% duplicates, 100% aggressor-flag coverage, longest gap < 10s.
- **License/cost:** Binance public data terms; $0.
- **Coverage:** archive runs 2017→present; ~2.48M trades/day, ~50 GB/yr
  compressed. **This run is deliberately bounded** (see §7) and disk is checked
  before any download (CLAUDE.md §14).
- **Known limitations:** spot, single venue, no resting-book reconstruction, no
  queue position. Funding/perp basis out of scope.

## 5. Expected gross Sharpe and capacity

- **Ex-ante expected gross (no-cost) daily Sharpe:** 1.0–2.0 (trade-flow imbalance
  is a well-documented short-horizon predictor; gross signal is expected to be
  real, consistent with the depth benchmark).
- **Ex-ante expected NET daily Sharpe after taker cost:** **< 0** is the honest
  prior. The depth signal died on taker cost; the pre-registered question is
  whether trade-flow's markout/horizon profile clears cost where depth did not.
- **Capacity:** tiny. Tick-horizon BTCUSDT flow signals are capacity-constrained
  to small notional; this is a research probe, not a capacity claim.

## 6. Cost assumptions

- **Taker baseline:** 1.0 bps/side fee + half-spread crossing + 0.5 bps slippage
  (matches the order-book benchmark, Binance spot taker realistic).
- **Passive/maker scenario (pre-registered, reported separately, never used for
  promotion):** maker fill earning the spread with a documented adverse-selection
  haircut and an explicit non-fill assumption. This is reported only to localize
  *why* a signal is or isn't tradable; it is NOT a promotion path (queue position
  is not modeled, so maker fills are optimistic).
- The pipeline applies the declared taker numbers **plus 2× cost stress plus a
  1-event execution-delay stress**. A strategy needing sub-taker costs is not a
  hedge-fund strategy.

## 7. Universe and history

- **Universe:** BTCUSDT spot (single instrument). Single-instrument is acceptable
  here because the information channel (tick flow), not cross-sectional breadth,
  is the hypothesis.
- **Bounded data plan (disk-checked before download):**
  - **Dev:** a contiguous recent window of aggTrades days (target ~40 trading
    days; bounded so total compressed download stays < ~10 GB and is logged).
  - **Embargo:** ≥ 1 full day gap between dev_end and holdout_start.
  - **Holdout:** a later contiguous window (~15 days), dev-only-guard enforced.
- **18-month-holdout rule:** that protocol default is a *daily-bar* convention to
  prevent calendar leakage; at tick horizons the label is a seconds-ahead
  markout, so label leakage is fully prevented by the ≥1-day embargo. The
  chronological dev→embargo→holdout ordering is preserved. This relaxation is
  pre-registered here, not applied post-hoc.
- The three already-audited sample days (2024-04-01, 2024-08-05, 2026-05-22) are
  used only as a smoke fixture, **not** as the dev/holdout set.

## 8. What would make this fail? (pre-registered)

1. **Markout < cost (most likely).** Trade-flow predicts direction but the
   per-event midprice markout is smaller than half-spread + fee, exactly as the
   depth signal failed. Cost-decomposition + the taker/maker split localize this.
2. **Horizon mismatch.** The edge lives at sub-second horizons that the audited
   aggTrades resolution + realistic latency cannot capture; the horizon sweep
   should show gross PnL concentrated at horizons too short to execute.
3. **Autocorrelated noise / PBO.** Apparent imbalance edge is microstructure
   noise (bid-ask bounce) that inflates in-sample fit; chronological-block CSCV
   PBO and the inverted-signal + shuffled-prediction null baselines catch it.

## 9. Promotion intent

`research_only`. The data is free Binance public data; `classify_perp_candidate`
**always appends a `free_data_research_only` blocker**, so the maximum reachable
status is research-only knowledge. `promotion_eligible` and `production_candidate`
are hardcoded `False`. No paper-trade or live-capital intent. No §11 promotion
gate is triggered.

## 10. Sign-off

Proposer: QuantLab research. Intake date: 2026-05-29. The strategy will be
subjected to the full validation gate (PBO ≤ 0.25, bootstrap CI lower > 0,
DSR ≥ 0.95, net daily Sharpe ≥ 1.5, positive net total, 2× cost stress, 1-event
delay stress, concentration) with **no post-hoc tuning after the holdout pass**.
The honest prior (§5) is that the net result will fail the cost gate; the value
of the run is to document *whether and why* trade-flow differs from the depth
signal, not to manufacture a pass.

## Mode B addendum (2026-05-29) — L1 quotes on Binance bookTicker

Ran first under **Mode B** (operator decision "B first, then A"): the finished
`perps` L1-quote pipeline needs best bid/ask + sizes, which the audited aggTrades
stream lacks. Mode B sources **Binance USDT-M futures bookTicker** archives
(`data/futures/um/daily/bookTicker/BTCUSDT/`, free, streamed capped per day), so
the declared information source for the Mode B run is **`microstructure_book`**
(L1 top-of-book), not `microstructure_tick`. All §6 cost rules and §8 failure
modes carry over unchanged. Mode A (aggressor-signed flow on aggTrades, the
literal `microstructure_tick` channel) remains to be built next.

**Mode B result (run `20260529-201427`, 6 days, 456k rows, walk-forward OOS):**
gross signal is real (ridge IC ≈ 0.097, directional accuracy 79–90% on non-zero
moves) but **untradable** — an edge-over-cost threshold sweep (k ∈ {none,1…4})
shows gross markout collapsing to ≈0 as selectivity rises, and **net return is
negative at every threshold** (best k=4: net −0.81%, 27 trades, net hit 0%).
Fails the cost gate, 2× cost, and 1-event delay. Failure mode #1 (markout < cost)
materialized exactly as pre-registered. `research_candidate=False`, hard-capped
`research_only`. Confirms the order-book L2 benchmark's "predictive but
untradable" finding on the L1-quote channel.

## Mode A result (2026-05-29) — aggressor-signed flow on spot aggTrades

Run `20260529-204946`, 6 days, 2.4M trades, walk-forward OOS (markout horizon 20
events, modeled 1 bp half-spread). The trade-flow signal is **strongly
predictive statistically** — ensemble IC **0.45**, zero-mean R² **+0.22**, 78%
directional accuracy (far above Mode B's IC 0.015). But it is **even more
untradable**: the edge-over-cost sweep makes **zero trades at k ≥ 1.5× cost** and
−0.09% on the single k=1 trade — the predicted markouts are almost always smaller
than the round-trip cost. The high IC is consistent with **bid-ask-bounce mean
reversion** (mechanically real, uncapturable without crossing the spread), which
the cost gate rejects by construction. Failure mode #1 (markout < cost) confirmed.
`research_candidate=False`, hard-capped `research_only`. The literal
`microstructure_tick` channel is now closed alongside Mode B.

**Conclusion across the microstructure arc:** three independent channels — L2
depth (order-book benchmark), L1 quotes (Mode B), and tick trade-flow (Mode A) —
all show genuinely predictive short-horizon signals that die on realistic taker
cost. The binding constraint is execution economics on free retail venues, not
signal absence. Any tradable microstructure edge would require a maker/queue
execution model and venue-level fee/rebate structure that free public data and a
taker-cost model cannot represent.

## What happens after this intake

1. This document is committed to `docs/research/intake/`.
2. A bounded contiguous aggTrades window is downloaded (disk-checked) into
   `data/raw/binance/aggTrades/BTCUSDT/` (gitignored).
3. `scripts/run_trade_flow_v1.py` wires: aggTrades → trade-flow features →
   `perps` walk-forward training → `perps` cost-aware backtest →
   `classify_perp_candidate` → report under
   `reports/signal_research/microstructure/trade_flow_v1/`.
4. The status classification is committed alongside the report. No re-tuning
   after holdout numbers are seen.
