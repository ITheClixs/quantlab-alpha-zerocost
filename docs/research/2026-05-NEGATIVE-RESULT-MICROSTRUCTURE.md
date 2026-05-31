# Negative-Result Note — Free-Data Crypto Microstructure (closed)

**Date:** 2026-05-30
**Status:** CLOSED. Do not continue free-data BTCUSDT microstructure variants.
**Scope:** the three-channel microstructure arc, May 2026.
**Author:** QuantLab research

## 0. Conclusion

Free-data crypto microstructure produced **genuinely predictive short-horizon
signals on every channel tested, all untradable after realistic taker cost**.
The binding constraint is **execution economics, not signal absence**. The arc
is closed; no further free-data BTCUSDT microstructure variants are authorized.

## 1. The three channels

| Channel | Implementation | Signal | Verdict |
|---|---|---|---|
| L2 depth | `orderbook_microstructure_benchmark` (HF Binance futures L2, BTC/ETH/SOL) | IC 0.19, ~72% dir-acc | predictive but untradable (markout < spread+fee) |
| L1 quotes | trade_flow_v1 **Mode B**, Binance USDT-M futures bookTicker (`run_trade_flow_v1.py`) | IC 0.015, 79–90% dir-acc | net < 0 at every edge-over-cost threshold |
| Tick trade-flow | trade_flow_v1 **Mode A**, Binance spot aggTrades, aggressor-signed (`run_trade_flow_v1_agg.py`) | **IC 0.45, R² +0.22, 78% dir-acc** | **zero trades above 1.5× cost**; markouts < round-trip cost |

References: `reports/orderbook_microstructure_benchmark_*.md`,
`reports/signal_research/microstructure/trade_flow_v1/`,
`reports/signal_research/microstructure/trade_flow_v1_agg/`,
intake `docs/research/intake/2026-05-29-trade-flow-v1.md` (Mode A + B addenda).

## 2. Why it is closed

- Mode A was the **strongest statistical signal the whole program has produced**
  (IC 0.45) and was still completely untradable: predicted markouts are almost
  always smaller than the round-trip spread+fee. The high IC is largely
  **bid-ask-bounce mean reversion** — mechanically real, uncapturable without
  crossing the spread. The cost gate rejected it by construction.
- The result reproduces across L2 depth, L1 quotes, and tick flow — three
  orthogonal microstructure representations. The negative result is structural.

## 3. What is explicitly NOT being done (operator decision, 2026-05-30)

- **No more free-data BTCUSDT microstructure variants.**
- **No maker-execution modeling yet.** Maker/passive-fill simulation becomes
  evidence-driven (not assumption-driven) only with: reliable order-book
  sequence data, data-grounded queue-position assumptions, fill-probability
  estimates, adverse-selection measurement, a venue fee/rebate schedule, latency
  assumptions, and enough event-level history to validate passive execution.
  Without those, maker simulation risks manufacturing tradability. Deferred.

## 4. Next direction

Pivot to the cheaper untested channel: **event-conditioned macro/calendar v1**
(`docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md`). The
program `/goal` remains: find **taker-tradable** alpha for QuantLab.
