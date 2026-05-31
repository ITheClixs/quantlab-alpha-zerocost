# EDGAR 10-K v1 — Baseline Comparison

**Built:** 2026-05-30T09:20:04.393908+00:00 | decision = text model must beat OHLCV/factor baselines on IC AND spread AND net PnL.

| metric | best text model | size baseline | event-ret baseline | placebo max |
|---|---:|---:|---:|---:|
| holdout mean IC | 0.0213 | 0.0005 | 0.0080 | 0.0153 |
| holdout mean spread | 0.0147 | 0.0045 | 0.0074 | — |
| net LS PnL | 3.20% | -0.29% | 0.87% | — |

- Beats baselines — IC: **True**, spread: **True**, net PnL: **True**.
- **Data caveat:** full momentum/vol/low-vol factor baselines are NOT computable — they need a
  survivorship-safe price panel the program lacks (equity-return audit `3f9a658`). Size (mkt_cap) is
  the binding factor baseline here; full factor subsumption is a v2 item gated on a clean price panel.
