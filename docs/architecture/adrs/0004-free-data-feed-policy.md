# ADR 0004: Free-data-only policy for live feeds

## Status
Accepted, 2026-05-14.

## Context
The operator stipulated unpaid resources. Paid market data ($1k-$50k/mo for equity SIP)
is out of scope. Equity HFT and free data are incompatible.

## Decision
Equity strategies run at 15-min bar minimum (Alpaca / Polygon free tier).
Crypto strategies run at tick frequency via Binance and Coinbase public WebSocket
(real-time, no auth required). yfinance is allowed for backtest research only — never
live. The README explicitly limits live equity trading to mid-frequency and crypto to
tick-frequency.

## Consequences
+ Zero monthly data cost.
+ Crypto strategies can be tick-level (real HFT-ish).
- Equity strategies cannot react sub-minute.
- "HFT-optimized" framing in marketing is unsupported and removed from docs.
