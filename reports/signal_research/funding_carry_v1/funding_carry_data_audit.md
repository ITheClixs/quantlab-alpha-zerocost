# Funding-Carry Data Audit

**Built:** 2026-05-30T16:10:43.156474+00:00  **Verdict:** **PASS**
**Binding question:** leak-safe, survivorship-clean crypto perp funding-carry panel from FREE data?

## 1. Funding-rate data (Binance Vision, free)

| symbol | rows | start | end | ann. funding (full) |
|---|---:|---|---|---:|
| BTCUSDT | 6936 | 2020-01-01 00:00:00+00:00 | 2026-04-30 16:00:00+00:00 | 12.2% |
| ETHUSDT | 6936 | 2020-01-01 00:00:00+00:00 | 2026-04-30 16:00:00+00:00 | 14.5% |

### Realized funding by year (regime honesty — carry is NOT constant)
- **BTCUSDT**: 2020:+17.2%, 2021:+30.6%, 2022:+4.2%, 2023:+7.9%, 2024:+11.9%, 2025:+5.1%, 2026:+0.4%
- **ETHUSDT**: 2020:+27.4%, 2021:+37.5%, 2022:+0.8%, 2023:+8.3%, 2024:+13.0%, 2025:+4.9%, 2026:-0.4%

- Funding is large+positive in leverage/bull regimes (longs pay shorts) and ~0/negative otherwise. A carry strategy's edge is regime-dependent — the v1 gate must test robustness across these years.

## 2. Timestamp / leakage
- funding settles every 8h; realized rate known at settlement t; signal uses funding<=t, earns the next settlement (t+1 interval) -> leak-safe.
- 8h settlements (00/08/16 UTC); `calc_time` = settlement; `last_funding_rate` realized at that time.

## 3. Spot / perp / basis
- btc_spot_on_disk: True
- perp_ohlcv_on_disk: True
- eth_spot_source: free (Binance Vision spot klines / yfinance ETH-USD) — fetch in P1
- perp_price_source: free (Binance Vision futures/um klines; 123olp on disk)
- basis_source: free (Binance Vision markPriceKlines + premiumIndexKlines)

## 4. Universe & survivorship
- BTCUSDT, ETHUSDT perps — never delisted -> survivorship-clean (cross-sectional top-N deferred).

## 5. Cost model (delta-neutral)
- delta-neutral carry = long spot / short perp; cost paid at entry/exit only, funding earned each 8h held -> NOT a per-trade taker bet (escapes the microstructure cost wall). perp taker ~5.0bps, spot taker ~10.0bps one-way.

## 6. Verdict
- **PASS** — funding (free, 2020-01..), spot, perp, and basis are all free + timestamp-clean + survivorship-clean for BTC/ETH. Proceed to funding-carry v1 intake + backtest (delta-neutral long-spot/short-perp + directional variants), with the regime-robustness gate.

## Constraints
- No strategy code in this audit. No paper/live. Funding carry escapes the cost wall (held, not taker) and the subsumption wall (carry != vol-timing); the gate must still beat buy-and-hold + survive regime/cost.
