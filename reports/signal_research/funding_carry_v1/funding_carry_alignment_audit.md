# Funding-Carry Data Alignment Audit (P1)

**Date:** 2026-05-30  
**Verdict:** **PASS**  
**Status:** research_only — no paper, no live.

Leak-safe daily carry panel for BTC/ETH from free Binance Vision archives (funding + spot + perp daily klines). Basis = perp/spot − 1 at the simultaneous UTC-midnight close; `funding_day` = total funding settled that UTC day.

## Per-symbol panel

| Symbol | rows | start | end | missing days | basis mean | basis |max| p95 | ann funding (full) |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2312 | 2020-01-01 | 2026-04-30 | 0 | -0.0146% | 0.7365% | 0.0954% | 12.17% |
| ETHUSDT | 2312 | 2020-01-01 | 2026-04-30 | 0 | -0.008% | 1.0295% | 0.1124% | 14.5% |

## Annualized funding by year (gross, %)

| Symbol | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 17.19 | 30.61 | 4.16 | 7.87 | 11.92 | 5.13 | 0.38 |
| ETHUSDT | 27.41 | 37.54 | 0.79 | 8.26 | 12.96 | 4.93 | -0.38 |

## BTC spot cross-check (Vision daily vs on-disk vaquum 1m)

- status: `ok`
- overlap days: 2312
- relative close diff: mean 0.0%, p99 0.0%, max 0.0%
- → Vision daily close ≈ on-disk last-of-day close (reuse-on-disk validated).

## Leakage rule

- Funding realized at settlement *t* is known at *t*; the short-perp leg held through UTC day *D* collects day *D*'s funding.
- Basis uses spot and perp closes at the **same** UTC instant (one source) → no stale-leg basis.
- P2 backtest applies a single explicit decision-*t* / earn-*t+1* shift; no contemporaneous funding in the position that earns it.

## Verdict

**PASS.** Panel coverage clean (gaps ≤ 5 days, >1500 rows/asset) and the on-disk cross-check agrees → proceed to P2 (delta-neutral carry backtest) under the intake §5 gate.
