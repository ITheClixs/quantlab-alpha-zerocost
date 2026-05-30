# Funding-Carry v1 — Delta-Neutral Carry Backtest (P2)

**Date:** 2026-05-30  
**Verdict:** **DO_NOT_ADVANCE**  
**Status:** research_only — **no paper, no live, no promotion**.

Strategy A: long spot / short USDT-M perp on BTC + ETH, equal-weight pooled, re-neutralized daily. Costs: spot taker 10.0bps + perp taker 5.0bps one-way, plus daily hedge-maintenance turnover and entry/exit. Returns per unit one-side notional (carry yield). Annualized at sqrt(365).

## ⚠️ Realism caveat — read before the headline

The pooled Sharpe below (8.6) is **not deployable-real** — it is the **carry illusion**. Spot and the perp track almost perfectly at the *same-venue daily close*, so the basis/price term has near-zero variance and the book is a smooth, low-variance positive funding drip. A tradable book carries risks this daily-close model omits: intraday basis gaps (the perp dislocated from spot by multiple % in the 2020-03 and 2021-05 crashes), funding spikes, **short-leg liquidation/margin risk**, and execution slippage off the close. A realistic funding-carry book runs Sharpe ~1-2, not ~8. **Trust the annual return and the per-year regime picture; do NOT trust the Sharpe.**

## Headline (pooled book, base cost)

- Sharpe **8.56** (illusory — see caveat), ann return **13.94%**, ann vol 1.53%, max DD -1.7% (understated), Calmar 8.32.
- Per asset: BTC Sharpe 8.61 (12.70%/yr); ETH Sharpe 7.65 (15.19%/yr).

## Per-year net return (pooled, %, after cost)

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 23.53 | 40.35 | 2.42 | 8.22 | 13.33 | 5.12 | -0.16 |

## Cost stress (pooled Sharpe / ann return)

| cost | Sharpe | ann return |
|---|---|---|
| 1x | 8.56 | 13.94% |
| 2x | 8.48 | 13.87% |
| 3x | 8.37 | 13.80% |

## Placebo separation

- Inverted book (long perp / short spot): ann -12.36% (must be **negative** — carry direction matters).
- Zero-funding (price/basis only): ann -0.28% (must be **~0** — confirms funding, not a price artifact, is the source).

## Statistical battery (pooled net)

- Stationary bootstrap Sharpe 95% CI: [5.713, 8.584] (per-period; lower bound must be > 0).
- Deflated Sharpe probability: 1.000 (trials=3).
- PBO probability: 0.314 (must be < 0.5).

## Concentration

- Pooled top-year PnL share 0.41, top-day 0.0078 (year share must be ≤ 0.60).
- BTC top-year 0.404, ETH top-year 0.416.

## Gate (intake §5)

| gate | result |
|---|---|
| regime_robust | FAIL |
| cost_survival | PASS |
| placebo_separation | PASS |
| statistical | PASS |
| concentration | PASS |

## Verdict

**DO_NOT_ADVANCE.** The carry is **real and persistent** — net-positive in 6 of 7 years, with clean placebo and statistical separation (funding, not price, is the source). But it **does not clear the pre-registered regime gate**: the most recent regime (2026 YTD, -0.16% over 120 days) is net-negative after cost, and the gate (no-weakening) requires non-negative 2022 *and* 2026. Separately and more importantly, the headline Sharpe is an **illusion** of the daily-close basis model (see caveat) — a realistic execution/risk model would materially lower the risk-adjusted edge. **DO_NOT_ADVANCE.** No paper/live.
