# Funding-Carry — Realism Upgrade (8h-marked, slippage, liquidation)

**Date:** 2026-05-30  
**Status:** research_only — no paper, no live.

Operator-requested realism pass. The P2 daily-close model smoothed away the intraday spot-perp basis variance, inflating the Sharpe to ~8.6. Here the carry is marked on the **8h funding-settlement grid** (true basis variance), with execution slippage (5.0bps/leg) and an isolated-margin liquidation model on the short perp.

## Daily illusion vs honest 8h (pooled BTC+ETH)

| | daily-close (illusion) | 8h-marked (honest) |
|---|---|---|
| Sharpe | 8.56 | **8.61** |
| ann return | 13.94% | 13.92% |
| ann vol | 1.53% | 1.52% |
| max DD | -1.68% | -1.6% |

Bootstrap Sharpe 95% CI (8h net): [3.278, 5.095] (per-bar).

## Honest per-year net (pooled, %, 8h-marked)

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 23.83 | 40.18 | 2.38 | 8.16 | 13.3 | 5.09 | -0.26 |

## Liquidation tail (intrabar adverse basis on the short perp)

| asset | max adverse Δbasis | p99.9 | liq events 3x / 5x / 10x |
|---|---|---|---|
| BTCUSDT | 46.95% | 21.627% | 5 / 10 / 93 |
| ETHUSDT | 276.611% | 28.713% | 5 / 25 / 191 |

## Liquidation-stressed pooled (conservative: lose posted margin + re-hedge)

| leverage | Sharpe | ann return | liq events |
|---|---|---|---|
| 3x | -0.47 | -17.00% | 10 |
| 5x | -1.50 | -37.84% | 35 |
| 10x | -5.17 | -89.60% | 284 |

## Honest conclusion (corrects the P2 'illusion' hypothesis)

- **8h marking did NOT deflate the Sharpe** (8.56 -> 8.61). The spot-perp basis is genuinely tight even at 8h, so the daily-close model was *not* an illusion on the basis-variance axis. The P2 guess that the realistic Sharpe was ~1-2 was wrong on the mechanism.
- The unlevered, fully-collateralized delta-neutral carry **genuinely** shows Sharpe ~8.61 / ~14%/yr over 2020-2026 — real, because funding is a steady positive drip and the hedge tracks tightly (low daily/8h variance).
- **The catch is the fat left tail, not the daily vol.** (1) At 1x it is capital-inefficient (100% margin on both legs). (2) Any leverage to fix that introduces short-perp **liquidation in crashes** — the stress table above goes negative at 3x (-17%/yr) and catastrophic at 10x (-90%). The high Sharpe does not price this tail.
- The liquidation proxy uses non-simultaneous intrabar high/low so it **overstates** adverse basis; the 8h-close model **understates** intrabar dislocation. Pricing the tail precisely needs intraday simultaneous spot+perp (1m/tick) data.
- **Verdict unchanged: DO_NOT_ADVANCE.** A real, free, market-neutral carry, but: (a) the P2 regime gate fails (2026 net-negative), (b) the edge is decaying with crowding, and (c) it is deployable only unlevered/capital-inefficient with an unpriced crash-liquidation tail. research_only — no paper, no live.
