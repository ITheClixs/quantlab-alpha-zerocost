# Negative/Partial Result — Crypto Perp Funding-Rate Carry v1

**Date:** 2026-05-30
**Status:** **DO_NOT_ADVANCE — research_only.** No paper. No live. No promotion.
**Branch:** reopened via close-out §5 condition #5 (genuinely new free information
source). Funding was new — the microstructure arc was spot/trade-flow only.
**Artifacts:** `crypto_research/funding/` (`data.py`, `prices.py`, `carry.py`),
`scripts/audit_funding_carry_data.py`, `scripts/audit_funding_carry_alignment.py`,
`scripts/run_funding_carry_v1.py`, `scripts/run_funding_carry_realism.py`,
`reports/signal_research/funding_carry_v1/*`, `manifests/funding_carry/*`.
Commits `f169ac0` (data audit), `3aa799b` (alignment), `b50d2ed` (P2), `eac6453` (P2.5).

## 1. What this was

Delta-neutral funding-rate carry: long spot / short USDT-M perp on BTC + ETH, collect
the 8h funding that longs persistently pay shorts. The first channel since the
zero-cost close-out to **clear its data audits**, and structurally the first to escape
the two walls that killed everything before it: it is *held* (not a per-trade taker
bet → no cost wall) and it is *carry* (not single-index vol-timing → no subsumption
wall). Data is free (Binance Vision), leak-safe (funding settled at *t* known at *t*),
and survivorship-clean (BTC/ETH perps never delisted).

## 2. What the evidence showed (honest, in order)

**Data audits — PASS.** Funding BTC/ETH 6,936 8h settlements, 2020-01..2026-04. Daily
carry panel: 2,312 rows/asset, zero missing days, basis mean ~0, |basis| p95 ~0.1%.
BTC spot Vision-daily == on-disk vaquum 1m to 0.0% over all 2,312 days.

**P2 delta-neutral backtest — net-positive 6 of 7 years** (pooled, after 10bps spot +
5bps perp taker + hedge maintenance): 2020 +23.5%, 2021 +40.4%, 2022 +2.4%, 2023
+8.2%, 2024 +13.3%, 2025 +5.1%, 2026 −0.16% (120d). Placebo separation clean (inverted
−12.4%/yr; zero-funding ≈0 → funding *is* the source, not a price artifact). Bootstrap
lower-CI Sharpe +5.7, DSR 1.0, PBO 0.31, top-year share 41%. **4 of 5 gates pass.**

**P2.5 realism pass — corrected the mechanism.** Re-marking on the 8h funding grid did
**not** deflate the Sharpe (8.56 → 8.61): the spot-perp basis is genuinely tight even
at 8h, so the daily-close model was *not* a basis-variance illusion (the P2 guess of
"realistic ~1-2" was wrong). The unlevered, fully-collateralized carry really earned
~14%/yr at a high Sharpe in 2020-2026. **The risk is the fat left tail, not the daily
vol:** at 1x it is capital-inefficient (100% margin both legs); any leverage to fix
that introduces short-perp **liquidation in crashes** — stress: 3x −17%/yr, 5x −38%,
10x −90%.

## 3. Why DO_NOT_ADVANCE

Two independent reasons, neither weakened:

1. **Pre-registered regime gate fails.** The intake required net non-negative in 2022
   *and* 2026. 2026 YTD is −0.16% — the most recent, thinnest-funding regime is
   net-negative after cost. The edge is **decaying with crowding** (2024 +13% → 2025
   +5% → 2026 ~0).
2. **No deployable risk/leverage point.** Unlevered it is real but capital-inefficient
   and still fails (1); levered it is killed by the crash-liquidation tail. The high
   Sharpe is a calm-regime figure that does not price the tail.

## 4. New failure class

This does not reduce to the prior four walls (cost / subsumption / data-access /
frequency). It is a new pattern: **a real, free, market-neutral carry that is
regime-decaying and has a fat crash-liquidation tail that only appears under the
leverage needed to make it capital-efficient.** High calm-period Sharpe; unpriced
left tail; "pennies in front of a steamroller."

## 5. Two bugs rigor caught (both would have fabricated a result)

- **8h join dropped ~45% of settlements** — Binance funding `calc_time` carries ms
  jitter (`00:00:00.002`) vs the 8h kline open at `00:00:00.000`; the exact-timestamp
  join silently lost the jittered settlements, halving the funding total and faking a
  4.46% "honest" return. Fixed (round the funding key to the hour) + coverage guard +
  regression test.
- **`pooled_book` annualized 8h data at 365 not 1095** — understated pooled return ~3x
  and Sharpe by √3. Fixed (periods-per-year param) + regression test.

Same lesson as the whole program: the validation harness exists to stop fabricated
edges, and it did again.

## 6. Reopen conditions (operator-authorized only)

1. **1m/tick simultaneous spot+perp** to price the crash-liquidation tail precisely
   (the 8h high/low proxy *overstates* adverse basis — non-simultaneous; the 8h-close
   model *understates* intrabar dislocation; the truth is bracketed). Would not change
   the regime-gate failure, only quantify the tail.
2. An explicit decision to deploy **unlevered/capital-inefficiently** and accept the
   ~14%-decaying-to-~0 yield (operator call; still fails the pre-registered gate).
3. **P3 directional carry-timing** (Strategy B) — pre-registered but lower-probability
   (adds price risk, likely disguised beta; carry signal already decaying). Held to the
   residual-Sharpe-after-stripping-BTC-beta gate.

## 7. Operating mode

research_only. No paper, no live, no `QUANTLAB_STAGE` change. The funding module +
8h/liquidation tooling are the durable deliverable, ready if 1m data is supplied.
Still **0 taker-tradable deployable strategies**; the program's gates held.
