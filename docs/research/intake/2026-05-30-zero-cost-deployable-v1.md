# Intake + Implementation Plan — Zero-Cost Deployable Strategy v1

**Date:** 2026-05-30
**Status:** PRE-REGISTRATION + implementation plan (no strategy code until accepted).
**Strategy name:** `zero_cost_deployable_v1`
**Proposer:** QuantLab research
**Promotion intent:** `paper_trade_after_pass` (single-index risk-timing exception
policy applies — `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`).
No live capital. No promotion language until paper trading.
**Constraint:** `docs/research/2026-05-30-ZERO-COST-CONSTRAINT.md` (zero-cost data only).

## 0. Honest framing (read first)

The program has shown, three times (VRP, HMM single-index, FOMC), that
**vol-targeting / regime exposure subsumes single-index risk-timing edges**. So the
*binding* test for this entire direction is: **does the regime/macro overlay beat a
plain vol-targeted buy-and-hold, net of cost and delay?** The prior is that this is
hard. We proceed anyway because (a) the goal is now a *deployable, paper-tradeable*
pipeline on free data — not the 1.5 production Sharpe — and (b) a long-flat
vol-targeted/regime allocator with better drawdown control is itself a legitimate
paper-trade deliverable even if it only matches BAH on raw return. The exception
policy and the strict delay/cost gates remain non-negotiable; the kill criterion
(§6) is explicit.

## 1. Instrument universe (directly tradable, free, no hidden constituents)

| Instrument | Free source | History | Role |
|---|---|---|---|
| SPY | on disk `data/processed/vrp/bars/SPY.parquet` | 2010-2026 | equity beta, longest clean history |
| QQQ | on disk `.../QQQ.parquet` | 2010-2026 | equity growth/tech |
| BTCUSDT | Binance public archive / yfinance (free) | 2017-2026 | crypto, orthogonal regime |
| ETHUSDT | Binance public / yfinance (free) | 2017-2026 | crypto |

Small **directly-traded basket** (4 names) — no cross-sectional selection, no hidden
universe. Equity + crypto sleeves diversify across regimes. All long-flat (no
shorting needed; deployable on the free/paper venues already wired:
`brokers/alpaca_paper.py` for ETFs, `brokers/*_paper.py` / Binance paper for crypto).

## 2. Free, timestamp-safe features

**2.1 Per-instrument OHLCV (own series, causal):** trailing realized vol (e.g. 20/60d),
downside vol, trend (12-1 momentum, SMA50/200 state), drawdown-from-peak, return
autocorr. All computed at close t, used at t+1.

**2.2 Regime:** reuse `signal_research/strategies/hmm_single_index.py` (HMM risk-on/off)
**and** a transparent vol-regime fallback (trailing vol vs its rolling median). HMM
used as an overlay, with the exception policy's refit-stability + delay gates applied
(HMM v1 failed those; we re-test honestly).

**2.3 Market-priced macro overlays (timestamp-safe at t+1; the key discipline):**
use only **daily market-observed, minimally-revised** series, never revised
aggregates. Allowed:
- **VIX** level + **VIX term structure** (VIX / VIX3M ratio) — contango/backwardation.
- **Yield-curve slope** (DGS10 − DGS2, FRED daily Treasury — published next-day, not revised).
- **Credit spread** (BAML US HY OAS, FRED daily).
- **USD trend** (DXY / UUP).
- **Cross-asset trend** (equity vs bonds/gold via free ETFs TLT/GLD).

**Forbidden (look-ahead / revision risk):** GDP, payrolls, CPI, and other revised
macro aggregates **unless** pulled as ALFRED point-in-time vintages (deferred). A
macro-feature **timestamp audit** is the first gate (§ implementation P0): each
series must be confirmed observable-at-t (market close) and used at t+1.

## 3. The ONE deployable candidate (frozen, predeclared)

`zero_cost_riskalloc_v1` — a long-flat **regime-and-vol-targeted risk allocator**:

1. **Vol target** each instrument to a fixed annual vol (SPY/QQQ 12%, BTC/ETH 12%
   with a hard leverage cap: ≤1.5 equity, ≤1.0 crypto), using trailing vol (causal).
2. **Risk-on/off overlay** = AND of: trend state (close > SMA200), regime (HMM
   risk-on OR vol-regime low), and a macro filter (VIX term structure not in steep
   backwardation AND yield-curve/credit not in stress). Risk-off → scale exposure
   toward 0.
3. **Basket** = fixed equal-risk weights across the 4 instruments (equity/crypto
   sleeves), rebalanced weekly (low turnover).
4. All decisions at close t, executed **t+1** (delay-safe by construction).

Baselines (must beat to survive): **vol-targeted buy-and-hold (no overlay)** — the
subsumption baseline; plain buy-and-hold; 60/40-style static; HMM-only; random/
inverted sanity.

## 4. Evaluation (reuse `ValidationPipeline` + exception-path gate)

Net PnL after **declared cost** (SPY/QQQ ~1 bp/side, BTC/ETH ~5-10 bp/side) +
**turnover** + **1- and 2-bar delay stress** + **2×/3× cost stress**. Metrics: net
Sharpe, Calmar, max drawdown, bootstrap CI, PBO/DSR over the variant pool,
concentration (by instrument, year, crisis), crisis removal (2018/2020/2022). The
**binding gate**: net Sharpe **AND** max-drawdown must **beat vol-targeted BAH** on
the chronological holdout — else `subsumed_by_vol_targeting` and kill.

## 5. Paper-trading readiness (only if §4 passes)

Wire the surviving signal into the existing stage-gated execution path (CLAUDE.md
S4): `QUANTLAB_STAGE=paper` → `brokers/*_paper.py`, `feeds/` for live bars,
`execution/` + append-only `logs/audit/`. No live. The exception policy governs the
paper-trade candidacy; in-process self-promotion remains forbidden.

## 6. Kill criterion (explicit)

Kill `zero_cost_deployable_v1` if ANY: it does not beat vol-targeted BAH on net
Sharpe AND max-drawdown on the holdout; OR loses > 0.5 Sharpe to 1-bar delay; OR is
refit-unstable (per exception §3); OR net PnL ≤ 0 after 2× cost; OR the edge is
concentrated in one instrument/one crisis. A clean kill (documented failure class)
is an acceptable outcome.

## 7. Pre-registered failure modes

`subsumed_by_vol_targeting`, `delay_sensitivity`, `refit_instability`,
`cost_failure`, `turnover_too_high`, `crisis_concentration`,
`macro_feature_lookahead`, `no_edge_over_baseline`.

## 8. Implementation sequence (phased; TDD; no code until accepted)

- **P0 — data + macro timestamp audit:** loaders for SPY/QQQ (on disk), BTC/ETH
  (free fetch, cached), and the macro series (FRED daily + VIX/VIX3M + ETFs); a
  `macro_timestamp_audit.md` confirming each feature is observable-at-t / used t+1.
  **Gate:** any feature that can't be shown timestamp-safe is dropped.
- **P1 — feature layer:** per-instrument OHLCV features + regime (reuse HMM) + macro
  overlays. Leak-safe, tested.
- **P2 — the frozen candidate** (`zero_cost_riskalloc_v1`) + baselines.
- **P3 — validation battery** (§4) → registry + reports + classification.
- **P4 — paper wiring** (only if §4 passes the §6 gate).

## 9. Constraints

- Zero-cost data only. Directly-traded instruments only. Long-flat (no shorting).
- No cross-sectional stock selection. No paid data. No hidden universe.
- No promotion language until paper trading; no live capital.
- Must beat vol-targeted BAH or be killed.

## What happens after this intake

1. Commit this intake.
2. P0: build the free data loaders + the macro-feature **timestamp audit** (the
   first hard gate — analogous to every prior data-audit-first step).
3. Proceed P1→P4 only as each gate passes; kill on §6.
