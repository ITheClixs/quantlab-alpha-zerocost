# Intake — Options-IV Features v1 (research_only)

**Date:** 2026-05-30
**Status:** PRE-REGISTRATION (intake submitted; cross-sectional track gated on a
secondary equity-return data audit; **no strategy code yet**)
**Strategy name:** `options_iv_features_v1`
**Proposer:** QuantLab research
**Promotion intent:** **`research_only`** — hard ceiling from the data audit (see §4)
**Program `/goal`:** find taker-tradable alpha for QuantLab.

## 0. Context

The data audit (`reports/signal_research/options_iv_v1/options_iv_data_audit_report.md`,
commit `34cc78f`) labelled the on-disk `gauss314__options-IV-SP500` dataset
`options_features_only` + `timestamp_uncertain` → **`RESEARCH_ONLY_FEATURES`**. It
is a daily per-underlying IV *summary* (no strikes/expiries/bid-ask), so options
cannot be traded directly; they are **features** for trading the underlying. The
genuinely-new angle the program has never tested is the **cross-sectional** one:
per-stock IV rank / skew / vol-spread across ~3,900 names → equity selection. This
is NOT a single-index risk-timing overlay, so it is not automatically subsumed by
vol-targeting (the failure mode that closed VRP, HMM, and FOMC).

## 1. Strategy name and one-line description

`options_iv_features_v1` — test whether cross-sectional options-implied-vol features
(IV rank, skew proxy, implied-minus-realized vol spread, put/call flow imbalance)
predict next-day equity returns, with SPY/QQQ IV-feature timing as a secondary track.

## 2. Hypothesis statement

Options-implied vol carries forward-looking information about the cross-section of
equity returns that price/volume alone does not. Two documented mechanisms: (a) the
**implied-minus-realized vol spread** (IV rank / VRP proxy) predicts subsequent
stock returns because option markets price a variance risk premium and impound
information ahead of the stock (Goyal & Saretto 2009; Bali & Hovakimian 2009); and
(b) **skew / demand-for-lottery** — richly-priced OTM call IV (positive skew demand)
is associated with subsequent underperformance (Bali, Cakici & Whitelaw 2011;
Ang, Hodrick, Xing & Zhang 2006 on idiosyncratic-vol). The hypothesis: a
cross-sectional rank on these IV features earns a return spread that is **not
reducible to OHLCV factors (size/momentum/low-vol)** and **survives daily-rebalance
costs**. The honest prior is that the cost wall (high cross-sectional turnover) and
the low-vol-factor-restatement risk are the binding threats.

## 3. Information source declaration (machine-readable)

- `options_implied_vol` — ATM IV, moneyness-bucket IV (DITM..DOTM), IV rank, RV−IV.
- `options_volume` — calls/puts contracts traded + open interest (flow imbalance).
- `ohlcv` — equities / SPY / QQQ as the **tradable instrument** (and for realized
  vol + return labels).

Driving channels are `options_implied_vol` + `options_volume` (non-OHLCV), so the
no-OHLCV-only-promotion rule is satisfied. (Promotion is moot here — research_only.)

## 4. Provenance, timestamp integrity, and the SECONDARY data gate

### 4.1 IV feature source (audited)
`gauss314__options-IV-SP500` — `RESEARCH_ONLY_FEATURES`, `timestamp_uncertain`,
daily EOD. **Hard rule: features are observed at EOD of day t and acted on at
day t+1** (1-trading-day shift enforced in code). Not current-constituent
survivorship-biased (delisted names retained). 2019-10-14 → 2023-07-28 (~3.8y).

### 4.2 Equity-return source — NEW data gate (must pass before the cross-sectional backtest)
The cross-sectional track needs **next-day total returns for each name in the IV
universe**, survivorship-safe and corporate-action-adjusted — the gauss314 dataset
has **no underlying prices**. Before any cross-sectional backtest, a secondary audit
(`equity_return_data_audit.md`) must establish, for the 2019-2023 IV universe:
- a return source covering the names **including those that delist mid-sample**
  (yfinance is survivorship-biased for delisted names → likely insufficient; check
  on-disk `HexQuant__Stocks-Daily-Price` and others first);
- corporate-action (split/dividend) adjustment aligned with the IV symbol;
- symbol-mapping between the IV dataset tickers and the return source;
- a PIT membership rule: trade only names present in the IV data on date t.
If a clean survivorship-safe return source is not available, the **cross-sectional
track is rejected on data grounds** and only the SPY/QQQ track (which needs only the
already-clean SPY/QQQ bars) proceeds.

## 5. Expected gross Sharpe and capacity

- Cross-sectional IV-spread/skew long-short, gross (pre-cost): **0.5–1.5** per the
  literature. **Net after daily-rebalance cost: honest prior < 0** unless turnover
  is controlled (weekly rebalance / banding). Capacity modest (needs liquid names).
- SPY/QQQ IV-feature timing (secondary): modest (~0.3–0.7), with the explicit
  expectation it must beat vol-targeted BAH or it is subsumed.

## 6. Cost assumptions

- Equities: 2 bps one-way (liquid) commission+spread; **cross-sectional daily
  rebalance turnover is the dominant cost risk** — variants must include a
  reduced-turnover (weekly / banded) version. Pipeline applies 2× cost + 1-bar delay.
- A liquidity filter (min option OI/volume + min equity ADV) restricts to tradable
  names; the illiquid long tail (~5-6% zero-volume rows) is excluded.

## 7. Universe and history

- IV universe: ~3,900 US optionable names, 2019-10-14 → 2023-07-28 (short history —
  pre-registered failure mode). PIT membership = present in IV data on date t.
- Chronological train / validation / holdout; dev-only-guard; given the short
  sample, the holdout is necessarily short → wide CIs (pre-registered).
- SPY/QQQ secondary track uses 2010-2026 bars where IV overlaps 2019-2023.

## 8. Pre-registered failure modes

1. **Cost wall** — gross cross-sectional spread real, dies on daily-rebalance cost.
2. **Low-vol/size restatement** — the IV signal is just the low-volatility or size
   factor in disguise (must beat an OHLCV vol/size cross-sectional baseline).
3. **Return-data survivorship** — no clean delisted-inclusive return source → the
   backtest is survivorship-biased (data gate §4.2 catches this).
4. **Short history** — ~3.8y, one regime (incl. 2020/2022) → wide CIs, crisis
   concentration; fails DSR/bootstrap.
5. **Timestamp** — same-day use leaks; only next-day is valid (enforced).
6. **Placebo** — shuffled-IV / random-rank baselines match the real signal.
7. **Subsumed (secondary track)** — SPY/QQQ IV timing subsumed by vol-targeting.

## 9. Promotion intent

`research_only` (hard ceiling: `options_features_only` + `timestamp_uncertain`). No
direct option trading. No promotion-style claims. No §11 promotion gate triggered.
If a cross-sectional variant is strikingly robust it may, at most, motivate a
*paid-data* options-chain follow-up (per the data-constraint review) — never a
promotion on this dataset.

## 10. Sign-off

Proposer: QuantLab research. Intake date: 2026-05-30. Cross-sectional track gated on
the §4.2 equity-return audit. Full validation gate, next-day execution enforced, no
post-hoc tuning after holdout. Research_only ceiling acknowledged.

---

## 11. Pre-declared design (frozen)

### 11.1 Features (EOD t → act t+1; all timestamp-shifted)
`atm_iv`, `iv_rank_252` (trailing percentile of ATM_IV per symbol), `skew_proxy`
(`DOTM_IV - DITM_IV`, and `OTM_IV - ITM_IV`), `rv_iv_spread` (`hv_20 - ATM_IV`,
`hv_60 - ATM_IV`), `pc_volume_imbalance` ((puts−calls)/(puts+calls) traded),
`pc_oi_imbalance`, `iv_change_5d`, `skew_change_5d`. **Not available:** term-structure
slope (no per-expiry IV).

### 11.2 Cross-sectional variants (primary, frozen; gated on §4.2)
1. long-short deciles on `rv_iv_spread` (high RV−IV long / low short) — VRP cross-section
2. long-short deciles on `iv_rank_252` (low-IV long / high-IV short) — low-vol cross-section
3. long-short deciles on `skew_proxy`
4. long-short on `pc_oi_imbalance`
5. each of the above in a **reduced-turnover** (weekly-rebalance / banded) version

### 11.3 SPY/QQQ timing variants (secondary, runnable now)
6. SPY/QQQ next-day exposure gated by index `rv_iv_spread` (VRP-proxy) state
7. SPY/QQQ exposure scaled by cross-sectional IV **dispersion** regime

### 11.4 Baselines (first-class, in the PBO pool)
buy-and-hold, vol-targeted BAH, **OHLCV low-vol / size cross-sectional factor**
(subsumption test), equal-weight universe, **shuffled-IV placebo**, **random-rank
placebo**, inverted-signal sanity.

### 11.5 Validation
ValidationPipeline; PBO / DSR; bootstrap CI; cost stress (2×) + **turnover-explicit
cost**; 1-bar delay; crisis exclusion (2020, 2022); concentration; placebo (shuffled-IV,
random-rank); attribution by feature. research_only; next-day execution enforced.

### 11.6 Decision rule
- If the §4.2 equity-return audit FAILS → run only the SPY/QQQ secondary track.
- Classify any failure: cost_wall / low_vol_or_size_restatement / survivorship /
  short_history / placebo_indistinguishable / subsumed_by_vol_targeting.
- research_only throughout; a strong result motivates a paid-chain follow-up only.

## What happens after this intake

1. This document is committed to `docs/research/intake/`.
2. Run the **§4.2 equity-return data audit** (gate for the cross-sectional track).
3. Build the EOD→next-day feature layer + frozen variants + baselines; run the
   validation battery; emit registry + validation + placebo + failure reports.
4. Commit the status classification. No re-tuning after holdout.
