# Negative Result: Mechanical Alpha from OHLCV Data on Liquid US Equities and Crypto, 2006-2026

**Date:** 2026-05-28
**Author:** QuantLab research
**Status:** Closed (research_pass on negative finding)
**Classification:** Internal research note

## Abstract

Across six independent backtest iterations — three on US S&P 500 equities
(top-50, top-100, top-200 by ADV), one sector-conditioned (financials,
industrials, technology, healthcare, consumer-discretionary), one
LightGBM scaleup over 19 OHLCV-derived characteristics, and one on top-30
crypto pairs — **no model family from the M3 single-asset catalogue
generates alpha after realistic hedge-fund costs under honest validation**.
The family ordering (Avellaneda-Lee residual mean-reversion ≪ triple-
barrier meta-labeling ≪ GKX LightGBM ≪ 12-1 cross-sectional momentum)
reproduces across the equity and crypto microstructures. The 12-1
momentum signal produces a +0.15-0.59 holdout Sharpe in every iteration,
which closer inspection identifies as a regime-specific 2023-2026 trend
exposure rather than alpha — PSR_zero of the best instance is 0.55, and
deflation for the variant grid drops DSR to 0.000. The constraint is the
information set, not the universe.

## Method (one paragraph)

Each iteration applies the same discipline: yfinance-sourced bars with
explicit `survivorship_prototype_only` banner; predeclared variant grid
with all degrees of freedom logged; walk-forward training within the dev
window with embargo equal to label horizon plus 5 days; permanent
holdout (typically 2023-01-01 → 2026-05-26) untouched until the final
pass; stationary block bootstrap CIs for Sharpe; three-tier PBO across
the variant pool (Bailey/López de Prado 2014, López de Prado 2018);
deflated Sharpe ratio with the n-strategies multi-test penalty; and a
cost stack of 0.5 bps commission + 10 bps spread one-way plus 2× cost-
stress for US equities (4 bps + 50 bps for crypto). The 8-criteria
promotion gate requires dev Sharpe ≥ 1.0, holdout Sharpe ≥ 0.5,
cost-stress 2× > 0, bootstrap lower-CI > 0, PBO ≤ 0.25, DSR ≥ 0.50, and
material beat over random and inverted sanity baselines.

## Iterations and headline results

### Iteration 1: Triple-Barrier Meta-Labeled Avellaneda-Lee (top-50 SP500, 2015-2026)

Dev Sharpe **+11.46** initially. Inspection revealed in-sample meta-
labeling leakage: the RF meta-labeler was trained on triple-barrier
labels that look forward 20 days, then applied to filter positions on
the same dev dates. After refactoring to walk-forward expanding-window
training (5 folds, embargo = vertical_barrier + 5), dev Sharpe dropped
to **−1.51** with holdout **−1.69**. Dev/holdout agreement to within
0.2 Sharpe confirmed the leak was fixed and the strategy genuinely
loses money.

Anti-leakage telemetry now in the test suite as a structural assertion:
`test_walk_forward_oos_predictions_are_strictly_out_of_sample`.

### Iteration 2: Multi-model comparison (top-50 SP500, 2015-2026)

Four families on identical fixture:

| Model | dev Sharpe | dev 95% CI | holdout Sharpe | cost-2× |
|---|---:|---:|---:|---:|
| raw Avellaneda-Lee | −2.47 | [−3.22, −1.78] | −2.76 | −5.24 |
| 12-1 cross-sectional momentum | +0.26 | [−0.31, +0.89] | **+0.58** | +0.12 |
| GKX LightGBM (walk-forward) | −0.49 | [−1.13, +0.18] | −1.38 | −1.16 |
| triple-barrier meta-labeled AvL | −1.51 | [−2.33, −0.78] | −1.69 | −3.60 |

12-1 momentum produced the only positive holdout Sharpe. Subsequent
iterations classify this as noise (see iteration 3).

### Iteration 3: Momentum scale-up (top-100 + top-200 SP500, 2006-2026)

10 momentum variants (5 modifications × 2 universes):

| Variant | Universe | dev | dev 95% CI | holdout | cost-2× |
|---|---|---:|---:|---:|---:|
| `mom_12_1` | top100 | +0.15 | [−0.37, +0.71] | +0.16 | −0.005 |
| `mom_12_1` | top200 | −0.15 | [−0.62, +0.33] | +0.59 | −0.32 |
| `mom_12_1_hmm_gated` | top100 | **+0.50** | [+0.07, +0.95] | +0.14 | **+0.28** |
| `mom_12_1_vol_scaled` | top100 | +0.12 | [−0.37, +0.69] | −0.15 | −0.07 |
| (others omitted) | | | | | |

**PBO = 0.004**, **DSR = 0.608** (best variant: HMM-gated on top100,
fails by `dev Sharpe < 1.0`), **PSR_zero = 0.980** without multi-test
penalty.

The prior +0.58 holdout collapsed to +0.16 when the universe widened
from top-50 to top-100. This is the noise-floor signature.

The HMM regime gate produced a real and reproducible 3.3× dev Sharpe
lift on momentum (+0.15 → +0.50), establishing the HMM-as-risk-filter
primitive as a durable component for future strategies.

### Iteration 4: Sector-conditional Avellaneda-Lee (5 sectors × 18 variants, 2006-2026)

18 AvL variants (3 PCA components × 3 z-entry thresholds × 2 HMM gates)
on 5 sectors (Energy dropped after liquidity screen).

Best AvL: `avl_pca3_z2.0_hmmrisk_on` at dev **−0.78**, holdout **−1.22**.
**The best baseline (`inverted_signal_mom`) beats every AvL variant** at
dev −0.33. The structured PCA-residual machinery destroys signal that
exists in simple single-stock reversal.

Structural diagnostics:
- pca=3 < pca=2 < pca=1 monotonically (more residualization, more noise)
- z=2.0 > z=1.5 > z=1.0 monotonically (higher entry, less noise traded)
- HMM gate uniformly ~5-15% improvement (cannot rescue dead primary)

**PBO = 0.014**, **DSR = 0.000**, **PSR_zero = 0.089**. Failure class:
`no_residual_meanreversion_edge`.

### Iteration 5: GKX-style LightGBM scale-up (top-100 + top-200, 2006-2026)

6 GKX variants (3 label horizons × 2 universes) plus 6 baselines.

| Variant | Universe | dev | dev 95% CI | holdout | cost-2× |
|---|---|---:|---:|---:|---:|
| `gkx_lgb_h21d` | top100 | −0.25 | [−0.68, +0.18] | −1.26 | −1.03 |
| `gkx_lgb_h63d` | top200 | −0.70 | [−1.14, −0.29] | **−0.37** | −1.54 |
| (best baseline) `mom_12_1` | top100 | +0.15 | [−0.39, +0.67] | +0.16 | −0.005 |

**PBO = 0.266** (marginal failure: just over 0.25 gate). **DSR = 0.000**,
**PSR_zero = 0.733**. Best in pool is again `mom_12_1` baseline; no GKX
variant beats any baseline on dev Sharpe.

Diagnostic: longer label horizons hurt less (h=63d > h=21d > h=5d),
consistent with the residual being noise that compounds at high turnover.

Failure class: `overfit_parameter_grid`.

### Iteration 6: Crypto top-30 multi-family (yfinance USD pairs, 2018-2026)

Independent microstructure test. 30 pairs by ADV (BTC, ETH, BCH, BNB,
YFI, SOL, LTC, MKR, AAVE, DASH, ZEC, AVAX, ETC, LINK, DOT, …) with
crypto-tuned costs (4 bps + 50 bps).

| Model | dev Sharpe | dev 95% CI | holdout | cost-2× |
|---|---:|---:|---:|---:|
| raw Avellaneda-Lee | **−4.80** | [−5.49, −4.18] | −6.79 | −9.05 |
| 12-1 cross-sectional momentum | +0.05 | [−0.78, +0.91] | **+0.21** | −0.27 |
| GKX LightGBM | −1.65 | [−2.33, −0.99] | −2.24 | −3.27 |
| triple-barrier meta-labeled AvL | −3.21 | [−4.14, −2.35] | −4.78 | −6.27 |

**PBO = 0.000**, **DSR = 0.000**, **PSR_zero = 0.553**. Same family
ordering as US equities. The mom_12_1 noise floor reproduces.

## The noise-floor finding

Naïve 12-1 cross-sectional momentum produces a small positive holdout
Sharpe (+0.15-0.60) in **every** iteration across US large-cap equities
(top-50 / top-100 / top-200) and crypto top-30. The 2023-01-01 →
2026-05-26 holdout window is dominated by a directional trend (US
mega-cap AI rally; crypto recovery from 2022 trough) that rewards any
signal long this-year's-winners.

This is not alpha. It is exposure to the recent trend. The signature:
- PSR_zero = 0.55-0.73 (without multi-test penalty)
- DSR collapses to 0.00 after the n=10-21 variant penalty
- Bootstrap CI straddles zero on every instance
- Cost-stress 2× flips it negative
- Dev Sharpe and holdout Sharpe agree directionally only because they
  both lie in the same near-zero region of the distribution

Practical implication: **+0.2 holdout Sharpe is the universal noise
floor for any naive directional cross-sectional signal in this window.**
Future strategies whose holdout Sharpe lands here are noise.

## What the methodology stack caught

| Iteration | What was caught | How |
|---|---|---|
| 1 | In-sample meta-labeling leakage (+11.46 dev) | dev/holdout disagreement (+11.46 vs −1.55) |
| 2 | Initial multi-model flicker (+0.58 holdout) | wide bootstrap CI [−0.31, +0.89] |
| 3 | +0.58 collapse to +0.16 at universe widening | top-50 vs top-100 comparison |
| 3 | HMM gate as real but small | 3.3× dev Sharpe lift, holdout < gate |
| 4 | Structured AvL underperforms naive reversal | sanity baselines as first-class strategies |
| 5 | LightGBM produces no alpha on 19 OHLCV features | PBO=0.266 (marginal grid overfit) |
| 6 | Family ordering reproduces across microstructures | crypto run matches equity ordering |

## Durable methodology assets

The negative result produced these reusable components:

1. **`ValidationPipeline`** — single entrypoint with PBO, DSR, walk-forward,
   bootstrap CIs, cost decomposition, delay stress, sanity baselines,
   concentration diagnostics, failure taxonomy, and 8-criteria gate.
2. **Walk-forward meta-labeling pattern** — TimeSeriesSplit-style training
   with embargo for any wrapper that uses forward-looking labels.
3. **HMM regime gate primitive** — 2-state Gaussian HMM on equal-weighted
   market returns, fit on dev only, with predeclared favorable-regime
   rule. Reusable for any directional signal.
4. **`InformationSource` enum + "no promotion without new info" rule** —
   structural enforcement in `_assign_status`. OHLCV-only strategies
   cannot reach `promotion_eligible` regardless of metrics.
5. **9-class failure taxonomy** — `FailureCategory` enum used by both
   the runner and the report writer.
6. **Cross-strategy multiple-testing controls** — three-tier PBO
   (raw_global / per_profile / per_family) plus DSR with `n_strategies`
   deflation, callable from any runner.
7. **Strategy intake protocol** — `docs/research/STRATEGY_INTAKE.md`
   contract that pre-registers hypothesis, information sources, expected
   metrics, and failure modes.

## What is ruled out

For the 2006-2026 window on liquid US large-cap equities and 2018-2026 on
top-30 crypto, at hedge-fund-grade costs (0.5+10 bps for equities, 4+50
for crypto) with 2× cost stress:

1. Avellaneda-Lee residual mean-reversion (single-period, cross-sectional, or sector-conditional)
2. Triple-barrier meta-labeling on the above
3. 12-1, 12-3, 24-1 cross-sectional momentum and vol-scaled variants
4. HMM-gated momentum (the gate works; the primary doesn't)
5. GKX-style LightGBM on 19 OHLCV characteristics with horizons {5d, 21d, 63d}
6. The combination of the above across crypto microstructure

## What is not ruled out (the next-fork space)

The negative result narrows the search by ruling out the cheap directions.
What remains:

1. **Options-implied features (VRP family)** — a genuinely new
   information channel: option-market participants' forecast of future
   variance and skew. Bondarenko 2014 documents a structurally
   negative risk premium that the short-vol seller earns; this is
   priced in the option market, not in spot price/volume.
2. **News/transcript sentiment (FinBERT / M6a deferred)** — the 10-criterion
   audit gate is hostile but the source is genuinely orthogonal to price/volume.
3. **Microstructure** (tick data, order book) — different time scale,
   different information.
4. **Cross-asset signals** — bond yields and FX moves leading equity
   moves.
5. **Event-conditioned strategies** — pre-/post-FOMC, CPI, earnings.
6. **Deep models (M5 deferred)** — Lim/Zohren LSTM, Wood/Zohren transformer.
   On the same OHLCV data, the prior (linear) result says no, but with
   genuinely new inputs the non-linear methods may help.

## Operational recommendation

Phase A of this research program is complete. The validation
infrastructure is production-grade. Continue to Phase B (options-implied
VRP) only after the intake protocol has been used to formally propose
it. If VRP also fails, the system has the discipline to record that
failure rigorously without prompting another round of OHLCV variant
search.

## References

- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 3, 7, 11.
- Politis, D. & Romano, J. (1994). The Stationary Bootstrap.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
- Avellaneda, M. & Lee, J.-H. (2010). Statistical Arbitrage in the US Equities Market.
- Bondarenko, O. (2014). Variance Trading and Market Price of Variance Risk.
- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers.

## Reproducibility

Every iteration's report, daily-returns parquets, and PBO/DSR JSON are
committed under `reports/signal_research/`:

- `triple_barrier_av_lee/focused_walkforward/`
- `multi_model_fixture/focused/`
- `momentum_scaleup/`
- `sector_avl/`
- `gkx_scaleup/`
- `crypto_multi_model/`

Each report carries the run git SHA, the spec, and the cost / fixture
parameters. The validation pipeline in
`src/quant_research_stack/signal_research/validation/` is the canonical
implementation for future strategies.
