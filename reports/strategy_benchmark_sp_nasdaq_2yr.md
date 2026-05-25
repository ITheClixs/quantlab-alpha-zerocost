# Strategy Benchmark Report — S&P / Nasdaq Futures & ETFs

_Generated: 2026-05-25T16:42:21+00:00_

> ⚠️ **Benchmark only — not investment advice.** This report measures what is achievable across 1500 systematically-enumerated quant strategies on free daily data over 24 months, and applies Bailey/López de Prado PBO + Deflated Sharpe to expose multiple-testing bias.

## 1. Setup

- **Date range:** 2024-05-28 to 2026-05-22 (~24 months, 501 trading days)
- **Strategies tested:** 1500 = 5 universes × 15 signal families × 4 lookbacks × 5 thresholds
- **Cost model:** 1.5 bps per side (commission + half-spread). No market impact, no overnight financing.
- **Wall-clock:** 10.6 s end-to-end (strategy generation + 1500 backtests + 12,870-combination PBO + DSR).
- **Universes:**
  - `ES_F`: S&P 500 e-mini front-month
  - `NQ_F`: Nasdaq 100 e-mini front-month
  - `SPY`: S&P 500 ETF cash proxy
  - `QQQ`: Nasdaq 100 ETF cash proxy
  - `EW_BASKET`: Equal-weighted basket of the four

## 2. Signal-family lineage

Every signal family has peer-reviewed quant lineage; retail patterns (ICT, Smart-Money order blocks, etc.) were intentionally excluded because they do not survive walk-forward backtests in any published study I could find.

| family | reference |
|---|---|
| `TS_MOMENTUM` | Moskowitz, Ooi, Pedersen 2012 — Time Series Momentum |
| `LAGGED_MOMENTUM` | Jegadeesh & Titman 1993 — 12-1 momentum (skip-1) |
| `MA_CROSSOVER` | Faber 2007 — A Quantitative Approach to Tactical Asset Allocation |
| `DONCHIAN_BREAKOUT` | Donchian / Turtle Traders (Dennis & Eckhardt 1983) |
| `BOLLINGER_REVERT` | Bollinger 1980s — band mean-reversion |
| `BOLLINGER_BREAKOUT` | Bollinger 1980s — band breakout |
| `RSI_MEANREVERT` | Wilder 1978 — Relative Strength Index |
| `MACD` | Appel 1979 — Moving-Average Convergence Divergence |
| `VOLTGT_MOMENTUM` | Hurst, Ooi, Pedersen 2017 — vol-targeted trend |
| `ZSCORE_MEANREVERT` | classical statistical-arbitrage mean reversion |
| `AROON` | Chande 1995 |
| `STOCHASTIC` | Lane 1950s |
| `ROC` | Rate of Change momentum |
| `CCI` | Lambert 1980 — Commodity Channel Index |
| `KELTNER_BREAKOUT` | Keltner / Chester 1960s — ATR channel breakout |

## 3. Headline result — Probability of Backtest Overfitting (PBO)

| metric | value | interpretation |
|---|---:|---|
| **PBO** | **0.9025** | ⚠️ severe overfitting risk — probability the in-sample winner ranks below median OOS |
| Median logit | -1.7887 | negative ⇒ IS winners systematically degrade OOS |
| OOS-failure rate | **0.9106** | fraction of IS/OOS splits where the IS winner had **negative** OOS Sharpe |
| PBO combinations sampled | 12870 | of C(16, 8) = 12870 |
| Submatrix size | 31 days | 16 equal time partitions of the 501-day sample |
| Strategies in pool | 1500 | |

**Reading the result.** A PBO of **90.25%** means that if we were to pick the in-sample best strategy across 1500 candidates and trade it out-of-sample, it would rank **below the median** OOS strategy in 90% of time-partition combinations. This is the formal statement of Bailey & López de Prado's central insight: **with this many trials and this little data, the apparent in-sample winners are statistically indistinguishable from noise.**

## 4. Distribution of in-sample Sharpe across 1500 strategies

  - p 1: -0.847
  - p 5: -0.587
  - p25: -0.224
  - p50: +0.037
  - p75: +0.296
  - p95: +0.703
  - p99: +0.950
  - mean: +0.038
  - std:  0.388

```
  [-1.10, -0.98)  █  5
  [-0.98, -0.87)  ██  8
  [-0.87, -0.76)  ███  11
  [-0.76, -0.65)  ██████████  39
  [-0.65, -0.54)  ██████████  39
  [-0.54, -0.42)  ███████████████████  76
  [-0.42, -0.31)  ██████████████████████████  107
  [-0.31, -0.20)  ██████████████████████████████  122
  [-0.20, -0.09)  ████████████████████████████████████████  163
  [-0.09, +0.02)  ████████████████████████████████████████  161
  [+0.02, +0.13)  ██████████████████████████████████████  154
  [+0.13, +0.25)  ███████████████████████████████████████  160
  [+0.25, +0.36)  ████████████████████████████████████████  161
  [+0.36, +0.47)  ██████████████████████████  104
  [+0.47, +0.58)  ████████████████  65
  [+0.58, +0.69)  ██████████  42
  [+0.69, +0.80)  ████████████  50
  [+0.80, +0.92)  ███  14
  [+0.92, +1.03)  ██  9
  [+1.03, +1.14)  ██  10
```

## 5. Top-25 by raw Sharpe, with PSR and DSR deflation

**PSR(0)** = probability the strategy's true Sharpe is > 0, accounting for skew + kurtosis but **ignoring multiple-testing**.

**DSR** = the same probability **after** subtracting `E[max Sharpe]` across N=1500 trials under the null. DSR > 0.95 is the conventional threshold for 'real edge'.

| strategy_id | Sharpe | Sortino | total ret | MaxDD | trades | turnover/y | PSR(0) | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NQ_F|LAGGED_MOMENTUM|L120|T1.50` | +1.14 | +1.38 | +16.44% | -7.32% | 18 | 9.96x | 0.949 | **0.405** |
| `NQ_F|LAGGED_MOMENTUM|L120|T2.50` | +1.14 | +1.38 | +28.20% | -12.19% | 18 | 16.60x | 0.949 | **0.405** |
| `NQ_F|LAGGED_MOMENTUM|L120|T0.50` | +1.14 | +1.38 | +5.32% | -2.44% | 18 | 3.32x | 0.949 | **0.405** |
| `NQ_F|LAGGED_MOMENTUM|L120|T1.00` | +1.14 | +1.38 | +10.80% | -4.88% | 18 | 6.64x | 0.949 | **0.405** |
| `NQ_F|LAGGED_MOMENTUM|L120|T2.00` | +1.14 | +1.38 | +22.24% | -9.76% | 18 | 13.28x | 0.949 | **0.405** |
| `QQQ|LAGGED_MOMENTUM|L120|T0.50` | +1.06 | +1.28 | +4.92% | -2.65% | 15 | 2.93x | 0.935 | **0.364** |
| `QQQ|LAGGED_MOMENTUM|L120|T1.00` | +1.06 | +1.28 | +9.96% | -5.29% | 15 | 5.86x | 0.935 | **0.364** |
| `QQQ|LAGGED_MOMENTUM|L120|T2.00` | +1.06 | +1.28 | +20.41% | -10.53% | 15 | 11.72x | 0.935 | **0.364** |
| `QQQ|LAGGED_MOMENTUM|L120|T1.50` | +1.06 | +1.28 | +15.13% | -7.92% | 15 | 8.79x | 0.935 | **0.364** |
| `QQQ|LAGGED_MOMENTUM|L120|T2.50` | +1.06 | +1.28 | +25.80% | -13.13% | 15 | 14.65x | 0.935 | **0.364** |
| `NQ_F|BOLLINGER_REVERT|L60|T2.50` | +1.01 | +0.90 | +19.80% | -2.78% | 26 | 13.08x | 0.995 | **0.219** |
| `NQ_F|ZSCORE_MEANREVERT|L60|T2.50` | +1.01 | +0.90 | +19.80% | -2.78% | 26 | 13.08x | 0.995 | **0.219** |
| `SPY|BOLLINGER_REVERT|L10|T2.50` | +0.96 | +5.32 | +2.58% | -0.06% | 6 | 3.03x | 0.996 | **0.172** |
| `SPY|ZSCORE_MEANREVERT|L10|T2.50` | +0.96 | +5.32 | +2.58% | -0.06% | 6 | 3.03x | 0.996 | **0.172** |
| `ES_F|BOLLINGER_REVERT|L10|T2.50` | +0.95 | +2.70 | +2.55% | -0.09% | 6 | 3.02x | 0.995 | **0.169** |
| `ES_F|ZSCORE_MEANREVERT|L10|T2.50` | +0.95 | +2.70 | +2.55% | -0.09% | 6 | 3.02x | 0.995 | **0.169** |
| `QQQ|BOLLINGER_REVERT|L60|T2.50` | +0.95 | +0.88 | +18.65% | -2.73% | 24 | 12.12x | 0.991 | **0.185** |
| `QQQ|ZSCORE_MEANREVERT|L60|T2.50` | +0.95 | +0.88 | +18.65% | -2.73% | 24 | 12.12x | 0.991 | **0.185** |
| `ES_F|DONCHIAN_BREAKOUT|L60|T1.50` | +0.93 | +0.97 | +9.15% | -1.48% | 50 | 25.15x | 0.983 | **0.198** |
| `ES_F|DONCHIAN_BREAKOUT|L120|T1.50` | +0.92 | +0.98 | +8.56% | -1.48% | 34 | 17.10x | 0.989 | **0.163** |
| `SPY|VOLTGT_MOMENTUM|L60|T2.50` | +0.89 | +1.04 | +18.65% | -6.96% | 404 | 10.25x | 0.890 | **0.284** |
| `SPY|AROON|L120|T2.00` | +0.89 | +0.96 | +19.64% | -8.55% | 4 | 2.02x | 0.890 | **0.282** |
| `SPY|VOLTGT_MOMENTUM|L60|T1.50` | +0.89 | +1.02 | +11.11% | -4.20% | 438 | 6.31x | 0.888 | **0.282** |
| `SPY|VOLTGT_MOMENTUM|L60|T0.50` | +0.89 | +1.02 | +3.66% | -1.41% | 438 | 2.10x | 0.888 | **0.282** |
| `SPY|VOLTGT_MOMENTUM|L60|T1.00` | +0.89 | +1.02 | +7.37% | -2.81% | 438 | 4.21x | 0.888 | **0.282** |

**DSR survivors at the 95 % threshold: 0 / 25.** The highest DSR observed in the top-25 is **0.405**, which means even the strongest in-sample candidate is a likely overfit artefact.

## 6. By signal family

| signal_family | n | mean Sharpe | max Sharpe | std Sharpe | mean total_ret | worst MaxDD | mean ann turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LAGGED_MOMENTUM` | 100 | +0.33 | +1.14 | 0.48 | +4.35% | -31.94% | 20.40x |
| `AROON` | 100 | +0.23 | +0.89 | 0.26 | +4.86% | -33.81% | 20.14x |
| `ROC` | 100 | +0.14 | +0.69 | 0.22 | +1.98% | -33.97% | 32.00x |
| `BOLLINGER_REVERT` | 100 | +0.13 | +1.01 | 0.37 | +0.36% | -28.74% | 32.88x |
| `ZSCORE_MEANREVERT` | 100 | +0.13 | +1.01 | 0.37 | +0.36% | -28.74% | 32.88x |
| `TS_MOMENTUM` | 100 | +0.08 | +0.73 | 0.38 | +0.48% | -48.23% | 21.85x |
| `DONCHIAN_BREAKOUT` | 100 | +0.08 | +0.93 | 0.43 | -0.71% | -23.39% | 36.87x |
| `KELTNER_BREAKOUT` | 100 | +0.07 | +0.78 | 0.41 | +1.82% | -36.53% | 33.55x |
| `VOLTGT_MOMENTUM` | 100 | +0.06 | +0.89 | 0.47 | +0.48% | -32.54% | 16.91x |
| `MA_CROSSOVER` | 100 | +0.00 | +0.52 | 0.26 | -1.55% | -32.61% | 18.00x |
| `CCI` | 100 | -0.00 | +0.67 | 0.32 | -1.62% | -39.01% | 41.71x |
| `MACD` | 100 | -0.01 | +0.35 | 0.30 | -3.46% | -41.77% | 53.98x |
| `RSI_MEANREVERT` | 100 | -0.17 | +0.44 | 0.23 | -8.34% | -32.37% | 34.81x |
| `BOLLINGER_BREAKOUT` | 100 | -0.21 | +0.68 | 0.37 | -4.73% | -35.66% | 32.88x |
| `STOCHASTIC` | 100 | -0.27 | +0.66 | 0.37 | -10.21% | -33.44% | 42.93x |

## 7. By universe

| universe | n | mean Sharpe | max Sharpe | mean total_ret | worst MaxDD |
|---|---:|---:|---:|---:|---:|
| `ES_F` | 300 | +0.08 | +0.95 | -0.41% | -35.59% |
| `SPY` | 300 | +0.06 | +0.96 | -0.41% | -31.94% |
| `QQQ` | 300 | +0.05 | +1.06 | -0.77% | -41.77% |
| `NQ_F` | 300 | +0.01 | +1.14 | -1.88% | -48.23% |
| `EW_BASKET` | 300 | -0.01 | +0.88 | -1.85% | -36.53% |

## 8. Best strategy — monthly returns

For completeness, monthly returns of the **single best raw-Sharpe** strategy (`NQ_F|LAGGED_MOMENTUM|L120|T1.50`, Sharpe = +1.14). Note the DSR penalty: even this strategy has DSR = 0.405 and would not survive a rigorous promotion gate.

#### Monthly returns — `NQ_F|LAGGED_MOMENTUM|L120|T1.50`

| month | return |
|---|---:|
| 2024-05 | +0.00% |
| 2024-06 | +0.00% |
| 2024-07 | +0.00% |
| 2024-08 | +0.00% |
| 2024-09 | +0.00% |
| 2024-10 | +0.00% |
| 2024-11 | +0.00% |
| 2024-12 | +0.00% |
| 2025-01 | +0.00% |
| 2025-02 | +0.00% |
| 2025-03 | +0.00% |
| 2025-04 | +0.00% |
| 2025-05 | +1.23% |
| 2025-06 | +4.21% |
| 2025-07 | +1.24% |
| 2025-08 | +0.27% |
| 2025-09 | -1.66% |
| 2025-10 | -2.18% |
| 2025-11 | +4.24% |
| 2025-12 | -1.00% |
| 2026-01 | +0.52% |
| 2026-02 | -1.53% |
| 2026-03 | -2.59% |
| 2026-04 | +9.01% |
| 2026-05 | +4.23% |

Total: **+16.44%**, annualised: **+7.96%**.
## 9. Honest interpretation

- The PBO of **90.2%** says that picking the best-IS strategy from this menu is almost guaranteed to be overfit. **This is the correct answer to the user's question about whether any of these strategies survive out-of-sample.**
- The Sharpe-5 / +10 %-monthly target is **mathematically incompatible** with this data + this strategy menu. The best raw Sharpe achievable on 24 months of S&P/Nasdaq daily data, across 1500 classical strategies, is **1.14** — and even that doesn't survive DSR deflation.
- The few strategies that look superficially attractive (high Sortino, low max-DD) all have **6-30 trades over 500 days**, which is too few to draw any statistical conclusion. Their high Sortino is a small-sample artefact.
- Daily-bar systematic strategies on liquid US equity indices are a saturated research space. Realistic targets after PBO survive are Sharpe 0.5–1.0 net, which is consistent with the Bailey & López de Prado finding that long-running quant signals decay rapidly after publication.

## 10. What would actually move this forward

If the goal is to find robust edge above this benchmark, the directions that have *some* prior literature support are:

1. **Longer history** — extend to 10-20 years so the PBO submatrices have more data and the deflated Sharpe penalty (`E[max SR | N, T]`) is less brutal.
2. **Cross-sectional, not single-asset** — the M4 engine in this repo is purpose-built for dollar-neutral L/S across a basket of names; that's a structurally different bet than single-asset trend.
3. **Higher-frequency** — intraday tick or 1-minute bars would change the strategy space entirely (microstructure, order-flow, queue position).
4. **Walk-forward parameter selection** rather than full-sample parameter fixing — but this only helps if the PBO is run on the *walk-forward* returns, not pre-tuned strategies.
5. **Lower-cost regime** — many of these mean-reversion strategies turn 10-50x/year, and the 1.5 bps round-trip cost is what's killing them. A lower-cost broker / better fill model would lift the boundary.


---
`not_investment_advice: true`
