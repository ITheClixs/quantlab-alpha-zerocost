# Intake — Strategy-Zoo Backtest-Overfitting Demonstration v1 (~100k strategies)

**Date:** 2026-06-04
**Status:** OPEN — research_only. **No paper. No live.** No promotion language. This is a
*methodology demonstration*, not a search for a deployable strategy.
**Branch / artifacts root:** `reports/signal_research/strategy_zoo_overfitting_v1/`
**Builds on:** the existing `strategy_benchmark/` framework (1,500-strategy run,
`reports/strategy_benchmark_sp_nasdaq_2yr.md`) and the §6 derivations in `README.md`
(expected-max Sharpe, McLean–Pontiff, Bailey–López de Prado).
**Data audit:** PENDING — `reports/signal_research/strategy_zoo_overfitting_v1/data_audit.md`.

## 1. Thesis (what we are proving, and the trap we are NOT falling into)

If you backtest enough strategies, the **best in-sample performer is guaranteed to look
excellent by pure chance**. For `N` zero-skill strategies the expected maximum Sharpe grows
as `E[max] ≈ √(2 ln N)` (Bailey–López de Prado): at `N=100,000`, `√(2 ln N) ≈ 4.8` standard
errors above the null mean. This branch runs a **~100,000-strategy grid** and demonstrates,
empirically, the four claims below — turning the §6 *theory* into a *measured* result on
real backtests:

1. The **distribution** of in-sample annualized Sharpe across the zoo matches a noise-plus-
   small-drift distribution; the **best** is large but exactly where chance predicts.
2. **Out-of-sample collapses.** The in-sample top strategies have near-zero (or negative)
   net-of-cost Sharpe on a purged held-out period.
3. **PBO (CSCV) ≈ 1** and the **Deflated Sharpe** passes ≈ 0 strategies once `SR_0` is
   inflated for the number of trials.
4. A **permutation/null control** (independently shuffle each strategy's returns in time)
   produces a "best strategy" statistically indistinguishable from the real run — proving
   the apparent winners are selection artifacts, not signal.

**The trap we explicitly avoid:** surfacing "the best of 100k" as a recommendation is the
exact data-mining fallacy this program exists to debunk. No strategy from this run is a
candidate for anything. The deliverable is the *demonstration* and the *reusable harness*.

## 2. Relation to prior work / scope

Reuses `strategy_benchmark/`: `signals.SIGNAL_FAMILIES`, `enumeration.enumerate_strategies`,
`backtest.run_single_asset_backtest`, `runner.run_benchmark`, `pbo.compute_pbo`,
`dsr.compute_dsr`/`expected_max_sharpe`, `runner.deflate_top_strategies`. **New work:** more
**cited single-asset signal families**; a **configurable grid** (extra axes); a **purged
walk-forward OOS split**; a **permutation null control**; **advanced figures**; scale and
performance to ~100k.

**Scope boundary (YAGNI):** v1 is **single-asset / time-series** families only. Cross-
sectional families (cross-sectional momentum, pairs/cointegration, factor momentum) need a
different backtest engine and are deferred to a **v2** note. The multiple-testing point is
fully made with single-asset families across a large grid.

## 3. Reaching ~100,000 honestly (grid, not 100k hand-coded ideas)

Strategies are **templates parameterized across a grid**. Target grid:

| Axis | v1 values | count |
|---|---|---|
| Universe | ~10 (ES_F, NQ_F, SPY, QQQ, IWM, DIA, sector ETFs XLK/XLF/XLE, EW basket) | 10 |
| Signal family | ~32 (15 existing + ~17 sourced, §4) | 32 |
| Lookback | 5, 10, 20, 40, 60, 120, 180, 252 | 8 |
| Threshold | 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 | 8 |
| Volatility estimator | close-to-close, Parkinson (1980), Yang–Zhang (2000) | 3 |
| Position mode | long-only, long-short | 2 |

`10 × 32 × 8 × 8 × 3 × 2 = 122,880` configurations → run a **tiered** sweep
`1k → 10k → 100k` via `--max-strategies`; report the full grid as the headline `N`. The
expected-max-Sharpe prediction is evaluated *per tier* to show the bar rising with `N`.

## 4. Signal families — cited lineage (single-asset)

Existing 15 (see `reports/strategy_benchmark_sp_nasdaq_2yr.md`): TS_MOMENTUM, LAGGED_MOMENTUM,
MA_CROSSOVER, DONCHIAN_BREAKOUT, BOLLINGER_REVERT, BOLLINGER_BREAKOUT, RSI_MEANREVERT, MACD,
VOLTGT_MOMENTUM, ZSCORE_MEANREVERT, AROON, STOCHASTIC, ROC, CCI, KELTNER_BREAKOUT.

Sourced additions (v1), each falsifiable with peer-reviewed or canonical lineage:

| family | reference |
|---|---|
| `VOLMANAGED_MOMENTUM` | Moreira & Muir (2017), *Volatility-Managed Portfolios*, J. Finance 72(4) |
| `TSMOM_SIGN` / `TSMOM_MACROSS` / `TSMOM_BREAKOUT` (trading-rule variants) | Baltas & Kosowski (2013/2020), *Demystifying Time-Series Momentum* |
| `DUAL_MOMENTUM_ABS` | Antonacci (2014), *Dual Momentum Investing* (absolute leg) |
| `ATR_TRAILING_TREND` | Wilder (1978), ATR trailing-stop trend |
| `RANGE_OSCILLATOR` | classical support/resistance range trading |
| `VWAP_REVERSION` | reuse of `signal_research/fingerprint_vwap` VWAP-band entry |
| `LOWVOL_TIMING` | Baker, Bradley & Wurgler (2011), low-volatility anomaly |
| `ROLLING_SHARPE_MOM` | risk-adjusted (Sharpe-of-returns) momentum |
| `KALMAN_TREND` | state-space/local-level trend filter |
| `TURN_OF_MONTH` | calendar/seasonality anomaly (Ariel 1987) |
| `DRAWDOWN_REVERSION` | mean reversion conditioned on rolling drawdown |
| `EWMA_CROSS` | exponential-MA crossover (RiskMetrics lineage) |
| `PSAR_TREND` | Wilder (1978), Parabolic SAR |
| `VOL_BREAKOUT` | volatility-expansion breakout (range compression → expansion) |
| `MOM_SKIP` | momentum with skip window (echo-effect control) |

Excluded (per the existing report's stance): ICT/Smart-Money/order-block and other
non-falsifiable retail patterns — they do not survive walk-forward in any published study.

## 5. Honest priors (stated before the run)

1. Best in-sample Sharpe at `N≈100k` will be **large (~4–5σ above the null mean)** and
   **meaningless** — chance, not skill.
2. **PBO ≈ 1**, **DSR pass count ≈ 0**, **OOS Sharpe of in-sample winners ≈ 0 or < 0**.
3. The **permutation control** will reproduce the same apparent best Sharpe.
4. A handful of survivors, if any, will be **vol-targeted trend** variants (the one family
   with the most robust published evidence) — and even those will not clear the deflated bar
   net-of-cost on this free-data sample. **Expected verdict: confirms the thesis.**

## 6. Method / pipeline

1. **Data audit (gate, first).** Universes from `strategy_benchmark.data` (cached daily
   bars); verify no missing/duplicate bars, PIT-reasonable membership, corporate-action
   consistency; **split** into in-sample (IS) and a purged, embargoed **held-out OOS** tail.
2. **Enumerate** the configurable grid (§3) → up to ~123k `StrategySpec`s.
3. **Backtest** each on IS via `run_single_asset_backtest` (cost-aware, 1.5 bps/side),
   clustering signal generation by `(family, lookback, threshold, vol_est)` for speed;
   assemble a `float32` `(T, N)` returns matrix (chunked).
4. **In-sample statistics:** Sharpe/Sortino/turnover/maxDD/hit-rate distribution.
5. **Multiple-testing gates:** `compute_pbo` (CSCV) + `compute_dsr` with `SR_0 =
   expected_max_sharpe(observations, trials=N)`; `deflate_top_strategies` on the top-K.
6. **OOS validation:** re-evaluate the IS top-K on the purged OOS tail; measure Sharpe decay.
7. **Permutation null control:** shuffle each strategy's IS returns in time (seeded), re-run
   the max-Sharpe and PBO; compare to the real run.
8. **Figures + report + verdict.**

## 7. Visualizations (advanced; reproducible; embedded in README.md)

Generated by a committed script (deterministic, from the run artifacts):

- **F1 — Sharpe distribution:** histogram of in-sample annualized Sharpe across all `N`
  strategies, with the empirical best marked and the **permutation-null** distribution
  overlaid (kernel density).
- **F2 — Expected-max vs empirical:** the theoretical `E[max]=√(2 ln N)·σ` curve vs the
  **measured** best Sharpe at tiers `N ∈ {1k, 10k, 100k}` (ties directly to README §6.1).
- **F3 — IS→OOS decay scatter:** in-sample Sharpe (x) vs out-of-sample Sharpe (y) for the
  top-K, with the 45° line and a regression — the visual collapse.
- **F4 — Overfitting panel:** PBO logit-rank histogram + DSR pass-count bar (≈0) +
  real-vs-permutation best-Sharpe.
- **F5 — Family/lookback heatmap:** median Sharpe by `(family × lookback)` — shows the
  pattern is noise, not a stable cell.

A short **README.md** section ("Empirical proof: 100k strategies, zero survivors") embeds
F1–F5 and links to the report + the competitive-landscape report.

## 8. Gates & pre-committed kill/PASS criteria

This is a *demonstration*; the "PASS" is methodological, not a tradable green light:

- **Demonstration PASS** (expected): PBO ≥ 0.9, DSR pass-count ≤ a handful, OOS Sharpe of IS
  top-K not materially positive net-of-cost, and the permutation control matching — i.e. the
  thesis is empirically confirmed. Write the result note.
- **Surprise branch** (if a family *does* clear PBO **and** Deflated Sharpe **and** holds
  net-of-cost on the purged OOS): do **not** celebrate — escalate it to its **own** intake
  with full survivorship/PIT/capacity audit before any further claim. One lucky cell in 100k
  is the null hypothesis until proven otherwise.

## 9. Caution / compute

- `(T, N)` matrix at `T≈2,500`, `N≈123k`, `float32` ≈ **1.2 GB** — chunk by universe; never
  hold all signals in memory at once.
- PBO via CSCV: `C(16,8)=12,870` splits × argmax over `N` — vectorized, fine.
- Tiered `--max-strategies` so 1k/10k validate correctness before the 100k headline run.
- Deterministic seeds for the permutation control; wall-clock budget recorded in the report.

## 10. Deliverables

`reports/signal_research/strategy_zoo_overfitting_v1/`: `data_audit.md`, `metrics.parquet`,
`returns_matrix` (chunked), `top_k_with_dsr.parquet`, `oos_decay.parquet`,
`permutation_control.json`, `report.md`, `VERDICT.md`, and `figures/` (F1–F5). Plus a new
README.md section embedding F1–F5. **Nothing here transfers to Prevalence** — it is a
methodology artifact for the research-paper codebase.

## 11. Non-actions / scope

- No paper, no live, no promotion. Do not present any zoo strategy as a candidate.
- Do not add non-falsifiable retail patterns to inflate `N`.
- Do not expand the grid mid-run to chase a survivor (that is the meta-overfit).
- Cross-sectional families (XS momentum, pairs, factor momentum) are **v2**, out of scope here.

## 12. References

- Bailey & López de Prado (2014), *Deflated Sharpe Ratio*, JPM 40(5); Bailey, Borwein,
  López de Prado & Zhu (2017), *Probability of Backtest Overfitting*, JCF 20(4).
- Moskowitz, Ooi & Pedersen (2012); Moreira & Muir (2017); Baltas & Kosowski (2013/2020);
  Antonacci (2014); Baker, Bradley & Wurgler (2011); Ariel (1987); Wilder (1978).
- Internal: `strategy_benchmark/*`, `README.md` §6, competitive-landscape report
  (`reports/2026-06-02-COMPETITIVE-LANDSCAPE-PUBLIC-QUANT.md`).
