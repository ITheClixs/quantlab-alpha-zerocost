# Signal-Research Program Review — May 2026

**Date:** 2026-05-29
**Status:** Program-level review; production-deployment pause requested by reviewer
**Author:** QuantLab research
**Audience:** internal review; basis for choosing the next research direction
**Scope:** the eight-iteration signal-research arc from triple-barrier meta-labeled
AvL through HMM single-index v1, inclusive

## 0. Executive summary

Over eight independent backtest iterations on US equities, crypto, and
SPY/QQQ index-level option-implied data, the program produced:

- **0 production candidates**
- **0 paper-trade candidates**
- **0 exception_review_required candidates**
- **research_pass candidates limited to:** the HMM single-index `*_full_dev_*`
  family (6 variants) and the VRP-on-SPY `vrp_long_only` variant (1 variant,
  archived as a research note per the VRP × HMM attribution test).
- **No live trading is authorized.** **No investment-advice product is authorized.**

The durable deliverable from the program is the validation infrastructure
itself, not any single strategy. The infrastructure repeatedly identified
attractive-looking metrics as fragile or noise, and the discipline of the
stricter exception-path gate correctly rejected the strongest static-fit
HMM result on grounds of execution-delay sensitivity and refit instability.

This review consolidates the full arc, summarizes what has been ruled out,
catalogues what has not, and lists the candidate next directions. It does
not propose action. A separate decision point follows.

## 1. OHLCV mechanical alpha search (Iterations 1-6)

Six independent search iterations on price/volume-derived strategies under
hedge-fund-grade cost discipline.

| # | Iteration | Universe | Variants | Best dev Sharpe | Best holdout Sharpe | Verdict | Commit |
|---|---|---|---:|---:|---:|---|---|
| 1 | Triple-Barrier Meta-Labeled AvL (walk-forward) | top-50 SP500 2015-2026 | 1 (after leakage fix) | −1.51 | −1.69 | FAIL | `6cfe784` |
| 2 | Multi-model (AvL / mom-12-1 / GKX / TB-meta AvL) | top-50 SP500 2015-2026 | 4 | +0.26 (mom_12_1) | +0.58 (mom_12_1, later noise) | FAIL | `0661216` |
| 3 | Momentum scale-up (12-1, 12-3, 24-1, vol-scaled, HMM-gated) | top-100/top-200 SP500 2006-2026 | 10 | +0.50 (HMM-gated top-100) | +0.16 (top-100) / +0.59 (top-200) | FAIL | `dd3720a` |
| 4 | Sector-conditional AvL (PCA × z-entry × HMM) | 5 SP500 sectors 2006-2026 | 18 | −0.78 (best AvL) | −1.22 | FAIL | `a114607` |
| 5 | GKX-style LightGBM (label horizons × universes) | top-100/top-200 SP500 2006-2026 | 6 + 6 baselines | −0.25 (best GKX) | −0.37 (best GKX) | FAIL (PBO=0.266 marginal) | `da9a58d` |
| 6 | Crypto top-30 multi-family (independent microstructure) | yfinance USD pairs 2018-2026 | 4 | +0.05 (mom_12_1) | +0.21 (mom_12_1) | FAIL | `5d2f2bf` |

**Pattern-level findings:**

- **Family ordering reproduces across orthogonal microstructures.** On US
  equity and on crypto top-30, the ordering raw_AvL ≪ TB-meta-AvL ≪
  GKX-LGB ≪ mom_12_1 is identical. The negative result is structural, not
  universe-specific.
- **The 2023-2026 holdout regime has a directional bias** of ≈ +0.15-0.60
  Sharpe for any naïve trend signal. PSR_zero on the best instance was
  0.55-0.73; DSR after multi-test deflation went to 0.000. This is the
  reproducible noise floor.
- **Walk-forward training and the dev-only guard caught one real leakage:**
  the in-sample meta-labeling bug in iteration 1 produced dev Sharpe +11.46
  that collapsed to dev −1.51 after the fix. Dev-holdout agreement to
  within 0.2 Sharpe was the smoking gun.
- **HMM as a gate primitive emerged in iteration 3** as a 3.3× dev Sharpe
  lift on momentum (mom_12_1 top-100 dev +0.15 → mom_12_1_hmm_gated dev
  +0.50). The primitive is real even though that specific composition
  failed the holdout gate.

**Conclusion (closed):** price/volume-derived cross-sectional alpha was
not found under realistic costs in any tested universe. See
`docs/research/2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` (commit `8cad7ce`)
for the iteration-level write-up.

## 2. VRP (Phase B + B.γ)

VRP was the first deliberate test of a non-OHLCV information channel,
pre-registered against Bondarenko 2014.

| Test | Result | Reference |
|---|---|---|
| VRP intake (pre-registration) | committed | `docs/research/intake/2026-05-28-vrp-index-v1.md` (commit `c9530ee`) |
| VRP standalone on SPY 2010-2026 | `vrp_long_only` dev Sharpe **+0.90**, holdout **+1.20**, cost-stress 2× **+0.88**, 1-bar delay **+0.79**, bootstrap CI [+0.45, +1.42] | `reports/signal_research/vrp/report.md` (commit `66e263b`) |
| VRP × HMM attribution | best interaction (`vrp_sized_by_hmm_prob`) holdout +1.77 vs HMM-only +1.76 (gap +0.006, threshold +0.25). Orthogonalized VRP residual Sharpe vs HMM = **+0.000**. `vrp_when_hmm_risk_off` dev = **−0.00**. | `reports/signal_research/vrp_hmm_interaction/` (commit `0d00266`) |

**Verdict (closed):**

VRP is empirically real on the SPY 2010-2026 fixture and matches Bondarenko
empirics (0.5-1.0 net Sharpe). It survives every stress test in the
standalone validation. But it does not clear the +1.5 dev-Sharpe gate, and
the orthogonalization test demonstrates that **VRP carries zero incremental
information beyond HMM regime**. The variance risk premium and the HMM
regime classification are detecting the same underlying market state from
two different sides; HMM does it without options data.

The VRP file is closed at **research_note** status. No promotion is
pursued; no follow-up VRP variant is opened.

## 3. HMM single-index v1 (Phase B.γ.3, under exception policy)

The HMM result emerged as the strongest standalone signal in the VRP × HMM
interaction test, which motivated the formal single-index risk-timing
exception policy. The exception policy was drafted, amended (seven
amendments), accepted, and then used to authorize a dedicated HMM
validation.

| Stage | Reference |
|---|---|
| Exception policy accepted (with amendments 1-7) | `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md` (commit `74ca502`) |
| HMM intake pre-registered (30-strategy pool frozen) | `docs/research/intake/2026-05-28-hmm-single-index-v1.md` (commit `ce6e8e3`) |
| §15 implementation (status enum, spec, pipeline, HMM module, cash leg, robustness, runner, CLI, 44 tests) | commits `25ef711`, `8e00e8c` |
| HMM v1 validation results | `reports/signal_research/hmm_single_index/` (commit `61ab215`) |

**Results under the stricter §9 24-criterion gate:**

| Status | Count | Variants |
|---|---:|---|
| `exception_review_required` | **0** | (none) |
| `research_pass` | **6** | `hmm_{2,3,4}_full_dev_spy`, `hmm_{2,3,4}_full_dev_qqq` |
| `none` (failed) | **12 HMM + 12 baselines** | the 12 multi-fit variants (expanding + rolling_5y) and all baselines |

**Best in-pool metric:** `hmm_4_full_dev_qqq` dev Sharpe **+2.36**, holdout
**+2.62**, cost-stress 2× **+2.34**. PBO = **0.000**, DSR = **0.9999**,
PSR_zero = **1.000**. There is no statistical evidence of overfitting in
the pool; the failure is **structural, not statistical**.

**Why no variant reached `exception_review_required`** — four converging
gate failures:

1. **Delay-stress sensitivity.** Every `*_full_dev_*` variant lost more
   than 0.5 Sharpe to 1-bar execution delay (gate ≤ 0.5). The strongest
   variant (`hmm_4_full_dev_qqq`) lost 1.10 Sharpe (+2.36 → +1.26). The
   HMM signal needs same-day execution to capture its edge; realistic
   close→next-open execution already introduces ~1 bar of effective delay.
2. **Refit instability.** 9 of 12 multi-fit variants exceeded the 20%
   economic-identity flip rate (per amendment 3). Higher state counts and
   rolling windows produced flip rates from 25% to 83%. The regime
   structure is not stable enough across reasonable training windows to
   trust live refits.
3. **Vol-targeted-BAH dominance.** 12 of 18 HMM variants failed to beat a
   vol-targeted buy-and-hold baseline on Sharpe AND max drawdown. Most of
   the "HMM advantage" can be replicated by simple vol targeting on the
   underlying without a regime classifier.
4. **Concentration / crisis dependence.** Multiple expanding / rolling
   variants concentrated PnL in single calendar years (one had 57% in a
   single year) and lost their edge when 2020 or 2022 was removed.

**Verdict (closed):**

HMM single-index timing is a **statistically real but operationally
fragile** primitive. Static full-dev fits produce strong dev and holdout
Sharpe; the edge is too sensitive to execution delay; dynamic refits are
not economically stable enough for deployment. **HMM is a research
primitive, not a paper-trade candidate.**

The HMM intake is **closed at research_pass status**. No `exception_review_required`,
no `paper_trade_candidate`, no `production_candidate`. The exception policy
is not amended in response to this result; the policy worked as designed.

## 4. Infrastructure — the durable deliverable

The methodology stack that emerged across the iterations is the strongest
single asset the program produced. Every component repeatedly demonstrated
its value by rejecting attractive but fragile results.

### Validation discipline assets

| Component | Module | Purpose |
|---|---|---|
| `ValidationPipeline` | `signal_research/validation/pipeline.py` | Single entrypoint for vetting any strategy under hedge-fund-grade discipline |
| `ValidationSpec` | `signal_research/validation/spec.py` | Information-source declaration, gates, exception fields |
| Three-tier PBO | `signal_research/methodology/pbo_extensions.py` | Cross-strategy multi-test inflation control |
| Deflated Sharpe (DSR) | `strategy_benchmark/dsr.py` | n_strategies penalty on best-in-pool |
| CPCV / walk-forward | `signal_research/methodology/cpcv.py` | Time-series-safe cross validation |
| Stationary block bootstrap | `signal_research/methodology/bootstrap_ci.py` | Honest CIs on Sharpe |
| Permanent holdout guard | `signal_research/methodology/dev_only_guard.py` | Code-level enforcement of dev-only access |
| Cost decomposition | `signal_research/validation/cost_decomposition.py` | no-cost / fee-only / spread-only / full / 2× |
| Delay stress | `signal_research/validation/delay_stress.py` | 1-bar and 2-bar shift |
| Sanity baselines | `signal_research/validation/sanity.py` | random + inverted signal as first-class strategies |
| Concentration diagnostics | `signal_research/validation/concentration.py` | Monthly / yearly PnL share |
| Failure-class taxonomy | `signal_research/methodology/failure_classifier.py` | 13-category structured failure reporting |
| Selection funnel | `signal_research/methodology/selection_funnel.py` | Per-stage candidate counts |
| Strategy intake protocol | `docs/research/STRATEGY_INTAKE.md` | Pre-registration contract with `InformationSource` enum |
| Validation runbook | `docs/research/VALIDATION_RUNBOOK.md` | Standard invocation and report shape |
| Exception policy (HMM-class) | `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md` | Narrow OHLCV exception with stricter gates |
| Exception-path code surface | `validation/spec.py`, `validation/pipeline.py`, `validation/cash_leg_reporting.py`, `validation/exception_robustness.py`, `strategies/hmm_single_index.py`, `strategies/hmm_runner.py` | Conditional exception path; default rule unchanged |
| 5-tier status enum | `signal_research/status.py` | `NONE → RESEARCH_PASS → EXCEPTION_REVIEW_REQUIRED \| PROMOTION_ELIGIBLE → PAPER_TRADE_CANDIDATE → PRODUCTION_CANDIDATE` |

### Evidence that the discipline worked

- **Iteration 1**: caught the in-sample meta-labeling leakage. Dev Sharpe
  collapsed from +11.46 to −1.51 after the walk-forward fix. The
  anti-leakage assertion is now in the test suite as a structural check.
- **Iteration 3**: caught the +0.58 multi-model flicker as noise when the
  universe widened from top-50 to top-100. The reproducible noise floor
  finding was a direct output.
- **Iteration 5**: PBO = 0.266 (just above the 0.25 gate) correctly flagged
  the GKX variant grid as overfit-on-the-margin.
- **Iteration 6**: cross-microstructure reproduction (crypto top-30
  matched the equity family ordering) provided independent confirmation
  that the family-level result is structural.
- **Phase B**: VRP × HMM interaction test produced the orthogonalization
  residual Sharpe = +0.000 result — the single cleanest piece of evidence
  in the whole arc that two apparently-different signals are detecting the
  same underlying state.
- **Phase B.γ.3**: the 24-criterion exception gate correctly rejected a
  strategy with dev Sharpe +2.36 and DSR 0.9999 on grounds of delay
  sensitivity and refit instability — exactly the failure modes a strict
  policy is supposed to catch.

This is the strongest success of the program: **the system rejects
attractive but fragile results**.

## 5. Current production status

| Tier | Count | Variants |
|---|---:|---|
| `production_candidate` | **0** | (none) |
| `paper_trade_candidate` | **0** | (none) |
| `exception_review_required` | **0** | (none) |
| `research_pass` | **7** | HMM `*_full_dev_*` × {SPY, QQQ} (6), VRP `vrp_long_only` on SPY (1, archived as research note) |
| `none` (failed) | many | the full back-catalogue of failed iterations |

**No live trading is authorized.**
**No paper trading is authorized.**
**No investment-advice product is authorized.**
**No user-facing strategy claims are authorized.**

Per CLAUDE.md §11 and the accepted exception policy §5, all four of the
above require explicit policy action that has not been taken.

## 6. What has been ruled out

The following hypotheses are recorded as closed at this date. Future
proposals against any of them must declare a fundamentally new
information channel or a materially different strategy shape to be
considered (per `docs/research/STRATEGY_INTAKE.md`).

1. **Liquid US large-cap OHLCV cross-sectional alpha.** Six iterations on
   top-50 / top-100 / top-200 SP500 (2006-2026) ruled this out at
   hedge-fund-grade costs.
2. **Crypto top-30 OHLCV cross-sectional alpha.** The independent
   microstructure run replicated the equity family ordering and produced
   no incremental result. The hypothesis "the failure was about US equity
   crowding" was directly tested and rejected.
3. **Avellaneda-Lee-style residual mean reversion on tested universes.**
   Sector-conditional AvL across 5 sectors with 18 variants confirmed
   the AvL family is structurally broken on liquid US equities in this
   regime.
4. **GKX-style OHLCV LightGBM on tested universes.** The 6-variant scale-up
   with 17 OHLCV characteristics produced no variant clearing the gates.
5. **Naïve broad momentum as alpha.** mom_12_1 / mom_12_3 / mom_24_1 / vol-
   scaled / HMM-gated all failed across multiple universes. The +0.15-0.60
   holdout flicker is the regime-bias noise floor, not alpha.
6. **VRP as independent incremental signal over HMM regime gate.** The
   orthogonalization residual Sharpe vs HMM is +0.000. VRP is real but
   redundant.
7. **HMM v1 as deployable under the accepted exception policy.** The HMM
   `*_full_dev_*` family is statistically strong but operationally fragile
   (delay sensitivity + refit instability + vol-targeted-BAH dominance).
   HMM stays at research_pass.

## 7. What has not been ruled out

The arc has not tested or has not exhaustively tested:

1. **Intraday microstructure / order-book alpha.** Tick-level data and
   limit-order-book imbalance signals have not been touched. Different
   time scale, different participants, different information content.
2. **Event-conditioned strategies.** Pre-/post-FOMC, CPI, earnings windows
   as conditioning. Cheap infrastructure addition relative to the existing
   stack. Specific known-time windows where retail/algorithmic interaction
   is distinct from baseline.
3. **Properly timestamped news / sentiment.** The default `STRATEGY_INTAKE.md`
   declares `SENTIMENT_NEWS` and `SENTIMENT_SOCIAL` as legitimate
   information channels if the timestamp audit is solvable. FinBERT was
   declared `research_only_default` in the M6a stub. Not exhaustively
   tested.
4. **Earnings / fundamentals with PIT data.** Requires CRSP / Compustat or
   similar paid feed. The PIT data-quality question is a known unsolved
   item in the current data layer.
5. **Cross-asset macro signals.** Equity ↔ bond / FX / commodity
   correlations as conditioning. Not yet tested.
6. **Options-chain features beyond index-level VIX proxies.** Per-stock
   implied vol surface, term structure, put-call skew, volume.
7. **Futures-specific carry / term-structure signals.** Requires the
   Tier-2 futures audit in the accepted exception policy §1. Not yet
   completed.
8. **HMM v2 with fundamentally different execution convention.** Only
   under a new intake that pre-registers a specific attack on delay
   sensitivity (e.g. signal computed at intraday timestamp, executed
   intraday with a documented cost model) AND refit instability (e.g. a
   regime classifier with a stability constraint baked into the training
   objective rather than evaluated post-hoc). The exception policy itself
   is not amended; HMM v2 must clear the same §3 gates.

## 8. Recommended next research directions

Ranked by the reviewer's stated preference and the cost / expected payoff
profile.

### A. Microstructure / order-book v1 (preferred-if-data-clean)

The strongest candidate if clean L2 (limit-order-book level 2) or
trade-by-trade data is available for at least one Tier-1-comparable liquid
instrument. The data is a fundamentally new information channel
(participant-level pricing intent, not just summarized end-of-day flow).
Infrastructure cost is high (tick storage, event-driven backtest, latency
modeling), but the alpha potential is the highest among the remaining
candidates because retail-equity OHLCV is the most crowded segment of the
search space.

**Decision contingency:** if a clean L2 or trade-data feed is available
for SPY, QQQ, or top-5 mega-cap names, proceed with this direction.
Otherwise, defer.

### B. Event-conditioned macro / earnings-calendar strategies (fallback)

Materially cheaper to test than microstructure. The infrastructure
addition is small: an event calendar (FOMC dates from FRED / Federal
Reserve API, CPI release dates from BLS, earnings dates from a free
aggregator). Strategies condition on event-window proximity using the
existing OHLCV bars feed.

The strict bar from the program: this is OHLCV plus a known timestamp
schedule. The information channel is the event timing itself. Promotion
under the default rule requires `InformationSource.EVENT_WINDOW` in the
intake. The default rule's no-OHLCV-only-promotion language applies; the
event channel must be declared and must drive the hypothesis.

### C. FinBERT / news (conditional on timestamp audit)

Sentiment is a real candidate but the 10-criterion FinBERT audit (per
M6a in the original spec) is hostile. The audit covers timestamp
normalization, deduplication, ticker mapping, source provenance,
look-ahead from headlines that summarize after-the-fact events, etc. The
infrastructure cost is medium; the audit cost is the binding constraint.

**Decision contingency:** open this direction only if a credible
timestamp audit pathway exists (e.g. a structured news feed with explicit
event timestamps separated from publication timestamps).

### D. Options-chain features beyond VIX proxies

The Phase B VRP work touched index-level VIX-family series. Per-stock
implied vol, vol surface skew, put-call ratios, options volume, are
genuinely different from spot OHLCV. Infrastructure cost is high if a
free feed cannot be sourced; the CBOE DataShop and OptionMetrics are paid.

### E. Futures carry / term-structure

The accepted exception policy §1 lists ES / NQ as Tier-2 instruments
gated behind a separate audit. The audit covers roll convention, roll
cost, margin and overnight financing, and a continuous-contract
construction documented in the manifest. The infrastructure cost is
medium; the analytical content is well-understood (term-structure carry,
basis trades).

### F. HMM v2

Only under a new intake that pre-registers attacks on delay sensitivity
and refit instability simultaneously. The §15 implementation work is
already done and is reusable. The strategy itself must be materially
different — for example: intraday signal computation with a documented
intraday execution path, or a regime classifier whose training objective
explicitly penalizes refit instability. The exception policy itself is
not amended; HMM v2 must clear the same §3 gates.

## 9. Explicit non-action

These items are recorded as out of scope until further notice:

9.1. **Do not continue broad OHLCV model searches.** The six prior
iterations are closed. New OHLCV-only proposals will be rejected under
the default rule.

9.2. **Do not run deep models (M5 Lim/Zohren LSTM, Wood/Zohren Momentum
Transformer) on the same OHLCV information set.** The GKX result already
demonstrated that nonlinear OHLCV models do not find what linear OHLCV
models cannot. Deep models with the same inputs are not authorized.

9.3. **Do not lower the §3 gate thresholds.** Specifically:
- dev Sharpe ≥ 1.5 (default) and dev AND holdout ≥ 1.5 (exception path)
- bootstrap CI lower > 0.5 (exception path)
- 2× AND 3× cost stress
- 1-bar AND 2-bar delay stress
- max DD ≥ −20% or Calmar > 1.0
- PBO ≤ 0.25
- DSR ≥ 0.5

None of these are negotiable on the basis of a near-miss.

9.4. **Do not promote HMM.** HMM v1 is closed at research_pass. HMM v2
requires a fresh intake with a substantively different hypothesis.

9.5. **Do not paper trade anything yet.** No strategy has reached
`exception_review_required`, which is a prerequisite for `paper_trade_candidate`
under both the default and exception paths. Paper trading is not authorized.

9.6. **Do not weaken the delay-stress gate.** The delay sensitivity is
the dominant HMM failure mode. Weakening this gate would directly
contradict the reviewer's recorded conclusion that the HMM primitive is
"too sensitive to execution delay" for deployment.

9.7. **Do not amend the exception policy in response to this result.**
The policy was accepted with seven amendments and applied to the HMM v1
validation. Post-hoc amendment in response to the result violates the
"no rule-changing after seeing results" non-goal in §7.4 of the policy.

9.8. **Do not start HMM v2 immediately.** HMM v2 (per §8.F) is one of
the recommended directions but is not the highest-ranked. The reviewer's
expressed preference is microstructure / order-book v1 if data exists,
event-conditioned otherwise.

## 10. Audit trail

Every iteration's results, intake document, validation report, daily-
returns parquet, and JSON summary are committed under the repository.
The reproducibility guarantees of `VALIDATION_RUNBOOK.md` apply.

| Iteration | Reports directory | Key commit |
|---|---|---|
| 1 — TB-meta-AvL (walk-forward) | `reports/signal_research/triple_barrier_av_lee/focused_walkforward/` | `6cfe784` |
| 2 — Multi-model fixture | `reports/signal_research/multi_model_fixture/focused/` | `0661216` |
| 3 — Momentum scale-up | `reports/signal_research/momentum_scaleup/` | `dd3720a` |
| 4 — Sector-conditional AvL | `reports/signal_research/sector_avl/` | `a114607` |
| 5 — GKX scale-up | `reports/signal_research/gkx_scaleup/` | `da9a58d` |
| 6 — Crypto top-30 multi-model | `reports/signal_research/crypto_multi_model/` | `5d2f2bf` |
| Phase B — VRP standalone | `reports/signal_research/vrp/` | `66e263b` |
| Phase B.γ — VRP × HMM interaction | `reports/signal_research/vrp_hmm_interaction/` | `0d00266` |
| Phase B.γ.3 — HMM v1 validation | `reports/signal_research/hmm_single_index/` | `61ab215` |
| Negative-result research note | `docs/research/2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` | `8cad7ce` |
| Strategy intake protocol | `docs/research/STRATEGY_INTAKE.md` | `748a5ec` |
| Validation runbook | `docs/research/VALIDATION_RUNBOOK.md` | `748a5ec` |
| Single-index risk-timing exception policy | `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md` | `74ca502` |
| VRP intake | `docs/research/intake/2026-05-28-vrp-index-v1.md` | `c9530ee` |
| HMM single-index v1 intake | `docs/research/intake/2026-05-28-hmm-single-index-v1.md` | `ce6e8e3` |
| Status / spec / pipeline exception-path code | `signal_research/status.py`, `signal_research/validation/spec.py`, `signal_research/validation/pipeline.py` | `25ef711` |
| §15 HMM strategy + cash leg + robustness + runner + 44 tests | `signal_research/strategies/`, `signal_research/validation/cash_leg_reporting.py`, `signal_research/validation/exception_robustness.py`, `scripts/run_hmm_single_index_v1.py`, `tests/signal_research/strategies/test_hmm_exception_path.py` | `8e00e8c` |

## 11. Closing note

The program produced no deployable alpha. The program produced a
production-grade validation infrastructure that correctly rejected every
strategy proposed under it, plus seven `research_pass` results that
document known-strong-but-not-deployable primitives. The infrastructure
is the deliverable. The intake protocol, the exception policy, the
24-criterion exception gate, the cash-leg reporting, the economic-identity
stability check, the 44 tests covering the exception-path code — all of
these will outlast any single strategy proposal and will be the
substrate for whatever direction the next research program takes.

The next decision point is the choice of direction per §8. The reviewer's
expressed preference is microstructure / order-book v1 if clean data
exists, with event-conditioned macro/earnings-calendar strategies as the
fallback. No code work begins until the direction is chosen and an intake
is committed.
