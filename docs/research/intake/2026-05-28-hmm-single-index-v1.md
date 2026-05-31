# Strategy Intake — HMM Single-Index Risk Timing v1

**Date:** 2026-05-28
**Status:** PRE-REGISTRATION (intake submitted; validation run pending)
**Strategy name:** `hmm_single_index_v1`
**Proposer:** QuantLab research
**Promotion intent:** `paper_trade_after_pass` (capped — see §11)
**Invokes exception policy:** YES
**Exception policy reference:** `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md` (ACCEPTED 2026-05-28, commit `74ca502`)
**Default rule status (unchanged):** the no-promotion-without-new-information-source rule in `docs/research/STRATEGY_INTAKE.md` §3 remains intact for all non-exception strategies

## 0. One-paragraph summary

This intake pre-registers a dedicated validation run of a Hidden Markov
Model timing strategy on Tier-1 single-index instruments (SPY and QQQ).
The strategy is long-or-cash on the underlying, conditioned on an HMM
regime classification of the underlying's own OHLCV-derived features.
The run invokes the accepted single-index risk-timing exception policy
(commit `74ca502`); per that policy, this strategy is permitted to
consume OHLCV-only inputs and may be flagged `exception_review_required`
if it clears the stricter §3 gates from the exception. **Acceptance of
the exception policy does not promote HMM.** This intake reserves no
status above `paper_trade_candidate` regardless of metrics; the
`production_candidate` status requires the §5.1 paper-trading workflow
plus the CLAUDE.md §11 stage-promotion review.

## 1. Scope

This validation is limited to **Tier-1 instruments only** (per exception
policy §1):

- **SPY** — SPDR S&P 500 ETF (directly traded, no constituent
  survivorship issue)
- **QQQ** — Invesco Nasdaq-100 ETF (directly traded, no constituent
  survivorship issue)

**Explicitly excluded from this validation:**

- BTCUSDT, ETHUSDT (Tier-2, require crypto venue/funding/data audit
  per exception policy §1)
- ES, NQ futures (Tier-2, require futures roll/financing/cost audit
  per exception policy §1)
- Any cross-sectional basket
- Any individual stock or crypto pair other than the two listed above

A separate Tier-2 audit proposal must be drafted and accepted before
any Tier-2 instrument is added. This intake does not authorize such an
extension.

## 2. Strategy hypothesis

**Hypothesis statement:**

A Hidden Markov Model fitted only on the instrument's own OHLCV-derived
features can identify persistent market regimes characterized by
differing return and volatility distributions. A long-or-cash exposure
rule conditioned on the HMM regime classification may improve
risk-adjusted returns by holding the underlying during favorable
(higher expected return) regimes and reducing exposure during
unfavorable regimes.

**Critical framing:**

- This is **single-index risk timing**, not cross-sectional alpha.
- The mechanism is drawdown avoidance during high-volatility regimes,
  not selection of which name to long.
- The strategy does not require the option market, the news flow, or
  any cross-asset information.
- Absolute returns are expected to be **comparable to or below**
  buy-and-hold; the value claim is risk-adjusted (Sharpe, drawdown,
  Calmar) and crash-protection, not return enhancement.

**Why the hypothesis is plausible:**

The 2026-05-28 VRP × HMM interaction test (commit `0d00266`) found
that an HMM-only gate on SPY 2010-2026 produced dev Sharpe +1.74,
holdout Sharpe +1.76, cost-stress 2× +1.74, bootstrap 95% CI [+1.26,
+2.24], max DD only −8.5%, PSR_zero = 1.000. The orthogonalization
test showed that VRP carries zero incremental information after
controlling for HMM. The HMM result was strong enough on a single
instrument to justify the exception policy that enables this dedicated
validation.

**What would falsify the hypothesis:**

Three pre-registered failure modes (also see §3 gates):

1. **Cross-instrument failure.** The HMM works on SPY but fails on QQQ
   (or vice versa). A regime-timing primitive that works on only one
   liquid index is suspect.
2. **Crisis-window dominance.** The HMM's edge collapses when 2020 or
   2022 is removed. If the dev Sharpe drops from +1.7 to ≤ +0.8 when
   either year is excluded, the strategy is a crisis-detector, not a
   general regime-timer.
3. **State instability.** Across HMM variants (state count × fit
   scheme), the predeclared risk-on identity (§5) flips its economic
   interpretation. If the "favorable" state is not consistent across
   reasonable model choices, the underlying regime structure is too
   weak to be a reliable signal.

## 3. Information source declaration

- `InformationSource.OHLCV`
- `exception_invoked = True`
- `exception_policy = single-index-risk-timing exception, ACCEPTED 2026-05-28 (commit 74ca502)`

**This declaration does not repeal the default no-OHLCV-promotion
rule.** The default rule remains in force for all non-exception
strategies. This intake invokes the narrow exception policy carved out
in `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`,
which permits a single-index timing strategy on a Tier-1 directly-traded
instrument to consume OHLCV-only inputs under stricter gates than the
default 8-criteria gate.

## 4. Instruments and variants (predeclared, frozen)

Run SPY and QQQ **separately**. The validation produces one report per
instrument; the cross-instrument robustness check per exception §4.1
compares both reports.

### 4.1 Predeclared HMM variant grid (9 variants per instrument)

| # | State count | Fit scheme |
|---|---|---|
| 1 | 2 | full-dev fit (single fit on all dev days) |
| 2 | 2 | expanding-window fit (refit annually with all prior dev data) |
| 3 | 2 | rolling 5-year fit (refit annually using prior 5y window only) |
| 4 | 3 | full-dev fit |
| 5 | 3 | expanding-window fit |
| 6 | 3 | rolling 5-year fit |
| 7 | 4 | full-dev fit |
| 8 | 4 | expanding-window fit |
| 9 | 4 | rolling 5-year fit |

**Total: 18 strategy entries in the PBO/DSR pool** (9 variants × 2
instruments).

**No variants may be added after seeing results.** Adding a 5-state HMM
or a 10-year rolling window after observing the metrics is a violation
of the pre-registration. If the run produces ambiguous results, that is
the result; the response is a follow-up intake with a fresh variant
grid, not a quiet expansion of the current one.

### 4.2 Variant identifiers

Each variant is identified by a deterministic name:

```
hmm_<states>_<fit_scheme>_<instrument>
```

Examples:
- `hmm_2_fulldev_spy`
- `hmm_2_expanding_qqq`
- `hmm_4_rolling5y_spy`

The validation registry uses these as the primary keys.

### 4.3 Baselines (required, separate from the 18 variants)

For each instrument, the validation must include:

- `buy_and_hold_<instrument>`
- `vol_targeted_buy_and_hold_<instrument>` (10% annualized vol target
  via trailing 60-day realized vol)
- `sma_50_200_gate_<instrument>` (long when 50-day SMA > 200-day SMA)
- `mom_12_1_<instrument>` (long when 252-day cumulative return minus
  21-day cumulative return > 0)
- `random_signal_<instrument>` (sanity)
- `inverted_signal_of_best_hmm_<instrument>` (sanity, sign-flipped
  version of the best HMM variant by dev Sharpe)

**Baselines × 2 instruments = 12 baseline entries** in the PBO/DSR
pool. **Combined pool: 18 HMM + 12 baselines = 30 strategies.**

## 5. Risk-on state definition (predeclared, per exception policy §4.5)

Per the accepted exception policy (§4.5 amendment 3), HMM state labels
are arbitrary permutations and must not be used directly. The economic
identity of the risk-on state is defined as follows:

### 5.1 Primary rule

```
risk_on_state = argmax over states of (mean log return on the
                fitting window for that state)
```

### 5.2 Tie-breaker

If two states have mean returns within an absolute tolerance of
**0.0001 daily** (~2.5% annualized), the tie is broken by:

```
choose the state with the LOWER realized volatility on the
fitting window (lower σ wins the tie)
```

### 5.3 Multi-state generalization

For 3-state and 4-state HMMs, the risk-on assignment must yield a
single state. The primary rule (argmax mean) selects one state. The
remaining states are classified as `neutral_1`, `neutral_2`,
`risk_off_1` according to a predeclared ordering by decreasing mean
return. The long-or-cash rule treats `risk_on` as the only state in
which the position is taken; all other states are out of market.

### 5.4 Stability requirements

For each refit (expanding-window and rolling-window schemes), the
validation reports:

- whether the predeclared rule places risk-on on a state whose
  (mean, vol) signature matches the previous fit's risk-on state
  (within a tolerance of ±0.0002 daily mean and ±0.005 daily vol)
- the rate of economic-identity flips across refits

A run with > 20% economic-identity-flip rate across its refits will
not satisfy the §3 gate even if metrics otherwise pass — economic
state identity must be stable.

### 5.5 Raw label permutations are explicitly NOT flips

A refit that returns state IDs (0, 1) with mean returns swapped
relative to the previous fit's (0, 1) is **not a flip** if the
predeclared rule correctly identifies the higher-mean state as risk-on
in both fits. This is the central point of the exception policy
§4.5(a)-(b) amendment.

## 6. Features (predeclared, frozen)

Only OHLCV-derived features of the instrument's own bars are permitted.
All features must be computable at signal time using only data
observable at or before close on day t.

### 6.1 Allowed features

| Feature | Window | Definition |
|---|---|---|
| `log_return` | 1 | log(close_t / close_{t-1}) |
| `realized_vol_21` | 21 | sqrt(252) × stdev(log_return) over trailing 21 days |
| `realized_vol_63` | 63 | sqrt(252) × stdev(log_return) over trailing 63 days |
| `drawdown_60` | 60 | (close_t / max(close over trailing 60 days)) − 1 |
| `drawdown_252` | 252 | (close_t / max(close over trailing 252 days)) − 1 |
| `trend_slope_50` | 50 | normalized slope of OLS regression of log(close) on time over trailing 50 days |
| `trend_slope_200` | 200 | same, 200-day window |
| `range_pct_20` | 20 | trailing 20-day mean of (high − low) / close |
| `volume_zscore_20` | 20 | (volume_t − mean(volume, 20)) / std(volume, 20) |

The HMM may consume any subset of these features. The chosen feature
set must be predeclared in the variant registry before the run.

**Default feature set (must be used unless otherwise stated):**
`log_return`, `realized_vol_21`, `drawdown_60`, `range_pct_20`.

### 6.2 Explicitly forbidden features in this run

The following are forbidden in this exception-path validation. None of
them are OHLCV-derived; including any of them would re-introduce the
information-source concern that the default rule was designed to police:

- VIX, VIX9D, VIX3M, VVIX, SKEW, VXN, or any option-implied feature
- VRP or any variant of implied-minus-realized variance
- Macro features (rates, FX, commodities, yield curves)
- Sentiment (news, transcripts, social, FinBERT)
- Earnings, fundamentals, analyst revisions
- Cross-asset features (bond yields, credit spreads, FX)
- Microstructure (tick data, order book imbalance)
- Calendar features beyond instrument-derived ones (no FOMC dates,
  no CPI dates, no earnings windows)

If a future iteration wants to test HMM with any of the above as
conditioning features, it must be submitted as a separate intake
under the default rule with the new InformationSource value declared.

## 7. Execution rule

### 7.1 Default execution

- Signal is computed at close of day `t` using bars through close of
  day `t` and no later data.
- Execution occurs at **the next-day open** (open of day `t+1`).
- Fill model: open price of day `t+1`. Implementation must use a fill
  model that matches this convention; the existing `alpha_eq.fills`
  `FillModel.OPEN` is the reference.
- Position rule: long-or-cash. `position(t+1) = 1` if HMM classifies
  day `t` as risk-on; `position(t+1) = 0` otherwise.
- **No shorting in v1.** A future v2 may propose adding a short leg
  via a separate intake.
- `position(t+1) ∈ {0, 1}` (binary in v1; a future variant may propose
  continuous sizing).

### 7.2 Delay stress (required per exception policy §3.8)

Same backtest is run with the signal shifted by `+1` and `+2` bars.
The §3 gate requires net Sharpe degradation ≤ 0.5 under both 1-bar
and 2-bar delay.

### 7.3 Costs

- Commission: 0.5 bps one-way (per exception policy §3 default; SPY
  and QQQ are at the tightest cost tier in our cost model).
- Spread: 0.5 bps one-way (per exception policy §3 default).
- Cost-stress multipliers: 2× and 3× (both required per exception
  policy §3.7).

### 7.4 Frictions intentionally not modeled in v1

- Borrow / financing on long-only position (zero by construction in
  v1; no short leg).
- Same-day intraday rebalance: the model assumes a single
  next-day-open execution per day.
- Discretionary intraday risk management: out of scope for the
  validation backtest.

These limitations must be documented in the report. If the strategy
clears the §3 gate, an extension intake may model these frictions.

## 8. Cash-leg assumptions (per exception policy §3.25 and §4.14)

For the long-or-cash strategy shape, the validation reports results
under **three** explicit cash-leg assumptions:

### 8.1 Zero cash return (stress floor)

Risk-off days earn exactly 0%. This is conservative. Used as a floor
in reporting.

### 8.2 T-bill / cash proxy return

Risk-off days earn the prevailing short rate. **Series: FRED `DTB3`**
(3-month T-bill, secondary market rate). Source: `signal_research.data.fred`
(already wired in commit `b1a689b`). Implementation:

- daily T-bill rate is divided by 365 to obtain the daily yield
- non-trading days are carried forward
- the daily yield is applied to the cash leg on every risk-off day

### 8.3 Conservative after-fee cash return (the gating assumption)

Risk-off days earn (T-bill rate − 25 bps annualized). The 25 bps
prime-broker / cash-sweep fee is the default per exception policy
§4.14(c). This is the assumption the §3 gates are evaluated against.

A strategy whose Sharpe is above the gate under assumptions (8.1) and
(8.2) but fails under (8.3) is recorded as **cash-return-dependent**
and does not clear §3 per the exception policy.

## 9. Required gates (per accepted exception policy §3)

The candidate must satisfy **every** gate below. All thresholds are
predeclared and are not negotiable based on observed metrics.

**Methodology gates:**

- 9.1. Instrument is SPY or QQQ (Tier-1 only).
- 9.2. Strategy shape: single-instrument scalar exposure in `[0, 1]`,
       no selection step.
- 9.3. Net Sharpe ≥ **1.5** on dev AND ≥ **1.5** on holdout.
- 9.4. DSR ≥ **0.5** in the 30-strategy pool (see §4).
- 9.5. PBO_profile < **0.25**, preferred < **0.10**.
- 9.6. Bootstrap 95% lower Sharpe bound > **0.5**, preferred ≥ **1.0**.
- 9.7. Positive net Sharpe under **2×** AND **3×** cost stress.
- 9.8. Net Sharpe degradation ≤ **0.5** under both 1-bar and 2-bar
       delay stress.
- 9.9. Max dev drawdown ≥ **−20%** (i.e. no worse than −20%), OR
       Calmar > **1.0** if drawdown is worse than −20%.

**Concentration / regime gates:**

- 9.10. No calendar year carries > **50%** of |PnL|.
- 9.11. No calendar quarter carries > **35%** of |PnL|.
- 9.12. At least **two** distinct dev years contribute positive net PnL.
- 9.13. Strategy survives **removal of 2020** with Sharpe ≥ **0.8** on
        the remaining dev sample.
- 9.14. Strategy survives **removal of 2022** with Sharpe ≥ **0.8**.
- 9.15. Strategy survives the **pre-2020 dev subsample** with Sharpe
        ≥ **0.8**.

**Baseline-domination gates (compute under the §8.3 conservative
after-fee cash assumption):**

- 9.16. Beats `buy_and_hold` on both Sharpe AND max drawdown.
- 9.17. Beats `vol_targeted_buy_and_hold` on Sharpe AND max drawdown.
- 9.18. Beats `sma_50_200_gate` on Sharpe AND max drawdown.
- 9.19. Beats `mom_12_1` on Sharpe AND max drawdown.
- 9.20. `random_signal` baseline FAILS the §3 gate (control).
- 9.21. `inverted_signal_of_best_hmm` baseline FAILS the §3 gate.

**Information-integrity gates:**

- 9.22. All features available strictly at signal time (no
        look-ahead, no overlapping CV, no HMM refits that use
        post-decision data).
- 9.23. No holdout tuning. Holdout window is touched exactly once.
- 9.24. Under the §8.3 conservative after-fee cash assumption, the
        Sharpe must still clear §9.3 (this is the §3.25 cash-leg
        robustness gate).
- 9.25. Cross-instrument robustness: at least one HMM variant on
        each Tier-1 instrument must clear §9.3-§9.24 independently.
        A strategy that clears only on SPY but fails on QQQ does
        not clear the §3 gate.

## 10. Required robustness diagnostics (per accepted exception policy §4)

These are reporting requirements, not pass/fail gates by themselves.
Their collective pattern decides reviewer acceptance even after the
§9 gates have passed.

| # | Diagnostic |
|---|---|
| 10.1 | Transition matrix for each HMM variant |
| 10.2 | Expected state duration (from `1 / (1 − P(stay))`) |
| 10.3 | State persistence histogram (empirical run lengths) |
| 10.4 | State-label raw stability across refits (informational only — must NOT trigger demotion per exception §4.5(b)) |
| 10.5 | Economic state identity stability across refits (this IS the demotion trigger; report the rate per §5.4) |
| 10.6 | Exposure-time fraction by regime, separately for dev and holdout |
| 10.7 | PnL contribution decomposition: dev PnL while in risk-on vs while in risk-off |
| 10.8 | False de-risking cost: cumulative buy-and-hold return missed during the strategy's risk-off periods |
| 10.9 | Crash-protection contribution: strategy net return during the 10 largest peak-to-trough buy-and-hold drawdowns |
| 10.10 | Re-entry timing quality: the strategy's return during the first 20 trading days after each risk-off → risk-on transition |
| 10.11 | Turnover (total position changes per year) |
| 10.12 | Cost drag (total commission + spread cost in bps annualized) |
| 10.13 | Delay-stress decomposition: Sharpe at 0-bar, 1-bar, 2-bar delay |
| 10.14 | Bootstrap 95% CI for Sharpe (n_resamples = 2000) |
| 10.15 | PBO_profile (and raw_global, per_family) |
| 10.16 | DSR with multi-test deflation across the 30-strategy pool |
| 10.17 | PSR_zero for the best variant |
| 10.18 | Failure classification (per `methodology.failure_classifier.FailureCategory`) if any variant fails |
| 10.19 | Cash-leg sensitivity: Sharpe under §8.1 / §8.2 / §8.3 separately |
| 10.20 | Cross-instrument summary: comparison table of best SPY variant vs best QQQ variant on every §9 gate |

## 11. Status outcomes

The validation pipeline assigns one of the following per variant:

| Status | When |
|---|---|
| `fail` | Any §9 gate is missed, OR variant flagged as a failure class in §10.18. |
| `research_pass` | Variant has positive metrics but does not clear the strict §9 gates (e.g. dev Sharpe < 1.5). Default rule's `RESEARCH_PASS` status. |
| `exception_review_required` | Variant clears all §9 gates AND §10 robustness diagnostics are complete. Awaits reviewer acceptance of exception invocation per exception policy §5.1 step 2. |
| `paper_trade_candidate` | Reviewer has accepted the exception invocation. Strategy is ready for the §5.1 paper-trading workflow. |

**The `production_candidate` status is NOT available from this
validation.** Reaching production requires:
- Successful 6-month paper trading under exception policy §5.1 step 7
  structural-break review
- CLAUDE.md §11 stage-promotion review (signed `stage_change.md`,
  updated `.env`, operator restart)
- Two-person review

This intake reserves no higher status than `paper_trade_candidate`
regardless of how strong the metrics turn out to be.

## 12. Explicit non-goals

This intake commits to the following non-goals. Any deviation requires
a fresh intake, not an extension of this one:

12.1. **This validation does not revive broad OHLCV alpha search.**
The six prior cross-sectional iterations remain closed.

12.2. **This validation does not permit cross-sectional stock or
crypto selection.** The exception is for single-instrument timing
only. Any future variant that adds a selection step is out of scope.

12.3. **This validation does not permit current-constituent universe
promotion.** SPY and QQQ are index proxies directly traded; their
"constituents" are handled by the ETF issuer. Any strategy that picks
constituents and equal-weights them is cross-sectional, not single-index,
and falls outside this exception.

12.4. **This validation does not allow live deployment.** Even a
`paper_trade_candidate` from this run requires the §5.1 paper-trading
workflow and the CLAUDE.md §11 stage-promotion process before any
real capital is involved.

12.5. **This validation does not change the default no-OHLCV rule for
non-exception strategies.** The default rule remains in force.

12.6. **This validation does not approve HMM automatically.**
Acceptance of the exception policy was necessary for this validation
to be initiated. Acceptance of THIS intake does not promote HMM;
promotion requires clearing the §9 gates and the §11 paper-trading
process in sequence.

12.7. **This validation does not lower the §3.3 Sharpe gate.** The
1.5 dev AND 1.5 holdout requirement stands. The previously observed
+1.74/+1.76 result from the VRP × HMM interaction commit `0d00266` is
suggestive but does not lower the bar for this dedicated run.

12.8. **This validation does not authorize features outside §6.1.**
The forbidden list in §6.2 is the closed list of off-limits inputs.

12.9. **This validation does not authorize variant grid expansion
after seeing results.** The 9-per-instrument grid in §4.1 is frozen
at intake submission.

## 13. Deliverables

The dedicated HMM validation run, once executed, must produce the
following artifacts under `reports/signal_research/hmm_single_index/`:

| File | Content |
|---|---|
| `hmm_single_index_registry.parquet` | One row per (variant × instrument) entry including all §10 diagnostics |
| `hmm_single_index_validation_report.md` | Headline summary, side-by-side table, §9 gate scorecard |
| `hmm_state_stability_report.md` | §10.4-§10.5 raw vs economic-identity stability across refits |
| `hmm_cash_leg_report.md` | §8 / §10.19 Sharpe and metrics under all three cash assumptions |
| `hmm_baseline_comparison_report.md` | §9.16-§9.21 comparison vs each baseline on every gate criterion |
| `hmm_exception_gate_report.md` | §9.1-§9.25 gate-by-gate pass/fail with the assigned status per §11 |
| `hmm_failure_classification.md` | §10.18 failure-class enumeration if any variant fails any §9 gate |

The intake document itself (`docs/research/intake/2026-05-28-hmm-single-index-v1.md`)
is committed before any of the above are generated.

## 14. Pre-execution sign-off (acknowledgement)

The proposer acknowledges:

- This strategy will be subjected to the §9 (24 + cross-instrument)
  promotion-gate criteria with no post-hoc tuning permitted after the
  holdout pass.
- The variant grid in §4 is frozen at intake submission.
- The feature set in §6 is frozen at intake submission.
- The cash-leg assumptions in §8 will be reported and the conservative
  after-fee one is the gating assumption.
- The maximum status reachable from this validation is
  `paper_trade_candidate`. `production_candidate` is reachable only
  via the §11 paper-trading workflow plus CLAUDE.md §11 stage promotion.
- If any §9 gate is missed, the failure class is recorded and the
  variant is closed. No silent variant expansion is permitted.

**Proposer:** QuantLab research (Phase B.γ follow-up)
**Intake submitted:** 2026-05-28
**Exception invocation:** policy ACCEPTED commit `74ca502`

## 15. Implementation prerequisites (out of scope of this intake)

The following code changes are required before this validation can be
executed. They are listed here for completeness, but the intake itself
is committed independently of them:

15.1. **Add `EXCEPTION_REVIEW_REQUIRED` to `signal_research.status.CandidateStatus`**
between `RESEARCH_PASS` and `PROMOTION_ELIGIBLE`. Update
`promote_if_eligible` to honor the new sequential-promotion order.

15.2. **Add `exception_invoked: bool` and `exception_policy_ref: str`
fields to `signal_research.validation.spec.ValidationSpec`** with
defaults `False` / `""`. When `exception_invoked = True`, the spec's
`information_sources` may contain OHLCV-only without disqualifying
the strategy from `exception_review_required`.

15.3. **Update `validation.pipeline._assign_status`** to honor the
exception path: if `spec.exception_invoked` is `True` and all §9
exception-policy gates pass, the status becomes
`EXCEPTION_REVIEW_REQUIRED`. If the default 8-criteria gates pass
(non-exception path), behavior is unchanged.

15.4. **Implement the §4 robustness-test harness** as
`signal_research.validation.exception_robustness` (new module). Must
compute every §10 diagnostic.

15.5. **Implement the §8 cash-leg reporting** as
`signal_research.validation.cash_leg_reporting` (new module). Must
support the three assumptions in §8.1-§8.3 with the FRED `DTB3` series.

15.6. **Implement the §10.5 economic-identity stability check** as a
helper in `signal_research.vrp.hmm_panel` (or a new
`signal_research.hmm` module) that applies the §5 predeclared rule
and reports whether the economic identity is preserved across refits.

15.7. **Implement the HMM strategy module** as
`signal_research.strategies.hmm_single_index` (new module) with the
§4 variant grid and §6 default feature set.

15.8. **CLI driver** as `scripts/run_hmm_single_index_v1.py`.

These items will be sequenced after this intake document is reviewed
and accepted. None of them require additional governance review beyond
the already-accepted exception policy.

## References

- `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`
  — accepted exception policy (commit `74ca502`)
- `docs/research/STRATEGY_INTAKE.md` — default intake protocol
- `docs/research/VALIDATION_RUNBOOK.md` — validation pipeline runbook
- `docs/research/2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` — six-iteration
  negative result
- `docs/research/intake/2026-05-28-vrp-index-v1.md` — VRP intake
- `reports/signal_research/vrp_hmm_interaction/` — VRP × HMM
  interaction reports (commit `0d00266`)
- `CLAUDE.md` §11 — production stage promotion runbook
