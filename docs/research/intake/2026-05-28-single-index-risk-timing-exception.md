# Single-Index Risk-Timing Exception Proposal

**Date:** 2026-05-28
**Status:** **ACCEPTED** (with seven amendments) — see §11 Acceptance Record
**Type:** Governance / intake-protocol amendment
**Author:** QuantLab research
**Supersedes:** none
**Amends:** `docs/research/STRATEGY_INTAKE.md` §3 ("Information source declaration")

## 0. One-paragraph summary

The no-promotion-without-new-information-source rule, established after
six failed OHLCV cross-sectional alpha iterations and codified in code at
`validation.pipeline._assign_status`, must remain intact. **This proposal
does not repeal that rule.** It carves out a narrow exception for one
specific use case — single-instrument risk timing on a directly-traded
liquid index — under stricter gates than the default 8-criteria
promotion gate. The exception is motivated by a categorically different
empirical result (HMM-only-gate on SPY: dev Sharpe +1.74, holdout +1.76,
cost-stress 2× +1.74, delay-1d +1.45, PSR_zero=1.000) and the orthogonality
finding that VRP and HMM detect the same regime structure. The proposal
requires explicit acceptance before any HMM strategy is run under it,
and even after acceptance HMM is not auto-promoted — it must pass a
dedicated stricter validation, then go through paper trading, before any
production claim.

## 1. Scope of exception

This exception applies ONLY to strategies satisfying every condition
below:

**Allowed instruments — tiered list (amendment 6, 2026-05-28):**

**Tier 1 (immediately eligible at exception acceptance):**

- `SPY` — S&P 500 SPDR ETF, directly traded
- `QQQ` — Nasdaq-100 Invesco ETF, directly traded

These have clean yfinance bars, well-understood cost structure, no
constituent-survivorship issue, and no roll convention. They are
immediately eligible for an exception-path validation under §3 / §4.

**Tier 2 (eligible only after a separate market-specific audit):**

- `BTCUSDT` — Bitcoin perpetual or spot, blocked pending: (a) venue
  selection with documented fee schedule and observable spread series,
  (b) funding-rate handling spec if perp, (c) data-quality audit of the
  bars feed, (d) session-time convention. The yfinance `BTC-USD` proxy
  used in the crypto top-30 iteration is **not** sufficient for Tier-2
  qualification — that feed lacks venue-specific cost detail.
- `ETHUSDT` — same conditions as BTCUSDT.
- `ES` / `NQ` continuous futures — blocked pending: (a) roll convention
  (calendar vs open-interest) audit, (b) roll-cost model with documented
  assumptions, (c) margining and overnight financing spec, (d) data feed
  with continuous-contract construction documented in the manifest.

Tier-2 instruments are **not eligible at exception-acceptance time**.
A separate proposal must follow that completes the market-specific
audit before a Tier-2 instrument is added. Until then, no exception-
path validation may be initiated for them.

**Strategy shape (all required):**

- Exactly one underlying instrument from the allowed list
- The output of the strategy at any time `t` is a scalar gross exposure
  `position(t) ∈ [-1, +1]` (or `[0, +1]` for long-only variants)
- No basket construction. No multi-instrument allocation. No
  cross-sectional ranking of any kind.
- The signal generator may consume only OHLCV-derived features of the
  same single instrument, plus possibly the instrument's own
  realized-variance features.

**Explicitly NOT allowed under this exception:**

- Current-constituent S&P 500 / Nasdaq 100 / FTSE / any equity index
  universe (survivorship issue)
- Top-N-by-ADV cross-sectional baskets in any market
- Individual stock selection of any kind
- Crypto top-N cross-sectional ranking
- Multi-asset baskets (SPY + GLD, SPY + bonds, etc.)
- Multi-instrument sector rotation
- Any strategy that relies on a hidden constituent-selection step,
  even if the final output is "one position size"
- Sentiment, fundamentals, news, alternative-data inputs without
  passing the existing data-quality audit (these have their own
  intake path through the default rule)
- Microstructure tick-level inputs (these are out of scope; they'd
  need a separate intake)

**Universe restriction summary:** the exception covers timing on a
*passively-known* liquid index proxy. The moment a strategy needs to
select *what* to be long, it falls outside the exception and back under
the default no-OHLCV-promotion rule.

## 2. Rationale

### 2.1 Why the default rule must stay

Six iterations on OHLCV-derived signals (top-50/100/200 SP500, sector-
conditional AvL, GKX LightGBM, crypto top-30 multi-family) found no
strategy that survives honest validation at hedge-fund-grade costs.
The mom_12_1 noise floor (+0.15-0.59 holdout Sharpe) reproduces across
equity and crypto microstructures and is the 2023-2026 trend exposure,
not alpha. The no-OHLCV-promotion rule was added because:

- Without it, the equity research would have flagged at least one
  variant of momentum, AvL, or GKX as "passing" before the multi-test
  deflation kicked in
- The DSR multi-test penalty is the only thing protecting against
  promoting overfit OHLCV variants
- An infrastructure-level rule that prevents OHLCV-only promotion is
  cheaper to maintain than human review of every variant

**This rule is load-bearing. It is not under review.**

### 2.2 Why HMM-only is categorically different

The HMM-only-gate result observed on SPY 2010-2026 is not the same
shape as the failed OHLCV iterations:

| Property | OHLCV cross-sectional (failed) | HMM single-index (this proposal) |
|---|---|---|
| Selection step | Yes (which name to long?) | None (always SPY) |
| Constituent survivorship | Yes (current SP500 only) | None (SPY is the index proxy) |
| Variant grid size | 10-21 strategies | 1 (binary state gate) |
| Multi-test deflation impact | DSR drops from 0.98 to 0.00 | DSR stays at 1.000 |
| PBO | 0.004-0.266 | 0.010 |
| Cost-stress 2× degradation | Often flips negative | +1.74 (essentially unchanged) |
| 1-bar delay degradation | Often 0.5+ Sharpe loss | 0.3 Sharpe loss (acceptable) |
| Holdout / dev agreement | Sometimes close, often noise | dev +1.74 ≈ holdout +1.76 |
| Max DD dev | -50% to -90% | -8.5% |
| Mechanism | "Pick winners after the fact" | "Avoid drawdowns by stepping out" |

The HMM strategy is mechanically risk *avoidance* (step out during
high-vol regimes) on a passively-known instrument. It is not making a
selection claim and therefore does not consume the "is OHLCV alone
enough to find alpha?" question that the default rule was designed to
police.

### 2.3 What the VRP × HMM interaction test added

The Phase B.γ interaction test (commit `0d00266`) directly tested
whether the option-implied VRP signal carries information independent
of HMM regime. Three results converge:

- `vrp_when_hmm_risk_off` dev Sharpe: **−0.00** (VRP without HMM
  confirmation is noise)
- `vrp_when_hmm_risk_on` ρ(HMM): **+0.96** (VRP applied within risk-on
  is essentially HMM-only)
- `orthogonalized_vrp` residual Sharpe vs HMM: **+0.000** (no
  incremental information after controlling for HMM)

VRP and HMM are detecting the same regime structure. HMM is cheaper
(no option data) and faster (price moves precede implied-vol moves in
most crises). This makes the HMM-only result *more* notable: the
non-OHLCV channel that "should" carry independent information turned
out to be subsumed.

### 2.4 What this proposal does NOT claim

This proposal does not claim:

- That OHLCV-only methods can find cross-sectional alpha (six failures say no)
- That the HMM result is alpha (it is risk timing, not alpha; absolute
  returns are comparable to buy-and-hold, what changes is Sharpe via
  drawdown avoidance)
- That HMM should be promoted without further validation
- That the gates in the existing default rule are wrong
- That single-index timing in general is easy or that we expect more
  such results

## 3. Proposed stricter gate for OHLCV-only single-index risk timing

A candidate strategy under this exception may be flagged
`exception_review_required` (see §6) only if **every** condition holds.
This gate is intentionally stricter than the default 8-criteria gate in
multiple ways.

**Methodology gates:**

3.1. The instrument is on the §1 allowed list.

3.2. Strategy shape conforms to §1 (single instrument, scalar exposure,
no selection step).

3.3. Net Sharpe ≥ **1.5** on the development window AND ≥ **1.5** on
the permanent holdout window. (Note: the default rule requires dev ≥ 1.5
and holdout ≥ 0.5. The exception requires both ≥ 1.5 — the holdout bar
is 3× stricter.)

3.4. DSR ≥ **0.5** in a multi-strategy pool of ≥ 5 baselines (see §4).
For a single-strategy submission, the pool is constructed from the
mandatory baselines listed in §4.

3.5. PBO_profile < **0.25**, with a preferred target of < 0.10.

3.6. Bootstrap 95% lower Sharpe bound > **0.5**, preferred ≥ 1.0.
(The default rule requires > 0; the exception requires > 0.5.)

3.7. Positive net Sharpe under **2× and 3× cost stress** independently.
(The default rule requires 2× only.)

3.8. **One-bar AND two-bar** delay stress: net Sharpe degradation ≤ 0.5
under both 1-bar and 2-bar delay. (The default rule tests 1-bar only.)

3.9. Maximum dev drawdown ≥ **−20%** (i.e. no worse than −20%, where
drawdown is reported as a negative percentage and the inequality is
read on the signed value), OR Calmar ratio > **1.0** if max drawdown
is worse than −20%. (Amendment 1, 2026-05-28: inequality direction
fixed — a −30% drawdown is worse than −20% and must not pass under
the first branch of this gate.)

**Concentration / regime gates:**

3.10. Performance not dominated by one calendar year (no year carries
> 50% of |PnL|).

3.11. Performance not dominated by one calendar quarter (no quarter
carries > 35% of |PnL|).

3.12. Performance not dominated by one crisis window or one volatility
regime (≥ 2 distinct dev years contribute positive net PnL).

3.13. Strategy survives **removal of 2020** (Sharpe excluding 2020
remains ≥ 0.8).

3.14. Strategy survives **removal of 2022** (Sharpe excluding 2022
remains ≥ 0.8).

3.15. Strategy survives the **pre-2020 development subsample**, with
Sharpe ≥ **0.8** on that subsample alone. (Amendment 4, 2026-05-28:
re-worded for clarity. The original phrasing "removal of the holdout
window in dev" was conceptually confusing because the holdout is not
part of dev. The intent is to verify the strategy worked before the
2020-2025 high-vol-then-rally period, not only during it. The 0.8
threshold is materially below the dev gate of 1.5 because pre-2020
spans the post-2008 low-vol regime where any timing strategy will see
its weakest contribution; we still require it to be net-positive and
clear of noise.)

**Baseline-domination gates:**

3.16. Beats buy-and-hold on the same instrument on risk-adjusted basis
(Sharpe AND max drawdown both better).

3.17. Beats vol-targeted buy-and-hold (e.g. 10% annualized vol target
via 60-day realized vol) on risk-adjusted basis.

3.18. Beats simple-moving-average-cross baselines (e.g. 50/200 SMA
gate) on risk-adjusted basis.

3.19. Beats simple-momentum baselines (e.g. mom_12_1 single-asset) on
risk-adjusted basis.

3.20. Random-signal sanity baseline fails the gate (control test).

3.21. Inverted-signal sanity baseline fails the gate (sign-flip test).

**Information-integrity gates:**

3.22. All features are available strictly at signal time. No
look-ahead from later data, no overlapping CV with the same data, no
HMM refitting that uses post-decision data.

3.23. No holdout tuning. The holdout window is touched exactly once
at final validation; any parameter that was changed after seeing
holdout metrics invalidates the run.

3.24. Live paper trading required before any production claim (see §5).

3.25. **Cash-leg robustness (amendment 5, 2026-05-28).** For long-or-cash
strategies (the typical exception-path shape — long the instrument when
the gate is on, otherwise sit in cash), §4.14 requires reporting under
three cash-leg assumptions. Under the **conservative after-fee cash
return** assumption, the strategy must still clear §3.3 (net Sharpe
≥ 1.5 on both dev and holdout). A strategy whose pass depends on
generous cash-return assumptions does not clear §3.

## 4. Additional robustness tests required before approval

For an HMM-class strategy specifically, the following must be reported
alongside the §3 gates. These are not pass/fail by themselves but their
collective pattern decides reviewer acceptance.

4.1. **Cross-instrument robustness.** Run the same strategy with
identical config on SPY, QQQ, BTCUSDT, and ETHUSDT (subject to the
allowed-list in §1; if BTCUSDT/ETHUSDT loader is not yet clean, those
runs are required-with-caveat). Report all four side by side. A
strategy that works only on SPY is suspect.

4.2. **HMM training-window robustness.** Test with at least three
different training-window definitions:
- Full dev window
- Expanding-window (recompute HMM every N years)
- Rolling-window (most recent 5 years)
Report the Sharpe of the strategy under each. A strategy whose Sharpe
falls by > 0.5 across these choices is over-dependent on a specific fit.

4.3. **HMM state-count robustness.** Test 2-state, 3-state, and
4-state HMMs as separate registered variants. Count each in the PBO/DSR
pool. The promoted variant should not exclusively be the best-fitting
state count.

4.4. **HMM fit-window scheme.** Test expanding-window vs rolling-window
fitting as separate registered variants. Same PBO requirement.

4.5. **Economic state identity, not raw label stability (amendment 3,
2026-05-28).** HMM state labels (state 0 vs state 1) are arbitrary
permutations of the same fit. The economic identity of each state is
what matters. The proposal therefore requires, in this order:

  (a) **Predeclared risk-on identification rule.** Risk-on must be
  defined by state statistics, not by raw state ID. Default rule: the
  state with higher mean return on the dev fit window, with tie-broken
  by lower realized volatility (state with smaller σ wins on near-tie
  means within 0.0001 daily). Submissions may propose a different
  predeclared rule with justification, but the rule must be stated
  ex-ante and applied consistently across every refit.

  (b) **Economic-identity stability across refits.** Report whether
  successive refits produce states whose (mean, vol) signature places
  them in the same economic identity, regardless of raw label index.
  A "flip" for the purposes of §5.2 reversal triggers and §5.3 refit
  policy means the *economic interpretation* flipped (e.g. the
  previously-higher-mean state now has lower mean than its
  counterpart). A flip of raw label index 0↔1 with both states
  preserving their economic ordering is NOT a flip.

  (c) **Transition matrix stability.** Report the transition matrix
  for each refit, with state identity normalized via (a). Stability of
  the P(stay) entries across refits is a quality indicator; bands
  shifting by > 0.15 absolute on the diagonal are a quality warning.

4.6. **Transition probabilities and state persistence.** For the
chosen HMM, report the transition matrix and the expected duration of
each state. Persistence < 5 trading days is suspicious (state is just
overfitting price noise).

4.7. **Exposure-time by regime.** What fraction of dev and holdout
days does the strategy spend in each state? A strategy that's in
risk-on 95% of the time is essentially buy-and-hold; a strategy in
risk-on 30% of the time is a heavy de-risker. Report the number.

4.8. **PnL decomposition by regime.** Report dev PnL contribution from
risk-on days vs risk-off days. The user's question: is the strategy
making money in risk-on, or is it making money by being out of market
in risk-off?

4.9. **False de-risking cost.** When the strategy is out of the market,
what is the cumulative buy-and-hold return it missed? Report this as
"missed upside" so the reviewer can see how much expected return was
sacrificed for drawdown reduction.

4.10. **Crash-protection contribution.** During the largest 10
drawdown windows on buy-and-hold (peak-to-trough on the instrument),
what was the strategy's net return in each? This directly measures the
crash-protection mechanism.

4.11. **Re-entry timing quality.** After a risk-off period ends, how
fast does the strategy re-enter, and what return is captured in the
subsequent rebound? A strategy that exits drawdowns well but re-enters
late may have a Sharpe-positive but absolute-return-negative profile.

4.12. **Futures-roll handling (if instrument is NQ/ES).** Document the
roll convention (calendar vs open-interest), the roll cost assumption,
and the impact on net Sharpe. Required before NQ/ES is added to the
allowed list.

4.13. **Crypto session and funding-rate handling (if instrument is
BTCUSDT/ETHUSDT).** Document the venue spec, the assumed funding
schedule for perps (if applicable), and the impact on net Sharpe.

4.14. **Cash-leg reporting (amendment 5, 2026-05-28).** For long-or-
cash strategies, the report must include Sharpe, max drawdown, and
cumulative return under three explicit cash-leg assumptions:

  (a) **Zero cash return.** Risk-off days earn 0%. Conservative; used
  as a stress floor.
  (b) **T-bill / cash proxy return.** Risk-off days earn the prevailing
  short-rate (3-month T-bill or equivalent currency cash rate). Use
  the FRED series `DTB3` or equivalent; document the series ID and
  carry forward / interpolation rule for non-trading days.
  (c) **Conservative after-fee cash return.** T-bill rate minus an
  explicit prime-broker/cash-sweep fee assumption (default: 25 bps
  annualized; document if different). This is the realistic figure
  for actual deployment.

The §3.25 gate is evaluated against assumption (c). Assumptions (a)
and (b) are reported for transparency. A strategy whose Sharpe is
above the gate under (a) and (b) but fails under (c) is recorded as
"cash-return-dependent" and does not clear §3.

4.15. **Tier-1-only at exception acceptance (amendment 6, 2026-05-28).**
Sections 4.1, 4.12, and 4.13 reference Tier-2 instruments (BTCUSDT,
ETHUSDT, ES/NQ futures). Until Tier-2 audits per §1 are completed,
the §4.1 cross-instrument robustness check is run on **SPY and QQQ
only**. A strategy that works on SPY but fails on QQQ is suspect;
both Tier-1 instruments must be reported. Tier-2 cross-instrument
reporting becomes required only after each instrument's audit is
accepted.

## 5. Governance rule

### 5.1 Sequencing

Even if this exception is accepted as policy, no HMM strategy is
automatically promoted. The workflow is:

1. **Draft this exception proposal** (this document; complete).
2. **Review and accept-or-reject the exception policy.** Reviewer
   notes any modifications. If accepted, the policy is recorded in
   `docs/research/STRATEGY_INTAKE.md` as an amendment with an explicit
   pointer to this document.
3. **Draft a dedicated HMM intake document** under
   `docs/research/intake/YYYY-MM-DD-hmm-single-index-v1.md` that
   declares the exception is being invoked.
4. **Run the dedicated HMM risk-timing validation** under the §3
   stricter gates and §4 robustness tests.
5. **If §3 gates and §4 robustness checks pass:** classify as
   `paper_trade_candidate` (not `production_candidate`).
6. **Paper trading / shadow deployment** — at least 6 calendar months
   of out-of-sample paper trading with daily PnL recorded in the
   audit log. Real-time data, real-time signal computation, real
   execution venue conditions (but no real capital).
7. **Structural-break check at end of paper-trading window
   (amendment 2, 2026-05-28).** The previous version of this step
   required "paper trading reproduces holdout Sharpe within ±0.5
   over the 6-month window". That hard rule is replaced because a
   6-month Sharpe estimate has standard error too large to make ±0.5
   the right gate — under a true Sharpe of 1.5, the 6-month sample
   Sharpe routinely lands in [0.3, 2.7] from sampling variance alone.
   The replacement check is a **structural-break review** with the
   following components, none of which is by itself a hard fail but
   their *collective* pattern decides reviewer acceptance:

   (i) Realized 6-month Sharpe is reported. No hard threshold; it is
   context, not a gate.

   (ii) Realized max drawdown is **within the expected envelope**
   (no larger than the worst dev drawdown × 1.5).

   (iii) Realized exposure-time fraction is **within expected bands**
   (within ±15 percentage points of dev exposure-time).

   (iv) Realized turnover and cost drag are **within expected bands**
   (within ±50% of dev turnover; within ±25 bps annualized of dev
   cost drag).

   (v) No data-quality failures recorded in the audit log (missing
   bars, stale signal, late market open handling).

   (vi) No unexplained execution drift (the difference between
   intended position and actual paper-trading position must stay
   within ±0.05 of gross exposure on > 95% of trading sessions).

   (vii) Realized HMM state distribution is consistent with backtest
   expectation (each state's empirical frequency within ±10
   percentage points of dev frequency, with state identity computed
   per §4.5).

   The reviewer evaluates this pattern. If (ii)-(vii) are all clear
   and (i) is not catastrophically below the backtested figure
   (informal floor: realized Sharpe > 0 over the window), the
   strategy is eligible for the §5.1 step-8 review. If any of
   (ii)-(vii) shows a clear break, the strategy is demoted and a
   review note is filed.
8. **Two-person review for production_candidate** per existing CLAUDE.md
   §11. Operator restart, .env update, signed stage_change.md per
   existing runbook.

### 5.2 Reversal triggers

If at any time during paper trading or production:

- A single drawdown exceeds the worst dev drawdown by > 50%
- The HMM **economic state identity flips** between successive refits
  (per §4.5(b) — not a raw-label flip, but the previously-higher-mean
  state now has lower mean than its counterpart, or the predeclared
  risk-on rule places risk-on on a different economic identity than
  before)
- Any data-quality issue is discovered in the input feed
- Structural-break check (per §5.1 step 7 components (ii)-(vii)) fires

then the strategy is automatically demoted to `research_pass` and a
review note is filed. No automatic re-promotion.

(Amendment 3, 2026-05-28: the original trigger "the HMM produces a
state-label flip relative to the deployed fit" was replaced with the
economic-identity flip rule per §4.5. Raw-label permutations between
fits are noise and must not cause demotion.)

(Amendment 2, 2026-05-28: the original trigger "Realized Sharpe falls
below 0.5 over a trailing 6-month window" was removed; 6-month Sharpe
sampling variance makes that an unreliable trigger. The structural-
break check (component (i) reporting Sharpe but not gating on it) is
the replacement.)

### 5.3 Periodic refit policy

The HMM must be refit at least every 12 months. The refit uses the
expanded dev window (original dev + new data through refit date,
excluding the original holdout). **Economic-identity stability**
(per §4.5(b)) is logged at every refit. If the economic interpretation
flips — not merely raw labels — the strategy is automatically demoted
pending review.

## 6. Status labels

This section amends the existing `CandidateStatus` enum semantics for
strategies submitted under this exception. The base enum stays
unchanged for non-exception strategies.

- **`research_pass`** (existing) — result is statistically interesting,
  no further action.
- **`exception_review_required`** (new) — strategy passes §3 gates and
  §4 robustness reports are complete; awaiting reviewer acceptance of
  the exception invocation.
- **`paper_trade_candidate`** (existing, scope-limited under exception)
  — exception review accepted, ready for shadow / paper trading. Note
  that this status under the exception is *not* equivalent to the same
  status under the default rule; the exception path requires the
  stricter §3 gates whereas the default path has lower thresholds.
- **`production_candidate`** (existing, scope-limited under exception)
  — only after the §5.1 step-6 paper trading completes with the
  required Sharpe reproducibility.

The validation code in `validation.pipeline._assign_status` would need
an additional argument `exception_invoked: bool` to distinguish the
two paths if this proposal is accepted. The code change is out of
scope for this document.

## 7. Explicit non-goals

This proposal is constrained by the following non-goals. Reviewer
acceptance of the proposal is conditional on each remaining true:

7.1. **This is not a revival of broad OHLCV alpha search.** The six
prior iterations (TB-meta-AvL, multi-model, momentum scaleup, sector
AvL, GKX scaleup, crypto multi-family) remain closed. Future strategy
proposals on cross-sectional baskets must declare a non-OHLCV
information source per the default rule.

7.2. **This does not allow cross-sectional OHLCV stock selection
promotion.** Any strategy whose final output requires choosing among
multiple candidates is outside this exception, regardless of how the
choice is made.

7.3. **This does not allow current-constituent survivorship-biased
promotion.** SPY and QQQ are *index proxies* directly traded; their
"constituent" composition is handled by the ETF issuer, not by the
strategy. The exception is exactly for instruments that have this
property.

7.4. **This does not allow rule-changing after seeing results.** The
exception is defined ex-ante and no further changes are permitted
during a candidate's validation run. If a candidate falls just short
of a §3 gate, the candidate fails — the gate is not negotiated.

7.5. **This does not permit live deployment without paper trading.**
The §5.1 paper-trading step is mandatory and cannot be skipped on the
basis of strong backtested metrics.

7.6. **This does not authorize lowering the default 1.5 Sharpe gate
for non-exception strategies.** VRP's failure at +0.9 dev Sharpe
remains a fail under the default rule. The default 8-criteria gate is
unchanged.

7.7. **This does not allow OHLCV-only strategies that *include* a
cross-sectional step to claim the exception by adding a final
aggregation.** Example: a strategy that picks top-5 SP500 names then
equal-weights them is cross-sectional selection regardless of the
final-aggregation step. The shape constraint in §1 is strict.

7.8. **This does not approve HMM by default.** Acceptance of the
exception policy is necessary but not sufficient for HMM promotion.
HMM still needs to clear the §3 gates and §4 robustness tests under a
dedicated intake.

7.9. **This proposal can be revoked.** If at any future point evidence
suggests the exception was a mistake (e.g. multiple exception-path
strategies fail in paper trading), the proposal can be revoked and
existing exception-path strategies are demoted to `research_pass`.

7.10. **Tax treatment is outside this proposal's scope (amendment 7,
2026-05-28).** All §3 gates and §4 robustness tests are computed on
pre-tax returns. Tax treatment — short-term vs long-term capital
gains, wash-sale rules, K-1 vs 1099 reporting, withholding for
non-resident accounts, and any jurisdiction-specific obligation —
is **not** part of this validation gate and is **not** considered by
the §5 governance workflow. Tax treatment must be reviewed
separately before any user-facing communication, any client-facing
advice, or any deployment of real capital. A strategy that clears
§3 and §5 has cleared the *backtest-validation* bar; it has not
cleared the *productization* bar. The productization review covers
tax treatment, custody and execution venue selection, regulatory
filing obligations, KYC/AML where applicable, and any other items
outside the scope of this proposal. Productization review is a
prerequisite for any communication or deployment that could
implicate user-facing or capital-deployment claims.

## 8. Reviewer checklist

Before accepting this proposal, the reviewer should confirm:

- [ ] The §1 scope of allowed instruments is acceptable
- [ ] The §3 gates are not too easy compared to the default rule
- [ ] The §4 robustness tests cover the failure modes of greatest concern
- [ ] The §5 governance sequencing is implementable
- [ ] The §6 status-label semantics are compatible with the existing
      `CandidateStatus` enum
- [ ] The §7 non-goals are well-defined and enforceable

## 9. If accepted: next steps

If the reviewer accepts this proposal:

1. Amend `docs/research/STRATEGY_INTAKE.md` §3 to point at this
   exception document
2. Add the `exception_review_required` value to the `CandidateStatus`
   enum (code change in `signal_research/status.py`)
3. Add an `exception_invoked: bool` parameter to
   `validation.pipeline.validate_strategy` and update
   `_assign_status` accordingly
4. Draft the dedicated HMM intake document
5. Implement the §4 robustness-test harness as additional analytics in
   the validation pipeline
6. Run the HMM validation

## 10. If rejected

If the reviewer rejects this proposal:

- The HMM-only result is recorded in the negative-result research
  note as a known-strong OHLCV-only baseline that the default rule
  forbids promoting
- No further exception proposals on the no-OHLCV rule are entertained
  until materially new evidence
- The next alpha-search direction continues from §B of the
  negative-result research note (FinBERT sentiment / microstructure /
  cross-asset / event-conditioned)

## 11. Acceptance Record

**Status:** ACCEPTED, 2026-05-28.

The proposal was accepted in direction with the following seven
amendments. Each amendment is referenced inline in the section where
it applies. The amendments are recorded here as a unified change log
for audit purposes.

| # | Section(s) amended | Summary |
|---|---|---|
| 1 | §3.9 | Max-drawdown inequality direction corrected: max drawdown ≥ −20% (no worse than −20%), not ≤ −20%. A −30% drawdown now correctly fails the first branch of the gate. |
| 2 | §5.1 step 7, §5.2 | Replaced the 6-month paper-trading "Sharpe within ±0.5" hard rule with a structural-break review (7 components: realized Sharpe reported; drawdown envelope; exposure time; turnover/cost drag; data-quality; execution drift; state distribution). Reason: 6-month Sharpe sampling variance is too wide for ±0.5 to be a sound gate. |
| 3 | §4.5, §5.2, §5.3 | HMM state-label flips are arbitrary permutations and must not cause demotion. "Flip" redefined to mean *economic-identity* flip per a predeclared risk-on rule (state with higher mean return on dev, tie-broken by lower vol). Refit policy and reversal triggers updated accordingly. |
| 4 | §3.15 | Re-worded "removal of the holdout window in dev" — conceptually confusing because holdout is not part of dev. Replaced with: strategy survives the pre-2020 development subsample, Sharpe ≥ 0.8. |
| 5 | §3.25 (new), §4.14 (new) | Cash-leg treatment added. For long-or-cash strategies, results must be reported under three explicit cash assumptions: zero cash return; T-bill / cash proxy return (FRED `DTB3` or equivalent); conservative after-fee cash return (T-bill minus 25 bps prime-broker fee default). §3 gate is evaluated against the conservative after-fee assumption. |
| 6 | §1 (restructured), §4.15 (new) | Allowed instruments split into Tier 1 (SPY, QQQ — immediately eligible) and Tier 2 (BTCUSDT, ETHUSDT, ES/NQ futures — eligible only after separate market-specific loader, venue, funding, roll, and cost audits). yfinance `BTC-USD` proxy explicitly disqualified from Tier-2 status. |
| 7 | §7.10 (new) | Productization limitation: tax treatment is outside the §3/§4 validation gate. Tax, custody, execution venue selection, regulatory filing, and KYC/AML must be reviewed separately before any user-facing communication or real-capital deployment. Backtest-validation pass ≠ productization pass. |

**Effect of acceptance:**

Acceptance of this policy does **not** promote HMM. Acceptance permits
a dedicated HMM single-index validation run to be initiated under the
amended §3 gates and §4 robustness tests, on Tier-1 instruments (SPY,
QQQ) only.

The HMM intake will be drafted as a separate document at
`docs/research/intake/YYYY-MM-DD-hmm-single-index-v1.md` referencing
this exception by path.

Code changes per §9 of the original proposal (the `exception_invoked`
flag in the validation pipeline, the `exception_review_required`
status value, and the §4 robustness-test harness) remain TODO and are
out of scope of this acceptance — they will accompany the HMM intake.

## References

- `docs/research/STRATEGY_INTAKE.md` — default intake protocol
- `docs/research/VALIDATION_RUNBOOK.md` — validation pipeline runbook
- `docs/research/2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md` — six-iteration
  negative result
- `docs/research/intake/2026-05-28-vrp-index-v1.md` — VRP intake
- `reports/signal_research/vrp/report.md` — VRP standalone results
- `reports/signal_research/vrp_hmm_interaction/` — VRP × HMM
  interaction reports
- `CLAUDE.md` §11 — production stage promotion runbook
