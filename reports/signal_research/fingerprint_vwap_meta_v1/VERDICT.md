# Fingerprint-VWAP Meta-Labeling v1 — VERDICT

**Date:** 2026-06-03
**Status:** research_only. **NEGATIVE RESULT — DO_NOT_ADVANCE.** No paper. No live. Do **not** transfer to Prevalence.
**Spec:** `docs/research/intake/2026-06-03-fingerprint-vwap-meta-v1.md`
**Plan:** `docs/superpowers/plans/2026-06-03-fingerprint-vwap-meta-v1.md`

## Outcome

The branch was implemented in full (VWAP proxy + primary, multi-window fingerprint
features, primary-edge eligibility gate, a backward-compatible generalization of the
meta-label walk-forward trainer, net-of-cost pipeline with lift-vs-baseline, and a
PBO/Deflated-Sharpe gate). All unit tests pass; ruff + mypy clean; the shared-engine
change preserved parity (23 pre-existing tests green).

A smoke run on a **5-ticker, non-point-in-time** universe (AAPL/AMZN/AMD/GOOGL/GOOG,
2018–2024) produced:

| Quantity | Value |
|---|---:|
| Primary VWAP entry — eligible? | yes (net Sharpe 1.117, 4,869 events) |
| Meta-labeled net Sharpe | 0.929 |
| Baseline (take-every-entry) net Sharpe | 1.117 |
| **Lift (meta − baseline)** | **−0.189** |
| Deflated-Sharpe probability | 0.343 (gate ≥ 0.95) |
| Return kurtosis | 20.7 (fat-tailed) |
| Failed gates | `lift`, `deflated_sharpe` |

## Why DO_NOT_ADVANCE (spec §8 kill criteria)

1. **Negative lift — the binding finding.** The fingerprint meta-classifier *removed
   value*: filtered Sharpe (0.929) < take-every-entry baseline (1.117). The regime
   fingerprint did not improve entry selection; it hurt it. This is decisive and is
   the cleanest failure mode in the spec.
2. **Deflated Sharpe fails.** Probability 0.343 ≪ 0.95 at 50 trials — the result is
   not distinguishable from selection luck, and kurtosis 20.7 means it is tail-driven.
3. **Survivorship contamination (audit §1 PENDING/FAIL).** The universe is today's
   constituent snapshot, not point-in-time. This *inflates* the gross numbers — yet the
   meta-filter still subtracts value on data biased *in its favor*. A clean PIT universe
   would only lower the figures, not rescue the negative lift.

## Honest reading

This reproduces the program's standing lessons: meta-labeling improves a primary's
*precision* only when the conditioning features carry real, orthogonal information about
entry quality — here the multi-window fingerprint did not, and the lift test (the gate
designed exactly for this) caught it. The negative lift on a survivorship-favorable
sample is a strong prior that a clean PIT run would not pass either.

## Reopen condition (not pursued now)

Only a point-in-time top-30 universe run that (a) passes the full `data_audit.md`
checklist, (b) shows **positive** lift over the take-every-entry baseline, and (c) clears
the Deflated-Sharpe gate would justify revisiting. Absent that, the branch is closed as a
negative result. The reusable deliverable is the gate-disciplined pipeline itself (and the
backward-compatible trainer generalization), which correctly rejected a fragile idea.
