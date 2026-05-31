# EDGAR 10-K v1 — Validation Report

**Built:** 2026-05-30T09:20:04.393908+00:00 | git label `fwd_ret_63` | research_only.
**Intake:** `docs/research/intake/2026-05-30-edgar-10k-text-features-v1.md`.

## Data
- 6,282 filings (label-valid), 727 companies, filing years 2010-2022.
- 57 classical text features (no embeddings/LLM/NN). LM lexicon is a compact subset
  (v1 approximation — pre-registered limitation).

## Leakage audit
- Signal = SEC filing date; cross-section by filing year; forward returns are LABELS ONLY (guarded in
  `features._assert_no_label_leak`). YoY features use only the same CIK's strictly-earlier filing.
- Univariate signal signs fixed on TRAIN only; models fit on TRAIN cohorts (≤2017); holdout 2020-2022.

## Pool-level multiple-testing control
- PBO (CSCV over 12 signals, holdout cohorts): **None**
- Best text model `model_lgbm_text`: holdout IC **0.0213** (t=1.26), DSR **0.0**, bootstrap spread CI lower **0.0**.

## Classification
- status **none** | failure_class **low_frequency_insufficient_sample** | research_candidate False | promotion_eligible False
- blockers: `placebo_indistinguishable, low_frequency_insufficient_sample, research_only`
