# Design — QuantLab Alpha MVP Deliverable (Research-Paper README + Reproducible Quickstart)

**Date:** 2026-05-30
**Status:** Approved design (Approach A). Pre-implementation spec.
**Branch:** `quant-llm-implementation` — final work before merge to `main`. **Do NOT open a
PR until the operator says so.**
**Author:** research engineer (Claude) under the QuantLab Alpha program discipline.

## 0. Goal

Produce the team-leader-facing MVP deliverable that finalizes the project: a
**research-paper-format `README.md`** documenting the full program (platform + ML
methodology with formulas/derivations + the honest experimental record), with
**in-repo hyperlinks** ("blue underlines") to every supporting document, plus a
**one-command reproducible quickstart** (`make mvp`) that regenerates the funding-carry
capstone result and its figures on cached free data.

**Hard honesty guardrail (program rule):** report real numbers only. S1 holdout weighted
zero-mean R² = 0.0055 (< the 0.012 release gate). 0 deployable / paper / live strategies.
Funding-carry = DO_NOT_ADVANCE. No fabricated performance, no gate-weakening, no
promotion language. research_only.

## 1. Approach (chosen: A — honest research / negative-results paper)

Frame QuantLab Alpha as a quant research paper whose central contribution is the
**validation infrastructure** and the rigorous, reproducible finding that under a
zero-cost data constraint no taker-tradable alpha survives honest out-of-sample
validation — with funding-carry as the closest lead. The negative result is the science.
Rejected: B (systems/MLOps framing — underuses methodology depth), C (single-strategy
capstone framing — overweights one branch, looks like a pitch).

## 2. Deliverables

1. `README.md` — rewritten as the research paper (sections in §3). Preserve the existing
   abstract/thesis/RQ content where still accurate; restructure and extend.
2. `scripts/make_mvp_figures.py` — generates committed PNG figures from existing/cached
   artifacts (no training, no new research).
3. `figures/` — four committed PNGs (§5).
4. `Makefile` `mvp` target (§6) — one command: run funding-carry pipeline on cached data,
   generate figures, print the gate verdict.
5. Design + plan docs under `docs/superpowers/` (this spec + the implementation plan).

Non-goals: no S1 retraining, no new strategy branch, no gate-chasing, no PR.

## 3. README section structure

1. **Title + badges** — `QUANTLAB_STAGE=paper`, kill-switch armed, tests passing, and an
   honest **`alpha: none-deployable (research-only)`** badge. Badges link to runbooks
   (existing pattern).
2. **Abstract** — the platform in two sentences + the honest headline finding (validation
   infra is the deliverable; data acquisition is the binding constraint; 0 deployable
   alpha under zero-cost data).
3. **§1 Introduction** — problem statement, two-layer thesis (S1 numeric ⊥ S2 governance),
   RQ1–RQ4 table (preserved, with evidence links).
4. **§2 Platform Architecture** — the S1→S2→S3→S4 pipeline as a `mermaid` flowchart;
   stage-gating (`paper`/`live_shadow`/`live`), kill-switch precedence, append-only audit.
   Links to ADRs + specs.
5. **§3 Machine-Learning Methodology** — the formulas/derivations section (§4 below). Each
   subsection links to the implementing module + the VALIDATION_RUNBOOK.
6. **§4 Experimental Results**
   - **4.1 S1 tabular predictor** — real holdout weighted zero-mean R² table (per-fold &
     per-model from `experiments/alpha_s1/20260523-160541/metrics.json`), figure
     `s1_fold_model_r2.png`, and the honest verdict (0.0055 < 0.012 gate).
   - **4.2 Signal-research ledger** — a table of the ~13 branches (OHLCV, VRP, HMM,
     microstructure L2/L1/tick, FOMC, futures carry, options-IV, EDGAR 10-K/10-Q,
     zero-cost allocators v1/v2) with verdict + evidence link each. Sourced from the
     close-out + program review.
   - **4.3 Funding-carry capstone** — the reproducible result: data-audit PASS →
     8h-marked delta-neutral backtest → liquidation stress → §5 gate. Three figures
     (equity, per-year, leverage stress), the per-year table, and the DO_NOT_ADVANCE
     verdict with the corrected "Sharpe is real but tail-dominated" finding.
7. **§5 Discussion — the Four Walls** — cost / subsumption / data-access / frequency;
   the meta-conclusion that the binding constraint is the information set, not method.
8. **§6 Reproducibility** — `make mvp` quickstart, environment (`uv`, `PYTHONPATH=src`),
   artifact SHA-256 manifests, audit-log replay, the testing/lint/type commands.
9. **§7 Limitations & honest disclosures** — no deployable alpha; S1 below gate; funding
   carry tail + regime decay; free-data scope; what would change the answer (paid data).
10. **§8 References & repository map** — the hyperlink index: every spec, plan, ADR,
    negative-result note, intake, runbook. This is the bulk of the "blue underlines."

## 4. Formulas & derivations (LaTeX in markdown; each links to its module)

All derivations must match the code. Cross-checked against `alpha/` and
`crypto_research/perps/validation.py`.

- **Purged & embargoed walk-forward / CPCV** (`alpha/cv.py`): purge any train sample whose
  label horizon overlaps a test sample; embargo a buffer after each test block. State the
  leakage these prevent.
- **Weighted zero-mean R²** (`alpha/metrics.py`): `R²_w = 1 − Σ wᵢ(yᵢ−ŷᵢ)² / Σ wᵢ yᵢ²`
  (zero-mean denominator → measures variance explained about 0, not about the mean; the
  Jane Street convention). Note why it can be negative.
- **Adversarial validation** (`alpha/adversarial.py`): train a classifier to separate
  train vs holdout rows; feature with AUC > 0.6 indicates distribution shift → drop or
  transform. Brief derivation of AUC-as-separability.
- **Noise-floor feature control** (CLAUDE.md §5.6): inject a seeded `N(0,1)` feature; any
  engineered feature ranked below it in ≥3/5 folds is removed (guards against spurious
  importance).
- **Stacking meta-learner** (`alpha/stacking.py`): build out-of-fold base predictions
  (no leakage), fit a linear meta-model on the OOF matrix; the stack must beat the best
  base learner on the OOF metric.
- **Probability of Backtest Overfitting (PBO)** (`validation.estimate_registry_pbo`):
  CSCV — partition the return matrix into blocks, for each combinatorial split compute the
  in-sample-best strategy's out-of-sample rank, logit-transform; `PBO = P(logit ≤ 0)`
  (fraction where the IS-best underperforms OOS-median). Gate ≤ 0.25–0.5.
- **Deflated Sharpe Ratio (DSR)** (`validation.deflated_sharpe_payload`):
  `DSR = Φ( (ŜR − SR₀)·√(T−1) / √(1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²) )`, with skew γ₃, kurtosis
  γ₄, and `SR₀` inflated for the number of trials (Bailey–López de Prado). Gate ≥ 0.5.
- **Stationary bootstrap Sharpe CI** (`validation.bootstrap_sharpe_payload`): Politis–
  Romano geometric-block resampling; report 95% CI; gate lower bound > 0.
- **Net return, turnover, cost** and the **funding-carry identity**:
  `rₜ = (r^spot_t − r^perp_t) + fₜ − cₜ` (long spot, short perp; short receives funding
  when fₜ>0); delta-neutral; the isolated-margin liquidation model
  (`carry.carry_liquidation_stressed`) and why the high unlevered Sharpe hides a crash
  tail.

## 5. Figures (committed PNGs via `scripts/make_mvp_figures.py`)

Deterministic, generated from cached/committed artifacts. matplotlib already a dep (3.10.9).

| file | content | source |
|---|---|---|
| `figures/funding_carry_equity.png` | pooled 8h-marked equity curve | re-run realism on cached data |
| `figures/funding_carry_per_year.png` | per-year net return bar (regime story) | realism manifest `honest_pooled_per_year` (8h-marked, consistent with equity/stress figures) |
| `figures/funding_carry_leverage_stress.png` | Sharpe & ann return vs leverage (the tail) | realism manifest |
| `figures/s1_fold_model_r2.png` | per-fold & per-model holdout R² | `experiments/alpha_s1/20260523-160541/metrics.json` |

Figures script reads the committed manifests where possible; only the funding pipeline is
re-run (seconds, cached). No network required if caches present; script degrades
gracefully (skips a figure with a warning if an artifact is missing, never fabricates).

## 6. `make mvp` quickstart

```
make mvp   # 1) run_funding_carry_v1.py  2) run_funding_carry_realism.py
           # 3) scripts/make_mvp_figures.py  4) echo the gate verdict + artifact paths
```

Runs on cached free data in seconds. README §6 documents this as the single entry point.
Also document the verification commands: `PYTHONPATH=src uv run pytest -q`,
`ruff check`, `mypy src`.

## 7. Blue-underline link map (evidence for every claim)

- Methodology → `docs/research/VALIDATION_RUNBOOK.md`, module files.
- Each branch verdict → its `docs/research/2026-05-NEGATIVE-RESULT-*.md` /
  `docs/research/intake/*` / `reports/signal_research/*`.
- Architecture → `docs/architecture/adrs/0001..0014`.
- Pipeline → `docs/superpowers/specs/*` + `plans/*`.
- Ops/safety → `docs/runbooks/*`.
- Capstone → `docs/research/2026-05-NEGATIVE-RESULT-FUNDING-CARRY.md`,
  `reports/signal_research/funding_carry_v1/*`, `manifests/funding_carry/*`.

Markdown links render as blue underlined text — these ARE the requested "blue underlines."

## 8. Testing / acceptance

- `make mvp` runs clean and regenerates the four figures + prints DO_NOT_ADVANCE.
- `scripts/make_mvp_figures.py` covered by a small smoke test (runs, writes PNGs to a
  tmp dir, handles a missing-artifact gracefully).
- `ruff check` + `mypy` clean on the new script.
- README links resolve to existing files (a link-check step in the plan).
- No claim in README contradicts the committed manifests/metrics (honesty review).

## 9. Risks / guardrails

- **Over-claiming** — mitigated by §0 guardrail + the honesty-review acceptance check.
- **Stale numbers** — figures/tables generated from committed artifacts, not hand-typed.
- **Scope creep into research** — explicitly out of scope (no retrain, no new branch).
- **PR discipline** — no PR until the operator says so; merge is operator-gated.
