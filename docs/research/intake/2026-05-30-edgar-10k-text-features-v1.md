# Intake — EDGAR 10-K Text Features v1 (research_only)

**Date:** 2026-05-30
**Status:** PRE-REGISTRATION · **`research_only`** · no strategy code until committed.
**Strategy name:** `edgar_10k_text_features_v1`
**Proposer:** QuantLab research
**Program `/goal`:** find taker-tradable alpha for QuantLab (this branch is research_only).

## 0. Context

The news/sentiment data audit (commit `fdad6cc`,
`reports/signal_research/news_sentiment_v1/`) found EDGAR 10-K to be the **first
dataset in the program that passes both hard data gates**: a true point-in-time
timestamp (SEC **filing date** = exact public-availability) and a
**survivorship-aware** historical-constituent universe (727 companies 2010-2022,
incl. delisted/merged names — TWITTER, CELGENE, XILINX, CERNER, ACTIVISION, MAXIM).
It ships full 10-K item text (Risk Factors, MD&A) and pre-computed forward-return
fields (usable as **labels only**). Every prior channel failed on cost,
vol-targeting subsumption, or data (paywall / survivorship / timestamp); this one
clears the data gate cleanly. This intake pre-registers a research_only test of it.

## 1. Strategy name and one-line description

`edgar_10k_text_features_v1` — test whether 10-K text features available at the SEC
filing date carry cross-sectional information for medium-horizon equity returns not
already captured by standard price/factor baselines.

## 2. Primary hypothesis

Annual 10-K textual information — especially **Risk Factors (item 1A)** and **MD&A
(item 7)** — contains slow-moving firm-level information about uncertainty, business
deterioration, litigation risk, liquidity stress, competitive pressure, and
management tone. These signals may predict **medium-horizon cross-sectional returns
after the filing date** (filing-drift / slow information diffusion; cf. Cohen,
Malloy & Nguyen 2020 on 10-K text changes; Loughran & McDonald 2011 on financial
sentiment lexicons). **This is not high-frequency news trading** — it is slow
post-filing information diffusion at a 1–12 month horizon.

## 3. Information source declaration

Driving channel (genuinely non-OHLCV): **SEC 10-K filing text**.
Tags: `sec_filing_text`, `edgar_10k`, `filing_timestamp_clean`,
`survivorship_safe_enough`, `research_only`. (Maps to the `earnings_fundamentals` /
text channels in the enum; a `sec_filing_text` value may be warranted. The
no-OHLCV-only-promotion rule is satisfied — the signal is filing text, not price.
Promotion is moot: research_only.)

## 4. Data source

- The EDGAR 10-K dataset audited in `news_sentiment_v1` (`jlohding__sp500-edgar-10k`).
- **SEC filing `date` is the signal timestamp.**
- Full 10-K item text available (`item_1` … `item_15`).
- Historical constituents incl. delisted / merged / renamed names (survivorship-aware).
- **Pre-computed forward-return fields** (`ret`, `1/3/5/10/20/40/60/80/100/150/252_day_return`,
  `mkt_cap`) — **treated as LABELS ONLY, never features.** (Build step must verify
  their null coverage and, ideally, spot-check against an independent price source;
  for research_only the dataset's own returns are the label source, which keeps the
  cross-section survivorship-safe — sidestepping the return-panel survivorship
  problem that closed options-IV.)

## 5. Signal-timestamp rule (binding)

- Filing date is **date t**.
- Text may be used **only after** the SEC filing date.
- Daily bars: **trade no earlier than t+1.**
- If exact filing time is unavailable, **enforce next-trading-day execution.**
- **Do NOT use the fiscal-period-end date as the signal date.**
- **Do NOT use pre-computed forward-return columns as features.**

## 6. Scope v1

Cross-sectional **equity research only**. No direct options trading. No intraday.
No same-day filing reaction (unless an explicit accepted-timestamp + market-hours
timing is available — not assumed in v1). No live or paper trading. **No promotion
language.**

## 7. Universe

- Names present in the audited EDGAR dataset (historical-constituent coverage).
- Trade only names with **valid forward-return label after the filing date**.
- **Do NOT filter by future survival.** Include delisted, acquired, renamed names
  when present. PIT membership = has a 10-K filing on date t with a valid label.
- Document mapping/label losses (no silent drops; emit a coverage table).

## 8. Text sections to use

- **Risk Factors** (item 1A), **MD&A** (item 7), **Business description** (item 1
  if available). **Full filing text — diagnostic only.**

## 9. Feature families (start simple and interpretable)

**9.1 Classical text features:** word/section length, year-over-year length change,
readability, uncertainty word count, litigation/risk word count, negative tone,
positive tone, modal/forward-looking language, numeric density, boilerplate
similarity. (Loughran-McDonald lexicons.)

**9.2 Change features:** YoY cosine similarity, YoY textual change, new risk-factor
language, deleted risk-factor language, MD&A tone change, abnormal length change.

**9.3 Embedding features (only AFTER the classical baseline is implemented):**
document/section embeddings from **raw text only**; no model may use future returns
or labels; embedding model/version recorded; **no fine-tuning on holdout**.

**9.4 LLM features (optional DIAGNOSTIC only in v1):** no black-box LLM sentiment as
a primary signal unless prompts, model, and deterministic settings are recorded;
**no future-aware summaries; no use of modern-LLM knowledge of later company
outcomes.**

## 10. Labels

Pre-declared horizons: **21-day, 63-day, 126-day, 252-day forward return.**
Primary target: **cross-sectional residualized return / rank return after the
filing date** — not raw market direction.

## 11. Baselines (first-class, in the PBO pool)

buy-and-hold / equal-weight universe; **size** (mkt_cap proxy); **momentum**;
**reversal**; **volatility**; **low-vol**; **sector-neutral random** (if sector
data, via `sic`); **OHLCV-only** baseline; **shuffled-text placebo**; **random-rank
placebo**; **inverted signal**.

## 12. Models (conservative first)

Ridge / ElasticNet → logistic or linear rank model (if classification) → **LightGBM
only after linear baselines are built**. **No neural nets in v1. No large-LLM
fine-tuning in v1.**

## 13. Validation

Chronological train / validation / holdout (no random splits except placebo
controls); PBO / DSR; bootstrap CIs; **rank IC**; **decile spread**; **long-short
net PnL**; cost model + **2× cost stress** + **1-bar delay**; concentration by
**stock, sector, month, year, and filing year**; crisis / regime checks; compare
against OHLCV + factor baselines; **no holdout tuning**.

## 14. Critical leakage rules

- Use SEC **filing date**, not fiscal-period-end.
- Text features at filing date t trade **no earlier than t+1** (unless an exact
  accepted timestamp supports same-day-after-filing execution — not assumed v1).
- **Forward-return columns are labels only.**
- Do not use restated/amended filings unless amendment handling is explicitly defined.
- Do not use future ticker mappings.
- Do not filter companies by future survival.
- Do not use LLM-generated features that may encode future company outcomes.
- Any embedding/LLM feature must be reproducible and recorded.

## 15. Pre-registered failure modes

`no_text_edge`, `text_signal_subsumed_by_ohlcv_or_factors`,
`low_frequency_insufficient_sample`, `cost_failure`, `high_turnover`,
`weak_rank_ic`, `decile_spread_not_monotonic`, `concentration_in_few_filing_years`,
`concentration_in_few_names`, `sector_concentration`, `embedding_overfit`,
`llm_feature_leakage_risk`, `holdout_failure`.

## 16. Decision rule

- If **classical** 10-K text features beat OHLCV/factor baselines on **rank IC,
  decile spread, AND net long-short PnL** → keep the branch alive; run a stricter v2.
- If only embeddings/LLM features work while classical fails → classify
  **research-interesting**, require a separate embedding/LLM audit before continuation.
- If text features do **not** beat OHLCV/factor baselines → close v1 as
  `no_text_edge` or `text_signal_subsumed_by_factors`.
- If the signal is concentrated in one filing year, one sector, or a few distressed
  names → classify as a **concentration failure**.

## 17. Promotion intent & sign-off

`research_only` throughout. No direct trading claims, no promotion language, no
live/paper. Full validation gate, chronological holdout, **no post-hoc tuning after
holdout**. Proposer: QuantLab research. Intake date: 2026-05-30.

## 18. Expected deliverables (after this intake is committed)

- `edgar_10k_text_features_intake.md` (this document)
- `edgar_10k_feature_registry.parquet`
- `edgar_10k_validation_report.md`
- `edgar_10k_rank_ic_report.md`
- `edgar_10k_decile_spread_report.md`
- `edgar_10k_baseline_comparison_report.md`
- `edgar_10k_failure_classification.md`

## What happens after this intake

1. This document is committed to `docs/research/intake/`.
2. Build the **classical** feature layer keyed to the filing date (t → t+1), with a
   coverage/label-validity table (no silent drops), forward returns as labels only.
3. Implement baselines + frozen classical variants; run the §13 validation battery;
   emit the §18 deliverables; classify per §16. Embeddings/LLM only after classical.
4. Commit the status classification. No re-tuning after holdout.
