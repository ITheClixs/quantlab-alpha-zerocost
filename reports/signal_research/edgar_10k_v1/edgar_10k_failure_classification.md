# EDGAR 10-K v1 — Failure Classification

**Primary failure class:** `low_frequency_insufficient_sample`
**Best text model:** `model_lgbm_text` (holdout IC 0.0213).

## Evidence
- Beats size/event-ret baselines: IC=True, spread=True, net PnL=True.
- Holdout IC 0.0213 vs placebo max 0.0153.
- Net LS PnL 3.20% (1x), 1.98% (2x cost).
- PBO None, DSR 0.0.

## Decision
Per the intake §16: classical 10-K text features did not clear the bar (beat OHLCV/factor baselines
on rank IC AND decile spread AND net long-short PnL). v1 closed at the stated failure class.
Embeddings/LLM features are NOT added in this run (would require a separate intake). The compact
LM-lexicon limitation is noted; a full-dictionary v2 could be reconsidered only with a fresh intake.
