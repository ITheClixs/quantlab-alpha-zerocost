# ADR 0001: Two-tier predictor — tabular originates, LLM governs

## Status
Accepted, 2026-05-14.

## Context
The operator's initial request framed the system as "an LLM trading bot". Jane Street's
competition target is `responder_6`, a numeric regression. LLMs predict numerics by
sampling text tokens that decode to numbers; this is empirically worse than tree
ensembles on tabular finance data. The risk is that a literal "LLM as predictor" build
loses to a LightGBM baseline.

## Decision
S1 — a tabular stack (Ridge / LightGBM / XGBoost / CatBoost / MLP / 1D-CNN / linear
stacker) — is the only source of numeric forecasts. S2 — Mistral 22B / Yi 34B / smaller
LoRA'd models — acts as a governor: retrieves cited paper chunks, emits constrained
JSON, and can only veto, pass, or return insufficient_evidence. S2 never originates
trades.

## Consequences
+ Highest probability of measurable Kaggle-leaderboard performance.
+ Honest about LLM strengths (explanation, retrieval) and weaknesses (numerics).
+ S2 latency budget (5-30 s) is decoupled from S1 latency (<1 ms).
- Operators expecting "the LLM trades" must be re-educated.
- Two model classes to maintain.
