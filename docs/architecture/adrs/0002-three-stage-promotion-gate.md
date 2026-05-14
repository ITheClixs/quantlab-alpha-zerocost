# ADR 0002: Three-stage promotion gate for real-money trading

## Status
Accepted, 2026-05-14.

## Context
The operator wants real-money commercial trading. The default CLAUDE.md bans live
trading; reversing that ban without structure is the fastest known way to lose money.

## Decision
A single environment variable `QUANTLAB_STAGE` controls the broker class loaded at
process start. Three stages:

- `paper` -> `brokers/*_paper.py`
- `live_shadow` -> `brokers/null_broker.py` + read-only real account
- `live` -> `brokers/*_live.py`

Promotion is human-only. The running process cannot promote itself. Each transition
requires a signed `docs/runbooks/stage_change.md` commit. Risk caps are cut to 50% for
the first 30 days after entering `live`.

## Consequences
+ Real-money path exists but is deliberate.
+ Every transition has an auditable artifact.
+ Risk caps + kill switch are stage-aware.
- Adds operational overhead to every promotion.
- Two-person review required for `configs/promotion.yaml` once `live_shadow`.
