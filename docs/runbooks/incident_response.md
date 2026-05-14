# Runbook: Incident response

## Triggers
- Daily realized DD > 5% of account equity
- Cumulative DD > 15% from peak
- Two consecutive minutes without market data (crypto)
- Broker reconciliation mismatch
- NTP drift > 1 s

## Immediate steps
1. Touch `KILL_TRADING` in the repo root (see `kill_switch.md`).
2. Capture state: `PYTHONPATH=src uv run python scripts/capture_state.py --reason <text>`.
3. Notify the operator (out-of-band).

## Investigation
1. Read the last 24 hours of `logs/audit/`.
2. Replay the audit log: `PYTHONPATH=src uv run python scripts/audit_replay_check.py last-day`.
3. Compare positions with the broker (manual login, screenshot saved).

## Resolution
Document root cause, mitigation, and re-arm procedure in
`docs/runbooks/incident_<date>.md`. Commit before re-arming.
