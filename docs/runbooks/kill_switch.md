# Runbook: Kill switch

## Purpose
Halt all live trading immediately and require human re-arm.

## Trigger
Create a file at the repository root named `KILL_TRADING`:

```bash
touch /Users/dmr/MachineLearning/KILL_TRADING
```

The file's presence is checked on every order-placement attempt and once per minute
even when idle. The kill flag survives reboots.

## What happens on kill
1. S4 cancels all open orders on the active broker.
2. Position book takes a final snapshot to `data/snapshots/kill_<timestamp>.parquet`.
3. Audit log writes `kill_trigger` records for the active stage and trigger reason.
4. Process exits with code 137.

## Re-arming
1. Investigate root cause; document in `docs/runbooks/incident_<date>.md`.
2. Run reconciliation: `PYTHONPATH=src uv run python scripts/reconcile_book.py`.
3. Delete the `KILL_TRADING` file.
4. Restart the process. It refuses to start with `KILL_TRADING` present.
