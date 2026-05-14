# Runbook: Disaster recovery

## Scenarios

### Process crash mid-day
1. On restart, the process refuses to trade until reconciliation succeeds.
2. Run `PYTHONPATH=src uv run python scripts/reconcile_book.py --resync`.
3. Replay the audit log to reach the last known state.
4. Resume only after position book matches broker.

### Data feed loss
1. Kill switch fires automatically on 2 min crypto gap or 30 min equity gap.
2. Switch to backup feed in `configs/feeds.yaml` if available.
3. Replay missed window from `data/live/parquet/<symbol>/`.

### Broker outage
1. Cancel open orders via the broker's web UI manually.
2. Mark the broker offline in `configs/brokers.yaml`.
3. The kill switch fires automatically on next reconciliation failure.

### Corrupted audit log
1. Stop all trading.
2. Restore the last clean rotation from `data/snapshots/`.
3. Re-derive position state from broker via reconciliation.
4. Open an incident; the corrupted file goes to `logs/audit_corrupted/` for forensics.
