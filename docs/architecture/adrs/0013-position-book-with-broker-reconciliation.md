# ADR 0013: In-memory position book with 60-second broker reconciliation

## Status
Accepted, 2026-05-20.

## Context
S4 needs to know its positions and equity at low latency to size the next trade.
Querying the broker on every order would add a network round-trip to the hot path.
Keeping an in-memory book without a check against the broker risks silent drift.

## Decision
Authoritative state lives in-memory inside the daemon process. Every 60 seconds
(configurable in `risk.yaml`) and on every fill, the book serializes to
`data/positions/<stage>/<YYYY-MM-DD>.parquet`. A separate async task queries the
broker's `account()` and `positions()` once per minute and compares equity:

```text
diff_bps = abs(book_equity - broker_equity) / broker_equity * 10000
```

If `diff_bps > 1.0` (configurable), we emit a `kill_trigger` audit row and exit
137. Snapshot files are chmod-a-w on date rotation so historical state is immutable.

## Alternatives considered
- Broker-as-source-of-truth: zero drift risk, but adds latency to every sizing
  decision and doesn't fit tick-level crypto cadence.
- Append-only ledger only (no snapshots): O(n) startup replay; cheap to write
  but unbounded to read.

## Consequences
Startup loads the latest snapshot, then reconciles. Drift > 1 bp at startup means
the daemon refuses to start until an operator investigates. The first
reconciliation pass after startup is a precondition for the first trade.
