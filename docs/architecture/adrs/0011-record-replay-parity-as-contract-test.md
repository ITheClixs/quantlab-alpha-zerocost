# ADR 0011: Record-replay parity is a contract test, not a development convention

## Status
Accepted, 2026-05-17.

## Context
The Recorder writes live events to disk. The Replayer reads them back as a
FeedAdapter. If they ever diverge — schema drift, dropped events, timestamp
rewriting, source-specific fields leaking into the recorded form — every downstream
test that uses the Replayer (which is most of S3's unit tests + the backtester) is
silently wrong.

## Decision
A single integration test, tests/integration/test_record_replay_parity.py, is the
contract test for the whole S3 abstraction:

  1. Connect a real FeedAdapter (BinanceWS by default; CoinbaseWS as a second variant).
  2. Record 60 seconds of events to a temp directory via Recorder.
  3. Disconnect.
  4. Read the same recorded Parquet shards through a Replayer at speed=0.
  5. Assert: event count matches, timestamp sequence matches byte-identically,
     no events dropped, no schema drift, no field added or removed.

If this test fails, S3 is broken regardless of what unit tests pass. The CI workflow
runs it on demand via `make s3-parity`.

The test is marked s3_integration and skipped from the default test run because it
requires network. It must pass before any release tag.

## Consequences
+ Single source of truth for the abstraction's correctness.
+ Catches schema drift the moment a vendor changes their wire format.
+ Catches recorder bugs that would otherwise corrupt every recorded hour silently.
- Requires network during CI execution; not run on every PR.
- A failing run blocks S4 development because S4 depends on this contract.
