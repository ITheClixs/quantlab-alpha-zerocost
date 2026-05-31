# Runbook: S3 recorder operations

## Purpose
Run the live event recorder. The recorder is the source of truth for replay,
backtests, and the S1 feature reconstruction layer (S3.1).

## Start the recorder
```bash
PYTHONPATH=src uv run python scripts/s3_record.py --config configs/feeds.yaml
```

The recorder:
- Subscribes to every venue+symbol declared in configs/feeds.yaml.
- Writes hour-rotated Parquet shards to data/live/parquet/<venue>/<symbol>/<date>/<hh>.parquet.
- Chmods each hour file read-only when it rotates.
- Logs flush latency + dropped count every minute.

S4 reads the latest `timestamp_utc` for each symbol from the same recorder root
through `RecordedFeedHeartbeat`. Start S3 before S4, or pass
`--feed-recording-root <root>` to `scripts/s4_execute.py` if the recorder writes
somewhere other than `data/live/parquet`.

## Stop the recorder
Send SIGTERM:
```bash
pkill -TERM -f "python.*s3_record.py"
```

The recorder drains its current minute, closes the active hour file, and exits.

## Disk hygiene
```bash
du -sh data/live/parquet/*
```
Each venue+symbol generates ~5-50 MB per hour at tick frequency.
At 24 symbols x 24 hours x 30 days that's roughly 100-300 GB/month.

Cleanup:
```bash
find data/live/parquet -mindepth 4 -maxdepth 4 -name "*.parquet" -mtime +30 -delete
```

## Failure modes
- WebSocket disconnect: the FeedAdapter reconnects automatically with exponential
  backoff (1 s -> 60 s, 10 attempts). After 10 failures the recorder logs an error
  and continues with other adapters. The failed adapter is restarted on the next
  hour rotation.
- Disk full: the recorder logs and drops events. The dropped_count stat surfaces
  the loss; if no fresh parquet events arrive, S4's feed-freshness gate blocks
  new orders.
- Schema drift: the parser tests catch this in CI before the recorder ever sees a
  malformed event.
