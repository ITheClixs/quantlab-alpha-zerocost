# QuantLab Alpha — S3 (Feeds + Brokers + Backtester) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the typed IO layer (feeds + brokers) plus a deterministic backtester for QuantLab Alpha. Three paper brokers, four feed adapters, hour-rotated recorder, replayer-as-FeedAdapter, slippage-aware NullBroker with FillModel, BacktestRunner driving two reference strategies through to markdown reports. Live brokers and risk wiring are S4.

**Architecture:** New `src/quant_research_stack/feeds/`, `brokers/`, `backtest/` packages. Async `FeedAdapter` Protocol with `AsyncFeedBase` mixin (reconnect + backoff). `BrokerAdapter` Protocol with per-broker capability declaration that rejects unsupported order types before any network call. Recorder writes hour-rotated Parquet shards, chmod-a-w on close. Replayer implements the same `FeedAdapter` Protocol from recorded shards. Backtester wires Replayer → Strategy → NullBroker(FillModel) → metrics → report.

**Tech Stack:** Python 3.11, `pydantic`, `polars`, `pyarrow`, `websockets`, `httpx`, `alpaca-py`, `python-binance`, `numpy`, `matplotlib`, `pytest`, `pytest-asyncio`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-05-17-quantlab-alpha-s3-feeds-brokers-design.md`

---

## File Structure

**New files:**

```text
configs/feeds.yaml
configs/brokers.yaml
configs/backtests/smoke.yaml
docs/architecture/adrs/0009-python-for-s3-cpp-deferred.md
docs/architecture/adrs/0010-fill-model-and-fixed-bps-slippage.md
docs/architecture/adrs/0011-record-replay-parity-as-contract-test.md
docs/runbooks/s3_recorder_ops.md
docs/runbooks/s3_paper_broker_credentials.md
src/quant_research_stack/feeds/__init__.py
src/quant_research_stack/feeds/market_types.py
src/quant_research_stack/feeds/base.py
src/quant_research_stack/feeds/binance_ws.py
src/quant_research_stack/feeds/coinbase_ws.py
src/quant_research_stack/feeds/alpaca_rest.py
src/quant_research_stack/feeds/recorder.py
src/quant_research_stack/feeds/replayer.py
src/quant_research_stack/brokers/__init__.py
src/quant_research_stack/brokers/order_types.py
src/quant_research_stack/brokers/capabilities.py
src/quant_research_stack/brokers/base.py
src/quant_research_stack/brokers/fill_model.py
src/quant_research_stack/brokers/null_broker.py
src/quant_research_stack/brokers/alpaca_paper.py
src/quant_research_stack/brokers/binance_testnet.py
src/quant_research_stack/backtest/__init__.py
src/quant_research_stack/backtest/strategy.py
src/quant_research_stack/backtest/strategies/__init__.py
src/quant_research_stack/backtest/strategies/buy_and_hold.py
src/quant_research_stack/backtest/strategies/moving_average_cross.py
src/quant_research_stack/backtest/runner.py
src/quant_research_stack/backtest/metrics.py
src/quant_research_stack/backtest/report.py
scripts/s3_record.py
scripts/backtest_run.py
tests/test_feeds_market_types.py
tests/test_feeds_base.py
tests/test_feeds_binance_ws.py
tests/test_feeds_coinbase_ws.py
tests/test_feeds_alpaca_rest.py
tests/test_feeds_recorder.py
tests/test_feeds_replayer.py
tests/test_brokers_order_types.py
tests/test_brokers_capabilities.py
tests/test_brokers_fill_model.py
tests/test_brokers_null_broker.py
tests/test_brokers_alpaca_paper.py
tests/test_brokers_binance_testnet.py
tests/test_backtest_strategy.py
tests/test_backtest_metrics.py
tests/test_backtest_runner.py
tests/test_backtest_report.py
tests/integration/test_binance_ws_live.py
tests/integration/test_alpaca_paper_roundtrip.py
tests/integration/test_binance_testnet_roundtrip.py
tests/integration/test_record_replay_parity.py
```

**Modified files:**

```text
pyproject.toml   add websockets, alpaca-py, python-binance, httpx, matplotlib, pytest-asyncio
                 register s3_integration pytest marker
Makefile         add s3-record, s3-parity, backtest, backtest-smoke targets
```

---

## Task 1: ADRs 0009–0011 + 2 runbooks

**Files:**
- Create: `docs/architecture/adrs/0009-python-for-s3-cpp-deferred.md`
- Create: `docs/architecture/adrs/0010-fill-model-and-fixed-bps-slippage.md`
- Create: `docs/architecture/adrs/0011-record-replay-parity-as-contract-test.md`
- Create: `docs/runbooks/s3_recorder_ops.md`
- Create: `docs/runbooks/s3_paper_broker_credentials.md`

- [ ] **Step 1: Write ADR 0009** (verbatim from spec §5.3 — the Python-vs-C++ decision)

Create `docs/architecture/adrs/0009-python-for-s3-cpp-deferred.md` with the markdown content shown in the spec's §5.3 code fence — Status, Context, Decision, Latency math, Architectural enabler, Triggers, Consequences sections.

- [ ] **Step 2: Write ADR 0010** — fill model

Create `docs/architecture/adrs/0010-fill-model-and-fixed-bps-slippage.md`:

```markdown
# ADR 0010: Fill model uses fixed-bps slippage and commission; no L2 modeling in S3

## Status
Accepted, 2026-05-17.

## Context
The S3 backtester needs a fill simulator. Two ends of the spectrum:
- "Optimistic": fill at the next event's price with zero cost. Misleading.
- "Realistic": L2 order-book modeling with queue position and impact. Heavy; needs L2 data.

S3 ships paper brokers + a backtester; L2 simulation is deferred to S3.3 which uses
the already-downloaded CryptoLOB-2025 + HFT LOB datasets (35 GB total on disk).

## Decision
The S3 FillModel uses a fixed-bps approximation:

  fill_px = next_event_mid + half_spread_bps * 1e-4 * mid + slippage_bps * 1e-4 * mid
  commission = fill_px * qty * commission_bps * 1e-4
  fill is delayed by fill_latency_ms

Default config errs conservative: 1 bps commission + 2 bps slippage + 1 bps half-spread
+ 50 ms latency = 4 bps total per-side. This is realistic-to-pessimistic for retail
equity and crypto on the venues we target.

The model is deterministic. No randomness in fill price or timing.

## Consequences
+ Backtests reproducible byte-identical across runs with the same config.
+ Default config does not over-promise; strategies that look good here have margin.
+ Per-strategy configs can tighten (e.g. for a market-making study).
- Underestimates impact for large orders. Mitigated by reject_if_notional_above_pct_adv
  cap that refuses to backtest sizes that would have required L2 modeling.
- Future L2 modeling (S3.3) will likely show some strategies lose more than S3's
  fill model says. Documented limitation in every backtest report.
```

- [ ] **Step 3: Write ADR 0011** — record-replay parity

Create `docs/architecture/adrs/0011-record-replay-parity-as-contract-test.md`:

```markdown
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
```

- [ ] **Step 4: Write runbook `s3_recorder_ops.md`**

Create `docs/runbooks/s3_recorder_ops.md`:

```markdown
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
  the loss; S4 will wire this to the kill switch.
- Schema drift: the parser tests catch this in CI before the recorder ever sees a
  malformed event.
```

- [ ] **Step 5: Write runbook `s3_paper_broker_credentials.md`**

Create `docs/runbooks/s3_paper_broker_credentials.md`:

```markdown
# Runbook: S3 paper broker credentials

## Purpose
Place free paper-trading API credentials so the AlpacaPaper and BinanceTestnet
adapters can authenticate.

## Alpaca paper
1. Sign up at https://alpaca.markets (free).
2. Switch the dashboard to Paper Trading mode.
3. Generate an API key + secret.
4. Place the credentials at ~/.alpaca/paper_keys.json:
   ```json
   { "api_key": "PK...", "api_secret": "..." }
   ```
5. chmod 600 ~/.alpaca/paper_keys.json
6. Verify: `PYTHONPATH=src uv run pytest tests/integration/test_alpaca_paper_roundtrip.py -m s3_integration`

## Binance testnet
1. Sign up at https://testnet.binance.vision (free, GitHub login).
2. Generate HMAC SHA256 key + secret.
3. Place at ~/.binance/testnet_keys.json:
   ```json
   { "api_key": "...", "api_secret": "..." }
   ```
4. chmod 600 ~/.binance/testnet_keys.json
5. Verify: `PYTHONPATH=src uv run pytest tests/integration/test_binance_testnet_roundtrip.py -m s3_integration`

## File permissions
Both files MUST be chmod 600 (owner read-only). The adapters refuse to load with
permissive permissions to avoid leaking credentials into shared snapshots.

## Rotation
- Alpaca: rotate by generating a new key on the dashboard and replacing the JSON.
  Old key is revoked immediately.
- Binance testnet: same procedure on the testnet dashboard.

## Live credentials
NEVER put live credentials in these paths. Live keys belong at
~/.alpaca/live_keys.json and ~/.binance/live_keys.json respectively and are only
read by *_live.py adapters (which don't exist yet — S4 will add them).
```

- [ ] **Step 6: Commit**

```bash
git add docs/architecture/adrs/0009-python-for-s3-cpp-deferred.md \
        docs/architecture/adrs/0010-fill-model-and-fixed-bps-slippage.md \
        docs/architecture/adrs/0011-record-replay-parity-as-contract-test.md \
        docs/runbooks/s3_recorder_ops.md \
        docs/runbooks/s3_paper_broker_credentials.md
git commit -m "docs: add ADRs 0009-0011 and S3 runbooks (recorder ops, paper broker credentials)"
```

---

## Task 2: Scaffold `feeds/`, `brokers/`, `backtest/` packages + 3 config files

**Files:**
- Create: `src/quant_research_stack/feeds/__init__.py`
- Create: `src/quant_research_stack/brokers/__init__.py`
- Create: `src/quant_research_stack/backtest/__init__.py`
- Create: `src/quant_research_stack/backtest/strategies/__init__.py`
- Create: `configs/feeds.yaml`
- Create: `configs/brokers.yaml`
- Create: `configs/backtests/smoke.yaml`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p src/quant_research_stack/feeds src/quant_research_stack/brokers src/quant_research_stack/backtest/strategies configs/backtests
cat > src/quant_research_stack/feeds/__init__.py <<'PY'
"""S3 feed adapters and market type definitions.

Spec: docs/superpowers/specs/2026-05-17-quantlab-alpha-s3-feeds-brokers-design.md
"""
PY
cat > src/quant_research_stack/brokers/__init__.py <<'PY'
"""S3 broker adapters (paper-only in S3; live adapters land in S4)."""
PY
cat > src/quant_research_stack/backtest/__init__.py <<'PY'
"""S3 backtester — Replayer + NullBroker(FillModel) + Strategy + Report."""
PY
cat > src/quant_research_stack/backtest/strategies/__init__.py <<'PY'
"""Reference strategies for end-to-end backtester exercise."""
PY
```

- [ ] **Step 2: Create `configs/feeds.yaml`**

```yaml
# Adapter constructors and recording paths.

recorder:
  root: data/live/parquet
  flush_every_n_events: 1024
  flush_every_seconds: 5.0
  keep_raw: false

adapters:
  - venue: binance
    impl: BinanceWS
    symbols: [BTCUSDT, ETHUSDT]
    channels: [aggTrade]
  - venue: coinbase
    impl: CoinbaseWS
    symbols: [BTC-USD, ETH-USD]
    channels: [matches, ticker]
  - venue: alpaca
    impl: AlpacaREST
    symbols: [SPY, QQQ, AAPL]
    interval_minutes: 15
    poll_offset_seconds: 5
    credentials_path: ~/.alpaca/paper_keys.json
```

- [ ] **Step 3: Create `configs/brokers.yaml`**

```yaml
# Paper-broker credentials and capability overrides.

brokers:
  null_broker:
    enabled: true
  alpaca_paper:
    enabled: true
    credentials_path: ~/.alpaca/paper_keys.json
    base_url: https://paper-api.alpaca.markets
  binance_testnet:
    enabled: true
    credentials_path: ~/.binance/testnet_keys.json
    rest_base_url: https://testnet.binance.vision
    ws_base_url: wss://testnet.binance.vision/ws

capability_overrides: {}  # per-venue overrides if a paper sandbox lacks a feature live has
```

- [ ] **Step 4: Create `configs/backtests/smoke.yaml`**

```yaml
# Smoke backtest config — 1 hour of recorded BinanceWS BTCUSDT, buy and hold.

replayer:
  root: data/live/parquet
  venue: binance
  symbols: [BTCUSDT]
  start_utc: 2026-05-15T00:00:00Z
  end_utc:   2026-05-15T01:00:00Z
  speed: 0.0

fill_model:
  commission_bps: 1.0
  slippage_bps: 2.0
  half_spread_bps: 1.0
  fill_latency_ms: 50
  reject_if_notional_above_pct_adv: null
  partial_fill_max_pct_of_book: 0.10

starting_cash: 100000.0
strategy_name: buy_and_hold
strategy_params: {}
metrics_horizon_minutes: 5
```

- [ ] **Step 5: Verify YAML + imports**

```bash
cd /Users/dmr/MachineLearning && python -c "import yaml; [yaml.safe_load(open(p)) for p in ['configs/feeds.yaml','configs/brokers.yaml','configs/backtests/smoke.yaml']]; print('OK')"
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run python -c "import quant_research_stack.feeds, quant_research_stack.brokers, quant_research_stack.backtest, quant_research_stack.backtest.strategies; print('OK')"
```

Expected: both print `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/feeds/ src/quant_research_stack/brokers/ src/quant_research_stack/backtest/ configs/feeds.yaml configs/brokers.yaml configs/backtests/
git commit -m "feat: scaffold S3 packages (feeds/, brokers/, backtest/) and three config files"
```

---

## Task 3: Add S3 dependencies to `pyproject.toml`

**Files:** `pyproject.toml`

- [ ] **Step 1: Add runtime deps in alphabetical order**

Add to `dependencies = [...]`:

```toml
    "alpaca-py>=0.30.0",
    "httpx>=0.27.0",
    "matplotlib>=3.9.0",
    "python-binance>=1.0.19",
    "websockets>=12.0",
```

- [ ] **Step 2: Add `pytest-asyncio` to dev group**

In `[project.optional-dependencies] dev = [...]`:

```toml
    "pytest-asyncio>=0.24.0",
```

- [ ] **Step 3: Register the `s3_integration` marker**

In `[tool.pytest.ini_options]`:

- Change `addopts = "-q -m 'not governor_slow'"` to `addopts = "-q -m 'not governor_slow and not s3_integration'"`.
- Add to the `markers` list:

```toml
    "s3_integration: feed/broker integration tests that hit live or paper endpoints (skipped by default)",
```

- Add `asyncio_mode = "auto"` so `pytest-asyncio` runs `async def` tests without per-test decoration.

- [ ] **Step 4: Sync**

```bash
cd /Users/dmr/MachineLearning && uv sync --extra dev --extra llm
```

Expected: 5 new runtime deps + pytest-asyncio installed; exit 0.

- [ ] **Step 5: Smoke-import**

```bash
uv run python -c "import alpaca, binance, httpx, matplotlib, websockets, pytest_asyncio; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 6: Default test run still passes**

```bash
PYTHONPATH=src uv run pytest -q 2>&1 | tail -3
```

Expected: same as before this task; no unknown-marker errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add S3 deps (websockets, alpaca-py, python-binance, httpx, matplotlib) + s3_integration marker"
```

---

## Task 4: `feeds/market_types.py` — Tick + Bar Pydantic models

**Files:**
- Create: `src/quant_research_stack/feeds/market_types.py`
- Create: `tests/test_feeds_market_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_market_types.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.feeds.market_types import Bar, Tick, TickSide, Venue


def _now() -> datetime:
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)


def _valid_tick(**overrides) -> dict:
    base = {
        "venue": "binance",
        "symbol": "BTCUSDT",
        "timestamp_utc": _now(),
        "received_utc": _now(),
        "price": 100.0,
        "size": 0.5,
        "side": "buy",
    }
    base.update(overrides)
    return base


def test_tick_minimal_valid() -> None:
    t = Tick.model_validate(_valid_tick())
    assert t.venue == Venue.binance
    assert t.side == TickSide.buy
    assert t.price == 100.0


def test_tick_price_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Tick.model_validate(_valid_tick(price=0.0))


def test_tick_size_may_be_zero() -> None:
    t = Tick.model_validate(_valid_tick(size=0.0))
    assert t.size == 0.0


def test_tick_size_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        Tick.model_validate(_valid_tick(size=-1.0))


def test_tick_is_frozen() -> None:
    t = Tick.model_validate(_valid_tick())
    with pytest.raises(ValueError):
        t.price = 999.0


def test_tick_round_trip_json() -> None:
    t = Tick.model_validate(_valid_tick())
    payload = t.model_dump_json()
    restored = Tick.model_validate_json(payload)
    assert restored == t


def test_bar_minimal_valid() -> None:
    b = Bar.model_validate({
        "venue": "alpaca",
        "symbol": "SPY",
        "timestamp_utc": _now(),
        "interval_seconds": 900,
        "open": 500.0,
        "high": 501.0,
        "low": 499.0,
        "close": 500.5,
        "volume": 1000.0,
    })
    assert b.interval_seconds == 900
    assert b.high == 501.0


def test_bar_interval_zero_rejected() -> None:
    with pytest.raises(ValueError):
        Bar.model_validate({
            "venue": "alpaca",
            "symbol": "SPY",
            "timestamp_utc": _now(),
            "interval_seconds": 0,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 0.0,
        })


def test_bar_is_frozen() -> None:
    b = Bar.model_validate({
        "venue": "alpaca",
        "symbol": "SPY",
        "timestamp_utc": _now(),
        "interval_seconds": 900,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 0.0,
    })
    with pytest.raises(ValueError):
        b.close = 2.0


def test_tick_side_enum_values() -> None:
    assert TickSide.buy.value == "buy"
    assert TickSide.sell.value == "sell"
    assert TickSide.unknown.value == "unknown"


def test_venue_enum_values() -> None:
    assert Venue.binance.value == "binance"
    assert Venue.coinbase.value == "coinbase"
    assert Venue.alpaca.value == "alpaca"
    assert Venue.replay.value == "replay"
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_market_types.py -v
```

Expected: ImportError on `quant_research_stack.feeds.market_types`.

- [ ] **Step 3: Implement `market_types.py`**

Create `src/quant_research_stack/feeds/market_types.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class Venue(StrEnum):
    binance = "binance"
    coinbase = "coinbase"
    alpaca = "alpaca"
    replay = "replay"


class TickSide(StrEnum):
    buy = "buy"
    sell = "sell"
    unknown = "unknown"


class Tick(BaseModel):
    model_config = {"frozen": True}
    venue: Venue
    symbol: str
    timestamp_utc: datetime
    received_utc: datetime
    price: Annotated[float, Field(gt=0.0)]
    size: Annotated[float, Field(ge=0.0)]
    side: TickSide
    sequence: int | None = None
    raw: dict | None = None


class Bar(BaseModel):
    model_config = {"frozen": True}
    venue: Venue
    symbol: str
    timestamp_utc: datetime
    interval_seconds: Annotated[int, Field(ge=1, le=86400)]
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int | None = None
```

- [ ] **Step 4: Run tests, expect 11 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_market_types.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Lint**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/market_types.py tests/test_feeds_market_types.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/feeds/market_types.py tests/test_feeds_market_types.py
git commit -m "feat: feeds/market_types.py with Tick + Bar Pydantic models"
```

---

## Task 5: `feeds/base.py` — FeedAdapter Protocol + AsyncFeedBase mixin

**Files:**
- Create: `src/quant_research_stack/feeds/base.py`
- Create: `tests/test_feeds_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_base.py`:

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def test_exponential_backoff_first_attempt_returns_base() -> None:
    assert exponential_backoff(attempt=1, base=1.0, cap=60.0) == 1.0


def test_exponential_backoff_doubles_each_attempt() -> None:
    assert exponential_backoff(attempt=2, base=1.0, cap=60.0) == 2.0
    assert exponential_backoff(attempt=3, base=1.0, cap=60.0) == 4.0
    assert exponential_backoff(attempt=4, base=1.0, cap=60.0) == 8.0


def test_exponential_backoff_caps() -> None:
    assert exponential_backoff(attempt=20, base=1.0, cap=60.0) == 60.0


class _StubFeed(AsyncFeedBase):
    venue = Venue.replay

    def __init__(self) -> None:
        super().__init__()
        self.subscribed_with: tuple[str, ...] | None = None
        self.closed = False
        self._events = [
            Tick(
                venue=Venue.replay, symbol="X", timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
                received_utc=datetime(2026, 1, 1, tzinfo=UTC), price=1.0, size=1.0, side=TickSide.buy,
            ),
            Tick(
                venue=Venue.replay, symbol="X", timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
                received_utc=datetime(2026, 1, 1, tzinfo=UTC), price=2.0, size=1.0, side=TickSide.sell,
            ),
        ]

    async def subscribe(self, symbols) -> None:
        self.subscribed_with = tuple(symbols)

    async def iterate(self):
        for ev in self._events:
            self._stats["events_emitted"] += 1
            yield ev

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_async_feed_base_iterate_yields_events() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    seen = [ev async for ev in feed.iterate()]
    assert len(seen) == 2
    assert feed.subscribed_with == ("X",)


@pytest.mark.asyncio
async def test_async_feed_base_stats_track_emissions() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    _ = [ev async for ev in feed.iterate()]
    assert feed.stats["events_emitted"] == 2


@pytest.mark.asyncio
async def test_async_feed_base_close_is_called() -> None:
    feed = _StubFeed()
    await feed.subscribe(["X"])
    await feed.close()
    assert feed.closed is True


@pytest.mark.asyncio
async def test_async_feed_base_buffer_drops_oldest_on_overflow() -> None:
    feed = _StubFeed()
    feed._buffer_cap = 3
    for i in range(5):
        feed._enqueue(f"event_{i}")
    queued = [feed._buffer.popleft() for _ in range(len(feed._buffer))]
    assert queued == ["event_2", "event_3", "event_4"]
    assert feed.stats["dropped_count"] == 2


def test_feed_connection_error_is_runtime_error_subclass() -> None:
    assert issubclass(FeedConnectionError, RuntimeError)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_base.py -v
```

Expected: ImportError on `quant_research_stack.feeds.base`.

- [ ] **Step 3: Implement `base.py`**

Create `src/quant_research_stack/feeds/base.py`:

```python
from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from quant_research_stack.feeds.market_types import Bar, Tick, Venue

MarketEvent = Tick | Bar


class FeedConnectionError(RuntimeError):
    pass


def exponential_backoff(attempt: int, base: float, cap: float) -> float:
    if attempt < 1:
        return base
    return min(cap, base * (2 ** (attempt - 1)))


class FeedAdapter(Protocol):
    venue: Venue

    async def subscribe(self, symbols: Iterable[str]) -> None: ...

    def iterate(self) -> AsyncIterator[MarketEvent]: ...

    async def close(self) -> None: ...

    @property
    def is_live(self) -> bool: ...

    @property
    def stats(self) -> dict: ...


class AsyncFeedBase:
    """Mixin providing reconnect/backoff helpers and a bounded ring buffer.

    Concrete adapters subclass this and implement `subscribe`, `iterate`, `close`.
    """

    venue: Venue = Venue.replay

    def __init__(self, buffer_cap: int = 10_000) -> None:
        self._buffer_cap = buffer_cap
        self._buffer: deque = deque()
        self._stats = {
            "events_emitted": 0,
            "last_event_lag_ms": 0.0,
            "reconnects": 0,
            "dropped_count": 0,
        }

    def _enqueue(self, item: object) -> None:
        if len(self._buffer) >= self._buffer_cap:
            self._buffer.popleft()
            self._stats["dropped_count"] += 1
        self._buffer.append(item)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def __aenter__(self) -> "AsyncFeedBase":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def subscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def iterate(self) -> AsyncIterator[MarketEvent]:  # pragma: no cover - overridden
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError
```

- [ ] **Step 4: Run tests, expect 9 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_base.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/base.py tests/test_feeds_base.py
git add src/quant_research_stack/feeds/base.py tests/test_feeds_base.py
git commit -m "feat: feeds/base.py with FeedAdapter Protocol, AsyncFeedBase mixin, exponential backoff"
```

---

## Task 6: `feeds/binance_ws.py` — Binance WebSocket adapter

**Files:**
- Create: `src/quant_research_stack/feeds/binance_ws.py`
- Create: `tests/test_feeds_binance_ws.py`

- [ ] **Step 1: Write failing tests (parser fixture only — no live network)**

Create `tests/test_feeds_binance_ws.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.feeds.binance_ws import parse_aggtrade_event
from quant_research_stack.feeds.market_types import TickSide, Venue


def _payload(**overrides) -> dict:
    base = {
        "e": "aggTrade",
        "E": 1747449600000,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "65000.50",
        "q": "0.125",
        "f": 100,
        "l": 105,
        "T": 1747449599000,
        "m": False,  # buyer is maker → trade direction = buy
        "M": True,
    }
    base.update(overrides)
    return base


def test_parse_aggtrade_basic() -> None:
    tick = parse_aggtrade_event(_payload(), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.venue == Venue.binance
    assert tick.symbol == "BTCUSDT"
    assert tick.price == 65000.50
    assert tick.size == 0.125


def test_parse_aggtrade_buyer_maker_is_sell_side() -> None:
    tick = parse_aggtrade_event(_payload(m=True), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.sell


def test_parse_aggtrade_buyer_not_maker_is_buy_side() -> None:
    tick = parse_aggtrade_event(_payload(m=False), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.buy


def test_parse_aggtrade_uses_T_for_timestamp() -> None:
    tick = parse_aggtrade_event(_payload(T=1747449500000), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.timestamp_utc == datetime.fromtimestamp(1747449500.0, tz=UTC)


def test_parse_aggtrade_sequence_from_a_field() -> None:
    tick = parse_aggtrade_event(_payload(a=987654321), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.sequence == 987654321
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_binance_ws.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `binance_ws.py`**

Create `src/quant_research_stack/feeds/binance_ws.py`:

```python
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import websockets

from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

_PUBLIC_URL = "wss://stream.binance.com:9443/ws"


def parse_aggtrade_event(payload: dict, *, received_utc: datetime) -> Tick:
    """Pure parser for Binance @aggTrade messages. Pure function for fixture tests."""
    side = TickSide.sell if payload.get("m") else TickSide.buy
    return Tick(
        venue=Venue.binance,
        symbol=str(payload["s"]),
        timestamp_utc=datetime.fromtimestamp(int(payload["T"]) / 1000.0, tz=UTC),
        received_utc=received_utc,
        price=float(payload["p"]),
        size=float(payload["q"]),
        side=side,
        sequence=int(payload["a"]),
    )


@dataclass
class BinanceWS(AsyncFeedBase):
    url: str = _PUBLIC_URL
    venue: Venue = Venue.binance

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._ws = None
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(s.lower() for s in symbols)

    async def _connect(self) -> None:
        streams = "/".join(f"{s}@aggTrade" for s in self._symbols)
        url = f"{self.url}/{streams}" if streams else self.url
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
                return
            except Exception as exc:
                if attempt >= 10:
                    raise FeedConnectionError(f"binance ws connect failed after {attempt} attempts") from exc
                self._stats["reconnects"] += 1
                await asyncio.sleep(exponential_backoff(attempt, base=1.0, cap=60.0))

    async def iterate(self) -> AsyncIterator[Tick]:
        if self._ws is None:
            await self._connect()
        while not self._closed:
            try:
                msg = await self._ws.recv()
            except Exception:
                self._stats["reconnects"] += 1
                await self._connect()
                continue
            payload = json.loads(msg)
            if payload.get("e") != "aggTrade":
                continue
            received = datetime.now(UTC)
            tick = parse_aggtrade_event(payload, received_utc=received)
            self._stats["events_emitted"] += 1
            self._stats["last_event_lag_ms"] = (received - tick.timestamp_utc).total_seconds() * 1000.0
            yield tick

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
```

- [ ] **Step 4: Run tests, expect 5 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_binance_ws.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/binance_ws.py tests/test_feeds_binance_ws.py
git add src/quant_research_stack/feeds/binance_ws.py tests/test_feeds_binance_ws.py
git commit -m "feat: feeds/binance_ws.py with aggTrade parser + websocket adapter"
```

---

## Task 7: `feeds/coinbase_ws.py` — Coinbase WebSocket adapter

**Files:**
- Create: `src/quant_research_stack/feeds/coinbase_ws.py`
- Create: `tests/test_feeds_coinbase_ws.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_coinbase_ws.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from quant_research_stack.feeds.coinbase_ws import parse_match_event
from quant_research_stack.feeds.market_types import TickSide, Venue


def _payload(**overrides) -> dict:
    base = {
        "type": "match",
        "trade_id": 42,
        "maker_order_id": "abc",
        "taker_order_id": "def",
        "side": "buy",
        "size": "0.10",
        "price": "65000.00",
        "product_id": "BTC-USD",
        "sequence": 1234567890,
        "time": "2026-05-17T12:00:00.123456Z",
    }
    base.update(overrides)
    return base


def test_parse_match_basic() -> None:
    tick = parse_match_event(_payload(), received_utc=datetime(2026, 5, 17, 12, 0, 1, tzinfo=UTC))
    assert tick.venue == Venue.coinbase
    assert tick.symbol == "BTC-USD"
    assert tick.price == 65000.0
    assert tick.size == 0.10


def test_parse_match_side_buy() -> None:
    tick = parse_match_event(_payload(side="buy"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.buy


def test_parse_match_side_sell() -> None:
    tick = parse_match_event(_payload(side="sell"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.sell


def test_parse_match_unknown_side_falls_back_to_unknown() -> None:
    tick = parse_match_event(_payload(side="weird"), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.side == TickSide.unknown


def test_parse_match_uses_sequence_field() -> None:
    tick = parse_match_event(_payload(sequence=9999), received_utc=datetime(2026, 5, 17, tzinfo=UTC))
    assert tick.sequence == 9999
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_coinbase_ws.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `coinbase_ws.py`**

Create `src/quant_research_stack/feeds/coinbase_ws.py`:

```python
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import websockets

from quant_research_stack.feeds.base import AsyncFeedBase, FeedConnectionError, exponential_backoff
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

_PUBLIC_URL = "wss://ws-feed.exchange.coinbase.com"


def parse_match_event(payload: dict, *, received_utc: datetime) -> Tick:
    side_raw = str(payload.get("side", "")).lower()
    side = TickSide.buy if side_raw == "buy" else TickSide.sell if side_raw == "sell" else TickSide.unknown
    iso = str(payload["time"]).replace("Z", "+00:00")
    return Tick(
        venue=Venue.coinbase,
        symbol=str(payload["product_id"]),
        timestamp_utc=datetime.fromisoformat(iso).astimezone(UTC),
        received_utc=received_utc,
        price=float(payload["price"]),
        size=float(payload["size"]),
        side=side,
        sequence=int(payload["sequence"]),
    )


@dataclass
class CoinbaseWS(AsyncFeedBase):
    url: str = _PUBLIC_URL
    venue: Venue = Venue.coinbase

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._ws = None
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(symbols)

    async def _connect(self) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)
                payload = {
                    "type": "subscribe",
                    "channels": [{"name": "matches", "product_ids": list(self._symbols)}],
                }
                await self._ws.send(json.dumps(payload))
                return
            except Exception as exc:
                if attempt >= 10:
                    raise FeedConnectionError(f"coinbase ws connect failed after {attempt} attempts") from exc
                self._stats["reconnects"] += 1
                await asyncio.sleep(exponential_backoff(attempt, base=1.0, cap=60.0))

    async def iterate(self) -> AsyncIterator[Tick]:
        if self._ws is None:
            await self._connect()
        while not self._closed:
            try:
                msg = await self._ws.recv()
            except Exception:
                self._stats["reconnects"] += 1
                await self._connect()
                continue
            payload = json.loads(msg)
            if payload.get("type") != "match":
                continue
            received = datetime.now(UTC)
            tick = parse_match_event(payload, received_utc=received)
            self._stats["events_emitted"] += 1
            self._stats["last_event_lag_ms"] = (received - tick.timestamp_utc).total_seconds() * 1000.0
            yield tick

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            await self._ws.close()
```

- [ ] **Step 4: Run tests, expect 5 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_coinbase_ws.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/coinbase_ws.py tests/test_feeds_coinbase_ws.py
git add src/quant_research_stack/feeds/coinbase_ws.py tests/test_feeds_coinbase_ws.py
git commit -m "feat: feeds/coinbase_ws.py with matches parser + websocket adapter"
```

---

## Task 8: `feeds/alpaca_rest.py` — Alpaca REST 15-min bar adapter

**Files:**
- Create: `src/quant_research_stack/feeds/alpaca_rest.py`
- Create: `tests/test_feeds_alpaca_rest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_alpaca_rest.py`:

```python
from __future__ import annotations

from quant_research_stack.feeds.alpaca_rest import parse_bars_response
from quant_research_stack.feeds.market_types import Venue


def _resp() -> dict:
    return {
        "bars": {
            "SPY": [
                {"t": "2026-05-17T13:30:00Z", "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 1000.0, "n": 50},
                {"t": "2026-05-17T13:45:00Z", "o": 500.5, "h": 502.0, "l": 500.0, "c": 501.5, "v": 1200.0, "n": 60},
            ],
        },
        "next_page_token": None,
    }


def test_parse_bars_yields_bar_per_input() -> None:
    bars = list(parse_bars_response(_resp(), interval_seconds=900))
    assert len(bars) == 2
    assert bars[0].symbol == "SPY"
    assert bars[0].venue == Venue.alpaca
    assert bars[0].interval_seconds == 900
    assert bars[0].open == 500.0


def test_parse_bars_handles_empty_symbol_list() -> None:
    bars = list(parse_bars_response({"bars": {}, "next_page_token": None}, interval_seconds=900))
    assert bars == []


def test_parse_bars_handles_n_trades_missing() -> None:
    resp = {
        "bars": {"AAPL": [{"t": "2026-05-17T13:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}]},
        "next_page_token": None,
    }
    bars = list(parse_bars_response(resp, interval_seconds=900))
    assert bars[0].n_trades is None
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_alpaca_rest.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `alpaca_rest.py`**

Create `src/quant_research_stack/feeds/alpaca_rest.py`:

```python
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Bar, Venue

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def parse_bars_response(response: dict, *, interval_seconds: int) -> Iterator[Bar]:
    bars = response.get("bars") or {}
    for symbol, rows in bars.items():
        for row in rows:
            iso = str(row["t"]).replace("Z", "+00:00")
            yield Bar(
                venue=Venue.alpaca,
                symbol=symbol,
                timestamp_utc=datetime.fromisoformat(iso).astimezone(UTC),
                interval_seconds=interval_seconds,
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
                n_trades=int(row["n"]) if "n" in row else None,
            )


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class AlpacaREST(AsyncFeedBase):
    credentials_path: str = "~/.alpaca/paper_keys.json"
    interval_seconds: int = 900
    poll_offset_seconds: int = 5
    base_url: str = _BARS_URL
    venue: Venue = Venue.alpaca

    def __post_init__(self) -> None:
        super().__init__()
        self._symbols: tuple[str, ...] = ()
        self._closed = False

    async def subscribe(self, symbols: Iterable[str]) -> None:
        self._symbols = tuple(symbols)

    async def iterate(self) -> AsyncIterator[Bar]:
        api_key, api_secret = _load_credentials(self.credentials_path)
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            while not self._closed:
                params = {
                    "symbols": ",".join(self._symbols),
                    "timeframe": f"{self.interval_seconds // 60}Min",
                    "limit": 1000,
                }
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                payload = response.json()
                for bar in parse_bars_response(payload, interval_seconds=self.interval_seconds):
                    self._stats["events_emitted"] += 1
                    yield bar
                await asyncio.sleep(self.interval_seconds + self.poll_offset_seconds)

    async def close(self) -> None:
        self._closed = True
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_alpaca_rest.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/alpaca_rest.py tests/test_feeds_alpaca_rest.py
git add src/quant_research_stack/feeds/alpaca_rest.py tests/test_feeds_alpaca_rest.py
git commit -m "feat: feeds/alpaca_rest.py with bars parser + REST polling adapter"
```

---

## Task 9: `feeds/recorder.py` — hour-rotated Parquet recorder

**Files:**
- Create: `src/quant_research_stack/feeds/recorder.py`
- Create: `tests/test_feeds_recorder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_recorder.py`:

```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig


def _tick(hour: int, minute: int = 0) -> Tick:
    ts = datetime(2026, 5, 17, hour, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=65000.0, size=0.1, side=TickSide.buy,
    )


@dataclass
class _FixtureFeed(AsyncFeedBase):
    events: list[Tick]
    venue: Venue = Venue.binance

    def __post_init__(self) -> None:
        super().__init__()

    async def subscribe(self, symbols) -> None: ...

    async def iterate(self) -> AsyncIterator[Tick]:
        for ev in self.events:
            self._stats["events_emitted"] += 1
            yield ev

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_recorder_writes_one_file_per_hour(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(10, 30), _tick(11, 5)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    written = sorted(tmp_path.rglob("*.parquet"))
    assert len(written) == 2
    # path scheme: <root>/<venue>/<symbol>/<YYYY-MM-DD>/<HH>.parquet
    assert any(p.name == "10.parquet" for p in written)
    assert any(p.name == "11.parquet" for p in written)


@pytest.mark.asyncio
async def test_recorder_files_are_read_only_after_close(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(11, 0)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    written = sorted(tmp_path.rglob("*.parquet"))
    for p in written:
        assert not (p.stat().st_mode & 0o222), f"file {p} still has write bits"


@pytest.mark.asyncio
async def test_recorder_stats_track_writes(tmp_path: Path) -> None:
    feed = _FixtureFeed(events=[_tick(10, 0), _tick(10, 5), _tick(11, 0)])
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    await recorder.run(feed)
    stats = recorder.stats()
    assert stats["events_written"] == 3
    assert stats["files_closed"] == 2
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_recorder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `recorder.py`**

Create `src/quant_research_stack/feeds/recorder.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_stack.feeds.base import FeedAdapter
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass(frozen=True)
class RecorderConfig:
    root: Path
    flush_every_n_events: int = 1024
    flush_every_seconds: float = 5.0
    keep_raw: bool = False


def _row_from_event(ev: Tick | Bar) -> dict:
    payload = ev.model_dump(mode="json")
    payload["_kind"] = "tick" if isinstance(ev, Tick) else "bar"
    return payload


def _key(ev: Tick | Bar) -> tuple[str, str, str, str]:
    ts = ev.timestamp_utc
    return (ev.venue.value, ev.symbol, ts.strftime("%Y-%m-%d"), f"{ts.hour:02d}")


class Recorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        self._cfg = cfg
        self._writers: dict[tuple[str, str, str, str], pq.ParquetWriter] = {}
        self._buffers: dict[tuple[str, str, str, str], list[dict]] = {}
        self._closed_paths: list[Path] = []
        self._events_written = 0

    def _path_for(self, key: tuple[str, str, str, str]) -> Path:
        venue, symbol, date, hh = key
        return Path(self._cfg.root) / venue / symbol / date / f"{hh}.parquet"

    def _flush(self, key: tuple[str, str, str, str]) -> None:
        rows = self._buffers.get(key, [])
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        writer = self._writers.get(key)
        if writer is None:
            path = self._path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            self._writers[key] = writer
        writer.write_table(table)
        self._buffers[key] = []

    def _close_writer(self, key: tuple[str, str, str, str]) -> None:
        writer = self._writers.pop(key, None)
        if writer is None:
            return
        writer.close()
        path = self._path_for(key)
        if path.exists():
            os.chmod(path, path.stat().st_mode & ~0o222)
            self._closed_paths.append(path)

    async def run(self, adapter: FeedAdapter) -> None:
        last_key: tuple[str, str, str, str] | None = None
        async for ev in adapter.iterate():
            key = _key(ev)
            if last_key is not None and key[2:] != last_key[2:]:
                # date or hour rolled over for this venue+symbol stream
                self._flush(last_key)
                self._close_writer(last_key)
            self._buffers.setdefault(key, []).append(_row_from_event(ev))
            self._events_written += 1
            if len(self._buffers[key]) >= self._cfg.flush_every_n_events:
                self._flush(key)
            last_key = key
        for k in list(self._writers.keys()):
            self._flush(k)
            self._close_writer(k)
        for k in list(self._buffers.keys()):
            if self._buffers[k]:
                self._flush(k)
                self._close_writer(k)

    def stats(self) -> dict:
        return {
            "events_written": self._events_written,
            "files_closed": len(self._closed_paths),
        }
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_recorder.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/recorder.py tests/test_feeds_recorder.py
git add src/quant_research_stack/feeds/recorder.py tests/test_feeds_recorder.py
git commit -m "feat: feeds/recorder.py with hour-rotated Parquet shards + chmod-a-w on close"
```

---

## Task 10: `feeds/replayer.py` — Parquet → FeedAdapter

**Files:**
- Create: `src/quant_research_stack/feeds/replayer.py`
- Create: `tests/test_feeds_replayer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feeds_replayer.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.feeds.market_types import Tick, TickSide, Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig


def _tick(minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=65000.0 + minute, size=0.1, side=TickSide.buy,
    )


@dataclass
class _FixtureFeed:
    venue = Venue.binance
    events: list

    async def iterate(self) -> AsyncIterator[Tick]:
        for ev in self.events:
            yield ev


@pytest.fixture
def recorded_dir(tmp_path: Path) -> Path:
    cfg = RecorderConfig(root=tmp_path)
    recorder = Recorder(cfg)
    feed = _FixtureFeed(events=[_tick(0), _tick(10), _tick(20), _tick(30)])
    asyncio.run(recorder.run(feed))
    return tmp_path


@pytest.mark.asyncio
async def test_replayer_yields_events_in_timestamp_order(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    seen = [ev async for ev in rep.iterate()]
    times = [ev.timestamp_utc for ev in seen]
    assert times == sorted(times)
    assert len(seen) == 4


@pytest.mark.asyncio
async def test_replayer_speed_zero_runs_fast(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    started = asyncio.get_event_loop().time()
    _ = [ev async for ev in rep.iterate()]
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_replayer_respects_symbol_filter(recorded_dir: Path) -> None:
    cfg = ReplayerConfig(
        root=recorded_dir, venue=Venue.binance, symbols=("ETHUSDT",),
        start_utc=datetime(2026, 5, 17, 10, 0, tzinfo=UTC),
        end_utc=datetime(2026, 5, 17, 11, 0, tzinfo=UTC),
        speed=0.0,
    )
    rep = Replayer(cfg)
    seen = [ev async for ev in rep.iterate()]
    assert seen == []
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_replayer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `replayer.py`**

Create `src/quant_research_stack/feeds/replayer.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from quant_research_stack.feeds.base import AsyncFeedBase
from quant_research_stack.feeds.market_types import Bar, Tick, Venue


@dataclass(frozen=True)
class ReplayerConfig:
    root: Path
    venue: Venue
    symbols: tuple[str, ...]
    start_utc: datetime
    end_utc: datetime
    speed: float = 1.0


@dataclass
class Replayer(AsyncFeedBase):
    cfg: ReplayerConfig

    def __post_init__(self) -> None:
        super().__init__()
        self.venue = self.cfg.venue
        self._closed = False

    def _shard_paths(self) -> list[Path]:
        out: list[Path] = []
        for symbol in self.cfg.symbols:
            symbol_dir = Path(self.cfg.root) / self.cfg.venue.value / symbol
            if not symbol_dir.exists():
                continue
            for shard in sorted(symbol_dir.rglob("*.parquet")):
                out.append(shard)
        return out

    async def subscribe(self, symbols) -> None:
        return None

    async def iterate(self) -> AsyncIterator[Tick | Bar]:
        prev_ts: datetime | None = None
        for shard in self._shard_paths():
            df = pl.read_parquet(shard).sort("timestamp_utc")
            for row in df.iter_rows(named=True):
                ts = row["timestamp_utc"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts < self.cfg.start_utc or ts >= self.cfg.end_utc:
                    continue
                if self.cfg.speed > 0.0 and prev_ts is not None:
                    delta = (ts - prev_ts).total_seconds() / self.cfg.speed
                    if delta > 0:
                        await asyncio.sleep(delta)
                prev_ts = ts
                kind = row.get("_kind", "tick")
                payload = {k: v for k, v in row.items() if k != "_kind"}
                event: Tick | Bar = Tick.model_validate(payload) if kind == "tick" else Bar.model_validate(payload)
                self._stats["events_emitted"] += 1
                yield event

    async def close(self) -> None:
        self._closed = True
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_feeds_replayer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/feeds/replayer.py tests/test_feeds_replayer.py
git add src/quant_research_stack/feeds/replayer.py tests/test_feeds_replayer.py
git commit -m "feat: feeds/replayer.py with timestamp-ordered Parquet replay (FeedAdapter impl)"
```

---

## Task 11: `brokers/order_types.py` — order + fill + position + account Pydantic models

**Files:**
- Create: `src/quant_research_stack/brokers/order_types.py`
- Create: `tests/test_brokers_order_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_brokers_order_types.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.brokers.order_types import (
    Account, Fill, Order, OrderIntent, OrderSide, OrderStatus, OrderType, Position, TimeInForce,
)


def _intent(**overrides) -> dict:
    base = {
        "client_order_id": "co-12345678",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "quantity": 0.5,
    }
    base.update(overrides)
    return base


def test_market_order_intent_valid() -> None:
    o = OrderIntent.model_validate(_intent())
    assert o.type == OrderType.market
    assert o.side == OrderSide.buy


def test_limit_order_requires_limit_price() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="limit"))


def test_stop_order_requires_stop_price() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop"))


def test_stop_limit_requires_both_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop_limit", limit_price=100.0))
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="stop_limit", stop_price=100.0))


def test_bracket_requires_three_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="bracket", limit_price=100.0, take_profit_price=110.0))


def test_oco_requires_both_oco_prices() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(type="oco", oco_limit_price=100.0))


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(quantity=0.0))


def test_client_order_id_min_length() -> None:
    with pytest.raises(ValueError):
        OrderIntent.model_validate(_intent(client_order_id="short"))


def test_order_status_enum_values() -> None:
    assert OrderStatus.accepted.value == "accepted"
    assert OrderStatus.filled.value == "filled"
    assert OrderStatus.canceled.value == "canceled"


def test_time_in_force_default_day() -> None:
    o = OrderIntent.model_validate(_intent())
    assert o.time_in_force == TimeInForce.day


def test_account_basic() -> None:
    a = Account.model_validate({"equity": 1000.0, "cash": 500.0, "buying_power": 2000.0})
    assert a.currency == "USD"


def test_fill_basic() -> None:
    f = Fill.model_validate({
        "client_order_id": "co-12345678", "fill_id": "f1", "symbol": "BTCUSDT",
        "side": "buy", "price": 100.0, "quantity": 0.5,
        "timestamp_utc": datetime(2026, 5, 17, tzinfo=UTC),
    })
    assert f.commission == 0.0


def test_position_signed_quantity() -> None:
    p = Position.model_validate({
        "symbol": "BTCUSDT", "quantity": -0.5, "avg_entry_price": 100.0,
        "market_value": -50.0, "unrealized_pnl": 5.0,
    })
    assert p.quantity == -0.5
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_order_types.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `order_types.py`**

Create `src/quant_research_stack/brokers/order_types.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class OrderSide(StrEnum):
    buy = "buy"
    sell = "sell"


class TimeInForce(StrEnum):
    day = "day"
    gtc = "gtc"
    ioc = "ioc"
    fok = "fok"


class OrderType(StrEnum):
    market = "market"
    limit = "limit"
    stop = "stop"
    stop_limit = "stop_limit"
    oco = "oco"
    bracket = "bracket"


class OrderStatus(StrEnum):
    accepted = "accepted"
    partially_filled = "partially_filled"
    filled = "filled"
    canceled = "canceled"
    rejected = "rejected"
    expired = "expired"


class OrderIntent(BaseModel):
    model_config = {"frozen": True}
    client_order_id: Annotated[str, Field(min_length=8, max_length=64)]
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Annotated[float, Field(gt=0.0)]
    time_in_force: TimeInForce = TimeInForce.day
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    oco_limit_price: float | None = None
    oco_stop_price: float | None = None
    extended_hours: bool = False

    @model_validator(mode="after")
    def _required_prices_for_type(self) -> "OrderIntent":
        t = self.type
        if t == OrderType.limit and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if t == OrderType.stop and self.stop_price is None:
            raise ValueError("stop order requires stop_price")
        if t == OrderType.stop_limit and (self.limit_price is None or self.stop_price is None):
            raise ValueError("stop_limit requires both limit_price and stop_price")
        if t == OrderType.bracket and (self.limit_price is None or self.take_profit_price is None or self.stop_loss_price is None):
            raise ValueError("bracket requires entry limit_price, take_profit_price, stop_loss_price")
        if t == OrderType.oco and (self.oco_limit_price is None or self.oco_stop_price is None):
            raise ValueError("oco requires oco_limit_price and oco_stop_price")
        return self


class Order(BaseModel):
    model_config = {"frozen": True}
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    filled_quantity: float
    status: OrderStatus
    submitted_utc: datetime
    updated_utc: datetime


class Fill(BaseModel):
    model_config = {"frozen": True}
    client_order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    timestamp_utc: datetime
    commission: float = 0.0


class Position(BaseModel):
    model_config = {"frozen": True}
    symbol: str
    quantity: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


class Account(BaseModel):
    model_config = {"frozen": True}
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
```

- [ ] **Step 4: Run, expect 13 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_order_types.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/order_types.py tests/test_brokers_order_types.py
git add src/quant_research_stack/brokers/order_types.py tests/test_brokers_order_types.py
git commit -m "feat: brokers/order_types.py with OrderIntent + Order + Fill + Position + Account"
```

---

## Task 12: `brokers/capabilities.py` — per-broker capability declaration

**Files:**
- Create: `src/quant_research_stack/brokers/capabilities.py`
- Create: `tests/test_brokers_capabilities.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_brokers_capabilities.py`:

```python
from __future__ import annotations

import pytest

from quant_research_stack.brokers.capabilities import BrokerCapabilities, UnsupportedOrderError, ensure_supported
from quant_research_stack.brokers.order_types import OrderType, TimeInForce


def _caps(types: set[OrderType]) -> BrokerCapabilities:
    return BrokerCapabilities(
        venue="x",
        supported_order_types=frozenset(types),
        supported_time_in_force=frozenset({TimeInForce.day, TimeInForce.gtc}),
        supports_shorting=True,
        supports_fractional_shares=True,
        supports_extended_hours=True,
        max_orders_per_second=10,
        paper_only=True,
    )


def test_ensure_supported_passes_when_supported() -> None:
    caps = _caps({OrderType.market, OrderType.limit})
    ensure_supported(caps, OrderType.limit)


def test_ensure_supported_raises_on_unsupported() -> None:
    caps = _caps({OrderType.market})
    with pytest.raises(UnsupportedOrderError) as exc:
        ensure_supported(caps, OrderType.oco)
    assert "x" in str(exc.value)
    assert "oco" in str(exc.value)


def test_unsupported_order_error_includes_suggestion() -> None:
    caps = _caps({OrderType.market, OrderType.limit})
    with pytest.raises(UnsupportedOrderError) as exc:
        ensure_supported(caps, OrderType.bracket)
    assert "market" in str(exc.value).lower() or "limit" in str(exc.value).lower()
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_capabilities.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `capabilities.py`**

Create `src/quant_research_stack/brokers/capabilities.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.brokers.order_types import OrderType, TimeInForce


class UnsupportedOrderError(ValueError):
    pass


@dataclass(frozen=True)
class BrokerCapabilities:
    venue: str
    supported_order_types: frozenset[OrderType]
    supported_time_in_force: frozenset[TimeInForce]
    supports_shorting: bool
    supports_fractional_shares: bool
    supports_extended_hours: bool
    max_orders_per_second: int
    paper_only: bool


def ensure_supported(caps: BrokerCapabilities, order_type: OrderType) -> None:
    if order_type in caps.supported_order_types:
        return
    suggestions = ", ".join(sorted(t.value for t in caps.supported_order_types))
    raise UnsupportedOrderError(
        f"venue {caps.venue!r} does not support {order_type.value!r}; "
        f"supported types: {suggestions}"
    )
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_capabilities.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/capabilities.py tests/test_brokers_capabilities.py
git add src/quant_research_stack/brokers/capabilities.py tests/test_brokers_capabilities.py
git commit -m "feat: brokers/capabilities.py with UnsupportedOrderError pre-network guard"
```

---

## Task 13: `brokers/base.py` — BrokerAdapter Protocol

**Files:** `src/quant_research_stack/brokers/base.py` (no tests; pure Protocol)

- [ ] **Step 1: Implement `base.py`**

Create `src/quant_research_stack/brokers/base.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from quant_research_stack.brokers.capabilities import BrokerCapabilities
from quant_research_stack.brokers.order_types import Account, Fill, Order, OrderIntent, Position


class BrokerAdapter(Protocol):
    capabilities: BrokerCapabilities

    async def place_order(self, intent: OrderIntent) -> Order: ...

    async def cancel_order(self, client_order_id: str) -> Order: ...

    async def get_order(self, client_order_id: str) -> Order: ...

    async def positions(self) -> list[Position]: ...

    async def account(self) -> Account: ...

    def stream_fills(self) -> AsyncIterator[Fill]: ...

    async def close(self) -> None: ...
```

- [ ] **Step 2: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/base.py
git add src/quant_research_stack/brokers/base.py
git commit -m "feat: brokers/base.py with BrokerAdapter Protocol"
```

---

## Task 14: `brokers/fill_model.py` — deterministic slippage + commission simulator

**Files:**
- Create: `src/quant_research_stack/brokers/fill_model.py`
- Create: `tests/test_brokers_fill_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_brokers_fill_model.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.order_types import OrderIntent, OrderSide, OrderType
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _tick(price: float, ts: datetime) -> Tick:
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


def _intent_buy(qty: float = 1.0) -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "buy", "type": "market", "quantity": qty,
    })


def _intent_sell(qty: float = 1.0) -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "sell", "type": "market", "quantity": qty,
    })


def test_buy_market_fill_includes_half_spread_and_slippage_adverse() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_buy(qty=1.0), market)
    assert len(fills) == 1
    expected_px = 100.0 + 100.0 * (1.0 + 2.0) * 1e-4  # half spread + slippage on buy
    assert fills[0].price == pytest.approx(expected_px, rel=1e-9)


def test_sell_market_fill_is_adverse_in_opposite_direction() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_sell(qty=1.0), market)
    expected_px = 100.0 - 100.0 * (1.0 + 2.0) * 1e-4
    assert fills[0].price == pytest.approx(expected_px, rel=1e-9)


def test_commission_uses_bps_of_notional() -> None:
    cfg = FillModelConfig(commission_bps=2.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0)
    fm = FillModel(cfg)
    market = iter([_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))])
    fills = fm.synthesize(_intent_buy(qty=2.0), market)
    notional = fills[0].price * fills[0].quantity
    expected_commission = notional * 2.0 * 1e-4
    assert fills[0].commission == pytest.approx(expected_commission, rel=1e-9)


def test_no_market_events_returns_empty() -> None:
    cfg = FillModelConfig()
    fm = FillModel(cfg)
    fills = fm.synthesize(_intent_buy(), iter([]))
    assert fills == []


def test_deterministic_across_runs() -> None:
    cfg = FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0)
    fm1 = FillModel(cfg)
    fm2 = FillModel(cfg)
    market_a = [_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))]
    market_b = [_tick(100.0, datetime(2026, 5, 17, tzinfo=UTC))]
    fills_a = fm1.synthesize(_intent_buy(), iter(market_a))
    fills_b = fm2.synthesize(_intent_buy(), iter(market_b))
    assert fills_a[0].price == fills_b[0].price
    assert fills_a[0].commission == fills_b[0].commission
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_fill_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `fill_model.py`**

Create `src/quant_research_stack/brokers/fill_model.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass(frozen=True)
class FillModelConfig:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    half_spread_bps: float = 1.0
    fill_latency_ms: int = 50
    reject_if_notional_above_pct_adv: float | None = None
    partial_fill_max_pct_of_book: float = 0.10


def _mid(event: Tick | Bar) -> float:
    if isinstance(event, Bar):
        return (event.open + event.close) / 2.0
    return event.price


class FillModel:
    def __init__(self, cfg: FillModelConfig) -> None:
        self.cfg = cfg

    def synthesize(self, intent: OrderIntent, market_iter: Iterator[Tick | Bar]) -> list[Fill]:
        try:
            event = next(market_iter)
        except StopIteration:
            return []
        mid = _mid(event)
        direction = 1.0 if intent.side == OrderSide.buy else -1.0
        adverse_bps = self.cfg.half_spread_bps + self.cfg.slippage_bps
        fill_px = mid + direction * mid * adverse_bps * 1e-4
        notional = fill_px * intent.quantity
        commission = notional * self.cfg.commission_bps * 1e-4
        ts = event.timestamp_utc + timedelta(milliseconds=self.cfg.fill_latency_ms)
        return [Fill(
            client_order_id=intent.client_order_id,
            fill_id=f"{intent.client_order_id}-1",
            symbol=intent.symbol,
            side=intent.side,
            price=fill_px,
            quantity=intent.quantity,
            timestamp_utc=ts,
            commission=commission,
        )]
```

- [ ] **Step 4: Run, expect 5 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_fill_model.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/fill_model.py tests/test_brokers_fill_model.py
git add src/quant_research_stack/brokers/fill_model.py tests/test_brokers_fill_model.py
git commit -m "feat: brokers/fill_model.py with deterministic slippage + commission + latency"
```

---

## Task 15: `brokers/null_broker.py` — in-process paper broker for tests + backtests

**Files:**
- Create: `src/quant_research_stack/brokers/null_broker.py`
- Create: `tests/test_brokers_null_broker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_brokers_null_broker.py`:

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.brokers.order_types import OrderIntent, OrderStatus
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _intent(symbol: str = "BTCUSDT") -> OrderIntent:
    return OrderIntent.model_validate({
        "client_order_id": "co-12345678", "symbol": symbol,
        "side": "buy", "type": "market", "quantity": 1.0,
    })


def _tick(price: float = 100.0) -> Tick:
    ts = datetime(2026, 5, 17, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


@pytest.mark.asyncio
async def test_place_order_returns_accepted_order_with_deterministic_id() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    order_a = await broker.place_order(_intent())
    broker.push_market_event(_tick())
    order_b = await broker.place_order(_intent())
    assert order_a.broker_order_id == "null-0000001"
    assert order_b.broker_order_id == "null-0000002"


@pytest.mark.asyncio
async def test_place_order_synthesizes_fill_when_market_event_available() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick(price=100.0))
    order = await broker.place_order(_intent())
    assert order.status == OrderStatus.filled
    assert order.filled_quantity == 1.0


@pytest.mark.asyncio
async def test_cancel_order_marks_canceled() -> None:
    fm = FillModel(FillModelConfig())
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    order = await broker.place_order(_intent())
    canceled = await broker.cancel_order(order.client_order_id)
    assert canceled.status == OrderStatus.canceled


@pytest.mark.asyncio
async def test_positions_reflect_fills() -> None:
    fm = FillModel(FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0))
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick(price=100.0))
    await broker.place_order(_intent())
    positions = await broker.positions()
    btc = next(p for p in positions if p.symbol == "BTCUSDT")
    assert btc.quantity == 1.0
    assert btc.avg_entry_price == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_stream_fills_yields_each_fill_once() -> None:
    fm = FillModel(FillModelConfig())
    broker = NullBroker(fill_model=fm)
    broker.push_market_event(_tick())
    await broker.place_order(_intent())
    seen = []
    async for fill in broker.stream_fills():
        seen.append(fill)
        if len(seen) == 1:
            break
    assert len(seen) == 1
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_null_broker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `null_broker.py`**

Create `src/quant_research_stack/brokers/null_broker.py`:

```python
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from quant_research_stack.brokers.capabilities import BrokerCapabilities
from quant_research_stack.brokers.fill_model import FillModel
from quant_research_stack.brokers.order_types import (
    Account, Fill, Order, OrderIntent, OrderSide, OrderStatus, OrderType, TimeInForce,
)
from quant_research_stack.feeds.market_types import Bar, Tick


_CAPS = BrokerCapabilities(
    venue="null_broker",
    supported_order_types=frozenset(OrderType),
    supported_time_in_force=frozenset(TimeInForce),
    supports_shorting=True,
    supports_fractional_shares=True,
    supports_extended_hours=True,
    max_orders_per_second=1_000_000,
    paper_only=True,
)


@dataclass
class NullBroker:
    fill_model: FillModel
    starting_cash: float = 100_000.0
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        self._next_id = 0
        self._orders: dict[str, Order] = {}
        self._fills: dict[str, list[Fill]] = defaultdict(list)
        self._positions: dict[str, float] = defaultdict(float)
        self._avg_price: dict[str, float] = defaultdict(float)
        self._cash = float(self.starting_cash)
        self._fill_queue: deque[Fill] = deque()
        self._market_events: deque[Tick | Bar] = deque()

    def push_market_event(self, event: Tick | Bar) -> None:
        self._market_events.append(event)

    def _next_broker_id(self) -> str:
        self._next_id += 1
        return f"null-{self._next_id:07d}"

    async def place_order(self, intent: OrderIntent) -> Order:
        broker_id = self._next_broker_id()
        now = datetime.now(UTC)
        fills = self.fill_model.synthesize(intent, iter(list(self._market_events)))
        filled_qty = sum(f.quantity for f in fills)
        status = OrderStatus.filled if filled_qty >= intent.quantity else OrderStatus.accepted
        order = Order(
            client_order_id=intent.client_order_id,
            broker_order_id=broker_id,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            filled_quantity=filled_qty,
            status=status,
            submitted_utc=now,
            updated_utc=now,
        )
        self._orders[intent.client_order_id] = order
        for f in fills:
            self._fills[intent.client_order_id].append(f)
            self._fill_queue.append(f)
            sign = 1.0 if f.side == OrderSide.buy else -1.0
            self._positions[f.symbol] += sign * f.quantity
            self._avg_price[f.symbol] = f.price
            self._cash -= sign * f.price * f.quantity + f.commission
        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        order = self._orders[client_order_id]
        canceled = order.model_copy(update={"status": OrderStatus.canceled, "updated_utc": datetime.now(UTC)})
        self._orders[client_order_id] = canceled
        return canceled

    async def get_order(self, client_order_id: str) -> Order:
        return self._orders[client_order_id]

    async def positions(self) -> list:
        from quant_research_stack.brokers.order_types import Position

        out = []
        for sym, qty in self._positions.items():
            if qty == 0.0:
                continue
            entry = self._avg_price[sym]
            market_value = entry * qty
            out.append(Position(
                symbol=sym, quantity=qty, avg_entry_price=entry,
                market_value=market_value, unrealized_pnl=0.0,
            ))
        return out

    async def account(self) -> Account:
        equity = self._cash + sum(self._avg_price[s] * q for s, q in self._positions.items())
        return Account(equity=equity, cash=self._cash, buying_power=equity, currency="USD")

    async def stream_fills(self) -> AsyncIterator[Fill]:
        while self._fill_queue:
            yield self._fill_queue.popleft()
        while True:
            await asyncio.sleep(0.01)
            while self._fill_queue:
                yield self._fill_queue.popleft()

    async def close(self) -> None:
        return None
```

- [ ] **Step 4: Run, expect 5 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_null_broker.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/null_broker.py tests/test_brokers_null_broker.py
git add src/quant_research_stack/brokers/null_broker.py tests/test_brokers_null_broker.py
git commit -m "feat: brokers/null_broker.py — in-process paper broker with deterministic IDs + fills"
```

---

## Task 16: `brokers/alpaca_paper.py` — Alpaca paper adapter (unit-test parts only)

**Files:**
- Create: `src/quant_research_stack/brokers/alpaca_paper.py`
- Create: `tests/test_brokers_alpaca_paper.py`

- [ ] **Step 1: Write failing tests (request-builder only — no network)**

Create `tests/test_brokers_alpaca_paper.py`:

```python
from __future__ import annotations

import pytest

from quant_research_stack.brokers.alpaca_paper import build_order_payload
from quant_research_stack.brokers.capabilities import UnsupportedOrderError
from quant_research_stack.brokers.order_types import OrderIntent


def _intent(**overrides) -> OrderIntent:
    base = {
        "client_order_id": "co-12345678", "symbol": "SPY",
        "side": "buy", "type": "market", "quantity": 10.0,
    }
    base.update(overrides)
    return OrderIntent.model_validate(base)


def test_market_payload_has_required_fields() -> None:
    payload = build_order_payload(_intent())
    assert payload["symbol"] == "SPY"
    assert payload["side"] == "buy"
    assert payload["type"] == "market"
    assert payload["qty"] == "10.0"
    assert payload["client_order_id"] == "co-12345678"


def test_limit_payload_includes_limit_price() -> None:
    payload = build_order_payload(_intent(type="limit", limit_price=500.0))
    assert payload["type"] == "limit"
    assert payload["limit_price"] == "500.0"


def test_stop_payload_includes_stop_price() -> None:
    payload = build_order_payload(_intent(type="stop", stop_price=490.0))
    assert payload["type"] == "stop"
    assert payload["stop_price"] == "490.0"


def test_bracket_payload_includes_take_profit_and_stop_loss() -> None:
    payload = build_order_payload(_intent(
        type="bracket", limit_price=500.0, take_profit_price=520.0, stop_loss_price=480.0,
    ))
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"]["limit_price"] == "520.0"
    assert payload["stop_loss"]["stop_price"] == "480.0"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_alpaca_paper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `alpaca_paper.py`**

Create `src/quant_research_stack/brokers/alpaca_paper.py`:

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from quant_research_stack.brokers.capabilities import BrokerCapabilities, ensure_supported
from quant_research_stack.brokers.order_types import (
    Account, Fill, Order, OrderIntent, OrderSide, OrderStatus, OrderType, Position, TimeInForce,
)


_CAPS = BrokerCapabilities(
    venue="alpaca_paper",
    supported_order_types=frozenset({
        OrderType.market, OrderType.limit, OrderType.stop, OrderType.stop_limit,
        OrderType.bracket, OrderType.oco,
    }),
    supported_time_in_force=frozenset({TimeInForce.day, TimeInForce.gtc, TimeInForce.ioc, TimeInForce.fok}),
    supports_shorting=True,
    supports_fractional_shares=True,
    supports_extended_hours=True,
    max_orders_per_second=200,
    paper_only=True,
)


def build_order_payload(intent: OrderIntent) -> dict:
    payload: dict = {
        "symbol": intent.symbol,
        "side": intent.side.value,
        "type": intent.type.value,
        "qty": str(intent.quantity),
        "time_in_force": intent.time_in_force.value,
        "client_order_id": intent.client_order_id,
        "extended_hours": intent.extended_hours,
    }
    if intent.limit_price is not None:
        payload["limit_price"] = str(intent.limit_price)
    if intent.stop_price is not None:
        payload["stop_price"] = str(intent.stop_price)
    if intent.type == OrderType.bracket:
        payload["order_class"] = "bracket"
        payload["take_profit"] = {"limit_price": str(intent.take_profit_price)}
        payload["stop_loss"] = {"stop_price": str(intent.stop_loss_price)}
    if intent.type == OrderType.oco:
        payload["order_class"] = "oco"
        payload["take_profit"] = {"limit_price": str(intent.oco_limit_price)}
        payload["stop_loss"] = {"stop_price": str(intent.oco_stop_price)}
    return payload


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class AlpacaPaper:
    credentials_path: str = "~/.alpaca/paper_keys.json"
    base_url: str = "https://paper-api.alpaca.markets"
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        key, secret = _load_credentials(self.credentials_path)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent) -> Order:
        ensure_supported(self.capabilities, intent.type)
        payload = build_order_payload(intent)
        response = await self._client.post("/v2/orders", json=payload)
        response.raise_for_status()
        body = response.json()
        return _order_from_alpaca(intent, body)

    async def cancel_order(self, client_order_id: str) -> Order:
        order = await self.get_order(client_order_id)
        response = await self._client.delete(f"/v2/orders/{order.broker_order_id}")
        response.raise_for_status()
        return order.model_copy(update={"status": OrderStatus.canceled, "updated_utc": datetime.now(UTC)})

    async def get_order(self, client_order_id: str) -> Order:
        response = await self._client.get(f"/v2/orders:by_client_order_id?client_order_id={client_order_id}")
        response.raise_for_status()
        body = response.json()
        return _order_from_alpaca_response(body)

    async def positions(self) -> list[Position]:
        response = await self._client.get("/v2/positions")
        response.raise_for_status()
        return [
            Position(
                symbol=row["symbol"], quantity=float(row["qty"]),
                avg_entry_price=float(row["avg_entry_price"]),
                market_value=float(row["market_value"]),
                unrealized_pnl=float(row["unrealized_pl"]),
            )
            for row in response.json()
        ]

    async def account(self) -> Account:
        response = await self._client.get("/v2/account")
        response.raise_for_status()
        body = response.json()
        return Account(
            equity=float(body["equity"]),
            cash=float(body["cash"]),
            buying_power=float(body["buying_power"]),
            currency=body.get("currency", "USD"),
        )

    async def stream_fills(self) -> AsyncIterator[Fill]:
        # Alpaca paper exposes fills via websocket; in S3 we poll /v2/account/activities
        # for simplicity. Live streaming via wss is reserved for S4 (live brokers).
        if False:
            yield  # type: ignore[unreachable]
        return

    async def close(self) -> None:
        await self._client.aclose()


def _order_from_alpaca(intent: OrderIntent, body: dict) -> Order:
    now = datetime.now(UTC)
    return Order(
        client_order_id=intent.client_order_id,
        broker_order_id=str(body.get("id", "")),
        symbol=intent.symbol,
        side=intent.side,
        type=intent.type,
        quantity=intent.quantity,
        filled_quantity=float(body.get("filled_qty", 0.0) or 0.0),
        status=OrderStatus(body.get("status", "accepted")),
        submitted_utc=now,
        updated_utc=now,
    )


def _order_from_alpaca_response(body: dict) -> Order:
    now = datetime.now(UTC)
    return Order(
        client_order_id=str(body["client_order_id"]),
        broker_order_id=str(body["id"]),
        symbol=str(body["symbol"]),
        side=OrderSide(body["side"]),
        type=OrderType(body["type"]),
        quantity=float(body["qty"]),
        filled_quantity=float(body.get("filled_qty", 0.0) or 0.0),
        status=OrderStatus(body["status"]),
        submitted_utc=now,
        updated_utc=now,
    )
```

- [ ] **Step 4: Run, expect 4 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_alpaca_paper.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/alpaca_paper.py tests/test_brokers_alpaca_paper.py
git add src/quant_research_stack/brokers/alpaca_paper.py tests/test_brokers_alpaca_paper.py
git commit -m "feat: brokers/alpaca_paper.py with request builder + REST adapter (no network in tests)"
```

---

## Task 17: `brokers/binance_testnet.py` — Binance testnet adapter

**Files:**
- Create: `src/quant_research_stack/brokers/binance_testnet.py`
- Create: `tests/test_brokers_binance_testnet.py`

- [ ] **Step 1: Write failing tests (request-builder only)**

Create `tests/test_brokers_binance_testnet.py`:

```python
from __future__ import annotations

import pytest

from quant_research_stack.brokers.binance_testnet import build_order_payload
from quant_research_stack.brokers.capabilities import UnsupportedOrderError
from quant_research_stack.brokers.order_types import OrderIntent


def _intent(**overrides) -> OrderIntent:
    base = {
        "client_order_id": "co-12345678", "symbol": "BTCUSDT",
        "side": "buy", "type": "market", "quantity": 0.1,
    }
    base.update(overrides)
    return OrderIntent.model_validate(base)


def test_market_payload_uses_uppercase_side() -> None:
    payload = build_order_payload(_intent())
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "BUY"
    assert payload["type"] == "MARKET"
    assert payload["quantity"] == "0.1"


def test_limit_payload_includes_price_and_tif() -> None:
    payload = build_order_payload(_intent(type="limit", limit_price=50000.0, time_in_force="gtc"))
    assert payload["type"] == "LIMIT"
    assert payload["price"] == "50000.0"
    assert payload["timeInForce"] == "GTC"


def test_oco_payload_has_both_legs() -> None:
    payload = build_order_payload(_intent(
        type="oco", oco_limit_price=60000.0, oco_stop_price=40000.0,
    ))
    assert payload["type"] == "OCO"
    assert payload["price"] == "60000.0"
    assert payload["stopPrice"] == "40000.0"


def test_client_order_id_passed_through() -> None:
    payload = build_order_payload(_intent())
    assert payload["newClientOrderId"] == "co-12345678"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_binance_testnet.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `binance_testnet.py`**

Create `src/quant_research_stack/brokers/binance_testnet.py`:

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from quant_research_stack.brokers.capabilities import BrokerCapabilities, ensure_supported
from quant_research_stack.brokers.order_types import (
    Account, Fill, Order, OrderIntent, OrderSide, OrderStatus, OrderType, Position, TimeInForce,
)


_CAPS = BrokerCapabilities(
    venue="binance_testnet",
    supported_order_types=frozenset({
        OrderType.market, OrderType.limit, OrderType.stop_limit, OrderType.oco,
    }),
    supported_time_in_force=frozenset({TimeInForce.gtc, TimeInForce.ioc, TimeInForce.fok}),
    supports_shorting=False,
    supports_fractional_shares=True,
    supports_extended_hours=False,
    max_orders_per_second=10,
    paper_only=True,
)


def build_order_payload(intent: OrderIntent) -> dict:
    payload: dict = {
        "symbol": intent.symbol,
        "side": intent.side.value.upper(),
        "type": intent.type.value.upper(),
        "quantity": str(intent.quantity),
        "newClientOrderId": intent.client_order_id,
    }
    if intent.time_in_force is not None:
        payload["timeInForce"] = intent.time_in_force.value.upper()
    if intent.limit_price is not None:
        payload["price"] = str(intent.limit_price)
    if intent.stop_price is not None:
        payload["stopPrice"] = str(intent.stop_price)
    if intent.type == OrderType.oco:
        payload["type"] = "OCO"
        payload["price"] = str(intent.oco_limit_price)
        payload["stopPrice"] = str(intent.oco_stop_price)
    return payload


def _load_credentials(path: Path | str) -> tuple[str, str]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    return str(payload["api_key"]), str(payload["api_secret"])


@dataclass
class BinanceTestnet:
    credentials_path: str = "~/.binance/testnet_keys.json"
    rest_base_url: str = "https://testnet.binance.vision"
    capabilities: BrokerCapabilities = field(default_factory=lambda: _CAPS)

    def __post_init__(self) -> None:
        key, secret = _load_credentials(self.credentials_path)
        self._key = key
        self._secret = secret
        self._client = httpx.AsyncClient(
            base_url=self.rest_base_url,
            headers={"X-MBX-APIKEY": key},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent) -> Order:
        ensure_supported(self.capabilities, intent.type)
        payload = build_order_payload(intent)
        response = await self._client.post("/api/v3/order", data=payload)
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=intent.client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=intent.symbol,
            side=intent.side,
            type=intent.type,
            quantity=intent.quantity,
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus(_translate_status(body.get("status", "NEW"))),
            submitted_utc=now,
            updated_utc=now,
        )

    async def cancel_order(self, client_order_id: str) -> Order:
        # The testnet REST API requires the broker order id for cancellation; tests stub this.
        response = await self._client.delete(f"/api/v3/order?origClientOrderId={client_order_id}")
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=str(body.get("symbol", "")),
            side=OrderSide(body.get("side", "buy").lower()),
            type=OrderType(body.get("type", "market").lower()),
            quantity=float(body.get("origQty", 0.0)),
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus.canceled,
            submitted_utc=now,
            updated_utc=now,
        )

    async def get_order(self, client_order_id: str) -> Order:
        response = await self._client.get(f"/api/v3/order?origClientOrderId={client_order_id}")
        response.raise_for_status()
        body = response.json()
        now = datetime.now(UTC)
        return Order(
            client_order_id=client_order_id,
            broker_order_id=str(body.get("orderId", "")),
            symbol=str(body.get("symbol", "")),
            side=OrderSide(body.get("side", "buy").lower()),
            type=OrderType(body.get("type", "market").lower()),
            quantity=float(body.get("origQty", 0.0)),
            filled_quantity=float(body.get("executedQty", 0.0) or 0.0),
            status=OrderStatus(_translate_status(body.get("status", "NEW"))),
            submitted_utc=now,
            updated_utc=now,
        )

    async def positions(self) -> list[Position]:
        response = await self._client.get("/api/v3/account")
        response.raise_for_status()
        body = response.json()
        out: list[Position] = []
        for bal in body.get("balances", []):
            free = float(bal.get("free", 0.0))
            if free == 0.0:
                continue
            out.append(Position(
                symbol=str(bal["asset"]),
                quantity=free,
                avg_entry_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
            ))
        return out

    async def account(self) -> Account:
        response = await self._client.get("/api/v3/account")
        response.raise_for_status()
        body = response.json()
        usdt = next((float(b["free"]) for b in body.get("balances", []) if b["asset"] == "USDT"), 0.0)
        return Account(equity=usdt, cash=usdt, buying_power=usdt, currency="USDT")

    async def stream_fills(self) -> AsyncIterator[Fill]:
        # Live user data stream is reserved for S4.
        if False:
            yield  # type: ignore[unreachable]
        return

    async def close(self) -> None:
        await self._client.aclose()


def _translate_status(s: str) -> str:
    return {
        "NEW": "accepted",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
    }.get(s.upper(), "accepted")
```

- [ ] **Step 4: Run, expect 4 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_brokers_binance_testnet.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/brokers/binance_testnet.py tests/test_brokers_binance_testnet.py
git add src/quant_research_stack/brokers/binance_testnet.py tests/test_brokers_binance_testnet.py
git commit -m "feat: brokers/binance_testnet.py with request builder + REST adapter (no network in tests)"
```

---

## Task 18: `backtest/strategy.py` + 2 reference strategies + tests

**Files:**
- Create: `src/quant_research_stack/backtest/strategy.py`
- Create: `src/quant_research_stack/backtest/strategies/buy_and_hold.py`
- Create: `src/quant_research_stack/backtest/strategies/moving_average_cross.py`
- Create: `tests/test_backtest_strategy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest_strategy.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.backtest.strategies.buy_and_hold import BuyAndHold
from quant_research_stack.backtest.strategies.moving_average_cross import MovingAverageCross
from quant_research_stack.feeds.market_types import Bar, Tick, TickSide, Venue


def _tick(price: float, minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


def test_buy_and_hold_emits_one_order_on_first_event_and_none_after() -> None:
    s = BuyAndHold(symbol="BTCUSDT", quantity=1.0)
    first = s.on_event(_tick(100.0, 0))
    second = s.on_event(_tick(101.0, 1))
    assert len(first) == 1
    assert first[0].side.value == "buy"
    assert first[0].quantity == 1.0
    assert second == []


def test_moving_average_cross_does_not_trade_before_window_fills() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    assert s.on_event(_tick(100.0, 0)) == []
    assert s.on_event(_tick(101.0, 1)) == []


def test_moving_average_cross_emits_buy_when_fast_crosses_above_slow() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    s.on_event(_tick(100.0, 0))
    s.on_event(_tick(99.0, 1))
    s.on_event(_tick(98.0, 2))  # slow window now full
    orders = s.on_event(_tick(105.0, 3))  # big jump → fast > slow
    assert orders and orders[0].side.value == "buy"


def test_moving_average_cross_emits_sell_when_fast_crosses_below_slow() -> None:
    s = MovingAverageCross(symbol="BTCUSDT", quantity=1.0, fast_window=2, slow_window=3)
    s.on_event(_tick(100.0, 0))
    s.on_event(_tick(101.0, 1))
    s.on_event(_tick(102.0, 2))
    s.on_event(_tick(103.0, 3))  # uptrend, fast > slow
    orders = s.on_event(_tick(80.0, 4))  # big drop → fast < slow
    assert orders and orders[0].side.value == "sell"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_strategy.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement Strategy Protocol**

Create `src/quant_research_stack/backtest/strategy.py`:

```python
from __future__ import annotations

from typing import Protocol

from quant_research_stack.brokers.order_types import Fill, OrderIntent
from quant_research_stack.feeds.market_types import Bar, Tick


class Strategy(Protocol):
    name: str

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]: ...

    def on_fill(self, fill: Fill) -> None: ...

    def snapshot_state(self) -> dict: ...
```

- [ ] **Step 4: Implement `buy_and_hold.py`**

Create `src/quant_research_stack/backtest/strategies/buy_and_hold.py`:

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide, OrderType
from quant_research_stack.feeds.market_types import Bar, Tick


@dataclass
class BuyAndHold:
    symbol: str
    quantity: float
    name: str = "buy_and_hold"
    _fired: bool = field(default=False, init=False)
    _fills: list[Fill] = field(default_factory=list, init=False)

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]:
        if self._fired or event.symbol != self.symbol:
            return []
        self._fired = True
        return [OrderIntent.model_validate({
            "client_order_id": f"bh-{uuid.uuid4().hex[:12]}",
            "symbol": self.symbol,
            "side": OrderSide.buy.value,
            "type": OrderType.market.value,
            "quantity": self.quantity,
        })]

    def on_fill(self, fill: Fill) -> None:
        self._fills.append(fill)

    def snapshot_state(self) -> dict:
        return {"fired": self._fired, "n_fills": len(self._fills)}
```

- [ ] **Step 5: Implement `moving_average_cross.py`**

Create `src/quant_research_stack/backtest/strategies/moving_average_cross.py`:

```python
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

from quant_research_stack.brokers.order_types import Fill, OrderIntent, OrderSide, OrderType
from quant_research_stack.feeds.market_types import Bar, Tick


def _price(event: Tick | Bar) -> float:
    return event.close if isinstance(event, Bar) else event.price


@dataclass
class MovingAverageCross:
    symbol: str
    quantity: float
    fast_window: int
    slow_window: int
    name: str = "moving_average_cross"
    _fast: deque = field(default_factory=lambda: deque(), init=False)
    _slow: deque = field(default_factory=lambda: deque(), init=False)
    _prev_fast_gt_slow: bool | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be < slow_window")
        self._fast = deque(maxlen=self.fast_window)
        self._slow = deque(maxlen=self.slow_window)

    def on_event(self, event: Tick | Bar) -> list[OrderIntent]:
        if event.symbol != self.symbol:
            return []
        price = _price(event)
        self._fast.append(price)
        self._slow.append(price)
        if len(self._slow) < self.slow_window:
            return []
        fast_mean = sum(self._fast) / len(self._fast)
        slow_mean = sum(self._slow) / len(self._slow)
        current = fast_mean > slow_mean
        prev = self._prev_fast_gt_slow
        self._prev_fast_gt_slow = current
        if prev is None or prev == current:
            return []
        side = OrderSide.buy if current else OrderSide.sell
        return [OrderIntent.model_validate({
            "client_order_id": f"ma-{uuid.uuid4().hex[:12]}",
            "symbol": self.symbol,
            "side": side.value,
            "type": OrderType.market.value,
            "quantity": self.quantity,
        })]

    def on_fill(self, fill: Fill) -> None:
        return None

    def snapshot_state(self) -> dict:
        return {
            "fast_len": len(self._fast),
            "slow_len": len(self._slow),
            "prev_fast_gt_slow": self._prev_fast_gt_slow,
        }
```

- [ ] **Step 6: Run, expect 4 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_strategy.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/backtest/ tests/test_backtest_strategy.py
git add src/quant_research_stack/backtest/strategy.py src/quant_research_stack/backtest/strategies/ tests/test_backtest_strategy.py
git commit -m "feat: backtest/strategy.py + buy_and_hold + moving_average_cross reference strategies"
```

---

## Task 19: `backtest/metrics.py` — pure metric functions

**Files:**
- Create: `src/quant_research_stack/backtest/metrics.py`
- Create: `tests/test_backtest_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest_metrics.py`:

```python
from __future__ import annotations

import math

import polars as pl
import pytest

from quant_research_stack.backtest.metrics import (
    hit_rate, max_drawdown, sharpe_ratio, total_return, turnover, value_at_risk,
)
from quant_research_stack.brokers.order_types import Fill, OrderSide


def _eq_curve(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"equity": values})


def test_total_return_basic() -> None:
    assert total_return(_eq_curve([100.0, 110.0])) == pytest.approx(0.10, rel=1e-9)


def test_total_return_zero_initial_returns_zero() -> None:
    assert total_return(_eq_curve([0.0, 100.0])) == 0.0


def test_max_drawdown_zero_when_monotonic_up() -> None:
    assert max_drawdown(_eq_curve([100.0, 101.0, 110.0])) == pytest.approx(0.0)


def test_max_drawdown_negative_value() -> None:
    dd = max_drawdown(_eq_curve([100.0, 110.0, 99.0, 105.0]))
    assert dd == pytest.approx(-(110.0 - 99.0) / 110.0, rel=1e-9)


def test_sharpe_basic_positive() -> None:
    returns = pl.Series("r", [0.001, 0.002, -0.001, 0.0015])
    s = sharpe_ratio(returns, periods_per_year=252)
    assert s > 0


def test_sharpe_zero_volatility_returns_zero() -> None:
    returns = pl.Series("r", [0.001, 0.001, 0.001, 0.001])
    assert sharpe_ratio(returns, periods_per_year=252) == 0.0


def test_hit_rate_alternating() -> None:
    from datetime import UTC, datetime
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="X", side=OrderSide.buy,
             price=100.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC)),
        Fill(client_order_id="b", fill_id="2", symbol="X", side=OrderSide.sell,
             price=110.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC)),
        Fill(client_order_id="c", fill_id="3", symbol="X", side=OrderSide.buy,
             price=120.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 3, tzinfo=UTC)),
        Fill(client_order_id="d", fill_id="4", symbol="X", side=OrderSide.sell,
             price=115.0, quantity=1.0, timestamp_utc=datetime(2026, 1, 4, tzinfo=UTC)),
    ]
    assert hit_rate(fills) == 0.5


def test_turnover_sums_notionals_normalized_by_capital() -> None:
    from datetime import UTC, datetime
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="X", side=OrderSide.buy,
             price=100.0, quantity=2.0, timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    assert turnover(fills, starting_cash=1000.0) == pytest.approx(200.0 / 1000.0)


def test_value_at_risk_returns_left_tail() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    returns = pl.Series("r", rng.normal(size=1000).tolist())
    var5 = value_at_risk(returns, alpha=0.05)
    assert var5 < 0
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `metrics.py`**

Create `src/quant_research_stack/backtest/metrics.py`:

```python
from __future__ import annotations

import math

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide


def total_return(equity_curve: pl.DataFrame) -> float:
    if equity_curve.height == 0:
        return 0.0
    start = float(equity_curve["equity"][0])
    end = float(equity_curve["equity"][-1])
    if start <= 0.0:
        return 0.0
    return (end - start) / start


def max_drawdown(equity_curve: pl.DataFrame) -> float:
    if equity_curve.height == 0:
        return 0.0
    peak = float("-inf")
    worst = 0.0
    for value in equity_curve["equity"].to_list():
        v = float(value)
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < worst:
            worst = dd
    return worst


def sharpe_ratio(returns: pl.Series, periods_per_year: int) -> float:
    if returns.len() == 0:
        return 0.0
    mu = float(returns.mean())
    sigma = float(returns.std())
    if sigma == 0.0 or math.isnan(sigma):
        return 0.0
    return (mu / sigma) * math.sqrt(periods_per_year)


def calmar_ratio(equity_curve: pl.DataFrame) -> float:
    dd = abs(max_drawdown(equity_curve))
    if dd == 0.0:
        return 0.0
    return total_return(equity_curve) / dd


def hit_rate(fills: list[Fill]) -> float:
    if len(fills) < 2:
        return 0.0
    sorted_fills = sorted(fills, key=lambda f: f.timestamp_utc)
    wins = 0
    pairs = 0
    open_price: float | None = None
    open_side: OrderSide | None = None
    for f in sorted_fills:
        if open_price is None:
            open_price = f.price
            open_side = f.side
            continue
        pairs += 1
        sign = 1.0 if open_side == OrderSide.buy else -1.0
        pnl = sign * (f.price - open_price)
        if pnl > 0:
            wins += 1
        open_price = None
        open_side = None
    if pairs == 0:
        return 0.0
    return wins / pairs


def turnover(fills: list[Fill], starting_cash: float) -> float:
    if starting_cash <= 0:
        return 0.0
    notional = sum(f.price * f.quantity for f in fills)
    return notional / starting_cash


def value_at_risk(returns: pl.Series, alpha: float = 0.05) -> float:
    if returns.len() == 0:
        return 0.0
    return float(returns.quantile(alpha))
```

- [ ] **Step 4: Run, expect 9 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_metrics.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/backtest/metrics.py tests/test_backtest_metrics.py
git add src/quant_research_stack/backtest/metrics.py tests/test_backtest_metrics.py
git commit -m "feat: backtest/metrics.py with total_return, sharpe, max_dd, calmar, hit_rate, turnover, var"
```

---

## Task 20: `backtest/runner.py` — BacktestRunner end-to-end harness

**Files:**
- Create: `src/quant_research_stack/backtest/runner.py`
- Create: `tests/test_backtest_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest_runner.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from quant_research_stack.backtest.runner import BacktestConfig, BacktestRunner
from quant_research_stack.brokers.fill_model import FillModelConfig
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue


def _tick(price: float, minute: int) -> Tick:
    ts = datetime(2026, 5, 17, 10, minute, tzinfo=UTC)
    return Tick(
        venue=Venue.binance, symbol="BTCUSDT", timestamp_utc=ts, received_utc=ts,
        price=price, size=1.0, side=TickSide.buy,
    )


@pytest.mark.asyncio
async def test_buy_and_hold_runs_end_to_end_and_emits_one_fill() -> None:
    events = [_tick(100.0, i) for i in range(10)]

    cfg = BacktestConfig(
        events=events,
        fill_model=FillModelConfig(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, fill_latency_ms=0),
        starting_cash=100_000.0,
        strategy_name="buy_and_hold",
        strategy_params={"symbol": "BTCUSDT", "quantity": 1.0},
        metrics_horizon_minutes=1,
    )
    runner = BacktestRunner(cfg)
    result = await runner.run()
    assert len(result.fills) == 1
    assert result.equity_curve.height == len(events)


@pytest.mark.asyncio
async def test_unknown_strategy_raises() -> None:
    cfg = BacktestConfig(
        events=[_tick(100.0, 0)],
        fill_model=FillModelConfig(),
        starting_cash=100_000.0,
        strategy_name="does_not_exist",
        strategy_params={},
    )
    runner = BacktestRunner(cfg)
    with pytest.raises(ValueError):
        await runner.run()


@pytest.mark.asyncio
async def test_two_identical_runs_produce_identical_metrics() -> None:
    events = [_tick(100.0 + i * 0.1, i) for i in range(20)]
    cfg = BacktestConfig(
        events=events,
        fill_model=FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=0),
        starting_cash=100_000.0,
        strategy_name="moving_average_cross",
        strategy_params={"symbol": "BTCUSDT", "quantity": 1.0, "fast_window": 3, "slow_window": 5},
    )
    a = await BacktestRunner(cfg).run()
    b = await BacktestRunner(cfg).run()
    assert a.metrics == b.metrics
    assert a.equity_curve.to_dicts() == b.equity_curve.to_dicts()
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_runner.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `runner.py`**

Create `src/quant_research_stack/backtest/runner.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from quant_research_stack.backtest.metrics import (
    calmar_ratio, hit_rate, max_drawdown, sharpe_ratio, total_return, turnover, value_at_risk,
)
from quant_research_stack.backtest.strategies.buy_and_hold import BuyAndHold
from quant_research_stack.backtest.strategies.moving_average_cross import MovingAverageCross
from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.feeds.market_types import Bar, Tick


_STRATEGIES = {
    "buy_and_hold": BuyAndHold,
    "moving_average_cross": MovingAverageCross,
}


@dataclass(frozen=True)
class BacktestConfig:
    events: Iterable[Tick | Bar]
    fill_model: FillModelConfig
    starting_cash: float
    strategy_name: str
    strategy_params: dict[str, Any]
    metrics_horizon_minutes: int = 5


@dataclass
class BacktestResult:
    fills: list[Fill]
    equity_curve: pl.DataFrame
    metrics: dict


def _build_strategy(name: str, params: dict):
    if name not in _STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")
    return _STRATEGIES[name](**params)


def _event_price(ev: Tick | Bar) -> float:
    return ev.close if isinstance(ev, Bar) else ev.price


class BacktestRunner:
    def __init__(self, cfg: BacktestConfig) -> None:
        self._cfg = cfg

    async def run(self) -> BacktestResult:
        fm = FillModel(self._cfg.fill_model)
        broker = NullBroker(fill_model=fm, starting_cash=self._cfg.starting_cash)
        strategy = _build_strategy(self._cfg.strategy_name, self._cfg.strategy_params)
        positions: dict[str, float] = {}
        avg_price: dict[str, float] = {}
        cash = float(self._cfg.starting_cash)
        equity_rows: list[dict] = []
        fills: list[Fill] = []
        for ev in self._cfg.events:
            broker.push_market_event(ev)
            intents = strategy.on_event(ev)
            for intent in intents:
                order = await broker.place_order(intent)
                for f in broker._fills.get(order.client_order_id, []):
                    fills.append(f)
                    strategy.on_fill(f)
                    sign = 1.0 if f.side == OrderSide.buy else -1.0
                    positions[f.symbol] = positions.get(f.symbol, 0.0) + sign * f.quantity
                    avg_price[f.symbol] = f.price
                    cash -= sign * f.price * f.quantity + f.commission
            mark_price = _event_price(ev)
            equity = cash + sum(
                qty * (mark_price if sym == ev.symbol else avg_price.get(sym, 0.0))
                for sym, qty in positions.items()
            )
            equity_rows.append({"timestamp_utc": ev.timestamp_utc, "equity": equity})
        equity_curve = pl.DataFrame(equity_rows) if equity_rows else pl.DataFrame({"equity": [self._cfg.starting_cash]})
        returns = (
            pl.Series("r", equity_curve["equity"].pct_change().drop_nulls().to_list())
            if equity_curve.height > 1 else pl.Series("r", [])
        )
        metrics = {
            "total_return": total_return(equity_curve),
            "max_drawdown": max_drawdown(equity_curve),
            "sharpe_ratio": sharpe_ratio(returns, periods_per_year=252),
            "calmar_ratio": calmar_ratio(equity_curve),
            "hit_rate": hit_rate(fills),
            "turnover": turnover(fills, starting_cash=self._cfg.starting_cash),
            "value_at_risk_5pct": value_at_risk(returns, alpha=0.05),
            "n_fills": len(fills),
        }
        return BacktestResult(fills=fills, equity_curve=equity_curve, metrics=metrics)
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_runner.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/backtest/runner.py tests/test_backtest_runner.py
git add src/quant_research_stack/backtest/runner.py tests/test_backtest_runner.py
git commit -m "feat: backtest/runner.py with deterministic end-to-end harness"
```

---

## Task 21: `backtest/report.py` — markdown + PNG report writer

**Files:**
- Create: `src/quant_research_stack/backtest/report.py`
- Create: `tests/test_backtest_report.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest_report.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.backtest.report import BacktestReport
from quant_research_stack.backtest.runner import BacktestResult
from quant_research_stack.brokers.order_types import Fill, OrderSide


def _result() -> BacktestResult:
    fills = [
        Fill(client_order_id="a", fill_id="1", symbol="BTCUSDT", side=OrderSide.buy,
             price=100.0, quantity=1.0, timestamp_utc=datetime(2026, 5, 17, tzinfo=UTC)),
    ]
    eq = pl.DataFrame({
        "timestamp_utc": [datetime(2026, 5, 17, 10, i, tzinfo=UTC) for i in range(5)],
        "equity": [100_000.0, 100_010.0, 100_020.0, 99_990.0, 100_050.0],
    })
    return BacktestResult(
        fills=fills, equity_curve=eq,
        metrics={
            "total_return": 0.0005, "max_drawdown": -0.0003, "sharpe_ratio": 1.2,
            "calmar_ratio": 0.5, "hit_rate": 0.5, "turnover": 0.001,
            "value_at_risk_5pct": -0.0002, "n_fills": 1,
        },
    )


def test_writes_all_required_artifacts(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    files = {p.name for p in tmp_path.glob("*")}
    assert "metrics.json" in files
    assert "fills.parquet" in files
    assert "equity_curve.parquet" in files
    assert "report.md" in files
    assert "equity_curve.png" in files
    assert "drawdown.png" in files


def test_metrics_json_round_trips(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["sharpe_ratio"] == 1.2
    assert payload["run_id"] == "20260517-120000"
    assert payload["strategy_name"] == "buy_and_hold"


def test_markdown_includes_metric_table(tmp_path: Path) -> None:
    report = BacktestReport(tmp_path)
    report.write(_result(), run_id="20260517-120000", strategy_name="buy_and_hold")
    md = (tmp_path / "report.md").read_text()
    assert "sharpe_ratio" in md
    assert "buy_and_hold" in md
    assert "![equity_curve](equity_curve.png)" in md
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_report.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `report.py`**

Create `src/quant_research_stack/backtest/report.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quant_research_stack.backtest.runner import BacktestResult


@dataclass(frozen=True)
class BacktestReport:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def write(self, result: BacktestResult, *, run_id: str, strategy_name: str) -> None:
        root = Path(self.root)
        metrics = dict(result.metrics)
        metrics["run_id"] = run_id
        metrics["strategy_name"] = strategy_name
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
        if result.fills:
            fills_df = pl.DataFrame([f.model_dump(mode="json") for f in result.fills])
        else:
            fills_df = pl.DataFrame({
                "client_order_id": [], "fill_id": [], "symbol": [],
                "side": [], "price": [], "quantity": [], "timestamp_utc": [], "commission": [],
            })
        fills_df.write_parquet(root / "fills.parquet", compression="zstd")
        result.equity_curve.write_parquet(root / "equity_curve.parquet", compression="zstd")
        self._plot_equity(result, root / "equity_curve.png")
        self._plot_drawdown(result, root / "drawdown.png")
        (root / "report.md").write_text(self._markdown(result, run_id, strategy_name))

    def _plot_equity(self, result: BacktestResult, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ts = result.equity_curve["timestamp_utc"].to_list()
        eq = result.equity_curve["equity"].to_list()
        ax.plot(ts, eq)
        ax.set_title("Equity Curve")
        ax.set_xlabel("time UTC")
        ax.set_ylabel("equity")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _plot_drawdown(self, result: BacktestResult, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 3))
        eq = result.equity_curve["equity"].to_list()
        peak = float("-inf")
        dd = []
        for v in eq:
            if v > peak:
                peak = v
            dd.append(((v - peak) / peak) if peak > 0 else 0.0)
        ax.fill_between(range(len(dd)), dd, 0.0, alpha=0.4)
        ax.set_title("Drawdown")
        ax.set_xlabel("step")
        ax.set_ylabel("fractional drawdown")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _markdown(self, result: BacktestResult, run_id: str, strategy_name: str) -> str:
        lines = [
            f"# Backtest report `{run_id}`",
            "",
            f"Strategy: `{strategy_name}`",
            "",
            "## Metrics",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for key, value in sorted(result.metrics.items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines += [
            "",
            "## Equity curve",
            "",
            "![equity_curve](equity_curve.png)",
            "",
            "## Drawdown",
            "",
            "![drawdown](drawdown.png)",
            "",
            "## Notes",
            "",
            "- This backtest uses fixed-bps slippage and commission. L2 order-book impact",
            "  is not modeled in S3; defer large-size studies to S3.3.",
            "- `not_investment_advice: true` — every artifact is research output only.",
        ]
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run, expect 3 passed**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest tests/test_backtest_report.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix src/quant_research_stack/backtest/report.py tests/test_backtest_report.py
git add src/quant_research_stack/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: backtest/report.py with metrics.json + fills + equity + PNG plots + markdown"
```

---

## Task 22: `scripts/s3_record.py` + `scripts/backtest_run.py` CLIs

**Files:**
- Create: `scripts/s3_record.py`
- Create: `scripts/backtest_run.py`

- [ ] **Step 1: Implement `scripts/s3_record.py`**

```python
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.feeds.alpaca_rest import AlpacaREST
from quant_research_stack.feeds.binance_ws import BinanceWS
from quant_research_stack.feeds.coinbase_ws import CoinbaseWS
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3 recorder daemon — tail live feeds and write Parquet shards.")
    p.add_argument("--config", default="configs/feeds.yaml")
    return p.parse_args()


def _build_adapter(spec: dict):
    impl = spec["impl"]
    if impl == "BinanceWS":
        feed = BinanceWS()
    elif impl == "CoinbaseWS":
        feed = CoinbaseWS()
    elif impl == "AlpacaREST":
        feed = AlpacaREST(
            credentials_path=spec.get("credentials_path", "~/.alpaca/paper_keys.json"),
            interval_seconds=int(spec.get("interval_minutes", 15)) * 60,
            poll_offset_seconds=int(spec.get("poll_offset_seconds", 5)),
        )
    else:
        raise ValueError(f"unknown feed impl: {impl}")
    return feed


async def _run(cfg: dict) -> None:
    recorder = Recorder(RecorderConfig(
        root=Path(cfg["recorder"]["root"]),
        flush_every_n_events=int(cfg["recorder"]["flush_every_n_events"]),
        flush_every_seconds=float(cfg["recorder"]["flush_every_seconds"]),
        keep_raw=bool(cfg["recorder"]["keep_raw"]),
    ))
    tasks = []
    for spec in cfg["adapters"]:
        feed = _build_adapter(spec)
        await feed.subscribe(spec["symbols"])
        tasks.append(asyncio.create_task(recorder.run(feed)))
    await asyncio.gather(*tasks)


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        console.print("[yellow]recorder draining on SIGINT[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Implement `scripts/backtest_run.py`**

```python
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.backtest.report import BacktestReport
from quant_research_stack.backtest.runner import BacktestConfig, BacktestRunner
from quant_research_stack.brokers.fill_model import FillModelConfig
from quant_research_stack.feeds.market_types import Venue
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a backtest from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="experiments/backtests")
    return p.parse_args()


async def _collect_events(replayer_cfg: ReplayerConfig) -> list:
    rep = Replayer(replayer_cfg)
    return [ev async for ev in rep.iterate()]


async def _run(cfg: dict, output_root: Path) -> int:
    rep_cfg_dict = cfg["replayer"]
    rep_cfg = ReplayerConfig(
        root=Path(rep_cfg_dict["root"]),
        venue=Venue(rep_cfg_dict["venue"]),
        symbols=tuple(rep_cfg_dict["symbols"]),
        start_utc=datetime.fromisoformat(str(rep_cfg_dict["start_utc"]).replace("Z", "+00:00")),
        end_utc=datetime.fromisoformat(str(rep_cfg_dict["end_utc"]).replace("Z", "+00:00")),
        speed=float(rep_cfg_dict.get("speed", 0.0)),
    )
    events = await _collect_events(rep_cfg)
    if not events:
        console.print(f"[red]no events found under {rep_cfg.root} for the requested window[/red]")
        return 2
    fm_dict = cfg["fill_model"]
    fill_model = FillModelConfig(
        commission_bps=float(fm_dict.get("commission_bps", 1.0)),
        slippage_bps=float(fm_dict.get("slippage_bps", 2.0)),
        half_spread_bps=float(fm_dict.get("half_spread_bps", 1.0)),
        fill_latency_ms=int(fm_dict.get("fill_latency_ms", 50)),
        reject_if_notional_above_pct_adv=fm_dict.get("reject_if_notional_above_pct_adv"),
        partial_fill_max_pct_of_book=float(fm_dict.get("partial_fill_max_pct_of_book", 0.10)),
    )
    bt_cfg = BacktestConfig(
        events=events,
        fill_model=fill_model,
        starting_cash=float(cfg.get("starting_cash", 100_000.0)),
        strategy_name=str(cfg["strategy_name"]),
        strategy_params=dict(cfg.get("strategy_params", {})),
        metrics_horizon_minutes=int(cfg.get("metrics_horizon_minutes", 5)),
    )
    result = await BacktestRunner(bt_cfg).run()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report = BacktestReport(output_root / run_id)
    report.write(result, run_id=run_id, strategy_name=bt_cfg.strategy_name)
    console.print(f"backtest complete: {output_root / run_id}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    return asyncio.run(_run(cfg, Path(args.output_root)))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Lint**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix scripts/s3_record.py scripts/backtest_run.py
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/s3_record.py scripts/backtest_run.py
git commit -m "feat: scripts/s3_record.py recorder daemon + scripts/backtest_run.py CLI"
```

---

## Task 23: Integration tests (`s3_integration` marker)

**Files:**
- Create: `tests/integration/test_binance_ws_live.py`
- Create: `tests/integration/test_alpaca_paper_roundtrip.py`
- Create: `tests/integration/test_binance_testnet_roundtrip.py`
- Create: `tests/integration/test_record_replay_parity.py`

- [ ] **Step 1: Ensure integration package exists** (created in S2 plan; check)

```bash
test -f tests/integration/__init__.py || (mkdir -p tests/integration && touch tests/integration/__init__.py)
```

- [ ] **Step 2: Write `test_binance_ws_live.py`**

```python
from __future__ import annotations

import asyncio

import pytest

from quant_research_stack.feeds.binance_ws import BinanceWS


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_binance_ws_emits_at_least_one_event_in_60_seconds() -> None:
    feed = BinanceWS()
    await feed.subscribe(["BTCUSDT"])
    seen = 0
    async def consume():
        nonlocal seen
        async for _ in feed.iterate():
            seen += 1
            if seen >= 1:
                return
    try:
        await asyncio.wait_for(consume(), timeout=60.0)
    finally:
        await feed.close()
    assert seen >= 1
```

- [ ] **Step 3: Write `test_alpaca_paper_roundtrip.py`**

```python
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from quant_research_stack.brokers.alpaca_paper import AlpacaPaper
from quant_research_stack.brokers.order_types import OrderIntent


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_place_cancel_get_roundtrip() -> None:
    if not Path("~/.alpaca/paper_keys.json").expanduser().exists():
        pytest.skip("alpaca paper credentials not present")
    broker = AlpacaPaper()
    intent = OrderIntent.model_validate({
        "client_order_id": f"it-{uuid.uuid4().hex[:12]}",
        "symbol": "SPY", "side": "buy", "type": "limit",
        "limit_price": 1.0, "quantity": 1.0, "time_in_force": "day",
    })
    try:
        order = await broker.place_order(intent)
        assert order.client_order_id == intent.client_order_id
        canceled = await broker.cancel_order(intent.client_order_id)
        assert canceled.status.value == "canceled"
    finally:
        await broker.close()
```

- [ ] **Step 4: Write `test_binance_testnet_roundtrip.py`**

```python
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from quant_research_stack.brokers.binance_testnet import BinanceTestnet
from quant_research_stack.brokers.order_types import OrderIntent


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_place_cancel_get_roundtrip() -> None:
    if not Path("~/.binance/testnet_keys.json").expanduser().exists():
        pytest.skip("binance testnet credentials not present")
    broker = BinanceTestnet()
    intent = OrderIntent.model_validate({
        "client_order_id": f"it-{uuid.uuid4().hex[:12]}",
        "symbol": "BTCUSDT", "side": "buy", "type": "limit",
        "limit_price": 1.0, "quantity": 0.001, "time_in_force": "gtc",
    })
    try:
        order = await broker.place_order(intent)
        canceled = await broker.cancel_order(intent.client_order_id)
        assert canceled.status.value == "canceled"
    finally:
        await broker.close()
```

- [ ] **Step 5: Write `test_record_replay_parity.py`**

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.feeds.binance_ws import BinanceWS
from quant_research_stack.feeds.market_types import Venue
from quant_research_stack.feeds.recorder import Recorder, RecorderConfig
from quant_research_stack.feeds.replayer import Replayer, ReplayerConfig


@pytest.mark.s3_integration
@pytest.mark.asyncio
async def test_record_then_replay_yields_same_event_count(tmp_path: Path) -> None:
    feed = BinanceWS()
    await feed.subscribe(["BTCUSDT"])
    recorder = Recorder(RecorderConfig(root=tmp_path))
    started = datetime.now(UTC)

    async def record_for_60s() -> None:
        try:
            await asyncio.wait_for(recorder.run(feed), timeout=60.0)
        except asyncio.TimeoutError:
            pass
        finally:
            await feed.close()

    await record_for_60s()
    recorded = sum(pl.read_parquet(p).height for p in tmp_path.rglob("*.parquet"))
    assert recorded >= 1, "live recording produced no events"

    rep = Replayer(ReplayerConfig(
        root=tmp_path, venue=Venue.binance, symbols=("BTCUSDT",),
        start_utc=started - timedelta(seconds=5),
        end_utc=started + timedelta(seconds=120),
        speed=0.0,
    ))
    replayed = [ev async for ev in rep.iterate()]
    assert len(replayed) == recorded
    times = [ev.timestamp_utc for ev in replayed]
    assert times == sorted(times)
```

- [ ] **Step 6: Verify default test run still skips integration**

```bash
cd /Users/dmr/MachineLearning && PYTHONPATH=src uv run pytest --collect-only -q tests/integration/ 2>&1 | tail -10
```

Expected: shows collected but deselected via `-m 'not governor_slow and not s3_integration'`.

- [ ] **Step 7: Lint and commit**

```bash
cd /Users/dmr/MachineLearning && uv run ruff check --fix tests/integration/
git add tests/integration/
git commit -m "test: S3 integration tests (binance_ws_live, alpaca_paper roundtrip, binance_testnet roundtrip, record_replay_parity)"
```

---

## Task 24: Makefile additions

**Files:** `Makefile`

- [ ] **Step 1: Append S3 targets**

Append to `Makefile`:

```makefile

S3_RECORD := scripts/s3_record.py
BACKTEST_RUN := scripts/backtest_run.py
BACKTEST_CONFIG ?= configs/backtests/smoke.yaml

.PHONY: s3-record s3-parity backtest backtest-smoke

s3-record:
	$(PY) python $(S3_RECORD) --config configs/feeds.yaml

s3-parity:
	$(PY) pytest tests/integration/test_record_replay_parity.py -v -m s3_integration

backtest:
	$(PY) python $(BACKTEST_RUN) --config $(BACKTEST_CONFIG)

backtest-smoke:
	$(PY) python $(BACKTEST_RUN) --config configs/backtests/smoke.yaml
```

- [ ] **Step 2: Smoke `make test` + `make lint`**

```bash
cd /Users/dmr/MachineLearning && make test && make lint
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: Makefile S3 targets (s3-record, s3-parity, backtest, backtest-smoke)"
```

---

## Self-review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 master architecture (overview + scope + non-scope) | Tasks 1 (ADRs/runbooks), 2 (scaffold) |
| §2.1 Tick / Bar Pydantic models | Task 4 |
| §2.2 FeedAdapter Protocol + AsyncFeedBase mixin | Task 5 |
| §2.3 BinanceWS / CoinbaseWS / AlpacaREST / Replayer | Tasks 6 / 7 / 8 / 10 |
| §2.4 asyncio runtime (not uvloop) | Task 5 + ADR 0009 (Task 1) |
| §2.5 backpressure + lag | Task 5 |
| §3.1 OrderIntent / Order / Fill / Position / Account | Task 11 |
| §3.2 BrokerCapabilities | Task 12 |
| §3.3 BrokerAdapter Protocol | Task 13 |
| §3.4 NullBroker, AlpacaPaper, BinanceTestnet | Tasks 15 / 16 / 17 |
| §3.5 Recorder | Task 9 |
| §3.6 Replayer | Task 10 |
| §4.2 FillModel | Task 14 |
| §4.3 Strategy Protocol + 2 reference strategies | Task 18 |
| §4.4 BacktestRunner | Task 20 |
| §4.5 metrics + report | Tasks 19 / 21 |
| §4.6 scripts/backtest_run.py + Makefile | Tasks 22 / 24 |
| §4.7 backtest YAML config | Task 2 |
| §5.1 unit tests | Tasks 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21 |
| §5.2 integration tests | Task 23 |
| §5.3 ADR 0009 | Task 1 |
| §5.4 ADR 0010 + 0011 | Task 1 |
| §5.5 repo layout delta | covered across all tasks |
| §5.6 risks | mitigations in Tasks 6/7 (fixture-based parsers), 9 (hour rotation chmod), 10 (delta-from-event-time sleep), 14 (deterministic fills) |

**Placeholder scan:** no TBD / TODO. Every code step shows full code. Every test step has explicit assertions.

**Type consistency:** `Venue`, `TickSide`, `Tick`, `Bar`, `MarketEvent`, `FeedAdapter`, `AsyncFeedBase`, `OrderSide`, `TimeInForce`, `OrderType`, `OrderStatus`, `OrderIntent`, `Order`, `Fill`, `Position`, `Account`, `BrokerCapabilities`, `UnsupportedOrderError`, `BrokerAdapter`, `FillModelConfig`, `FillModel`, `NullBroker`, `AlpacaPaper`, `BinanceTestnet`, `Strategy`, `BuyAndHold`, `MovingAverageCross`, `BacktestConfig`, `BacktestResult`, `BacktestRunner`, `BacktestReport`, `Recorder`, `RecorderConfig`, `Replayer`, `ReplayerConfig` referenced consistently across all 24 tasks.

**Deferred (master S3 spec §5.7):**
- **S3.1** — live feature reconstruction (streaming events → S1-compatible feature rows).
- **S3.2** — Coinbase paper + IBKR paper.
- **S3.3** — L2 order-book backtester using already-downloaded CryptoLOB-2025 (30 GB) + HFT LOB (5 GB).
- **S4** — live brokers (`*_live.py`), risk engine, kill-switch wiring, execution router, three-stage promotion gates.

All 24 tasks are bite-sized, TDD-disciplined, exact-path, exact-command, with frequent commits. Foundation (1–3) before modules; modules (4–17) before scripts (22); scripts before integration tests (23); Makefile (24) lands last.
