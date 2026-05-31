# QuantLab Alpha — S3 (Feeds + Brokers + Backtester) Design

**Date:** 2026-05-17
**Status:** Approved (sections 1–5 walked through with operator)
**Project:** QuantLab Alpha (`/Users/dmr/MachineLearning`)
**Master spec:** `docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md` (§3.2)
**Predecessor specs:** S1 design (2026-05-14), S2 governor design (2026-05-16)

S3 is the IO layer of the QuantLab Alpha platform. It owns the typed boundary between market reality and our models — feeds, brokers, recorder, replayer — plus a deterministic backtester that runs strategies against recorded data with realistic transaction-cost modeling. S3 ships **paper-broker adapters only**; live `*_live.py` adapters and the kill-switch / risk-engine plumbing land in S4.

---

## 1. Master Architecture

```text
                     ┌─────────────────────────────────────────────┐
                     │                S3 BOUNDARY                  │
                     │                                             │
   crypto WS  ──────►│ FeedAdapter (BinanceWS) ──┐                 │
                     │                           ▼                 │
   coinbase WS ─────►│ FeedAdapter (CoinbaseWS)─►│  TickStream     │──► Recorder ──► data/live/parquet/<venue>/<symbol>/<YYYY-MM-DD>/<HH>.parquet
                     │                           │  (Tick | Bar    │
   alpaca REST ─────►│ FeedAdapter (AlpacaREST)─►│   normalized)   │──► Subscribers (S1, S2, future S3.1)
                     │                           │                 │
   replayer file ───►│ FeedAdapter (Replayer)───►┘                 │
                     │                                             │
                     │     ┌──────────────────────────┐            │
                     │     │ BrokerAdapter interface  │◄─── orders from S4 / strategy harness
                     │     │  - place_order           │
                     │     │  - cancel_order          │
                     │     │  - positions             │
                     │     │  - account               │
                     │     │  - capabilities          │
                     │     └──────────────┬───────────┘            │
                     │                    │                        │
                     │  ┌─────────────────┼─────────────────────┐  │
                     │  ▼                 ▼                     ▼  │
                     │ AlpacaPaper   BinanceTestnet         NullBroker
                     │                                             │
                     └─────────────────────────────────────────────┘

                     Backtester (S3, lives on top of Replayer + NullBroker):
                       BacktestConfig ──► Replayer ──events──► Strategy ──orders──► NullBroker(FillModel) ──fills──► PnL ──► Report
```

**In S3 scope:**
- `feeds/`: typed `FeedAdapter` Protocol + concrete adapters (`BinanceWS`, `CoinbaseWS`, `AlpacaREST`, `Replayer`).
- `brokers/`: typed `BrokerAdapter` Protocol + `AlpacaPaper`, `BinanceTestnet`, `NullBroker`.
- `recorder.py`: writes every live event to Parquet shards rotated hourly, `chmod a-w` on close.
- `replayer.py`: replays recorded files via the FeedAdapter interface at configurable speed.
- `market_types.py`: shared `Tick`, `Bar` Pydantic types.
- `order_types.py`: shared `OrderIntent`, `Order`, `Fill`, `Position`, `Account` Pydantic types.
- `capabilities.py`: per-broker capability declaration.
- `backtest/`: deterministic backtester (slippage-aware `NullBroker` + `BacktestRunner` + metrics + report).
- 3 ADRs (0009, 0010, 0011) + 2 runbooks.

**Explicitly NOT in S3 scope (each is its own future spec):**
- **S3.1** — Live feature reconstruction (streaming events → S1-compatible feature rows).
- **S3.2** — Coinbase paper and IBKR paper broker adapters.
- **S3.3** — L2 order-book backtesting with realistic queue position. Uses already-downloaded 35 GB of LOB data (`qkxuuuu/cryptolob-2025` 30 GB + `martinsn/high-frequency-crypto-limit-order-book-data` 5 GB).
- **S4** — Live brokers (`*_live.py`), risk engine, kill-switch wiring, execution router, three-stage promotion gates.

**Per-event normalization contract:**
Every adapter emits the same `Tick` and `Bar` Pydantic models regardless of source — Binance's `aggTrade`, Coinbase's `matches`, Alpaca's REST bars, and replayed Parquet rows all flatten to the same shape. Adapters own the translation; nothing downstream knows the source.

---

## 2. Market Types and FeedAdapter Interface

### 2.1 Shared market types

A single Pydantic module that every adapter, recorder, and replayer agrees on. Strict, immutable, JSON-serializable, free of any source-specific fields.

```python
# src/quant_research_stack/feeds/market_types.py

from __future__ import annotations
from enum import StrEnum
from typing import Annotated
from datetime import datetime
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
    timestamp_utc: datetime           # exchange timestamp, UTC, microsecond precision
    received_utc: datetime            # local clock at receipt — used for drift checks
    price: Annotated[float, Field(gt=0.0)]
    size: Annotated[float, Field(ge=0.0)]
    side: TickSide
    sequence: int | None = None       # venue sequence number if available
    raw: dict | None = None           # original payload kept only when --keep-raw is set

class Bar(BaseModel):
    model_config = {"frozen": True}
    venue: Venue
    symbol: str
    timestamp_utc: datetime           # bar start (left-edge convention)
    interval_seconds: Annotated[int, Field(ge=1, le=86400)]
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int | None = None
```

Locked-in conventions:
- All times are **UTC microsecond**, never local.
- All prices and sizes are **float64 base units** of the symbol (BTC, not satoshis).
- Bars are **left-edge** (a 09:30 bar covers 09:30:00–09:30:59).
- `received_utc - timestamp_utc` is the adapter-attributed lag; the recorder logs it; future S4 wires large drifts to the NTP-drift kill trigger.

### 2.2 FeedAdapter Protocol

```python
# src/quant_research_stack/feeds/base.py

from collections.abc import AsyncIterator, Iterable
from typing import Protocol
from quant_research_stack.feeds.market_types import Bar, Tick, Venue

MarketEvent = Tick | Bar

class FeedAdapter(Protocol):
    venue: Venue
    async def subscribe(self, symbols: Iterable[str]) -> None: ...
    async def iterate(self) -> AsyncIterator[MarketEvent]: ...
    async def close(self) -> None: ...
    @property
    def is_live(self) -> bool: ...
    @property
    def stats(self) -> dict: ...        # {events_emitted, last_event_lag_ms, reconnects, dropped_count}
```

Every adapter is **async-iterator-shaped**. Consumers use a uniform pattern:

```python
async with adapter:                                  # __aenter__ subscribes
    async for event in adapter.iterate():            # MarketEvent stream
        await handle(event)
```

`__aenter__` / `__aexit__` are implemented in a base `AsyncFeedBase` mixin so adapters only fill in the Protocol methods. Reconnect-with-exponential-backoff also lives in the mixin (default: 1 s → 60 s, max 10 attempts before raising `FeedConnectionError`).

### 2.3 Concrete adapters in S3

| Adapter | File | Frequency | Auth | Notes |
|---|---|---|---|---|
| `BinanceWS` | `feeds/binance_ws.py` | tick | none | Public `wss://stream.binance.com:9443/ws`; `aggTrade` channel |
| `CoinbaseWS` | `feeds/coinbase_ws.py` | tick | none | Public `wss://ws-feed.exchange.coinbase.com`; `matches` + `ticker` channels |
| `AlpacaREST` | `feeds/alpaca_rest.py` | 15-min bars | free API key | `https://data.alpaca.markets/v2/stocks/bars` polled at bar close |
| `Replayer` | `feeds/replayer.py` | matches input | none | Reads `data/live/parquet/<venue>/<symbol>/<YYYY-MM-DD>/<HH>.parquet`, emits in timestamp order at configurable speed |

`Replayer` is the test fixture for the rest of the stack — record-then-replay parity (§5.1) is the integration test that proves the abstraction is sound.

### 2.4 Async runtime: `asyncio` only (no `uvloop` dep)

`asyncio` on Python 3.11 is fast enough for our event rates. `uvloop` would add a build-time dep and save microseconds per yield — same logic as ADR 0009.

### 2.5 Backpressure and lag

Each `iterate()` yields events to a single consumer. If the consumer is slow, events queue in the adapter's internal buffer (capped at 10 000 per symbol). Buffer overflow drops the **oldest** event and increments `dropped_count` (better to lose a stale tick than block on a hot path). Stats are surfaced via `adapter.stats`; S4 wires the count to the kill switch later.

---

## 3. BrokerAdapter Interface, Capabilities, Recorder, Replayer

### 3.1 Order types and broker-agnostic schema

```python
# src/quant_research_stack/brokers/order_types.py

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
    take_profit_price: float | None = None        # for bracket
    stop_loss_price: float | None = None          # for bracket
    oco_limit_price: float | None = None          # for oco — the other leg's limit
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
    quantity: float                    # signed; positive long, negative short
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

### 3.2 Capabilities — per-broker truth in code

```python
# src/quant_research_stack/brokers/capabilities.py

from __future__ import annotations
from dataclasses import dataclass
from quant_research_stack.brokers.order_types import OrderType, TimeInForce

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
```

Each broker module declares its capabilities at module level. Capabilities are checked at `place_order()` entry; an unsupported `OrderType` raises `UnsupportedOrderError(venue, type, suggestion)` **before** any network call.

Initial capability declarations:

| Venue | Order types | Shorting | Fractional | Extended hours |
|---|---|---|---|---|
| `alpaca_paper` | market, limit, stop, stop_limit, bracket, oco | yes | yes | yes |
| `binance_testnet` | market, limit, stop_limit, oco | spot: no | yes | n/a |
| `null_broker` | all | yes | yes | yes |

### 3.3 BrokerAdapter Protocol

```python
# src/quant_research_stack/brokers/base.py

from collections.abc import AsyncIterator
from typing import Protocol
from quant_research_stack.brokers.order_types import Account, Fill, Order, OrderIntent, Position
from quant_research_stack.brokers.capabilities import BrokerCapabilities

class BrokerAdapter(Protocol):
    capabilities: BrokerCapabilities

    async def place_order(self, intent: OrderIntent) -> Order: ...
    async def cancel_order(self, client_order_id: str) -> Order: ...
    async def get_order(self, client_order_id: str) -> Order: ...
    async def positions(self) -> list[Position]: ...
    async def account(self) -> Account: ...
    async def stream_fills(self) -> AsyncIterator[Fill]: ...
    async def close(self) -> None: ...
```

`stream_fills()` is the only push channel. Everything else is request/response.

### 3.4 Three concrete brokers in S3 scope

| File | Network | Auth | Notes |
|---|---|---|---|
| `brokers/null_broker.py` | none | none | Records intent only; assigns deterministic `broker_order_id`. Used in tests, S4's `live_shadow` stage, and the backtester. Fills via the `FillModel` (§4.2). |
| `brokers/alpaca_paper.py` | `https://paper-api.alpaca.markets` | API key + secret in `~/.alpaca/paper_keys.json` (chmod 600) | `alpaca-py` SDK; polls order status; fills via REST. Full equity order types supported. |
| `brokers/binance_testnet.py` | `wss://testnet.binance.vision/ws` + `https://testnet.binance.vision` | API key + secret in `~/.binance/testnet_keys.json` | Spot only (no shorting). OCO supported. `python-binance` SDK in async mode. |

Coinbase paper and IBKR paper are deferred to S3.2.

### 3.5 Recorder

```python
# src/quant_research_stack/feeds/recorder.py

@dataclass(frozen=True)
class RecorderConfig:
    root: Path                    # data/live/parquet
    flush_every_n_events: int = 1024
    flush_every_seconds: float = 5.0
    keep_raw: bool = False        # if True, persist Tick.raw / Bar.raw too

class Recorder:
    """Subscribes to a FeedAdapter and writes events to disk in append-only Parquet shards.

    Path scheme: <root>/<venue>/<symbol>/<YYYY-MM-DD>/<HH>.parquet
    One file per hour per symbol per venue. Each file is closed and chmod a-w
    at the end of its hour. The recorder is the source of truth for replay.
    """

    async def run(self, adapter: FeedAdapter) -> None: ...
    def stats(self) -> dict: ...   # {events_written, files_closed, last_flush_lag_ms}
```

Append-only and hour-rotated to mirror the S2 audit-log discipline. Files are immutable after close. A separate `pyarrow.parquet.ParquetWriter` per (venue, symbol, hour) — opened lazily on first event, closed on hour rollover.

### 3.6 Replayer — the integrity test for the entire abstraction

```python
# src/quant_research_stack/feeds/replayer.py

@dataclass(frozen=True)
class ReplayerConfig:
    root: Path                    # data/live/parquet
    venue: Venue
    symbols: tuple[str, ...]
    start_utc: datetime
    end_utc: datetime
    speed: float = 1.0            # 1.0 = real time; 10.0 = 10x; 0.0 = as-fast-as-possible

class Replayer(FeedAdapter):
    """Implements FeedAdapter by reading recorded Parquet shards and yielding events in
    timestamp order. At speed=1.0 it sleeps between events to match wall-clock pacing;
    at speed=0.0 it iterates as fast as the consumer drains.
    """
```

The **record-then-replay parity invariant** is the integration test for the whole S3 layer:

```text
record 60 min of live BinanceWS BTCUSDT
    → recorded.parquet
replay recorded.parquet at speed=0.0 through the same downstream code path
    → events_seen, events_dropped, lag_stats
assert: events_seen == recorded.parquet row count
assert: no dropped events
assert: timestamp order strictly monotonic
```

If parity fails, the abstraction is leaking source-specific behavior. This is the test that catches every "but Binance does X differently" bug.

### 3.7 What this section does *not* include

- Live brokers (`*_live.py`) — S4.
- Risk engine pre-trade checks (position size, gross exposure) — S4.
- Order routing across multiple brokers — S4.
- Strategy harness / signal-to-order translation for the live S1 model — S3.1.

---

## 4. Backtesting (folded into S3)

The replayer plus a null broker already gets you historical replay, but null fills at the next tick's price is not a backtest — it's tape playback. A *proper* backtest needs realistic costs, a strategy interface, and a report.

### 4.1 What "proper backtesting" means here

| Concern | In S3 scope | Deferred (which spec) |
|---|---|---|
| No look-ahead bias | yes (Replayer yields in strict timestamp order) | — |
| Realistic transaction costs (commission + slippage) | yes (slippage-aware NullBroker) | — |
| Latency simulation (signal-to-fill delay) | yes (configurable per-fill delay in ms) | — |
| Walk-forward validation | yes (already in S1's `cv.py`) | — |
| Bid-ask spread modeling | yes, fixed-bps approximation | L2 order-book backtester → S3.3 |
| Market impact / queue position | no | S3.3 (uses CryptoLOB-2025 + HFT LOB datasets) |
| Capacity constraints (max-size, ADV%) | yes, declarative caps | — |
| Multi-asset portfolio PnL | yes | — |
| Performance metrics (Sharpe, max DD, hit rate, turnover) | yes | — |
| Equity curve + tearsheet plots | yes (matplotlib PNG only, no notebooks) | — |
| Walk-forward parameter optimization on backtest | no (S1's CV already does this for the model) | — |

### 4.2 Slippage + commission model (extends `NullBroker`)

```python
# src/quant_research_stack/brokers/fill_model.py

@dataclass(frozen=True)
class FillModelConfig:
    commission_bps: float = 1.0           # per-side commission, basis points of notional
    slippage_bps: float = 2.0             # per-side adverse slippage, bps of notional
    half_spread_bps: float = 1.0          # half the typical bid-ask spread
    fill_latency_ms: int = 50             # synthetic delay between order submission and fill
    reject_if_notional_above_pct_adv: float | None = None   # capacity gate; None disables
    partial_fill_max_pct_of_book: float = 0.10              # cap per-fill share of next bar's volume

class FillModel:
    """Deterministic fill simulator. Given an OrderIntent and a stream of subsequent
    market events, returns Fill(s) with realistic price + commission. No randomness;
    backtests are reproducible from the same input."""

    def synthesize(self, intent: OrderIntent, market_iter: Iterator[Tick | Bar]) -> list[Fill]: ...
```

Fill price formula (long-buy market order):

```text
fill_px    = next_event_mid + half_spread_bps * 1e-4 * next_event_mid
                            + slippage_bps     * 1e-4 * next_event_mid
commission = fill_px * qty * commission_bps * 1e-4
```

`NullBroker` is wired with a `FillModel` instance. Backtests can swap a `ZeroCostFillModel` (commission=0, slippage=0) in for sanity checks; the default config is realistic-to-pessimistic.

### 4.3 Strategy Protocol

```python
# src/quant_research_stack/backtest/strategy.py

from typing import Protocol
from quant_research_stack.brokers.order_types import Fill, OrderIntent
from quant_research_stack.feeds.market_types import Bar, Tick

class Strategy(Protocol):
    name: str
    def on_event(self, event: Tick | Bar) -> list[OrderIntent]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def snapshot_state(self) -> dict: ...   # for the audit log
```

S3 ships **two reference strategies** so the harness has something to run against:

```text
backtest/strategies/buy_and_hold.py           buy max size at first event, never sell
backtest/strategies/moving_average_cross.py   classic MA crossover, parameterizable windows
```

These exist purely to exercise the backtester end-to-end. The real S1-driven strategy waits for S3.1 (feature reconstruction) and S4 (risk + routing).

### 4.4 BacktestRunner

```python
# src/quant_research_stack/backtest/runner.py

@dataclass(frozen=True)
class BacktestConfig:
    replayer: ReplayerConfig
    fill_model: FillModelConfig
    starting_cash: float = 100_000.0
    strategy_name: str                       # "buy_and_hold" | "moving_average_cross" | "external"
    strategy_params: dict
    metrics_horizon_minutes: int = 60        # bucket size for time-series metrics

class BacktestRunner:
    async def run(self) -> BacktestResult: ...
        # 1. Construct Replayer + NullBroker(FillModel)
        # 2. Construct Strategy from name + params
        # 3. for event in replayer: on_event → place_order → fills → on_fill
        # 4. Mark-to-market every event; capture equity curve
        # 5. Emit BacktestResult
```

### 4.5 Metrics and report

```python
# src/quant_research_stack/backtest/metrics.py — pure functions

def total_return(equity_curve: pl.DataFrame) -> float: ...
def sharpe_ratio(returns: pl.Series, periods_per_year: int) -> float: ...
def max_drawdown(equity_curve: pl.DataFrame) -> float: ...
def calmar_ratio(equity_curve: pl.DataFrame) -> float: ...
def hit_rate(fills: list[Fill]) -> float: ...
def turnover(fills: list[Fill], starting_cash: float) -> float: ...
def value_at_risk(returns: pl.Series, alpha: float = 0.05) -> float: ...
```

```python
# src/quant_research_stack/backtest/report.py

class BacktestReport:
    """Writes metrics.json, fills.parquet, equity_curve.parquet, report.md,
    equity_curve.png, drawdown.png to experiments/backtests/<run_id>/."""
```

Markdown reports diff cleanly in PRs and render on GitHub. Plots are matplotlib PNG only (no Plotly / no JS) for reproducibility and small artifact size.

### 4.6 Entry point and Makefile targets

```text
scripts/backtest_run.py            # CLI: --config configs/backtests/<name>.yaml
```

Makefile additions:

```makefile
s3-record:
	$(PY) python scripts/s3_record.py --config configs/feeds.yaml

s3-parity:
	$(PY) pytest tests/integration/test_record_replay_parity.py -v -m s3_integration

backtest:
	$(PY) python scripts/backtest_run.py --config $(BACKTEST_CONFIG)

backtest-smoke:
	$(PY) python scripts/backtest_run.py --config configs/backtests/smoke.yaml
```

### 4.7 Backtest configuration — declarative YAML

```yaml
# configs/backtests/smoke.yaml

replayer:
  root: data/live/parquet
  venue: binance
  symbols: [BTCUSDT]
  start_utc: 2026-05-15T00:00:00Z
  end_utc:   2026-05-15T01:00:00Z
  speed: 0.0                      # as fast as possible

fill_model:
  commission_bps: 1.0
  slippage_bps: 2.0
  half_spread_bps: 1.0
  fill_latency_ms: 50

starting_cash: 100000.0
strategy_name: buy_and_hold
strategy_params: {}
metrics_horizon_minutes: 5
```

### 4.8 Strategy gap and follow-ups

The backtester *runs* a `Strategy`. The two reference strategies prove the harness works. The S1 model itself **cannot** be plugged in as a `Strategy` yet — that's the S3.1 spec, because S1 needs features that don't exist on a raw event stream. Docs are explicit: "The backtester is complete; the live-S1 strategy is one spec away."

---

## 5. Testing, Success Criteria, ADRs, Repo Layout, Risks, Transition

### 5.1 Testing strategy

**Unit tests (no network, no models — fast):**

```text
tests/test_feeds_market_types.py        Tick/Bar field validation; UTC enforcement; immutability
tests/test_feeds_base.py                FeedAdapter Protocol surface; AsyncFeedBase backoff + reconnect math
tests/test_feeds_binance_ws.py          parse_aggtrade_event() on 20 fixture payloads → exact Tick output
tests/test_feeds_coinbase_ws.py         parse_match_event() on 20 fixture payloads → exact Tick output
tests/test_feeds_alpaca_rest.py         parse_bars_response() on fixture JSON; pagination handling
tests/test_feeds_replayer.py            monotonic order, speed=0 drains, speed=1 sleeps; symbol filter
tests/test_feeds_recorder.py            hourly rotation, chmod a-w on close, schema stable across hours
tests/test_brokers_order_types.py       OrderIntent validation (each type's required fields enforced)
tests/test_brokers_capabilities.py      UnsupportedOrderError raised before network on unsupported type
tests/test_brokers_null_broker.py       deterministic broker_order_id; stream_fills drains
tests/test_brokers_alpaca_paper.py      request-builder unit tests only (real API hits → integration)
tests/test_brokers_binance_testnet.py   request-builder unit tests only
tests/test_brokers_fill_model.py        slippage math, commission math, latency delay, capacity rejection
tests/test_backtest_strategy.py         buy_and_hold + ma_cross strategies on fixture event stream
tests/test_backtest_runner.py           end-to-end on a 100-event fixture; deterministic equity curve
tests/test_backtest_metrics.py          hand-computed values for sharpe, max_dd, hit_rate, turnover
tests/test_backtest_report.py           writes all required artifacts; markdown is valid
```

**Integration tests (`s3_integration` marker, skipped by default):**

```text
tests/integration/test_binance_ws_live.py         60 s real connection, asserts >= 1 event
tests/integration/test_alpaca_paper_roundtrip.py  place + cancel + get + positions on paper account
tests/integration/test_binance_testnet_roundtrip.py    place + cancel + get + positions on testnet
tests/integration/test_record_replay_parity.py    THE invariant — record 60 s live, replay, compare traces
```

`tests/integration/test_record_replay_parity.py` is the contract test for the whole abstraction. Runs in CI on demand via `make s3-parity`. If it fails, S3 is broken regardless of what unit tests pass.

### 5.2 Success criteria

| # | Criterion | Measurable how | Threshold |
|---|---|---|---|
| 1 | All adapters parse fixture payloads correctly | unit tests | 0 failures across 80+ fixtures |
| 2 | Record-replay parity | integration test | bytes-identical event sequence and counts |
| 3 | Replayer monotonic timestamps | unit test | strictly increasing within and across files |
| 4 | Backpressure does not block | unit test | full buffer drops oldest, increments counter |
| 5 | Reconnect with backoff | unit test | 10 attempts with exponential 1 s → 60 s; fails cleanly |
| 6 | Recorder hourly rotation | unit test | exactly N hour-files for an N-hour fixture |
| 7 | Recorder chmod a-w on close | unit test | written hour files are read-only |
| 8 | UnsupportedOrderError thrown pre-network | unit test | OCO on a venue that lacks OCO raises before any HTTP |
| 9 | NullBroker fills are deterministic | unit test | same input → same fills, byte-identical |
| 10 | Backtest reproducibility | unit + integration | identical config produces identical metrics across two runs |
| 11 | Fill model commission/slippage math | unit test | hand-computed values match within 1e-9 |
| 12 | Smoke backtest completes < 30 s | `make backtest-smoke` | wall-clock < 30 s on M4 for the 1-hour BTCUSDT replay |
| 13 | Live adapters never imported by tests | grep CI guard | no `from quant_research_stack.brokers import *_live` in `tests/` |

Criterion 13 — no live brokers exist in S3, so this guard is a contract to honor when S4 lands them.

### 5.3 ADR 0009 — Python for S3, C++ deferred

```markdown
# ADR 0009: Python for S3 (feeds + brokers); C++ adapters deferred until measured benefit

## Status
Accepted, 2026-05-17.

## Context
C++ is the conventional language for low-latency quant infrastructure. The operator
asked whether S3 (real-time feeds + broker abstraction) should be written in C++ to
match industry practice.

## Decision
S3 is implemented in Python. C++ adapters are deferred until profiling on real
production traffic shows a measurable latency win.

## Latency math at decision time
- Hardware: MacBook Air M4 in Istanbul (not a colo cabinet).
- Feeds: Binance / Coinbase public WebSocket, network RTT ~50-200 ms.
- Event rate at peak: Binance BTCUSDT aggTrade ~500-2 000 events/sec.
- Python json.loads on each event: ~5 microseconds.

A C++ feed handler would save ~5 microseconds per event. The network RTT (~50-100 ms)
is 20 000x larger. The latency win is unmeasurable on this network path.

## Architectural enabler
The FeedAdapter Protocol boundary is language-agnostic. A future C++ adapter wrapped
via pybind11 can drop in behind the same Protocol with zero downstream changes to
S1 / S2 / S4 / backtester.

## Triggers that would justify revisiting
- Co-location with the exchange (network RTT drops to single-digit microseconds).
- Migration to ITCH / OUCH / FIX-FAST binary feeds (parsing becomes the bottleneck).
- Aggregating L2 / L3 order books across >= 5 venues simultaneously.
- Production profiling shows p99 event-handler latency > 100 ms attributable to Python.

## Consequences
+ Faster iteration: hot-reload strategy modules in seconds.
+ Smaller build surface: no cmake / vcpkg / pybind11 in the default install path.
+ The heavy compute (Polars, NumPy, LightGBM, llama.cpp) is already C++ underneath;
  Python is the orchestration layer where its speed cost is invisible.
- A future migration to C++ for a specific bottleneck is non-trivial (pybind11 wrappers,
  separate test surface). Mitigated by keeping the Protocol boundary clean.
```

### 5.4 Other ADRs

ADR 0010 documents the fill model assumptions (fixed-bps slippage, half-spread, latency); ADR 0011 documents record-replay parity as a contract test rather than a development convention. Both are short and land in Task 1 of the implementation plan.

### 5.5 Repository layout delta from S2

```text
configs/
  feeds.yaml                                        NEW — adapter constructors, recording paths
  brokers.yaml                                      NEW — paper-broker credentials paths, capability overrides
  backtests/smoke.yaml                              NEW — first reference backtest config
src/quant_research_stack/feeds/
  __init__.py                                       NEW
  market_types.py                                   NEW
  base.py                                           NEW (Protocol + AsyncFeedBase mixin)
  binance_ws.py                                     NEW
  coinbase_ws.py                                    NEW
  alpaca_rest.py                                    NEW
  replayer.py                                       NEW
  recorder.py                                       NEW
src/quant_research_stack/brokers/
  __init__.py                                       NEW
  order_types.py                                    NEW
  capabilities.py                                   NEW
  base.py                                           NEW (Protocol)
  null_broker.py                                    NEW
  alpaca_paper.py                                   NEW
  binance_testnet.py                                NEW
  fill_model.py                                     NEW
src/quant_research_stack/backtest/
  __init__.py                                       NEW
  strategy.py                                       NEW (Protocol + 2 reference strategies)
  strategies/buy_and_hold.py                        NEW
  strategies/moving_average_cross.py                NEW
  runner.py                                         NEW
  metrics.py                                        NEW
  report.py                                         NEW
scripts/
  s3_record.py                                      NEW (long-running recorder CLI)
  backtest_run.py                                   NEW
docs/architecture/adrs/
  0009-python-for-s3-cpp-deferred.md                NEW
  0010-fill-model-and-fixed-bps-slippage.md         NEW
  0011-record-replay-parity-as-contract-test.md     NEW
docs/runbooks/
  s3_recorder_ops.md                                NEW — start/stop the recorder, disk hygiene
  s3_paper_broker_credentials.md                    NEW — where to drop keys, chmod rules
Makefile                                            MODIFY — add s3-record, s3-parity, backtest, backtest-smoke targets
```

### 5.6 Risks this spec carries

| Risk | Mitigation |
|---|---|
| Binance / Coinbase WebSocket schemas change | Fixture-based parser tests; CI re-runs against a frozen fixture; vendor change surfaces as a test failure not a silent corruption. |
| Recorder runs out of disk | `s3_recorder_ops.md` runbook covers disk hygiene; recorder logs `dropped_count` if writes block. |
| Replayer pacing drifts under load | Replayer uses `asyncio.sleep` with delta-from-event-time semantics, not fixed `sleep(0.001)`. Drift bounded by the consumer's `iterate()` speed. |
| Paper broker behavior diverges from live | Capability flags declare known divergences; cross-broker contract test runs the same scenarios against `null_broker` and each `*_paper`. |
| Fill model is too optimistic | Default config errs conservative (1 bps commission + 2 bps slippage + 1 bps half-spread + 50 ms latency = 4 bps total per-side, realistic-to-pessimistic for retail equity). Per-strategy configs can tighten or relax. |
| L2 backtester deferred to S3.3 means S3's backtester underestimates impact for large sizes | Documented limitation in the report; backtest config has `reject_if_notional_above_pct_adv` cap to refuse tests that should have used L2. |
| Paper API endpoints rate-limit during testing | Each paper-broker test uses a small budget of orders per test class; module-level fixture coordinates the limit. |

### 5.7 What's deferred (explicit follow-up specs)

- **S3.1** — Live feature reconstruction (streaming events → S1-compatible feature rows). Blocks live S1 execution.
- **S3.2** — Coinbase and IBKR paper brokers.
- **S3.3** — L2 order-book backtesting with realistic queue position. Uses already-downloaded 35 GB of LOB data (`qkxuuuu/cryptolob-2025` 30 GB + `martinsn/high-frequency-crypto-limit-order-book-data` 5 GB).
- **S4** — Live brokers (`*_live.py`), risk engine, kill-switch wiring, execution router, three-stage promotion gates.

### 5.8 Spec doc transition

After this spec is approved:

1. Inline self-review (placeholder scan, internal consistency, scope, ambiguity).
2. Spec committed.
3. Operator reviews the written spec.
4. On approval, `superpowers:writing-plans` produces the detailed S3 implementation plan.

The S3 plan follows the same TDD-discipline pattern as S1 and S2 — foundation modules first (`market_types`, `order_types`, `capabilities`), then adapters (`binance_ws`, `coinbase_ws`, `alpaca_rest`), then recorder + replayer, then brokers (`null_broker`, `alpaca_paper`, `binance_testnet`), then backtest layer (`fill_model`, `strategy`, `runner`, `metrics`, `report`). ADRs 0009 / 0010 / 0011 land as Task 1.
