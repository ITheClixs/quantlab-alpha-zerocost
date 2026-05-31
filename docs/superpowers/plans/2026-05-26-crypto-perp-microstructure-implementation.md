# Crypto Perp Microstructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free-data-first Binance perpetual futures microstructure research path for BTCUSDT and ETHUSDT with historical ingestion, event-driven bid/ask backtests, strict validation, live recorder/replayer practice, and paper-only shadow execution.

**Architecture:** Add an isolated `crypto_research/perps` package that normalizes Binance/HF perpetual event data into typed parquet tables, builds timestamp-safe features/labels, trains tabular walk-forward models, and evaluates them with executable bid/ask costs. Extend the existing Binance feed surface for public stream parsing/recording and reuse existing backtest, broker, execution, PBO, bootstrap, and reporting patterns where practical.

**Tech Stack:** Python 3.13, Polars, NumPy, scikit-learn, websockets, existing `quant_research_stack` feed/broker/execution modules, pytest, ruff, mypy.

---

## File Structure

Create:

- `src/quant_research_stack/crypto_research/perps/__init__.py`
  Public exports for the perps package.
- `src/quant_research_stack/crypto_research/perps/events.py`
  Typed normalized event schemas and parser helpers for aggTrade, bookTicker, depth, mark-price, and liquidation records.
- `src/quant_research_stack/crypto_research/perps/manifest.py`
  Dataset manifest creation, hashing, schema capture, and quality labels.
- `src/quant_research_stack/crypto_research/perps/normalize.py`
  Historical HF/Binance file normalization into event parquet tables.
- `src/quant_research_stack/crypto_research/perps/features.py`
  Timestamp-safe feature and label builder from normalized trade/book data.
- `src/quant_research_stack/crypto_research/perps/backtest.py`
  Event-driven bid/ask backtest, per-trade audit, and cost stress diagnostics.
- `src/quant_research_stack/crypto_research/perps/training.py`
  Chronological walk-forward tabular model training and prediction artifacts.
- `src/quant_research_stack/crypto_research/perps/validation.py`
  PBO/DSR/bootstrap/concentration gate helpers for the perps registry.
- `src/quant_research_stack/crypto_research/perps/realtime.py`
  Raw WebSocket recorder and replay metadata for Binance public streams.
- `src/quant_research_stack/crypto_research/perps/reports.py`
  Markdown/JSON reports and status semantics.
- `scripts/crypto_perp_microstructure_loop.py`
  End-to-end CLI for normalize → feature → train → backtest → validate → report.
- `scripts/crypto_perp_record_live.py`
  Short-running Binance public WebSocket recorder CLI.
- `scripts/crypto_perp_replay_live.py`
  Replay recorded raw events through parser and feature builder.

Modify:

- `src/quant_research_stack/feeds/binance_ws.py`
  Add pure parsers for bookTicker and depth events while keeping existing aggTrade behavior.
- `src/quant_research_stack/crypto_research/__init__.py`
  Export the `perps` package if local conventions require it.

Tests:

- `tests/crypto_research/perps/test_events.py`
- `tests/crypto_research/perps/test_manifest.py`
- `tests/crypto_research/perps/test_normalize.py`
- `tests/crypto_research/perps/test_features.py`
- `tests/crypto_research/perps/test_backtest.py`
- `tests/crypto_research/perps/test_training.py`
- `tests/crypto_research/perps/test_validation.py`
- `tests/crypto_research/perps/test_realtime.py`
- `tests/crypto_research/perps/test_cli_smoke.py`
- update or add `tests/test_feeds_binance_ws.py` only if such a file already exists; otherwise create `tests/test_feeds_binance_ws_parsers.py`.

---

### Task 1: Normalized Event Schemas And Parsers

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/__init__.py`
- Create: `src/quant_research_stack/crypto_research/perps/events.py`
- Modify: `src/quant_research_stack/feeds/binance_ws.py`
- Test: `tests/crypto_research/perps/test_events.py`
- Test: `tests/test_feeds_binance_ws_parsers.py`

- [ ] **Step 1: Write parser tests**

Add tests with concrete Binance-like payloads:

```python
from datetime import UTC, datetime

from quant_research_stack.crypto_research.perps.events import (
    normalize_agg_trade,
    normalize_book_ticker,
    normalize_depth_update,
)


def test_normalize_agg_trade_preserves_event_time_and_aggressor_side() -> None:
    payload = {
        "e": "aggTrade",
        "E": 1710000000123,
        "s": "BTCUSDT",
        "a": 101,
        "p": "70000.5",
        "q": "0.25",
        "T": 1710000000100,
        "m": True,
    }
    row = normalize_agg_trade(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))
    assert row["event_type"] == "agg_trade"
    assert row["symbol"] == "BTCUSDT"
    assert row["price"] == 70000.5
    assert row["size"] == 0.25
    assert row["aggressor_side"] == "sell"
    assert row["trade_id"] == 101


def test_normalize_book_ticker_has_positive_spread() -> None:
    payload = {
        "u": 400900217,
        "s": "ETHUSDT",
        "b": "3500.10",
        "B": "12.5",
        "a": "3500.20",
        "A": "11.0",
    }
    row = normalize_book_ticker(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))
    assert row["event_type"] == "book_ticker"
    assert row["best_bid"] == 3500.10
    assert row["best_ask"] == 3500.20
    assert row["best_ask"] > row["best_bid"]


def test_normalize_depth_update_keeps_levels_and_update_ids() -> None:
    payload = {
        "e": "depthUpdate",
        "E": 1710000000200,
        "s": "BTCUSDT",
        "U": 157,
        "u": 160,
        "b": [["70000.0", "1.0"], ["69999.5", "0.5"]],
        "a": [["70000.5", "0.75"]],
    }
    row = normalize_depth_update(payload, received_utc=datetime(2026, 5, 26, tzinfo=UTC))
    assert row["event_type"] == "depth_update"
    assert row["first_update_id"] == 157
    assert row["last_update_id"] == 160
    assert row["bids"][0] == [70000.0, 1.0]
    assert row["asks"][0] == [70000.5, 0.75]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_events.py -q
```

Expected: fails because `crypto_research.perps.events` does not exist.

- [ ] **Step 3: Implement minimal event normalizers**

Create `events.py` with pure functions:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _ms_to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)


def _levels(raw: list[list[str]] | None) -> list[list[float]]:
    return [[float(price), float(size)] for price, size in (raw or [])]


def normalize_agg_trade(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "agg_trade",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("T") or payload.get("E")),
        "exchange_event_time": _ms_to_utc(payload.get("E")),
        "received_utc": received_utc,
        "trade_id": int(payload["a"]),
        "price": float(payload["p"]),
        "size": float(payload["q"]),
        "aggressor_side": "sell" if bool(payload.get("m")) else "buy",
    }


def normalize_book_ticker(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "book_ticker",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("E")) or received_utc,
        "received_utc": received_utc,
        "update_id": int(payload["u"]),
        "best_bid": float(payload["b"]),
        "best_bid_size": float(payload["B"]),
        "best_ask": float(payload["a"]),
        "best_ask_size": float(payload["A"]),
    }


def normalize_depth_update(payload: dict[str, Any], *, received_utc: datetime) -> dict[str, Any]:
    return {
        "source": "binance_public",
        "event_type": "depth_update",
        "symbol": str(payload["s"]).upper(),
        "event_time": _ms_to_utc(payload.get("E")) or received_utc,
        "received_utc": received_utc,
        "first_update_id": int(payload["U"]),
        "last_update_id": int(payload["u"]),
        "bids": _levels(payload.get("b")),
        "asks": _levels(payload.get("a")),
    }
```

- [ ] **Step 4: Extend `feeds/binance_ws.py` parser surface**

Import and wrap the new pure parsers without changing the existing `Tick` iterator:

```python
from quant_research_stack.crypto_research.perps.events import (
    normalize_book_ticker,
    normalize_depth_update,
)


def parse_book_ticker_event(payload: dict, *, received_utc: datetime) -> dict:
    return normalize_book_ticker(payload, received_utc=received_utc)


def parse_depth_update_event(payload: dict, *, received_utc: datetime) -> dict:
    return normalize_depth_update(payload, received_utc=received_utc)
```

- [ ] **Step 5: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_events.py tests/test_feeds_binance_ws_parsers.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps src/quant_research_stack/feeds/binance_ws.py tests/crypto_research/perps/test_events.py tests/test_feeds_binance_ws_parsers.py
git commit -m "feat(crypto): normalize Binance perp events"
```

---

### Task 2: Dataset Manifests And Historical Normalization

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/manifest.py`
- Create: `src/quant_research_stack/crypto_research/perps/normalize.py`
- Test: `tests/crypto_research/perps/test_manifest.py`
- Test: `tests/crypto_research/perps/test_normalize.py`

- [ ] **Step 1: Write manifest and normalization tests**

Use tiny local parquet/JSONL fixtures. The tests must not require network:

```python
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.perps.manifest import build_dataset_manifest
from quant_research_stack.crypto_research.perps.normalize import normalize_book_ticker_frame


def test_manifest_records_hash_schema_and_timestamp_semantics(tmp_path: Path) -> None:
    data_path = tmp_path / "book.parquet"
    pl.DataFrame({"symbol": ["BTCUSDT"], "best_bid": [70000.0], "best_ask": [70000.5]}).write_parquet(data_path)
    manifest = build_dataset_manifest(
        dataset_id="unit-book",
        source="unit",
        paths=[data_path],
        symbols=["BTCUSDT"],
        timestamp_semantics="event_time from exchange milliseconds",
        quality_label="unit_test",
    )
    assert manifest["dataset_id"] == "unit-book"
    assert manifest["symbols"] == ["BTCUSDT"]
    assert manifest["files"][0]["sha256"]
    assert "best_bid" in manifest["files"][0]["schema"]


def test_normalize_book_ticker_frame_writes_required_columns() -> None:
    frame = pl.DataFrame(
        {
            "u": [1],
            "s": ["ETHUSDT"],
            "b": ["3500.0"],
            "B": ["10"],
            "a": ["3500.5"],
            "A": ["11"],
        }
    )
    out = normalize_book_ticker_frame(frame, received_utc=datetime(2026, 5, 26, tzinfo=UTC))
    assert {"symbol", "event_time", "best_bid", "best_ask", "relative_spread"}.issubset(out.columns)
    assert out["relative_spread"][0] > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_manifest.py tests/crypto_research/perps/test_normalize.py -q
```

Expected: fails because modules do not exist.

- [ ] **Step 3: Implement manifest builder**

Implement:

- SHA-256 per file.
- Polars schema capture for parquet/csv when readable.
- row count when readable.
- source, dataset id, symbols, timestamp semantics, quality label.

- [ ] **Step 4: Implement bookTicker/trade/depth frame normalizers**

Functions:

- `normalize_book_ticker_frame(frame: pl.DataFrame, received_utc: datetime) -> pl.DataFrame`
- `normalize_agg_trade_frame(frame: pl.DataFrame, received_utc: datetime) -> pl.DataFrame`
- `normalize_depth_frame(frame: pl.DataFrame, received_utc: datetime) -> pl.DataFrame`
- `write_normalized_events(frame, output_path, manifest_path, manifest_payload)`

All output frames must sort by `symbol,event_time` and include `dataset_id` when supplied by caller.

- [ ] **Step 5: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_manifest.py tests/crypto_research/perps/test_normalize.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/manifest.py src/quant_research_stack/crypto_research/perps/normalize.py tests/crypto_research/perps/test_manifest.py tests/crypto_research/perps/test_normalize.py
git commit -m "feat(crypto): add perp dataset manifests and normalization"
```

---

### Task 3: Microstructure Feature And Label Builder

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/features.py`
- Test: `tests/crypto_research/perps/test_features.py`

- [ ] **Step 1: Write timestamp-safety and label tests**

```python
from datetime import datetime, timedelta, UTC

import polars as pl

from quant_research_stack.crypto_research.perps.features import build_l1_features


def test_l1_features_use_only_current_and_past_rows() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "event_time": [t0 + timedelta(seconds=i) for i in range(4)],
            "best_bid": [100.0, 101.0, 102.0, 103.0],
            "best_ask": [100.2, 101.2, 102.2, 103.2],
            "best_bid_size": [10.0, 20.0, 30.0, 40.0],
            "best_ask_size": [15.0, 25.0, 35.0, 45.0],
        }
    )
    out = build_l1_features(frame, horizons=(1, 2), rolling_windows=(2,))
    assert out["mid_price"][0] == 100.1
    assert out["future_mid_return_1"][0] == out["mid_price"][1] / out["mid_price"][0] - 1.0
    assert out["mid_return_1"][0] is None
    assert out["mid_return_1"][1] == out["mid_price"][1] / out["mid_price"][0] - 1.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_features.py -q
```

Expected: fails because `features.py` does not exist.

- [ ] **Step 3: Implement `build_l1_features`**

Function contract:

`build_l1_features(book_ticker: pl.DataFrame, *, horizons: tuple[int, ...] = (1, 5, 15, 60, 300), rolling_windows: tuple[int, ...] = (10, 50, 200)) -> pl.DataFrame`

Required columns:

- `mid_price`
- `spread`
- `relative_spread`
- `l1_imbalance`
- `microprice`
- `microprice_deviation`
- `mid_return_1`
- `realized_vol_<window>`
- `event_count_<window>`
- `future_mid_return_<horizon>`
- `future_best_bid_<horizon>`
- `future_best_ask_<horizon>`
- `future_taker_long_return_<horizon>`
- `future_taker_short_return_<horizon>`

Use Polars `.shift()` over `symbol`; all historical features use positive shifts or rolling windows ending at current row. Labels may use negative shifts and must be named as future labels.

- [ ] **Step 4: Run feature tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_features.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/features.py tests/crypto_research/perps/test_features.py
git commit -m "feat(crypto): build perp microstructure features"
```

---

### Task 4: Event-Driven Bid/Ask Backtest

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/backtest.py`
- Test: `tests/crypto_research/perps/test_backtest.py`

- [ ] **Step 1: Write executable-cost tests**

```python
from datetime import datetime, timedelta, UTC

import polars as pl

from quant_research_stack.crypto_research.perps.backtest import (
    PerpBacktestConfig,
    run_event_backtest,
)


def test_event_backtest_long_uses_entry_ask_and_exit_bid() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "event_time": [t0, t0 + timedelta(seconds=1)],
            "prediction": [1.0, 0.0],
            "best_bid": [100.0, 101.0],
            "best_ask": [100.2, 101.2],
            "future_best_bid_1": [101.0, None],
            "future_best_ask_1": [101.2, None],
            "relative_spread": [0.002, 0.002],
            "best_bid_size": [10.0, 10.0],
            "best_ask_size": [10.0, 10.0],
        }
    )
    result = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, fee_bps=0.0, slippage_bps=0.0))
    trade = result.trades.row(0, named=True)
    assert trade["entry_price"] == 100.2
    assert trade["exit_price"] == 101.0
    assert trade["gross_return"] == 101.0 / 100.2 - 1.0


def test_event_backtest_cost_multiplier_reduces_net_return() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "event_time": [t0, t0 + timedelta(seconds=1)],
            "prediction": [1.0, 0.0],
            "best_bid": [100.0, 101.0],
            "best_ask": [100.2, 101.2],
            "future_best_bid_1": [101.0, None],
            "future_best_ask_1": [101.2, None],
            "relative_spread": [0.002, 0.002],
            "best_bid_size": [10.0, 10.0],
            "best_ask_size": [10.0, 10.0],
        }
    )
    low_cost = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, fee_bps=1.0, cost_multiplier=1.0))
    high_cost = run_event_backtest(frame, config=PerpBacktestConfig(horizon=1, fee_bps=1.0, cost_multiplier=3.0))
    assert high_cost.metrics["net_total_return"] < low_cost.metrics["net_total_return"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_backtest.py -q
```

Expected: fails because `backtest.py` does not exist.

- [ ] **Step 3: Implement backtest config and result types**

Implement dataclasses:

- `PerpBacktestConfig`
- `PerpBacktestResult`

Required config fields:

- `prediction_column`
- `horizon`
- `min_signal_abs`
- `min_edge_to_cost_ratio`
- `max_relative_spread`
- `min_top_of_book_depth`
- `fee_bps`
- `slippage_bps`
- `cost_multiplier`
- `latency_events`
- `invert_signal`

- [ ] **Step 4: Implement `run_event_backtest`**

Behavior:

- Long if prediction > threshold.
- Short if prediction < -threshold.
- Apply `latency_events` by shifting the signal forward per symbol.
- Long entry at current ask, exit at future bid.
- Short entry at current bid, exit at future ask.
- Net return subtracts taker fees and slippage on entry and exit.
- Skip rows with missing future bid/ask.
- Apply spread/depth/liquidity filters before trade creation.
- Metrics include trade count, gross/net total return, event-trade Sharpe, hit rates, profit factor, max drawdown, long/short split.

- [ ] **Step 5: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_backtest.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/backtest.py tests/crypto_research/perps/test_backtest.py
git commit -m "feat(crypto): add perp event-driven backtest"
```

---

### Task 5: Walk-Forward Microstructure Training

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/training.py`
- Test: `tests/crypto_research/perps/test_training.py`

- [ ] **Step 1: Write chronological split test**

```python
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_research_stack.crypto_research.perps.training import (
    PerpWalkForwardConfig,
    train_perp_walk_forward,
)


def test_perp_walk_forward_never_trains_on_or_after_test_rows() -> None:
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    rows = []
    for i in range(160):
        rows.append(
            {
                "symbol": "BTCUSDT",
                "event_time": t0 + timedelta(seconds=i),
                "relative_spread": 0.0001,
                "l1_imbalance": (i % 10 - 5) / 10.0,
                "microprice_deviation": 0.00001 * (i % 7),
                "mid_return_1": 0.0001 * ((i % 3) - 1),
                "realized_vol_10": 0.001,
                "future_mid_return_5": 0.0002 * ((i % 5) - 2),
                "best_bid": 100.0 + i * 0.01,
                "best_ask": 100.1 + i * 0.01,
                "future_best_bid_5": 100.2 + i * 0.01,
                "future_best_ask_5": 100.3 + i * 0.01,
                "best_bid_size": 10.0,
                "best_ask_size": 10.0,
            }
        )
    result = train_perp_walk_forward(
        pl.DataFrame(rows),
        config=PerpWalkForwardConfig(
            target_column="future_mid_return_5",
            min_train_rows=60,
            test_rows=25,
            step_rows=25,
            max_folds=2,
        ),
    )
    assert result.predictions.height > 0
    for fold in result.fold_specs:
        assert fold["train_end_time"] < fold["test_start_time"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_training.py -q
```

Expected: fails because `training.py` does not exist.

- [ ] **Step 3: Implement walk-forward trainer**

Use scikit-learn tabular baselines:

- Ridge with `StandardScaler`
- HistGradientBoostingRegressor
- ensemble mean of available predictions

Required result fields:

- `feature_columns`
- `predictions`
- `fold_specs`
- `fold_metrics`
- `model_metrics`

Fold construction must sort by `event_time`, train strictly before test, and support `embargo_rows`.

- [ ] **Step 4: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_training.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/training.py tests/crypto_research/perps/test_training.py
git commit -m "feat(crypto): train perp microstructure models"
```

---

### Task 6: Strict Validation, Registry, And Reports

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/validation.py`
- Create: `src/quant_research_stack/crypto_research/perps/reports.py`
- Test: `tests/crypto_research/perps/test_validation.py`

- [ ] **Step 1: Write validation gate tests**

```python
import polars as pl

from quant_research_stack.crypto_research.perps.validation import (
    classify_perp_candidate,
    estimate_registry_pbo,
)


def test_candidate_cannot_promote_without_pbo_and_bootstrap() -> None:
    status = classify_perp_candidate(
        {
            "net_daily_sharpe": 3.0,
            "net_total_return": 0.5,
            "pbo_probability": None,
            "bootstrap_ci_lower_95": -1.0,
            "cost_2x_net_total_return": 0.1,
            "delay_1_event_net_total_return": 0.1,
        }
    )
    assert status["promotion_eligible"] is False
    assert "missing_or_high_pbo" in status["blockers"]


def test_registry_pbo_returns_probability_for_variant_matrix() -> None:
    returns = pl.DataFrame(
        {
            "event_index": list(range(40)),
            "strategy_a": [0.001] * 40,
            "strategy_b": [0.001 if i % 2 == 0 else -0.001 for i in range(40)],
            "strategy_c": [-0.001] * 40,
        }
    )
    pbo = estimate_registry_pbo(returns, strategy_columns=["strategy_a", "strategy_b", "strategy_c"], n_partitions=4)
    assert 0.0 <= pbo["pbo_probability"] <= 1.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_validation.py -q
```

Expected: fails because validation module does not exist.

- [ ] **Step 3: Implement validation helpers**

Implement:

- `estimate_registry_pbo`
- `bootstrap_sharpe_payload`
- `deflated_sharpe_payload`
- `concentration_payload`
- `classify_perp_candidate`
- `write_perp_reports`

`classify_perp_candidate` must never return `production_candidate=True` in this free-data slice.

- [ ] **Step 4: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_validation.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/validation.py src/quant_research_stack/crypto_research/perps/reports.py tests/crypto_research/perps/test_validation.py
git commit -m "feat(crypto): validate perp microstructure candidates"
```

---

### Task 7: End-To-End Research CLI

**Files:**
- Create: `scripts/crypto_perp_microstructure_loop.py`
- Test: `tests/crypto_research/perps/test_cli_smoke.py`

- [ ] **Step 1: Write CLI smoke test**

The smoke test creates a tiny synthetic book-ticker parquet and verifies artifacts are written:

```python
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl


def test_crypto_perp_microstructure_loop_smoke(tmp_path: Path, subprocess_env: dict[str, str]) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    t0 = datetime(2026, 5, 26, tzinfo=UTC)
    pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 180,
            "event_time": [t0 + timedelta(seconds=i) for i in range(180)],
            "best_bid": [100.0 + i * 0.01 for i in range(180)],
            "best_ask": [100.1 + i * 0.01 for i in range(180)],
            "best_bid_size": [10.0] * 180,
            "best_ask_size": [11.0] * 180,
        }
    ).write_parquet(raw / "book_ticker.parquet")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/crypto_perp_microstructure_loop.py",
            "--input-book-ticker",
            str(raw / "book_ticker.parquet"),
            "--output-root",
            str(out),
            "--symbols",
            "BTCUSDT",
            "--max-rows",
            "180",
            "--min-train-rows",
            "60",
            "--test-rows",
            "25",
            "--max-folds",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=subprocess_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "strategy_registry.parquet").exists()
    assert (out / "all_backtests.parquet").exists()
    assert (out / "best_candidates_report.md").exists()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_cli_smoke.py -q
```

Expected: fails because CLI does not exist.

- [ ] **Step 3: Implement CLI**

CLI arguments:

- `--input-book-ticker`
- `--input-depth`
- `--output-root`
- `--symbols`
- `--max-rows`
- `--horizons`
- `--min-train-rows`
- `--test-rows`
- `--step-rows`
- `--max-folds`
- `--fee-bps`
- `--slippage-bps`
- `--edge-to-cost-k`
- `--cost-multipliers`
- `--latency-events`

Artifacts:

- `dataset_manifest.json`
- `features.parquet`
- `predictions.parquet`
- `strategy_registry.parquet`
- `all_backtests.parquet`
- `per_trade_audit.parquet`
- `pbo_report.json`
- `pbo_report.md`
- `cost_sensitivity_report.md`
- `holdout_report.md`
- `best_candidates_report.md`
- `failure_report.md` when no candidate passes
- `reproduce.sh`

- [ ] **Step 4: Run smoke test**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_cli_smoke.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/crypto_perp_microstructure_loop.py tests/crypto_research/perps/test_cli_smoke.py
git commit -m "feat(crypto): add perp microstructure research loop"
```

---

### Task 8: Binance Public WebSocket Recorder And Replayer

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/realtime.py`
- Create: `scripts/crypto_perp_record_live.py`
- Create: `scripts/crypto_perp_replay_live.py`
- Test: `tests/crypto_research/perps/test_realtime.py`

- [ ] **Step 1: Write recorder/replayer tests without network**

```python
from pathlib import Path

from quant_research_stack.crypto_research.perps.realtime import (
    replay_raw_events,
    write_raw_event,
)


def test_raw_event_writer_and_replayer_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_raw_event(path, {"stream": "btcusdt@bookTicker", "data": {"s": "BTCUSDT", "u": 1}})
    write_raw_event(path, {"stream": "ethusdt@aggTrade", "data": {"s": "ETHUSDT", "a": 2}})
    events = list(replay_raw_events(path))
    assert events[0]["stream"] == "btcusdt@bookTicker"
    assert events[1]["data"]["s"] == "ETHUSDT"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_realtime.py -q
```

Expected: fails because `realtime.py` does not exist.

- [ ] **Step 3: Implement raw event writer/replayer**

Implement:

- append-only JSONL writer
- replay iterator
- manifest writer with streams, start/end UTC, event count, parser version
- stream-name builder for `aggTrade`, `bookTicker`, and optional `depth@100ms`

- [ ] **Step 4: Implement recorder CLI**

`scripts/crypto_perp_record_live.py` should:

- connect to public Binance streams
- write raw JSONL under `data/live/crypto/binance/<run_id>/raw_events.jsonl`
- stop after `--seconds` or `--max-events`
- write `manifest.json`

Default command:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run python scripts/crypto_perp_record_live.py --symbols BTCUSDT,ETHUSDT --streams aggTrade,bookTicker --seconds 30 --output-root data/live/crypto/binance
```

- [ ] **Step 5: Implement replay CLI**

`scripts/crypto_perp_replay_live.py` should:

- read raw JSONL
- parse known streams through event normalizers
- write normalized parquet and replay manifest

- [ ] **Step 6: Run tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_realtime.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/realtime.py scripts/crypto_perp_record_live.py scripts/crypto_perp_replay_live.py tests/crypto_research/perps/test_realtime.py
git commit -m "feat(crypto): record and replay Binance perp streams"
```

---

### Task 9: Paper-Only Shadow Trading Integration

**Files:**
- Create: `src/quant_research_stack/crypto_research/perps/paper.py`
- Create: `scripts/crypto_perp_shadow_paper.py`
- Test: `tests/crypto_research/perps/test_paper.py`

- [ ] **Step 1: Write shadow paper test**

```python
from datetime import UTC, datetime

from quant_research_stack.crypto_research.perps.paper import (
    PaperFillConfig,
    simulate_taker_fill,
)


def test_simulate_taker_fill_uses_ask_for_buy_and_bid_for_sell() -> None:
    snapshot = {
        "symbol": "BTCUSDT",
        "event_time": datetime(2026, 5, 26, tzinfo=UTC),
        "best_bid": 100.0,
        "best_ask": 100.2,
    }
    buy = simulate_taker_fill(snapshot, side="buy", quantity=1.0, config=PaperFillConfig(fee_bps=1.0))
    sell = simulate_taker_fill(snapshot, side="sell", quantity=1.0, config=PaperFillConfig(fee_bps=1.0))
    assert buy["fill_price"] == 100.2
    assert sell["fill_price"] == 100.0
    assert buy["fee"] > 0.0
    assert sell["fee"] > 0.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_paper.py -q
```

Expected: fails because `paper.py` does not exist.

- [ ] **Step 3: Implement local paper fill simulation**

Implement:

- `PaperFillConfig`
- `simulate_taker_fill`
- `shadow_signal_to_order_intent`
- audit-row builder with signal, order, fill, risk decision, and market snapshot

The implementation must not call Binance live or testnet APIs.

- [ ] **Step 4: Implement shadow paper CLI**

`scripts/crypto_perp_shadow_paper.py` should:

- read replayed normalized book-ticker parquet
- load predictions or a simple threshold signal file
- simulate paper fills
- write `shadow_orders.parquet`, `shadow_fills.parquet`, `shadow_audit.parquet`, and `shadow_report.md`

- [ ] **Step 5: Run paper tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps/test_paper.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/quant_research_stack/crypto_research/perps/paper.py scripts/crypto_perp_shadow_paper.py tests/crypto_research/perps/test_paper.py
git commit -m "feat(crypto): add paper-only perp shadow execution"
```

---

### Task 10: Run Real Benchmark And Verification

**Files:**
- Generated only under `experiments/crypto_perp_microstructure/<run_id>/`
- Generated only under `reports/` if a concise stable report should be committed later by operator decision.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest tests/crypto_research/perps -q
```

Expected: pass.

- [ ] **Step 2: Run lint and typing**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run ruff check src/quant_research_stack/crypto_research/perps src/quant_research_stack/feeds/binance_ws.py scripts/crypto_perp_*.py tests/crypto_research/perps
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run mypy src/quant_research_stack/crypto_research/perps src/quant_research_stack/feeds/binance_ws.py scripts/crypto_perp_microstructure_loop.py scripts/crypto_perp_record_live.py scripts/crypto_perp_replay_live.py scripts/crypto_perp_shadow_paper.py
```

Expected: both pass.

- [ ] **Step 3: Run historical benchmark on available local HF/Binance perp data**

Start with existing local HF order-book data if present:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run python scripts/crypto_perp_microstructure_loop.py \
  --output-root experiments/crypto_perp_microstructure/$(date +%Y%m%d)-btc-eth-perps \
  --symbols BTCUSDT,ETHUSDT \
  --max-rows 250000 \
  --horizons 1,5,15,60,300 \
  --min-train-rows 60000 \
  --test-rows 15000 \
  --step-rows 15000 \
  --max-folds 4 \
  --fee-bps 4.0 \
  --slippage-bps 1.0 \
  --edge-to-cost-k 1.0,1.5,2.0,2.5,3.0,4.0 \
  --cost-multipliers 1.0,2.0,3.0 \
  --latency-events 0,1,5
```

Expected: writes the required backtest and validation artifacts. If no local input is available, stop with a clear dataset-missing report rather than fabricating data.

- [ ] **Step 4: Run short live recorder smoke if network is available**

Run with escalation if sandbox DNS blocks WebSockets:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run python scripts/crypto_perp_record_live.py \
  --symbols BTCUSDT,ETHUSDT \
  --streams aggTrade,bookTicker \
  --seconds 30 \
  --output-root data/live/crypto/binance
```

Expected: writes raw JSONL and manifest. If network is blocked, record the exact failure and keep the code path test-covered by fixture tests.

- [ ] **Step 5: Replay the recorder output**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run python scripts/crypto_perp_replay_live.py \
  --input data/live/crypto/binance/<run_id>/raw_events.jsonl \
  --output-root experiments/crypto_perp_live_replay/<run_id>
```

Expected: writes normalized replay parquet and replay manifest.

- [ ] **Step 6: Run full verification**

Run:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run ruff check src scripts tests
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run mypy src
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run pytest -q
```

Expected: all pass. If network-backed tests fail in sandbox DNS, rerun with approved network access and document both runs.

- [ ] **Step 7: Final commit and push**

If all code is committed task-by-task and verification passes:

```bash
git status --short --branch
git push
```

Expected: branch is pushed to `origin/quant-llm-implementation`.

---

## Self-Review

- Spec coverage: the plan covers perps-only scope, historical normalization, feature/label generation, event-driven bid/ask backtest, model training, PBO/DSR/bootstrap validation, live recorder/replayer practice, and paper-only execution.
- Placeholder scan: no unresolved `TBD`, `TODO`, or placeholder ellipses remain.
- Type consistency: the plan consistently uses `PerpBacktestConfig`, `PerpBacktestResult`, `PerpWalkForwardConfig`, `train_perp_walk_forward`, `build_l1_features`, `normalize_*`, and `run_event_backtest`.
