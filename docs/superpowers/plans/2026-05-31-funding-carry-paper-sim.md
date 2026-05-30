# Funding-Carry Paper-Trading Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the funding-carry delta-neutral strategy (long spot / short perp, BTC+ETH, 1× unlevered) forward through a dedicated `CarryLoop` that reuses the S4 safety components on real public Binance data, `QUANTLAB_STAGE=paper`, kill switch armed — producing a running paper bot + append-only audit log + a live-vs-model reconciliation report.

**Architecture:** A small new module `execution/paper_sim/` with focused units: a public-REST market-data poller, a target-position carry strategy, an 8h funding-accrual engine, a reconciliation report, and a `CarryLoop` runner that wires them through the reused `NullBroker`/`FillModel`/`PositionBook`/`AuditLog`/`KillSwitchWatcher` + `configs/risk.yaml` caps. `S4Loop` is NOT used (it is forecast/single-leg; see spec §1/§5).

**Tech Stack:** Python 3.11, `uv`, `httpx` (already a dep, used by `binance_testnet.py`), `pydantic`, `polars`, `pytest`, `ruff`, `mypy`. Reuses `quant_research_stack.{brokers,execution,feeds}` + `crypto_research.funding.carry` (for the carry identity, referenced in comments).

**Guardrails (every task):** observation-only; `QUANTLAB_STAGE=paper` enforced; kill switch armed; 1× unlevered; no live broker import; no promotion language; **no PR/merge until the operator authorizes.** Do NOT modify `configs/promotion.yaml` or any `brokers/*_live.py`.

---

## Verified reused interfaces (do not re-derive — use these exactly)

- `brokers/order_types.py`: `OrderIntent(client_order_id:str[8..64], symbol:str, side:OrderSide, type:OrderType, quantity:float>0, time_in_force:TimeInForce=day, ...)` (pydantic frozen); `OrderSide.{buy,sell}`; `OrderType.market`; `TimeInForce.ioc`; `Fill(client_order_id, fill_id, symbol, side, price, quantity, timestamp_utc, commission)`; `Position(symbol, quantity, avg_entry_price, market_value, unrealized_pnl)`.
- `brokers/fill_model.py`: `FillModelConfig(commission_bps=1.0, slippage_bps=2.0, half_spread_bps=1.0, fill_latency_ms=50, ...)`; `FillModel(cfg).synthesize(intent, market_iter)->[Fill]` (uses the FIRST event's mid; `_mid(Tick)=tick.price`).
- `brokers/null_broker.py`: `NullBroker(fill_model, starting_cash=100_000.0)`; `.push_market_event(Tick|Bar)`; `await .place_order(intent)->Order` (synthesizes fills from `iter(list(self._market_events))` — FIRST event); `await .positions()->[Position]`; `await .account()->Account`; `.stream_fills()` async gen; `await .close()`. **Internal deque `._market_events` is FIFO and never auto-cleared** — see Task 5 note.
- `feeds/market_types.py`: `Tick(venue:Venue, symbol:str, timestamp_utc:datetime, received_utc:datetime, price:float>0, size:float>=0, side:TickSide, sequence:int|None=None)`; `Venue.binance`; `TickSide.unknown`.
- `execution/position_book.py`: `PositionBook(snapshot_root:Path, stage:str, starting_equity:Decimal)`; `.apply_fill(fill)`; `.daily_realized_pnl:Decimal`; `.peak_equity:Decimal`; `.per_symbol_notional(mid:dict[str,Decimal])->dict[str,float]`; `.gross_exposure(mid:dict[str,Decimal])->float`; `.snapshot()->Path`; `.load_latest_snapshot()`.
- `execution/audit.py`: `AuditLog(root:Path|str, rotation="daily", chmod_after_close=True)`; `.append(event:str, payload:dict)`.
- `execution/kill_switch.py`: `KillSwitchWatcher(flag_path:Path, poll_interval_s:float, audit:AuditLog, on_kill:Callable[[str],Awaitable[None]])`; `.install_signal_handlers()`; `await .run()`; `.stop()`.

---

## File Structure

- Create: `src/quant_research_stack/execution/paper_sim/__init__.py`
- Create: `src/quant_research_stack/execution/paper_sim/config.py` — `PaperSimConfig` (pydantic) + loader.
- Create: `src/quant_research_stack/execution/paper_sim/market_data.py` — `MarketSnapshot` + public-REST poller + pure parsers.
- Create: `src/quant_research_stack/execution/paper_sim/strategy.py` — `FundingCarryStrategy`.
- Create: `src/quant_research_stack/execution/paper_sim/funding_accrual.py` — `FundingAccrual`.
- Create: `src/quant_research_stack/execution/paper_sim/reconciliation.py` — `ReconReport`.
- Create: `src/quant_research_stack/execution/paper_sim/runner.py` — `CarryLoop` + stage guard.
- Create: `scripts/run_funding_carry_paper.py` — CLI.
- Create: `configs/paper_sim.yaml` — symbols, notional, fill bps, cadence.
- Create tests under `tests/execution/paper_sim/`.

---

## Task 1: Config (`PaperSimConfig`)

**Files:** Create `src/quant_research_stack/execution/paper_sim/__init__.py` (empty), `src/quant_research_stack/execution/paper_sim/config.py`, `configs/paper_sim.yaml`; Test `tests/execution/paper_sim/test_config.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_config.py
from __future__ import annotations

from pathlib import Path

from quant_research_stack.execution.paper_sim.config import PaperSimConfig, load_paper_sim_config


def test_defaults_are_one_x_unlevered_and_paper() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"])
    assert cfg.leverage == 1.0
    assert cfg.total_notional_usd > 0
    assert cfg.max_data_gap_seconds == 120


def test_load_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "paper_sim.yaml"
    p.write_text(
        "symbols: [BTCUSDT]\n"
        "total_notional_usd: 20000\n"
        "starting_equity_usd: 100000\n"
        "half_spread_bps: 1.0\n"
        "slippage_bps: 4.0\n"
        "commission_bps: 1.0\n"
        "rebalance_drift_bps: 25.0\n"
        "poll_interval_s: 10.0\n"
    )
    cfg = load_paper_sim_config(p)
    assert cfg.symbols == ["BTCUSDT"]
    assert cfg.slippage_bps == 4.0
    assert cfg.leverage == 1.0  # not in yaml -> default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/execution/paper_sim/test_config.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/config.py
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PaperSimConfig(BaseModel):
    model_config = {"frozen": True}
    symbols: list[str] = Field(min_length=1)
    total_notional_usd: float = Field(default=20_000.0, gt=0.0)
    starting_equity_usd: float = Field(default=100_000.0, gt=0.0)
    leverage: float = Field(default=1.0, gt=0.0, le=1.0)  # 1x only (spec §0)
    half_spread_bps: float = Field(default=1.0, ge=0.0)
    slippage_bps: float = Field(default=4.0, ge=0.0)
    commission_bps: float = Field(default=1.0, ge=0.0)
    rebalance_drift_bps: float = Field(default=25.0, ge=0.0)
    poll_interval_s: float = Field(default=10.0, gt=0.0)
    max_data_gap_seconds: int = Field(default=120, ge=1)


def load_paper_sim_config(path: Path | str) -> PaperSimConfig:
    data = yaml.safe_load(Path(path).read_text())
    return PaperSimConfig.model_validate(data)
```

```yaml
# configs/paper_sim.yaml
symbols: [BTCUSDT, ETHUSDT]
total_notional_usd: 20000      # 1x unlevered, split across legs/assets
starting_equity_usd: 100000
half_spread_bps: 1.0
slippage_bps: 4.0              # ~matches backtest perp/spot taker assumptions
commission_bps: 1.0
rebalance_drift_bps: 25.0      # rebalance a leg when it drifts >25bps from target
poll_interval_s: 10.0
max_data_gap_seconds: 120      # crypto data-gap kill (CLAUDE.md §11)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/execution/paper_sim/test_config.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/__init__.py src/quant_research_stack/execution/paper_sim/config.py configs/paper_sim.yaml tests/execution/paper_sim/test_config.py
git commit -m "feat(paper-sim): PaperSimConfig + configs/paper_sim.yaml (1x unlevered)"
```

---

## Task 2: Market data (`MarketSnapshot` + public-REST poller)

**Files:** Create `src/quant_research_stack/execution/paper_sim/market_data.py`; Test `tests/execution/paper_sim/test_market_data.py`.

Binance public endpoints (no key): spot `GET https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT` → `{"symbol","price"}`; perp `GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT` → `{"markPrice","indexPrice","lastFundingRate","nextFundingTime"}`. Parsers are pure (tested on fixtures); the async `poll` is thin.

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_market_data.py
from __future__ import annotations

from quant_research_stack.execution.paper_sim.market_data import (
    MarketSnapshot,
    parse_premium_index,
    parse_spot_price,
)


def test_parse_spot_price() -> None:
    assert parse_spot_price({"symbol": "BTCUSDT", "price": "65000.50"}) == 65000.50


def test_parse_premium_index() -> None:
    mark, funding, next_ts = parse_premium_index({
        "markPrice": "65010.00", "indexPrice": "65005.00",
        "lastFundingRate": "0.0001", "nextFundingTime": 1717200000000,
    })
    assert mark == 65010.0
    assert funding == 0.0001
    assert next_ts == 1717200000000


def test_snapshot_basis() -> None:
    snap = MarketSnapshot(symbol="BTCUSDT", ts_ms=1, spot_price=100.0,
                          perp_mark=101.0, funding_rate=0.0002, next_funding_ms=8)
    assert abs(snap.basis - 0.01) < 1e-9
```

- [ ] **Step 2: Run** `PYTHONPATH=src uv run pytest tests/execution/paper_sim/test_market_data.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/market_data.py
from __future__ import annotations

from dataclasses import dataclass

import httpx

_SPOT_URL = "https://api.binance.com/api/v3/ticker/price"
_PERP_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    ts_ms: int
    spot_price: float
    perp_mark: float
    funding_rate: float
    next_funding_ms: int

    @property
    def basis(self) -> float:
        return self.perp_mark / self.spot_price - 1.0


def parse_spot_price(payload: dict) -> float:
    return float(payload["price"])


def parse_premium_index(payload: dict) -> tuple[float, float, int]:
    return (
        float(payload["markPrice"]),
        float(payload["lastFundingRate"]),
        int(payload["nextFundingTime"]),
    )


class MarketDataPoller:
    """Polls free Binance public REST for spot price + perp mark/funding. No API key."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def snapshot(self, symbol: str, *, now_ms: int) -> MarketSnapshot:
        spot = await self._client.get(_SPOT_URL, params={"symbol": symbol})
        perp = await self._client.get(_PERP_URL, params={"symbol": symbol})
        spot.raise_for_status()
        perp.raise_for_status()
        spot_price = parse_spot_price(spot.json())
        mark, funding, next_ms = parse_premium_index(perp.json())
        return MarketSnapshot(symbol=symbol, ts_ms=now_ms, spot_price=spot_price,
                              perp_mark=mark, funding_rate=funding, next_funding_ms=next_ms)

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run** the test → PASS (3 tests). Then `ruff check` + `mypy` the file → clean.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/market_data.py tests/execution/paper_sim/test_market_data.py
git commit -m "feat(paper-sim): public-REST market-data poller (spot price + perp mark/funding)"
```

---

## Task 3: Strategy (`FundingCarryStrategy`)

**Files:** Create `src/quant_research_stack/execution/paper_sim/strategy.py`; Test `tests/execution/paper_sim/test_strategy.py`.

The carry identity (from `crypto_research/funding/carry.py`): hold long spot + short perp at equal notional per asset; P&L = funding − basis drift − cost. The live strategy is a target-position rule: per asset, target long-spot qty = `leg_notional/spot_price`, target short-perp qty = `−leg_notional/perp_mark`, where `leg_notional = total_notional * leverage / (2 * n_assets)`. It emits a rebalancing `OrderIntent` for any leg whose notional drift exceeds `rebalance_drift_bps`. Spot symbol = `BTCUSDT`; perp symbol = `BTCUSDTPERP` (distinct keys in the PositionBook).

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_strategy.py
from __future__ import annotations

from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import FundingCarryStrategy, perp_symbol


def _snap(sym: str) -> MarketSnapshot:
    return MarketSnapshot(symbol=sym, ts_ms=0, spot_price=100.0, perp_mark=100.0,
                          funding_rate=0.0001, next_funding_ms=0)


def test_perp_symbol_distinct() -> None:
    assert perp_symbol("BTCUSDT") == "BTCUSDTPERP"


def test_from_flat_opens_both_legs_delta_neutral() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"], total_notional_usd=20000.0)
    strat = FundingCarryStrategy(cfg)
    intents = strat.rebalance_intents(_snap("BTCUSDT"), positions={}, cycle=0)
    # leg_notional = 20000 * 1 / (2 * 2) = 5000 per leg; at price 100 -> 50 units
    by_sym = {i.symbol: i for i in intents}
    assert by_sym["BTCUSDT"].side.value == "buy"
    assert abs(by_sym["BTCUSDT"].quantity - 50.0) < 1e-6
    assert by_sym["BTCUSDTPERP"].side.value == "sell"
    assert abs(by_sym["BTCUSDTPERP"].quantity - 50.0) < 1e-6


def test_no_trade_within_drift_band() -> None:
    cfg = PaperSimConfig(symbols=["BTCUSDT", "ETHUSDT"], total_notional_usd=20000.0,
                         rebalance_drift_bps=50.0)
    strat = FundingCarryStrategy(cfg)
    # already at target (50 long spot, 50 short perp) -> no intents
    pos = {"BTCUSDT": 50.0, "BTCUSDTPERP": -50.0}
    assert strat.rebalance_intents(_snap("BTCUSDT"), positions=pos, cycle=1) == []
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/strategy.py
from __future__ import annotations

from quant_research_stack.brokers.order_types import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot


def perp_symbol(spot_symbol: str) -> str:
    return f"{spot_symbol}PERP"


class FundingCarryStrategy:
    """Delta-neutral target-position rule: long spot + short perp at equal notional.

    1x unlevered (spec §0). Emits rebalancing market intents when a leg drifts beyond
    `rebalance_drift_bps` from its target notional.
    """

    def __init__(self, cfg: PaperSimConfig) -> None:
        self._cfg = cfg
        self._leg_notional = cfg.total_notional_usd * cfg.leverage / (2.0 * len(cfg.symbols))

    def rebalance_intents(self, snap: MarketSnapshot, *, positions: dict[str, float],
                          cycle: int) -> list[OrderIntent]:
        out: list[OrderIntent] = []
        drift = self._cfg.rebalance_drift_bps * 1e-4
        legs = (
            (snap.symbol, snap.spot_price, +self._leg_notional / snap.spot_price),       # long spot
            (perp_symbol(snap.symbol), snap.perp_mark, -self._leg_notional / snap.perp_mark),  # short perp
        )
        for leg_sym, price, target_qty in legs:
            cur = positions.get(leg_sym, 0.0)
            delta = target_qty - cur
            if abs(delta) * price < drift * self._leg_notional:
                continue
            out.append(OrderIntent(
                client_order_id=f"carry-{leg_sym}-{cycle:08d}",
                symbol=leg_sym,
                side=OrderSide.buy if delta > 0 else OrderSide.sell,
                type=OrderType.market,
                quantity=abs(delta),
                time_in_force=TimeInForce.ioc,
            ))
        return out
```

- [ ] **Step 4: Run** → PASS (3 tests). `ruff` + `mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/strategy.py tests/execution/paper_sim/test_strategy.py
git commit -m "feat(paper-sim): FundingCarryStrategy — delta-neutral target-position rule"
```

---

## Task 4: Funding accrual (`FundingAccrual`)

**Files:** Create `src/quant_research_stack/execution/paper_sim/funding_accrual.py`; Test `tests/execution/paper_sim/test_funding_accrual.py`.

A short perp of notional `N = |perp_qty| * perp_mark` receives `rate * N` when `rate > 0`. `funding_pnl = -perp_qty * perp_mark * rate` (positive when short and rate>0). Settles once per 8h boundary (dedup by `next_funding_ms`).

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_funding_accrual.py
from __future__ import annotations

from quant_research_stack.execution.paper_sim.funding_accrual import FundingAccrual
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import perp_symbol


def _snap(rate: float, next_ms: int) -> MarketSnapshot:
    return MarketSnapshot(symbol="BTCUSDT", ts_ms=0, spot_price=100.0, perp_mark=100.0,
                          funding_rate=rate, next_funding_ms=next_ms)


def test_short_receives_positive_funding_once_per_settlement() -> None:
    acc = FundingAccrual()
    pos = {perp_symbol("BTCUSDT"): -50.0}  # short 50 @ 100 -> notional 5000
    pnl = acc.maybe_settle(_snap(0.0001, next_ms=8), positions=pos)
    assert abs(pnl - 0.5) < 1e-9            # 0.0001 * 5000
    # same settlement window -> no double-count
    assert acc.maybe_settle(_snap(0.0001, next_ms=8), positions=pos) == 0.0
    # new settlement -> accrues again
    assert abs(acc.maybe_settle(_snap(0.0001, next_ms=16), positions=pos) - 0.5) < 1e-9


def test_no_funding_when_flat() -> None:
    acc = FundingAccrual()
    assert acc.maybe_settle(_snap(0.0001, next_ms=8), positions={}) == 0.0
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/funding_accrual.py
from __future__ import annotations

from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import perp_symbol


class FundingAccrual:
    """Credits/debits the short-perp leg at each 8h settlement from the REAL rate.

    Short receives funding when rate > 0: pnl = -perp_qty * perp_mark * rate.
    Dedups by the settlement boundary (`next_funding_ms`) so it accrues once per window.
    """

    def __init__(self) -> None:
        self._settled: set[int] = set()

    def maybe_settle(self, snap: MarketSnapshot, *, positions: dict[str, float]) -> float:
        if snap.next_funding_ms in self._settled:
            return 0.0
        self._settled.add(snap.next_funding_ms)
        perp_qty = positions.get(perp_symbol(snap.symbol), 0.0)
        return -perp_qty * snap.perp_mark * snap.funding_rate
```

- [ ] **Step 4: Run** → PASS (2 tests). `ruff` + `mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/funding_accrual.py tests/execution/paper_sim/test_funding_accrual.py
git commit -m "feat(paper-sim): FundingAccrual — 8h funding P&L on the short leg (dedup per settlement)"
```

---

## Task 5: Runner (`CarryLoop`) + stage guard

**Files:** Create `src/quant_research_stack/execution/paper_sim/runner.py`; Test `tests/execution/paper_sim/test_runner.py`.

`CarryLoop` owns the control loop. Key integration facts (verified): `NullBroker.place_order` synthesizes the fill from the FIRST event in its `_market_events` deque, which is never auto-cleared — so **before each `place_order`, clear and push the current leg's `Tick` at the leg price** so the fill prices correctly. Funding P&L is tracked in the loop's own ledger (PositionBook only tracks fill-realized PnL). Equity = `starting + book.daily_realized_pnl + funding_pnl`. Stage guard: refuse unless `QUANTLAB_STAGE == "paper"`. The loop accepts an injected `snapshot_source` (async callable `(symbol, now_ms) -> MarketSnapshot`) so tests drive it deterministically without network.

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_runner.py
from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.runner import CarryLoop, ensure_paper_stage


def test_ensure_paper_stage_rejects_non_paper(monkeypatch) -> None:
    monkeypatch.setenv("QUANTLAB_STAGE", "live")
    with pytest.raises(SystemExit):
        ensure_paper_stage()
    monkeypatch.setenv("QUANTLAB_STAGE", "paper")
    ensure_paper_stage()  # no raise


@pytest.mark.asyncio
async def test_loop_opens_delta_neutral_and_accrues_funding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTLAB_STAGE", "paper")
    cfg = PaperSimConfig(symbols=["BTCUSDT"], total_notional_usd=10000.0,
                         rebalance_drift_bps=10.0, poll_interval_s=0.0)

    snaps = iter([
        MarketSnapshot("BTCUSDT", 1, 100.0, 100.0, 0.0001, next_funding_ms=8),
        MarketSnapshot("BTCUSDT", 2, 100.0, 100.0, 0.0001, next_funding_ms=16),
    ])

    async def source(symbol: str, now_ms: int) -> MarketSnapshot:
        return next(snaps)

    loop = CarryLoop(cfg, audit_root=tmp_path / "audit", snapshot_root=tmp_path / "book",
                     snapshot_source=source)
    await loop.run(max_cycles=2)

    # leg_notional = 10000/(2*1) = 5000 @100 -> 50 units long spot, 50 short perp
    pos = loop.positions()
    assert abs(pos["BTCUSDT"] - 50.0) < 1.0
    assert abs(pos["BTCUSDTPERP"] + 50.0) < 1.0
    # funding accrued at least once (short 50@100 * 0.0001 = 0.5 per settlement)
    assert loop.funding_pnl() > 0.0
    # audit file written
    assert any((tmp_path / "audit").glob("*.jsonl"))
```

Add `asyncio_mode = "auto"` is already set if `pytest-asyncio` is configured; if the test errors on the async marker, the implementer should confirm `pytest-asyncio` is available (it is used elsewhere in `tests/`) and mark accordingly.

- [ ] **Step 2: Run** `PYTHONPATH=src uv run pytest tests/execution/paper_sim/test_runner.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/runner.py
from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.fill_model import FillModel, FillModelConfig
from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.paper_sim.config import PaperSimConfig
from quant_research_stack.execution.paper_sim.funding_accrual import FundingAccrual
from quant_research_stack.execution.paper_sim.market_data import MarketSnapshot
from quant_research_stack.execution.paper_sim.strategy import FundingCarryStrategy, perp_symbol
from quant_research_stack.feeds.market_types import Tick, TickSide, Venue

SnapshotSource = Callable[[str, int], Awaitable[MarketSnapshot]]


def ensure_paper_stage() -> None:
    """Refuse to run unless QUANTLAB_STAGE=paper (observation-only; no live)."""
    stage = os.environ.get("QUANTLAB_STAGE", "paper")
    if stage != "paper":
        sys.stderr.write(f"REFUSING: QUANTLAB_STAGE={stage!r}, paper-sim runs only at 'paper'\n")
        raise SystemExit(2)


def _leg_price(snap: MarketSnapshot, leg_symbol: str) -> float:
    return snap.perp_mark if leg_symbol.endswith("PERP") else snap.spot_price


class CarryLoop:
    """Observation-only funding-carry paper sim. Reuses NullBroker/FillModel + AuditLog.

    NOT validation or promotion; the strategy is DO_NOT_ADVANCE (spec §0). 1x unlevered.
    """

    def __init__(self, cfg: PaperSimConfig, *, audit_root: Path, snapshot_root: Path,
                 snapshot_source: SnapshotSource) -> None:
        ensure_paper_stage()
        self._cfg = cfg
        self._source = snapshot_source
        self._audit = AuditLog(audit_root)
        self._audit.append("paper_sim_start",
                            {"observation_only": True, "strategy": "funding_carry",
                             "verdict": "DO_NOT_ADVANCE", "leverage": cfg.leverage,
                             "symbols": cfg.symbols})
        self._broker = NullBroker(
            fill_model=FillModel(FillModelConfig(
                commission_bps=cfg.commission_bps,
                slippage_bps=cfg.slippage_bps,
                half_spread_bps=cfg.half_spread_bps)),
            starting_cash=cfg.starting_equity_usd)
        self._strategy = FundingCarryStrategy(cfg)
        self._accrual = FundingAccrual()
        self._positions: dict[str, float] = {}
        self._funding_pnl = 0.0
        snapshot_root.mkdir(parents=True, exist_ok=True)

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def funding_pnl(self) -> float:
        return self._funding_pnl

    async def _place(self, intent, price: float) -> None:
        # NullBroker fills from the FIRST market event; reset deque so this leg prices right.
        self._broker._market_events.clear()  # noqa: SLF001 - sim-only price injection
        self._broker.push_market_event(Tick(
            venue=Venue.binance, symbol=intent.symbol,
            timestamp_utc=datetime.now(UTC), received_utc=datetime.now(UTC),
            price=price, size=0.0, side=TickSide.unknown))
        order = await self._broker.place_order(intent)
        sign = 1.0 if intent.side.value == "buy" else -1.0
        self._positions[intent.symbol] = self._positions.get(intent.symbol, 0.0) + sign * intent.quantity
        self._audit.append("trade_placed", {
            "order_id": order.client_order_id, "symbol": intent.symbol,
            "side": intent.side.value, "qty": intent.quantity, "price": price})

    async def run(self, *, max_cycles: int | None = None) -> None:
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                for symbol in self._cfg.symbols:
                    snap = await self._source(symbol, cycle)
                    for intent in self._strategy.rebalance_intents(
                            snap, positions=self._positions, cycle=cycle):
                        await self._place(intent, _leg_price(snap, intent.symbol))
                    fpnl = self._accrual.maybe_settle(snap, positions=self._positions)
                    if fpnl != 0.0:
                        self._funding_pnl += fpnl
                        self._audit.append("funding_settled", {
                            "symbol": symbol, "rate": snap.funding_rate,
                            "perp_mark": snap.perp_mark, "funding_pnl": fpnl,
                            "basis": snap.basis})
                cycle += 1
        finally:
            self._audit.append("paper_sim_stop",
                               {"cycles": cycle, "funding_pnl": self._funding_pnl,
                                "positions": self._positions})
            await self._broker.close()
```

(Equity/DD/feed-gap kill wiring and the real REST source are added in Task 7; the loop here is the testable core.)

- [ ] **Step 4: Run** → PASS (2 tests). `ruff` + `mypy` clean (the `# noqa: SLF001` documents the deliberate private access).

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/runner.py tests/execution/paper_sim/test_runner.py
git commit -m "feat(paper-sim): CarryLoop control loop + paper-stage guard (observation-only)"
```

---

## Task 6: Reconciliation report (`ReconReport`)

**Files:** Create `src/quant_research_stack/execution/paper_sim/reconciliation.py`; Test `tests/execution/paper_sim/test_reconciliation.py`.

Aggregates the loop's outcome into a live-vs-model report (observation-only): cumulative funding collected, basis stats (mean/max) vs the backtest daily-close basis (~0 / <0.1% p95), realized funding vs the backtest mean, equity delta, number of rebalances. Pure aggregation + markdown render.

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/paper_sim/test_reconciliation.py
from __future__ import annotations

from quant_research_stack.execution.paper_sim.reconciliation import ReconReport


def test_report_renders_observation_only_and_numbers() -> None:
    rep = ReconReport(
        cycles=10, n_rebalances=4, funding_pnl=1.25,
        basis_samples=[0.0005, -0.0003, 0.0011], equity_start=100000.0, equity_end=100001.25)
    md = rep.render()
    assert "observation-only" in md.lower()
    assert "1.25" in md
    assert "DO_NOT_ADVANCE" in md
    assert rep.basis_mean_pct() == __import__("pytest").approx(
        (0.0005 - 0.0003 + 0.0011) / 3 * 100, rel=1e-6)
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/execution/paper_sim/reconciliation.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconReport:
    cycles: int
    n_rebalances: int
    funding_pnl: float
    basis_samples: list[float]
    equity_start: float
    equity_end: float

    def basis_mean_pct(self) -> float:
        return (sum(self.basis_samples) / len(self.basis_samples) * 100.0) if self.basis_samples else 0.0

    def basis_max_abs_pct(self) -> float:
        return (max(abs(b) for b in self.basis_samples) * 100.0) if self.basis_samples else 0.0

    def render(self) -> str:
        eq_delta = self.equity_end - self.equity_start
        return "\n".join([
            "# Funding-Carry Paper Sim — Live-vs-Model Reconciliation",
            "",
            "**Observation-only.** Strategy verdict: **DO_NOT_ADVANCE**. Not validation, "
            "not a step toward live (CLAUDE.md §7, §11).",
            "",
            f"- cycles: {self.cycles}  |  rebalances: {self.n_rebalances}",
            f"- funding P&L collected: {self.funding_pnl:.2f} USD",
            f"- equity: {self.equity_start:.2f} -> {self.equity_end:.2f} "
            f"(delta {eq_delta:+.2f})",
            f"- live basis mean: {self.basis_mean_pct():.4f}%  |  "
            f"max |basis|: {self.basis_max_abs_pct():.4f}%  "
            f"(backtest daily-close model: ~0% mean, <0.1% p95)",
            "",
            "Funding/basis are REAL (public mainnet); fills are simulated (FillModel). "
            "Compare live funding/basis above against the backtest cost-and-tail models in "
            "`reports/signal_research/funding_carry_v1/funding_carry_realism_results.md`.",
        ])
```

- [ ] **Step 4: Run** → PASS. `ruff` + `mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/paper_sim/reconciliation.py tests/execution/paper_sim/test_reconciliation.py
git commit -m "feat(paper-sim): ReconReport — live-vs-model reconciliation (observation-only)"
```

---

## Task 7: CLI + real REST source + kill switch wiring

**Files:** Create `scripts/run_funding_carry_paper.py`; Modify `src/quant_research_stack/execution/paper_sim/runner.py` (add the real REST snapshot source factory + kill-flag handling); Test `tests/execution/paper_sim/test_cli_smoke.py`.

Wire `MarketDataPoller` as the default `snapshot_source`; arm `KillSwitchWatcher(flag_path=repo_root/"KILL_TRADING", ...)`; CLI flags `--max-cycles`, `--duration-seconds`, `--config`. The CLI calls `ensure_paper_stage()` first.

- [ ] **Step 1: Write the failing smoke test** (replay/deterministic — no network)

```python
# tests/execution/paper_sim/test_cli_smoke.py
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_funding_carry_paper", Path(__file__).resolve().parents[3] / "scripts" / "run_funding_carry_paper.py")
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_cli_exposes_main_and_build_loop() -> None:
    assert hasattr(cli, "main")
    assert hasattr(cli, "build_rest_source")
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** the CLI

```python
# scripts/run_funding_carry_paper.py
"""Funding-carry paper-trading simulation CLI (observation-only, QUANTLAB_STAGE=paper).

Runs the delta-neutral carry through the CarryLoop on REAL public Binance data with
simulated fills. NOT validation/promotion (strategy is DO_NOT_ADVANCE). No live broker.

Run: QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/run_funding_carry_paper.py \
       --config configs/paper_sim.yaml --max-cycles 5
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from quant_research_stack.execution.paper_sim.config import load_paper_sim_config
from quant_research_stack.execution.paper_sim.market_data import MarketDataPoller, MarketSnapshot
from quant_research_stack.execution.paper_sim.runner import CarryLoop, SnapshotSource, ensure_paper_stage


def build_rest_source(poller: MarketDataPoller) -> SnapshotSource:
    async def source(symbol: str, now_ms: int) -> MarketSnapshot:
        return await poller.snapshot(symbol, now_ms=now_ms)
    return source


async def _run(args: argparse.Namespace) -> None:
    ensure_paper_stage()
    cfg = load_paper_sim_config(args.config)
    poller = MarketDataPoller()
    loop = CarryLoop(cfg, audit_root=Path("logs/audit/paper_sim"),
                     snapshot_root=Path("logs/paper_sim_book"),
                     snapshot_source=build_rest_source(poller))
    try:
        await loop.run(max_cycles=args.max_cycles)
    finally:
        await poller.close()
    print(f"done. funding_pnl={loop.funding_pnl():.2f} positions={loop.positions()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Funding-carry paper sim (observation-only)")
    ap.add_argument("--config", default="configs/paper_sim.yaml")
    ap.add_argument("--max-cycles", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** the smoke test → PASS. `ruff check scripts/run_funding_carry_paper.py` + `mypy` clean. (Do NOT run against the live network in CI; the smoke test only imports.)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_funding_carry_paper.py tests/execution/paper_sim/test_cli_smoke.py
git commit -m "feat(paper-sim): CLI + REST snapshot source (observation-only, paper-stage guarded)"
```

---

## Task 8: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full gate**

```bash
PYTHONPATH=src uv run pytest tests/execution/paper_sim/ -q
PYTHONPATH=src uv run ruff check src/quant_research_stack/execution/paper_sim/ scripts/run_funding_carry_paper.py tests/execution/paper_sim/
PYTHONPATH=src uv run mypy src/quant_research_stack/execution/paper_sim/ scripts/run_funding_carry_paper.py
```
Expected: all paper_sim tests pass; ruff + mypy clean.

- [ ] **Step 2: Bounded live smoke (optional, network)** — confirm it actually runs end-to-end:

```bash
QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/run_funding_carry_paper.py --max-cycles 2
```
Expected: prints `done. funding_pnl=... positions={...}`; writes `logs/audit/paper_sim/*.jsonl`. If the network/endpoints are unavailable, note it — the deterministic tests are the gate, not this.

- [ ] **Step 3: Confirm guardrails** — `QUANTLAB_STAGE=live PYTHONPATH=src uv run python scripts/run_funding_carry_paper.py --max-cycles 1` exits non-zero (refuses). No `brokers/*_live.py` imported anywhere in `paper_sim/` (`grep -rn "_live" src/quant_research_stack/execution/paper_sim/` → empty).

- [ ] **Step 4: Final commit (if pending) — DO NOT open a PR or merge**

```bash
git add -A && git commit -m "chore(paper-sim): finalize funding-carry paper simulation" || echo "nothing to commit"
```
**Do NOT run `gh pr create`, do NOT push, do NOT merge. The operator opens the PR/merge when they say so.**

---

## Self-Review (completed by plan author)

- **Spec coverage:** §1 architecture → Tasks 5/7 (CarryLoop wiring); §2 market data → Task 2; §3 strategy → Task 3; §4 funding accrual → Task 4; §5 execution wiring (components, no S4Loop, no governor) → Task 5; §6 risk/safety (paper guard, kill flag) → Tasks 5/7; §7 observation metrics → Task 6; §8 run modes → Tasks 5 (`max_cycles`) / 7 (CLI); §9 file structure → all tasks; §10 testing → each task's tests + Task 8; §11 honesty guardrails → Task 5 audit banner + Task 6 report + Task 8 guardrail check. No gaps.
- **Placeholder scan:** every code step has complete code; no TBD/TODO.
- **Type consistency:** `MarketSnapshot` fields, `perp_symbol()`, `FundingCarryStrategy.rebalance_intents(snap, positions, cycle)`, `FundingAccrual.maybe_settle(snap, positions)`, `CarryLoop(cfg, audit_root, snapshot_root, snapshot_source)` + `.run(max_cycles=)` / `.positions()` / `.funding_pnl()`, `ensure_paper_stage()`, `SnapshotSource` type alias, `build_rest_source()` — all used consistently across tasks and tests.
- **Known integration risk (flagged for the implementer):** `NullBroker._market_events` is reset before each `place_order` (Task 5, `# noqa: SLF001`) because its fill synthesis reads the FIRST event; if a future NullBroker change alters this, the loop's `_place` must adapt. `pytest-asyncio` is assumed available (used elsewhere in `tests/`); confirm the async marker works in Task 5.
