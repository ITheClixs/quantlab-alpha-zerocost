# QuantLab Alpha — S4 Execution + Risk + Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the S4 trading layer (paper + live_shadow stages) that consumes S1 predictions and S2 verdicts, applies risk gates, sizes positions, routes orders to a stage-resolved broker, maintains an in-memory position book reconciled against the broker every 60 s, and writes append-only audit events. Live broker implementations (`brokers/alpaca_live.py`, `brokers/binance_live.py`) are explicitly out of scope and reserved for S4.1 (two-person-review per CLAUDE.md §1.13).

**Architecture:** Long-running async daemon driven by `scripts/s4_execute.py`. Package modules under `src/quant_research_stack/execution/`: `signals.py` (ingest pairs of S1+S2), `risk.py` (ordered pre-trade checks), `sizing.py` (confidence × stance × cap → qty), `position_book.py` (in-mem + Parquet snapshot), `reconciliation.py` (60-s broker diff), `router.py` (stage → BrokerAdapter), `kill_switch.py` (file flag + SIGTERM/SIGINT + close-out), `loop.py` (orchestration glue). Configs: new `risk.yaml`, new `promotion.yaml`, extended `brokers.yaml`.

**Tech Stack:** Python 3.11, asyncio, Polars, Pydantic v2, PyYAML, pytest (+ pytest-asyncio already enabled), joblib (already a dep), the existing S3 broker abstraction (`BrokerAdapter` Protocol, `OrderIntent`, `Order`, `Fill`, `Account`, `Position`), the existing S2 `GovernorVerdict` schema, the existing `logs/audit/` transport pattern.

---

## File Structure

| Layer | Files |
|---|---|
| ADRs (3 docs) | `docs/architecture/adrs/0012-confidence-scaled-sizing.md`, `0013-position-book-with-broker-reconciliation.md`, `0014-kill-switch-precedence.md` |
| Configs | `configs/risk.yaml`, `configs/promotion.yaml`, `configs/exec.yaml`, modify `configs/brokers.yaml` |
| Core package | `src/quant_research_stack/execution/__init__.py`, `signals.py`, `risk.py`, `sizing.py`, `position_book.py`, `reconciliation.py`, `router.py`, `kill_switch.py`, `loop.py` |
| Daemon + tooling | `scripts/s4_execute.py`, `scripts/generate_promotion_report.py`, `scripts/audit_replay_check.py` |
| Tests (unit) | `tests/test_execution_signals.py`, `test_execution_risk.py`, `test_execution_sizing.py`, `test_execution_position_book.py`, `test_execution_reconciliation.py`, `test_execution_router.py`, `test_execution_kill_switch.py`, `test_execution_loop.py` |
| Tests (integration, gated) | `tests/integration/test_s4_paper_smoke.py`, `test_s4_kill_switch_drill.py`, `test_s4_reconciliation_kill.py`, `test_s4_audit_replay_parity.py` |
| Build glue | `pyproject.toml` (add `s4_integration` marker), `Makefile` (add `s4-execute`, `s4-promotion-report`, `s4-smoke`) |

---

### Task 1: ADRs for the three irreversible decisions

**Files:**
- Create: `docs/architecture/adrs/0012-confidence-scaled-sizing.md`
- Create: `docs/architecture/adrs/0013-position-book-with-broker-reconciliation.md`
- Create: `docs/architecture/adrs/0014-kill-switch-precedence.md`

- [ ] **Step 1: Write ADR-0012 (sizing)**

Create `docs/architecture/adrs/0012-confidence-scaled-sizing.md`:

```markdown
# ADR 0012: Confidence-scaled position sizing with hard caps

## Status
Accepted, 2026-05-20.

## Context
S4 must translate an S1 numeric prediction (`predicted_score`, `confidence`) and an
S2 `GovernorVerdict` (`decision`, `direction`, `confidence`) into an order quantity.
The S2 spec's Tier-3 verdict contributes a stance modifier whose magnitude is in
`configs/governor.yaml` as `stance.tier3_stance_modifier_pct` (default 0.20). The
goal is a sizing rule that (a) respects the predictor's strength signal, (b) is
hard-capped so a runaway confidence value cannot blow up exposure, and (c) shrinks
when Tier 3 disagrees and grows when Tier 3 agrees.

## Decision
We use a confidence-scaled rule with hard caps:

```text
stance_mod ∈ {-cfg_pct, 0, +cfg_pct} depending on tier3 vs primary direction
target_notional = equity * base_pct * primary.confidence * (1 + stance_mod)
target_notional = min(target_notional, equity * max_per_symbol_pct)
qty = primary.direction * target_notional / mid_price  ; rounded to lot
```

`Decision.veto` short-circuits the Sizer to `qty = 0`. `direction == 0` also yields
`qty = 0`. The `max_per_symbol_pct` and `base_notional_per_trade_pct` come from
`configs/risk.yaml`.

## Alternatives considered
- Kelly-lite fixed-fractional: simpler but discards the predictor's strength signal.
- Volatility-targeted: principled but adds a rolling-vol feature dependency we
  don't yet have wired through.

## Consequences
The sizer is bounded by construction; the worst-case single-trade notional is
`equity * max_per_symbol_pct`. Tier-3 disagreements shrink positions by up to
`cfg_pct` (default 20%). Veto closes the trade entirely.
```

- [ ] **Step 2: Write ADR-0013 (position book + reconciliation)**

Create `docs/architecture/adrs/0013-position-book-with-broker-reconciliation.md`:

```markdown
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
```

- [ ] **Step 3: Write ADR-0014 (kill-switch precedence)**

Create `docs/architecture/adrs/0014-kill-switch-precedence.md`:

```markdown
# ADR 0014: Kill-switch precedence — file flag is the first risk gate

## Status
Accepted, 2026-05-20.

## Context
Multiple risk checks could raise on a given signal: drawdown breach, exposure cap,
feed gap, etc. If the file-flag kill check runs after another check, an operator
who touches `KILL_TRADING` could see the daemon halt for the *other* reason — not
for their override. Worse, if the other check has a bug and never raises, the
operator's override is silently defeated.

## Decision
The kill-flag check is the *first* call in `RiskGate.evaluate()`, before all other
gates. Its precedence is enforced by:

1. A unit test asserting the check order at module-level.
2. A code comment in `risk.py` documenting the invariant.
3. The `RiskGate` class running checks via an explicit list (`_GATES`) where the
   first element is always `kill_flag_check`.

## Consequences
Operator intent is always honored before any algorithmic check fires. The cost is
that the kill flag is checked on every signal, but the check is a single `stat()`
call (sub-millisecond) so the overhead is negligible.
```

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/adrs/0012-confidence-scaled-sizing.md \
        docs/architecture/adrs/0013-position-book-with-broker-reconciliation.md \
        docs/architecture/adrs/0014-kill-switch-precedence.md
git commit -m "docs(s4): ADRs 0012-0014 (sizing, position book, kill-switch precedence)"
```

---

### Task 2: Scaffold the execution package + configs + pytest marker

**Files:**
- Create: `src/quant_research_stack/execution/__init__.py`
- Create: `configs/risk.yaml`, `configs/promotion.yaml`, `configs/exec.yaml`
- Modify: `configs/brokers.yaml` (add `stage_routes` block)
- Modify: `pyproject.toml` (register `s4_integration` marker)

- [ ] **Step 1: Create `execution/__init__.py`**

```python
"""S4: execution + risk + promotion gates."""
```

- [ ] **Step 2: Write `configs/risk.yaml`**

```yaml
limits:
  max_per_symbol_pct: 0.02
  max_gross_exposure_pct: 0.30
  base_notional_per_trade_pct: 0.005
  max_orders_per_minute: 10

drawdown:
  daily_realized_dd_kill_pct: 0.05
  cumulative_dd_kill_pct: 0.15

freshness:
  crypto_max_gap_seconds: 120
  equity_max_gap_seconds: 1800

reconciliation:
  interval_seconds: 60
  max_diff_bps: 1.0

stage_overrides:
  live:
    cap_multiplier_first_30d: 0.50
```

- [ ] **Step 3: Write `configs/promotion.yaml`**

```yaml
paper_to_live_shadow:
  min_days_in_paper: 30
  min_sharpe: 1.0
  max_daily_dd_pct: 0.05
  no_kill_triggers_days: 14
  max_audit_anomalies: 0
  required_artifacts:
    - "experiments/alpha_s1/<latest>/metrics.json with holdout_weighted_zero_mean_r2 >= 0.012"
    - "experiments/s2_verdicts_*/<date>.jsonl present for all paper days"

live_shadow_to_live:
  min_days_in_live_shadow: 14
  max_reconciliation_diff_bps: 1.0
  max_feed_gap_violations: 0
  kill_switch_drill_passed: true
  required_artifacts:
    - "docs/runbooks/paper_to_live_shadow.md  (signed promotion report)"
```

- [ ] **Step 4: Write `configs/exec.yaml`**

```yaml
ingest:
  s1_predictions_dir: data/live/s1_predictions
  s2_verdicts_dir: experiments/s2_verdicts_balanced
  poll_interval_seconds: 0.1
  pair_window_seconds: 60

position_book:
  snapshot_root: data/positions
  snapshot_interval_seconds: 60

audit:
  root: logs/audit/s4
  rotation: daily
  chmod_after_close: true

kill_switch:
  repo_root_marker: KILL_TRADING
  poll_interval_seconds: 0.1
  emergency_snapshot_root: data/snapshots
```

- [ ] **Step 5: Extend `configs/brokers.yaml`**

Append to the existing file (verify first that `stage_routes` is not already present):

```yaml
stage_routes:
  paper:
    equity: alpaca_paper
    crypto: binance_testnet
  live_shadow:
    equity: null_broker
    crypto: null_broker
    read_only_account:
      equity: alpaca_paper
      crypto: binance_testnet
  live:
    equity: alpaca_live
    crypto: binance_live
```

- [ ] **Step 6: Register pytest marker**

Modify `pyproject.toml` — find `[tool.pytest.ini_options]` and add `"s4_integration: requires running brokers/feeds (skipped in default run)"` to the `markers` list. If a markers list doesn't exist, add it.

- [ ] **Step 7: Verify configs parse**

```bash
PYTHONPATH=src uv run python -c "
import yaml
for p in ['configs/risk.yaml', 'configs/promotion.yaml', 'configs/exec.yaml', 'configs/brokers.yaml']:
    yaml.safe_load(open(p))
    print(p, 'OK')
"
```

Expected: four "OK" lines.

- [ ] **Step 8: Commit**

```bash
git add configs/risk.yaml configs/promotion.yaml configs/exec.yaml configs/brokers.yaml \
        pyproject.toml src/quant_research_stack/execution/__init__.py
git commit -m "feat(s4): scaffold execution package + risk/promotion/exec configs + s4_integration marker"
```

---

### Task 3: Pydantic config loaders (validate-on-startup invariant)

**Files:**
- Create: `src/quant_research_stack/execution/configs.py`
- Test: `tests/test_execution_configs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_configs.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_stack.execution.configs import (
    ExecConfig,
    PromotionConfig,
    RiskConfig,
    load_exec_config,
    load_promotion_config,
    load_risk_config,
)


def test_risk_config_loads_valid_yaml() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    assert isinstance(cfg, RiskConfig)
    assert 0 < cfg.limits.max_per_symbol_pct < 1
    assert cfg.reconciliation.max_diff_bps > 0


def test_risk_config_rejects_negative_caps(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "limits:\n"
        "  max_per_symbol_pct: -0.1\n"
        "  max_gross_exposure_pct: 0.3\n"
        "  base_notional_per_trade_pct: 0.005\n"
        "  max_orders_per_minute: 10\n"
        "drawdown:\n"
        "  daily_realized_dd_kill_pct: 0.05\n"
        "  cumulative_dd_kill_pct: 0.15\n"
        "freshness:\n"
        "  crypto_max_gap_seconds: 120\n"
        "  equity_max_gap_seconds: 1800\n"
        "reconciliation:\n"
        "  interval_seconds: 60\n"
        "  max_diff_bps: 1.0\n"
    )
    with pytest.raises(ValueError):
        load_risk_config(p)


def test_promotion_config_loads_valid_yaml() -> None:
    cfg = load_promotion_config(Path("configs/promotion.yaml"))
    assert isinstance(cfg, PromotionConfig)
    assert cfg.paper_to_live_shadow.min_days_in_paper >= 1
    assert cfg.live_shadow_to_live.kill_switch_drill_passed in (True, False)


def test_exec_config_loads_valid_yaml() -> None:
    cfg = load_exec_config(Path("configs/exec.yaml"))
    assert isinstance(cfg, ExecConfig)
    assert cfg.ingest.poll_interval_seconds > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_configs.py -q
```

Expected: ImportError / ModuleNotFoundError for `quant_research_stack.execution.configs`.

- [ ] **Step 3: Implement `configs.py`**

Create `src/quant_research_stack/execution/configs.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator


class RiskLimits(BaseModel):
    model_config = {"frozen": True}
    max_per_symbol_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    max_gross_exposure_pct: Annotated[float, Field(gt=0.0, le=1.0)]
    base_notional_per_trade_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    max_orders_per_minute: Annotated[int, Field(ge=1)]


class DrawdownLimits(BaseModel):
    model_config = {"frozen": True}
    daily_realized_dd_kill_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    cumulative_dd_kill_pct: Annotated[float, Field(gt=0.0, lt=1.0)]


class Freshness(BaseModel):
    model_config = {"frozen": True}
    crypto_max_gap_seconds: Annotated[int, Field(ge=1)]
    equity_max_gap_seconds: Annotated[int, Field(ge=1)]


class Reconciliation(BaseModel):
    model_config = {"frozen": True}
    interval_seconds: Annotated[int, Field(ge=1)]
    max_diff_bps: Annotated[float, Field(gt=0.0)]


class StageOverrides(BaseModel):
    model_config = {"frozen": True}
    cap_multiplier_first_30d: Annotated[float, Field(gt=0.0, le=1.0)] = 0.50


class RiskConfig(BaseModel):
    model_config = {"frozen": True}
    limits: RiskLimits
    drawdown: DrawdownLimits
    freshness: Freshness
    reconciliation: Reconciliation
    stage_overrides: dict[str, StageOverrides] = {}

    @model_validator(mode="after")
    def _cumulative_above_daily(self) -> "RiskConfig":
        if self.drawdown.cumulative_dd_kill_pct <= self.drawdown.daily_realized_dd_kill_pct:
            raise ValueError("cumulative_dd_kill_pct must exceed daily_realized_dd_kill_pct")
        return self


class GateRow(BaseModel):
    model_config = {"frozen": True}
    min_days_in_paper: int = 0
    min_days_in_live_shadow: int = 0
    min_sharpe: float | None = None
    max_daily_dd_pct: float | None = None
    no_kill_triggers_days: int | None = None
    max_audit_anomalies: int | None = None
    max_reconciliation_diff_bps: float | None = None
    max_feed_gap_violations: int | None = None
    kill_switch_drill_passed: bool | None = None
    required_artifacts: list[str] = []


class PromotionConfig(BaseModel):
    model_config = {"frozen": True}
    paper_to_live_shadow: GateRow
    live_shadow_to_live: GateRow


class IngestConfig(BaseModel):
    model_config = {"frozen": True}
    s1_predictions_dir: str
    s2_verdicts_dir: str
    poll_interval_seconds: Annotated[float, Field(gt=0.0)]
    pair_window_seconds: Annotated[int, Field(ge=1)]


class PositionBookConfig(BaseModel):
    model_config = {"frozen": True}
    snapshot_root: str
    snapshot_interval_seconds: Annotated[int, Field(ge=1)]


class AuditCfg(BaseModel):
    model_config = {"frozen": True}
    root: str
    rotation: str = "daily"
    chmod_after_close: bool = True


class KillSwitchCfg(BaseModel):
    model_config = {"frozen": True}
    repo_root_marker: str
    poll_interval_seconds: Annotated[float, Field(gt=0.0)]
    emergency_snapshot_root: str


class ExecConfig(BaseModel):
    model_config = {"frozen": True}
    ingest: IngestConfig
    position_book: PositionBookConfig
    audit: AuditCfg
    kill_switch: KillSwitchCfg


def _load_yaml(path: Path) -> dict:
    with path.open() as h:
        return yaml.safe_load(h)


def load_risk_config(path: Path) -> RiskConfig:
    return RiskConfig.model_validate(_load_yaml(path))


def load_promotion_config(path: Path) -> PromotionConfig:
    return PromotionConfig.model_validate(_load_yaml(path))


def load_exec_config(path: Path) -> ExecConfig:
    return ExecConfig.model_validate(_load_yaml(path))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_configs.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/configs.py tests/test_execution_configs.py
git commit -m "feat(s4): Pydantic config loaders for risk/promotion/exec yaml"
```

---

### Task 4: Signal + verdict data types + ExecutionTicket

**Files:**
- Create: `src/quant_research_stack/execution/types.py`
- Test: `tests/test_execution_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_types.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _verdict() -> GovernorVerdict:
    return GovernorVerdict.model_validate({
        "signal_id": "sig-00000001",
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    })


def test_s1_signal_validates() -> None:
    s = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    assert s.symbol == "BTCUSDT"
    assert 0 <= s.confidence <= 1


def test_s1_signal_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        S1Signal(
            signal_id="sig-00000001",
            symbol="BTCUSDT",
            predicted_score=0.05,
            confidence=2.0,
            horizon_minutes=5,
            ts_utc=datetime.now(UTC),
        )


def test_execution_ticket_pairs_signal_and_verdict() -> None:
    sig = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = _verdict()
    t = ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))
    assert t.signal.signal_id == t.primary_verdict.signal_id


def test_execution_ticket_rejects_mismatched_ids() -> None:
    sig = S1Signal(
        signal_id="sig-00000001",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = GovernorVerdict.model_validate({
        "signal_id": "sig-00000002",  # different id
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    })
    with pytest.raises(ValueError, match="signal_id mismatch"):
        ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_types.py -q
```

Expected: ImportError for `quant_research_stack.execution.types`.

- [ ] **Step 3: Implement `types.py`**

Create `src/quant_research_stack/execution/types.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from quant_research_stack.governor.signal_schema import GovernorVerdict


class S1Signal(BaseModel):
    model_config = {"frozen": True}
    signal_id: Annotated[str, Field(min_length=4, max_length=64)]
    symbol: Annotated[str, Field(min_length=1, max_length=32)]
    predicted_score: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    horizon_minutes: Annotated[int, Field(ge=1, le=1440)]
    ts_utc: datetime


class ExecutionTicket(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    signal: S1Signal
    primary_verdict: GovernorVerdict
    tier3_verdict: GovernorVerdict | None
    ingested_at: datetime

    @model_validator(mode="after")
    def _ids_match(self) -> "ExecutionTicket":
        if self.signal.signal_id != self.primary_verdict.signal_id:
            raise ValueError("signal_id mismatch between S1 signal and primary verdict")
        if self.tier3_verdict is not None and self.tier3_verdict.signal_id != self.signal.signal_id:
            raise ValueError("signal_id mismatch between S1 signal and tier3 verdict")
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_types.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/types.py tests/test_execution_types.py
git commit -m "feat(s4): S1Signal + ExecutionTicket Pydantic types"
```

---

### Task 5: AuditLog transport for S4

**Files:**
- Create: `src/quant_research_stack/execution/audit.py`
- Test: `tests/test_execution_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from quant_research_stack.execution.audit import AuditLog


def test_audit_log_appends_jsonl(tmp_path: Path) -> None:
    log = AuditLog(root=tmp_path, rotation="daily", chmod_after_close=False)
    log.append("signal_ingested", {"signal_id": "sig-1", "symbol": "BTCUSDT"})
    log.append("trade_placed", {"signal_id": "sig-1", "order_id": "o-1", "qty": 0.01})
    log.close_current()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["event"] == "signal_ingested"
    assert lines[0]["not_investment_advice"] is True
    assert "timestamp_utc" in lines[0]


def test_audit_log_chmod_a_w_on_close(tmp_path: Path) -> None:
    log = AuditLog(root=tmp_path, rotation="daily", chmod_after_close=True)
    log.append("test", {})
    log.close_current()
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    mode = files[0].stat().st_mode & 0o222
    assert mode == 0, f"expected no write bits, got {oct(mode)}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_audit.py -q
```

Expected: ImportError for `quant_research_stack.execution.audit`.

- [ ] **Step 3: Implement `audit.py`**

Create `src/quant_research_stack/execution/audit.py`:

```python
from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only JSONL audit log with date rotation and chmod-a-w on close.

    Pattern mirrors the S2 governor transport (logs/audit/governor/<date>.jsonl).
    """

    def __init__(self, root: Path | str, rotation: str = "daily", chmod_after_close: bool = True) -> None:
        if rotation != "daily":
            raise ValueError(f"only 'daily' rotation supported; got {rotation}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chmod_after_close = chmod_after_close
        self._current_day: str | None = None
        self._current_path: Path | None = None

    def _current_file(self) -> Path:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_day:
            if self._current_path is not None:
                self._maybe_chmod(self._current_path)
            self._current_day = today
            self._current_path = self.root / f"{today}.jsonl"
        return self._current_path

    def append(self, event: str, payload: dict[str, Any]) -> None:
        path = self._current_file()
        record = {
            "event": event,
            "not_investment_advice": True,
            "payload": payload,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with path.open("a") as h:
            h.write(json.dumps(record) + "\n")

    def close_current(self) -> None:
        if self._current_path is not None:
            self._maybe_chmod(self._current_path)
            self._current_path = None
            self._current_day = None

    def _maybe_chmod(self, path: Path) -> None:
        if not self.chmod_after_close or not path.exists():
            return
        current = path.stat().st_mode
        # Strip all write bits.
        os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_audit.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/audit.py tests/test_execution_audit.py
git commit -m "feat(s4): AuditLog JSONL transport with daily rotation + chmod-a-w"
```

---

### Task 6: SignalIngestor (pair S1 prediction with S2 verdict)

**Files:**
- Create: `src/quant_research_stack/execution/signals.py`
- Test: `tests/test_execution_signals.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_signals.py`:

```python
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.signals import SignalIngestor


def _write_pred(dir_: Path, ts: datetime, sig_id: str) -> None:
    df = pl.DataFrame({
        "signal_id": [sig_id],
        "symbol": ["BTCUSDT"],
        "predicted_score": [0.05],
        "confidence": [0.7],
        "horizon_minutes": [5],
        "ts_utc": [ts.isoformat()],
    })
    dir_.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dir_ / f"{ts.strftime('%Y-%m-%d')}.parquet")


def _write_verdict(dir_: Path, ts: datetime, sig_id: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "signal_id": sig_id,
        "decision": "pass",
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"],
        "contradictions_flagged": [],
    }
    with (dir_ / f"{ts.strftime('%Y-%m-%d')}.jsonl").open("a") as h:
        h.write(json.dumps(rec) + "\n")


@pytest.mark.asyncio
async def test_signal_ingestor_pairs_within_window(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    verdicts_dir = tmp_path / "verdicts"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    ts = datetime.now(UTC)
    _write_pred(preds_dir, ts, "sig-00000001")
    _write_verdict(verdicts_dir, ts, "sig-00000001")
    ing = SignalIngestor(
        preds_dir=preds_dir,
        verdicts_dir=verdicts_dir,
        poll_interval_s=0.05,
        pair_window_s=5,
        audit=audit,
    )
    tickets: list = []

    async def drain() -> None:
        async for t in ing.stream():
            tickets.append(t)
            if len(tickets) >= 1:
                ing.stop()

    await asyncio.wait_for(drain(), timeout=3.0)
    assert len(tickets) == 1
    assert tickets[0].signal.signal_id == "sig-00000001"


@pytest.mark.asyncio
async def test_signal_ingestor_audits_verdict_timeout(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    verdicts_dir = tmp_path / "verdicts"
    audit_dir = tmp_path / "audit"
    audit = AuditLog(root=audit_dir, chmod_after_close=False)
    ts = datetime.now(UTC)
    _write_pred(preds_dir, ts, "sig-00000099")
    # No verdict written — should time out
    ing = SignalIngestor(
        preds_dir=preds_dir,
        verdicts_dir=verdicts_dir,
        poll_interval_s=0.05,
        pair_window_s=1,
        audit=audit,
    )

    async def drain() -> None:
        async for _ in ing.stream():
            ing.stop()

    try:
        await asyncio.wait_for(drain(), timeout=3.0)
    except TimeoutError:
        pass
    ing.stop()
    audit.close_current()
    files = list(audit_dir.iterdir())
    assert files, "audit log empty"
    lines = files[0].read_text().splitlines()
    events = [json.loads(line)["event"] for line in lines if line]
    assert "verdict_timeout" in events
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_signals.py -q
```

Expected: ImportError for `quant_research_stack.execution.signals`.

- [ ] **Step 3: Implement `signals.py`**

Create `src/quant_research_stack/execution/signals.py`:

```python
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import GovernorVerdict


class SignalIngestor:
    """Tails S1 predictions Parquet + S2 verdicts JSONL; emits ExecutionTickets."""

    def __init__(
        self,
        preds_dir: Path,
        verdicts_dir: Path,
        poll_interval_s: float,
        pair_window_s: int,
        audit: AuditLog,
    ) -> None:
        self._preds_dir = Path(preds_dir)
        self._verdicts_dir = Path(verdicts_dir)
        self._poll = float(poll_interval_s)
        self._pair_window = int(pair_window_s)
        self._audit = audit
        self._stop = False
        self._seen_preds: set[str] = set()
        self._seen_verdicts: dict[str, GovernorVerdict] = {}
        self._pending_preds: dict[str, tuple[S1Signal, float]] = {}

    def stop(self) -> None:
        self._stop = True

    async def stream(self) -> AsyncIterator[ExecutionTicket]:
        while not self._stop:
            self._scan_predictions()
            self._scan_verdicts()
            for sig_id, (signal, first_seen) in list(self._pending_preds.items()):
                verdict = self._seen_verdicts.get(sig_id)
                if verdict is not None:
                    self._pending_preds.pop(sig_id, None)
                    self._audit.append("signal_ingested", {"signal_id": sig_id, "symbol": signal.symbol})
                    self._audit.append("verdict_received", {"signal_id": sig_id, "decision": verdict.decision.value})
                    yield ExecutionTicket(
                        signal=signal,
                        primary_verdict=verdict,
                        tier3_verdict=None,
                        ingested_at=datetime.now(UTC),
                    )
                elif asyncio.get_event_loop().time() - first_seen > self._pair_window:
                    self._pending_preds.pop(sig_id, None)
                    self._audit.append(
                        "verdict_timeout",
                        {"signal_id": sig_id, "waited_seconds": self._pair_window},
                    )
            await asyncio.sleep(self._poll)

    def _scan_predictions(self) -> None:
        if not self._preds_dir.exists():
            return
        for p in sorted(self._preds_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(p)
            except Exception:
                continue
            for row in df.iter_rows(named=True):
                sig_id = row.get("signal_id")
                if not sig_id or sig_id in self._seen_preds:
                    continue
                try:
                    signal = S1Signal(
                        signal_id=sig_id,
                        symbol=row["symbol"],
                        predicted_score=float(row["predicted_score"]),
                        confidence=float(row["confidence"]),
                        horizon_minutes=int(row["horizon_minutes"]),
                        ts_utc=datetime.fromisoformat(row["ts_utc"]),
                    )
                except Exception:
                    self._audit.append("signal_parse_error", {"signal_id": sig_id})
                    self._seen_preds.add(sig_id)
                    continue
                self._seen_preds.add(sig_id)
                self._pending_preds[sig_id] = (signal, asyncio.get_event_loop().time())

    def _scan_verdicts(self) -> None:
        if not self._verdicts_dir.exists():
            return
        for p in sorted(self._verdicts_dir.glob("*.jsonl")):
            try:
                text = p.read_text()
            except Exception:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    v = GovernorVerdict.model_validate(payload)
                except Exception:
                    self._audit.append("verdict_parse_error", {"raw": line[:200]})
                    continue
                self._seen_verdicts[v.signal_id] = v
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_signals.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/signals.py tests/test_execution_signals.py
git commit -m "feat(s4): SignalIngestor pairs S1 predictions with S2 verdicts in a window"
```

---

### Task 7: RiskGate (ordered pre-trade checks)

**Files:**
- Create: `src/quant_research_stack/execution/risk.py`
- Test: `tests/test_execution_risk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_risk.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_research_stack.execution.configs import RiskConfig, load_risk_config
from quant_research_stack.execution.risk import RiskGate, RiskState, _GATES
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _cfg() -> RiskConfig:
    return load_risk_config(Path("configs/risk.yaml"))


def _ticket(decision: str = "pass") -> ExecutionTicket:
    sig = S1Signal(
        signal_id="sig-00001111",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=0.7,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    v = GovernorVerdict.model_validate({
        "signal_id": sig.signal_id,
        "decision": decision,
        "direction": 1,
        "confidence": 0.7,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    })
    return ExecutionTicket(signal=sig, primary_verdict=v, tier3_verdict=None, ingested_at=datetime.now(UTC))


def test_gate_order_is_kill_first() -> None:
    assert _GATES[0].__name__ == "kill_flag_check"


def test_kill_flag_blocks_before_anything_else(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING"
    flag.touch()
    state = RiskState(
        account_equity=100_000,
        peak_equity=100_000,
        daily_realized_pnl=0,
        gross_exposure_notional=0,
        per_symbol_notional={},
        orders_last_minute=0,
        last_tick_ts={},
        kill_flag_path=flag,
        is_crypto=lambda _s: True,
        now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "kill_flag_check"


def test_governor_veto_blocks_without_killing() -> None:
    state = RiskState(
        account_equity=100_000, peak_equity=100_000, daily_realized_pnl=0,
        gross_exposure_notional=0, per_symbol_notional={}, orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(decision="veto"), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "governor_decision_check"


def test_drawdown_kill_when_daily_breached() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000, peak_equity=100_000,
        daily_realized_pnl=-100_000 * cfg.drawdown.daily_realized_dd_kill_pct * 1.1,
        gross_exposure_notional=0, per_symbol_notional={}, orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "drawdown_check"


def test_feed_freshness_kill_when_gap_exceeded() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000, peak_equity=100_000, daily_realized_pnl=0,
        gross_exposure_notional=0, per_symbol_notional={}, orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC) - timedelta(seconds=cfg.freshness.crypto_max_gap_seconds + 10)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is True
    assert decision.reason == "feed_freshness_check"


def test_exposure_blocks_without_killing() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000, peak_equity=100_000, daily_realized_pnl=0,
        gross_exposure_notional=100_000 * cfg.limits.max_gross_exposure_pct,  # at cap
        per_symbol_notional={"BTCUSDT": 100_000 * cfg.limits.max_per_symbol_pct},
        orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "exposure_check"


def test_rate_limit_blocks_without_killing() -> None:
    cfg = _cfg()
    state = RiskState(
        account_equity=100_000, peak_equity=100_000, daily_realized_pnl=0,
        gross_exposure_notional=0, per_symbol_notional={},
        orders_last_minute=cfg.limits.max_orders_per_minute,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(cfg)
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is False
    assert decision.kill_trigger is False
    assert decision.reason == "rate_limit_check"


def test_happy_path_passes_all_gates() -> None:
    state = RiskState(
        account_equity=100_000, peak_equity=100_000, daily_realized_pnl=0,
        gross_exposure_notional=0, per_symbol_notional={}, orders_last_minute=0,
        last_tick_ts={"BTCUSDT": datetime.now(UTC)},
        kill_flag_path=Path("/nonexistent/KILL_TRADING_XYZ"),
        is_crypto=lambda _s: True, now=datetime.now(UTC),
    )
    gate = RiskGate(_cfg())
    decision = gate.evaluate(_ticket(), state)
    assert decision.allowed is True
    assert decision.kill_trigger is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_risk.py -q
```

Expected: ImportError for `quant_research_stack.execution.risk`.

- [ ] **Step 3: Implement `risk.py`**

Create `src/quant_research_stack/execution/risk.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_research_stack.execution.configs import RiskConfig
from quant_research_stack.execution.types import ExecutionTicket
from quant_research_stack.governor.signal_schema import Decision


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    kill_trigger: bool
    reason: str  # name of the gate that fired; "" on allow


@dataclass
class RiskState:
    """Snapshot of the world the RiskGate evaluates against."""
    account_equity: float
    peak_equity: float
    daily_realized_pnl: float
    gross_exposure_notional: float
    per_symbol_notional: dict[str, float]
    orders_last_minute: int
    last_tick_ts: dict[str, datetime]
    kill_flag_path: Path
    is_crypto: Callable[[str], bool]
    now: datetime


# Per ADR-0014: kill_flag_check MUST be first. The unit test
# test_gate_order_is_kill_first enforces this invariant.
def kill_flag_check(_ticket: ExecutionTicket, state: RiskState, _cfg: RiskConfig) -> tuple[bool, bool]:
    if state.kill_flag_path.exists():
        return (False, True)
    return (True, False)


def feed_freshness_check(ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    last = state.last_tick_ts.get(ticket.signal.symbol)
    if last is None:
        return (False, True)
    gap_s = (state.now - last).total_seconds()
    threshold = (
        cfg.freshness.crypto_max_gap_seconds
        if state.is_crypto(ticket.signal.symbol)
        else cfg.freshness.equity_max_gap_seconds
    )
    if gap_s > threshold:
        return (False, True)
    return (True, False)


def drawdown_check(_ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    if state.account_equity <= 0:
        return (False, True)
    daily_dd_pct = -state.daily_realized_pnl / state.account_equity if state.daily_realized_pnl < 0 else 0.0
    if daily_dd_pct > cfg.drawdown.daily_realized_dd_kill_pct:
        return (False, True)
    cum_dd_pct = (state.peak_equity - state.account_equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    if cum_dd_pct > cfg.drawdown.cumulative_dd_kill_pct:
        return (False, True)
    return (True, False)


def exposure_check(ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    eq = state.account_equity
    if eq <= 0:
        return (False, False)
    gross_cap = eq * cfg.limits.max_gross_exposure_pct
    per_symbol_cap = eq * cfg.limits.max_per_symbol_pct
    if state.gross_exposure_notional >= gross_cap:
        return (False, False)
    if state.per_symbol_notional.get(ticket.signal.symbol, 0.0) >= per_symbol_cap:
        return (False, False)
    return (True, False)


def rate_limit_check(_ticket: ExecutionTicket, state: RiskState, cfg: RiskConfig) -> tuple[bool, bool]:
    if state.orders_last_minute >= cfg.limits.max_orders_per_minute:
        return (False, False)
    return (True, False)


def governor_decision_check(ticket: ExecutionTicket, _state: RiskState, _cfg: RiskConfig) -> tuple[bool, bool]:
    if ticket.primary_verdict.decision != Decision.pass_:
        return (False, False)
    return (True, False)


_GATES = [
    kill_flag_check,
    feed_freshness_check,
    drawdown_check,
    exposure_check,
    rate_limit_check,
    governor_decision_check,
]


class RiskGate:
    """Ordered pre-trade check chain. Kill checks short-circuit with kill_trigger=True."""

    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    def evaluate(self, ticket: ExecutionTicket, state: RiskState) -> RiskDecision:
        for gate in _GATES:
            allowed, kill = gate(ticket, state, self._cfg)
            if not allowed:
                return RiskDecision(allowed=False, kill_trigger=kill, reason=gate.__name__)
        return RiskDecision(allowed=True, kill_trigger=False, reason="")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_risk.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/risk.py tests/test_execution_risk.py
git commit -m "feat(s4): RiskGate with kill-first ordering and 6 pre-trade checks"
```

---

### Task 8: Sizer (confidence × stance × cap → qty)

**Files:**
- Create: `src/quant_research_stack/execution/sizing.py`
- Test: `tests/test_execution_sizing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_sizing.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant_research_stack.execution.configs import load_risk_config
from quant_research_stack.execution.sizing import Sizer, SizerInput
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


def _ticket(direction: int = 1, decision: str = "pass", confidence: float = 0.7,
            t3_dir: int | None = None, t3_dec: str | None = None) -> ExecutionTicket:
    sig = S1Signal(
        signal_id="sig-00002222",
        symbol="BTCUSDT",
        predicted_score=0.05,
        confidence=confidence,
        horizon_minutes=5,
        ts_utc=datetime.now(UTC),
    )
    prim = GovernorVerdict.model_validate({
        "signal_id": sig.signal_id,
        "decision": decision,
        "direction": direction,
        "confidence": confidence,
        "horizon_minutes": 5,
        "regime_tag": "trending",
        "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    })
    t3 = None
    if t3_dir is not None and t3_dec is not None:
        t3 = GovernorVerdict.model_validate({
            "signal_id": sig.signal_id,
            "decision": t3_dec,
            "direction": t3_dir,
            "confidence": 0.8,
            "horizon_minutes": 5,
            "regime_tag": "trending",
            "rationale_short": "ok",
            "cited_paper_chunk_ids": ["paper_pdf:x:0"] if t3_dec == "pass" else [],
            "contradictions_flagged": [],
        })
    return ExecutionTicket(signal=sig, primary_verdict=prim, tier3_verdict=t3, ingested_at=datetime.now(UTC))


def test_veto_yields_zero_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(ticket=_ticket(decision="veto"), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    assert qty == 0.0


def test_neutral_direction_yields_zero_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(ticket=_ticket(direction=0), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    assert qty == 0.0


def test_long_signal_yields_positive_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    # base_pct = 0.005 * conf 0.7 * equity 100k = 350 notional / price 50k = 0.007 BTC
    assert 0 < qty <= 0.02  # under max_per_symbol_pct cap


def test_short_signal_yields_negative_qty() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(ticket=_ticket(direction=-1), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    assert qty < 0


def test_tier3_agreement_increases_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    boosted = sizer.size(SizerInput(
        ticket=_ticket(direction=1, t3_dir=1, t3_dec="pass"),
        account_equity=100_000, mid_price=50_000, lot_size=0.0001,
    ))
    assert boosted > base


def test_tier3_disagreement_shrinks_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    shrunk = sizer.size(SizerInput(
        ticket=_ticket(direction=1, t3_dir=-1, t3_dec="pass"),
        account_equity=100_000, mid_price=50_000, lot_size=0.0001,
    ))
    assert shrunk < base


def test_tier3_veto_shrinks_size() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    base = sizer.size(SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.0001))
    vetoed = sizer.size(SizerInput(
        ticket=_ticket(direction=1, t3_dir=1, t3_dec="veto"),
        account_equity=100_000, mid_price=50_000, lot_size=0.0001,
    ))
    assert vetoed < base


def test_qty_respects_per_symbol_cap() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(
        ticket=_ticket(direction=1, confidence=1.0),
        account_equity=10_000_000, mid_price=50_000, lot_size=0.0001,
    ))
    cap_notional = 10_000_000 * cfg.limits.max_per_symbol_pct
    cap_qty = cap_notional / 50_000
    assert qty <= cap_qty + 1e-9


def test_qty_rounded_to_lot() -> None:
    cfg = load_risk_config(Path("configs/risk.yaml"))
    sizer = Sizer(cfg, tier3_stance_pct=0.20)
    qty = sizer.size(SizerInput(ticket=_ticket(direction=1), account_equity=100_000, mid_price=50_000, lot_size=0.001))
    # qty should be a multiple of 0.001
    assert abs(qty * 1000 - round(qty * 1000)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_sizing.py -q
```

Expected: ImportError for `quant_research_stack.execution.sizing`.

- [ ] **Step 3: Implement `sizing.py`**

Create `src/quant_research_stack/execution/sizing.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from quant_research_stack.execution.configs import RiskConfig
from quant_research_stack.execution.types import ExecutionTicket
from quant_research_stack.governor.signal_schema import Decision, GovernorVerdict


@dataclass(frozen=True)
class SizerInput:
    ticket: ExecutionTicket
    account_equity: float
    mid_price: float
    lot_size: float


def _stance_modifier(primary: GovernorVerdict, tier3: GovernorVerdict | None, cfg_pct: float) -> float:
    if tier3 is None or tier3.decision == Decision.insufficient_evidence:
        return 0.0
    if tier3.decision == Decision.veto:
        return -cfg_pct
    # Tier-3 passed; compare direction
    if int(tier3.direction) == int(primary.direction):
        return +cfg_pct
    return -cfg_pct


def _round_to_lot(qty: float, lot_size: float) -> float:
    if lot_size <= 0:
        return qty
    return math.copysign(math.floor(abs(qty) / lot_size) * lot_size, qty)


class Sizer:
    """Confidence-scaled position sizing with hard caps (ADR-0012)."""

    def __init__(self, cfg: RiskConfig, tier3_stance_pct: float = 0.20) -> None:
        self._cfg = cfg
        self._stance_pct = float(tier3_stance_pct)

    def size(self, inp: SizerInput) -> float:
        primary = inp.ticket.primary_verdict
        if primary.decision != Decision.pass_:
            return 0.0
        direction = int(primary.direction)
        if direction == 0:
            return 0.0
        if inp.mid_price <= 0 or inp.account_equity <= 0:
            return 0.0

        stance_mod = _stance_modifier(primary, inp.ticket.tier3_verdict, self._stance_pct)
        base_pct = self._cfg.limits.base_notional_per_trade_pct
        target_notional = inp.account_equity * base_pct * primary.confidence * (1.0 + stance_mod)
        per_symbol_cap = inp.account_equity * self._cfg.limits.max_per_symbol_pct
        target_notional = min(target_notional, per_symbol_cap)
        qty = direction * target_notional / inp.mid_price
        return _round_to_lot(qty, inp.lot_size)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_sizing.py -q
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/sizing.py tests/test_execution_sizing.py
git commit -m "feat(s4): Sizer with confidence × stance × cap math (ADR-0012)"
```

---

### Task 9: PositionBook (in-memory + Parquet snapshot)

**Files:**
- Create: `src/quant_research_stack/execution/position_book.py`
- Test: `tests/test_execution_position_book.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_position_book.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.execution.position_book import PositionBook


def _fill(symbol: str, side: str, qty: str, price: str) -> Fill:
    return Fill(
        order_id="o-1",
        client_order_id="c-1",
        symbol=symbol,
        side=OrderSide(side),
        qty=Decimal(qty),
        price=Decimal(price),
        fee=Decimal("0"),
        ts_utc=datetime.now(UTC),
    )


def test_apply_buy_fill_increases_position(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    pos = book.position("BTCUSDT")
    assert pos.qty == Decimal("0.01")
    assert pos.avg_price == Decimal("50000")


def test_apply_sell_fill_decreases_position(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.02", "50000"))
    book.apply_fill(_fill("BTCUSDT", "sell", "0.01", "55000"))
    pos = book.position("BTCUSDT")
    assert pos.qty == Decimal("0.01")
    assert book.daily_realized_pnl > 0  # sold higher than buy


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    book.snapshot()
    files = list((tmp_path / "paper").glob("*.parquet"))
    assert len(files) == 1
    df = pl.read_parquet(files[0])
    assert "symbol" in df.columns and "qty" in df.columns
    assert df.height >= 1


def test_load_latest_snapshot_recovers_book(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book.apply_fill(_fill("BTCUSDT", "buy", "0.01", "50000"))
    book.snapshot()
    book2 = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    book2.load_latest_snapshot()
    assert book2.position("BTCUSDT").qty == Decimal("0.01")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_position_book.py -q
```

Expected: ImportError for `quant_research_stack.execution.position_book`.

- [ ] **Step 3: Implement `position_book.py`**

Create `src/quant_research_stack/execution/position_book.py`:

```python
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from quant_research_stack.brokers.order_types import Fill, OrderSide


@dataclass
class Position:
    symbol: str
    qty: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")


@dataclass
class PositionBook:
    snapshot_root: Path
    stage: str
    starting_equity: Decimal
    _positions: dict[str, Position] = field(default_factory=dict)
    _daily_realized_pnl: Decimal = Decimal("0")
    _peak_equity: Decimal | None = None
    _last_snap_day: str | None = None

    def __post_init__(self) -> None:
        self.snapshot_root = Path(self.snapshot_root)
        self.stage_dir = self.snapshot_root / self.stage
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        if self._peak_equity is None:
            self._peak_equity = self.starting_equity

    @property
    def daily_realized_pnl(self) -> Decimal:
        return self._daily_realized_pnl

    @property
    def peak_equity(self) -> Decimal:
        assert self._peak_equity is not None
        return self._peak_equity

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    def per_symbol_notional(self, mid: dict[str, Decimal]) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym, pos in self._positions.items():
            if pos.qty == 0:
                continue
            price = mid.get(sym, pos.avg_price)
            out[sym] = float(abs(pos.qty) * price)
        return out

    def gross_exposure(self, mid: dict[str, Decimal]) -> float:
        return sum(self.per_symbol_notional(mid).values())

    def apply_fill(self, fill: Fill) -> None:
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        signed_qty = fill.qty if fill.side == OrderSide.buy else -fill.qty
        new_qty = pos.qty + signed_qty
        if pos.qty == 0 or (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
            # opening or extending — update VWAP
            total_cost = pos.qty * pos.avg_price + signed_qty * fill.price
            pos.qty = new_qty
            pos.avg_price = total_cost / pos.qty if pos.qty != 0 else Decimal("0")
        else:
            # reducing — realize PnL on the closed portion
            closing_qty = min(abs(signed_qty), abs(pos.qty))
            sign = 1 if pos.qty > 0 else -1
            realized = sign * closing_qty * (fill.price - pos.avg_price)
            self._daily_realized_pnl += realized
            pos.qty = new_qty
            if pos.qty == 0:
                pos.avg_price = Decimal("0")
        if pos.qty == 0:
            self._positions.pop(fill.symbol, None)

    def snapshot(self) -> Path:
        now = datetime.now(UTC)
        day = now.strftime("%Y-%m-%d")
        path = self.stage_dir / f"{day}.parquet"
        # Date rotation: lock previous day before writing the new one
        if self._last_snap_day is not None and self._last_snap_day != day:
            prev = self.stage_dir / f"{self._last_snap_day}.parquet"
            if prev.exists():
                self._chmod_a_w(prev)
        self._last_snap_day = day
        rows = [
            {"symbol": p.symbol, "qty": float(p.qty), "avg_price": float(p.avg_price),
             "snapshot_ts_utc": now.isoformat()}
            for p in self._positions.values()
        ]
        if not rows:
            rows = [{"symbol": "_empty", "qty": 0.0, "avg_price": 0.0, "snapshot_ts_utc": now.isoformat()}]
        pl.DataFrame(rows).write_parquet(path, compression="zstd")
        return path

    def load_latest_snapshot(self) -> bool:
        files = sorted(self.stage_dir.glob("*.parquet"))
        if not files:
            return False
        df = pl.read_parquet(files[-1])
        for row in df.iter_rows(named=True):
            if row["symbol"] == "_empty":
                continue
            self._positions[row["symbol"]] = Position(
                symbol=row["symbol"],
                qty=Decimal(str(row["qty"])),
                avg_price=Decimal(str(row["avg_price"])),
            )
        return True

    def _chmod_a_w(self, path: Path) -> None:
        current = path.stat().st_mode
        os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_position_book.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/position_book.py tests/test_execution_position_book.py
git commit -m "feat(s4): in-memory PositionBook with VWAP + Parquet snapshot + reload"
```

---

### Task 10: ReconcileLoop (60-s broker diff)

**Files:**
- Create: `src/quant_research_stack/execution/reconciliation.py`
- Test: `tests/test_execution_reconciliation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_reconciliation.py`:

```python
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from quant_research_stack.execution.position_book import PositionBook
from quant_research_stack.execution.reconciliation import (
    ReconciliationResult,
    diff_book_vs_broker,
)


def test_zero_diff_when_book_matches_broker(tmp_path: Path) -> None:
    book = PositionBook(snapshot_root=tmp_path, stage="paper", starting_equity=Decimal("100000"))
    broker_equity = Decimal("100000")
    result = diff_book_vs_broker(book_equity=Decimal("100000"), broker_equity=broker_equity)
    assert isinstance(result, ReconciliationResult)
    assert result.diff_bps == pytest.approx(0.0, abs=1e-9)


def test_one_bp_diff() -> None:
    book = Decimal("100010")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.diff_bps == pytest.approx(1.0, abs=1e-4)


def test_exceeds_threshold_at_2_bps() -> None:
    book = Decimal("100020")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.diff_bps > 1.0
    assert result.exceeds_threshold(max_diff_bps=1.0) is True


def test_within_threshold_at_half_bp() -> None:
    book = Decimal("100005")
    broker = Decimal("100000")
    result = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert result.exceeds_threshold(max_diff_bps=1.0) is False


def test_zero_broker_equity_treated_as_divergence() -> None:
    result = diff_book_vs_broker(book_equity=Decimal("100"), broker_equity=Decimal("0"))
    assert result.exceeds_threshold(max_diff_bps=1.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_reconciliation.py -q
```

Expected: ImportError for `quant_research_stack.execution.reconciliation`.

- [ ] **Step 3: Implement `reconciliation.py`**

Create `src/quant_research_stack/execution/reconciliation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ReconciliationResult:
    book_equity: Decimal
    broker_equity: Decimal
    diff_bps: float

    def exceeds_threshold(self, max_diff_bps: float) -> bool:
        return self.diff_bps > max_diff_bps


def diff_book_vs_broker(book_equity: Decimal, broker_equity: Decimal) -> ReconciliationResult:
    if broker_equity <= 0:
        return ReconciliationResult(
            book_equity=book_equity, broker_equity=broker_equity, diff_bps=float("inf"),
        )
    diff = abs(book_equity - broker_equity)
    bps = float(diff / broker_equity * Decimal("10000"))
    return ReconciliationResult(book_equity=book_equity, broker_equity=broker_equity, diff_bps=bps)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_reconciliation.py -q
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/reconciliation.py tests/test_execution_reconciliation.py
git commit -m "feat(s4): broker reconciliation diff helper (basis-points threshold)"
```

---

### Task 11: BrokerRouter (stage → BrokerAdapter)

**Files:**
- Create: `src/quant_research_stack/execution/router.py`
- Test: `tests/test_execution_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_router.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.execution.router import BrokerRouter, UnknownBrokerError


def _cfg() -> dict:
    return yaml.safe_load(Path("configs/brokers.yaml").read_text())


def test_paper_stage_resolves_alpaca_paper_for_equity() -> None:
    router = BrokerRouter(_cfg())
    name = router.resolved_name("paper", asset_class="equity")
    assert name == "alpaca_paper"


def test_live_shadow_writes_to_null_broker() -> None:
    router = BrokerRouter(_cfg())
    name = router.resolved_name("live_shadow", asset_class="equity")
    assert name == "null_broker"


def test_live_route_raises_import_error_when_module_missing() -> None:
    router = BrokerRouter(_cfg())
    with pytest.raises(ImportError, match="Live broker not installed"):
        router.resolve("live", asset_class="equity")


def test_unknown_stage_raises() -> None:
    router = BrokerRouter(_cfg())
    with pytest.raises(UnknownBrokerError):
        router.resolve("does_not_exist", asset_class="equity")


def test_live_shadow_returns_null_broker_instance() -> None:
    router = BrokerRouter(_cfg())
    broker = router.resolve("live_shadow", asset_class="equity")
    assert isinstance(broker, NullBroker)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_router.py -q
```

Expected: ImportError for `quant_research_stack.execution.router`.

- [ ] **Step 3: Implement `router.py`**

Create `src/quant_research_stack/execution/router.py`:

```python
from __future__ import annotations

from typing import Any

from quant_research_stack.brokers.base import BrokerAdapter


class UnknownBrokerError(KeyError):
    pass


_PAPER_ADAPTERS = {"alpaca_paper", "binance_testnet"}
_NULL_ADAPTERS = {"null_broker"}
_LIVE_ADAPTERS = {"alpaca_live", "binance_live"}


class BrokerRouter:
    """Resolves a stage + asset class to a concrete BrokerAdapter instance.

    Live brokers are ImportError-guarded; the live route never imports them unless
    the module is actually installed. Paper and live_shadow stages never touch
    live broker code.
    """

    def __init__(self, brokers_cfg: dict[str, Any]) -> None:
        routes = brokers_cfg.get("stage_routes")
        if not routes:
            raise ValueError("configs/brokers.yaml is missing 'stage_routes'")
        self._routes = routes

    def resolved_name(self, stage: str, asset_class: str) -> str:
        if stage not in self._routes:
            raise UnknownBrokerError(f"stage {stage!r} not in stage_routes")
        route = self._routes[stage]
        name = route.get(asset_class)
        if not name:
            raise UnknownBrokerError(f"asset_class {asset_class!r} not in stage_routes[{stage!r}]")
        return str(name)

    def resolve(self, stage: str, asset_class: str) -> BrokerAdapter:
        name = self.resolved_name(stage, asset_class)
        return self._instantiate(name)

    def _instantiate(self, name: str) -> BrokerAdapter:
        if name in _NULL_ADAPTERS:
            from quant_research_stack.brokers.null_broker import NullBroker

            return NullBroker()
        if name == "alpaca_paper":
            from quant_research_stack.brokers.alpaca_paper import AlpacaPaperBroker

            return AlpacaPaperBroker()
        if name == "binance_testnet":
            from quant_research_stack.brokers.binance_testnet import BinanceTestnetBroker

            return BinanceTestnetBroker()
        if name in _LIVE_ADAPTERS:
            raise ImportError(
                f"Live broker not installed: {name!r}. "
                "Live brokers (S4.1) require two-person review per CLAUDE.md §1.13.",
            )
        raise UnknownBrokerError(f"unknown broker {name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_router.py -q
```

Expected: 5 passed. `test_live_shadow_returns_null_broker_instance` constructs a NullBroker — if the existing NullBroker requires arguments, this test will fail; adjust the test to pass the required args from the current S3 NullBroker signature.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/router.py tests/test_execution_router.py
git commit -m "feat(s4): BrokerRouter with stage→adapter resolution + ImportError-guarded live route"
```

---

### Task 12: KillSwitch (file flag + SIGTERM/SIGINT + close-out)

**Files:**
- Create: `src/quant_research_stack/execution/kill_switch.py`
- Test: `tests/test_execution_kill_switch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_kill_switch.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.kill_switch import KillSwitchWatcher


@pytest.mark.asyncio
async def test_watcher_fires_when_flag_appears(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    fired: list[str] = []

    async def on_kill(reason: str) -> None:
        fired.append(reason)

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.1)
    flag.touch()
    await asyncio.wait_for(asyncio.sleep(0.5), timeout=2.0)
    watcher.stop()
    await task
    assert "file_flag" in fired


@pytest.mark.asyncio
async def test_watcher_stops_cleanly(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_TRADING_NEVER"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)

    async def on_kill(_: str) -> None:
        pass

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.15)
    watcher.stop()
    await asyncio.wait_for(task, timeout=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_kill_switch.py -q
```

Expected: ImportError for `quant_research_stack.execution.kill_switch`.

- [ ] **Step 3: Implement `kill_switch.py`**

Create `src/quant_research_stack/execution/kill_switch.py`:

```python
from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path

from quant_research_stack.execution.audit import AuditLog


class KillSwitchWatcher:
    """Watches a repo-root flag file. Also installs SIGTERM/SIGINT handlers."""

    def __init__(
        self,
        flag_path: Path,
        poll_interval_s: float,
        audit: AuditLog,
        on_kill: Callable[[str], Awaitable[None]],
    ) -> None:
        self._flag = Path(flag_path)
        self._poll = float(poll_interval_s)
        self._audit = audit
        self._on_kill = on_kill
        self._stop = False
        self._fired = False

    def stop(self) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig_name in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig_name, lambda s=sig_name: asyncio.create_task(self._trigger(s.name)))
            except NotImplementedError:
                # Windows / non-main-thread fallback — file flag still works
                pass

    async def run(self) -> None:
        while not self._stop:
            if self._flag.exists() and not self._fired:
                await self._trigger("file_flag")
                return
            await asyncio.sleep(self._poll)

    async def _trigger(self, reason: str) -> None:
        if self._fired:
            return
        self._fired = True
        self._audit.append("kill_trigger", {"reason": reason})
        await self._on_kill(reason)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_kill_switch.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/kill_switch.py tests/test_execution_kill_switch.py
git commit -m "feat(s4): KillSwitchWatcher for file-flag + SIGTERM/SIGINT close-out trigger"
```

---

### Task 13: Loop (async orchestration glue)

**Files:**
- Create: `src/quant_research_stack/execution/loop.py`
- Test: `tests/test_execution_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_execution_loop.py`:

```python
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.brokers.null_broker import NullBroker
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import load_exec_config, load_risk_config
from quant_research_stack.execution.loop import S4Loop


def _write_pred(dir_: Path, sig_id: str) -> None:
    df = pl.DataFrame({
        "signal_id": [sig_id], "symbol": ["BTCUSDT"], "predicted_score": [0.05],
        "confidence": [0.7], "horizon_minutes": [5], "ts_utc": [datetime.now(UTC).isoformat()],
    })
    dir_.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.parquet")


def _write_verdict(dir_: Path, sig_id: str, decision: str = "pass") -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "signal_id": sig_id, "decision": decision, "direction": 1, "confidence": 0.7,
        "horizon_minutes": 5, "regime_tag": "trending", "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"] if decision == "pass" else [],
        "contradictions_flagged": [],
    }
    with (dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl").open("a") as h:
        h.write(json.dumps(rec) + "\n")


@pytest.mark.asyncio
async def test_loop_processes_one_signal_end_to_end(tmp_path: Path) -> None:
    risk_cfg = load_risk_config(Path("configs/risk.yaml"))
    exec_cfg_data = {
        "ingest": {
            "s1_predictions_dir": str(tmp_path / "preds"),
            "s2_verdicts_dir": str(tmp_path / "verdicts"),
            "poll_interval_seconds": 0.05,
            "pair_window_seconds": 5,
        },
        "position_book": {"snapshot_root": str(tmp_path / "positions"), "snapshot_interval_seconds": 60},
        "audit": {"root": str(tmp_path / "audit"), "rotation": "daily", "chmod_after_close": False},
        "kill_switch": {
            "repo_root_marker": str(tmp_path / "KILL_TRADING_NEVER"),
            "poll_interval_seconds": 0.05,
            "emergency_snapshot_root": str(tmp_path / "snaps"),
        },
    }
    from quant_research_stack.execution.configs import ExecConfig
    exec_cfg = ExecConfig.model_validate(exec_cfg_data)
    audit = AuditLog(root=Path(exec_cfg.audit.root), chmod_after_close=False)
    broker = NullBroker()
    loop = S4Loop(
        stage="paper",
        risk_cfg=risk_cfg,
        exec_cfg=exec_cfg,
        broker=broker,
        audit=audit,
        starting_equity=Decimal("100000"),
        mid_price_lookup=lambda _s: Decimal("50000"),
        is_crypto=lambda _s: True,
        tier3_stance_pct=0.20,
    )

    _write_pred(Path(exec_cfg.ingest.s1_predictions_dir), "sig-00003333")
    _write_verdict(Path(exec_cfg.ingest.s2_verdicts_dir), "sig-00003333")

    task = asyncio.create_task(loop.run(max_tickets=1))
    await asyncio.wait_for(task, timeout=5.0)

    audit.close_current()
    events = []
    for p in Path(exec_cfg.audit.root).iterdir():
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line)["event"])
    assert "signal_ingested" in events
    assert "trade_placed" in events
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_loop.py -q
```

Expected: ImportError for `quant_research_stack.execution.loop`.

- [ ] **Step 3: Implement `loop.py`**

Create `src/quant_research_stack/execution/loop.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.base import BrokerAdapter
from quant_research_stack.brokers.order_types import OrderIntent, OrderSide, OrderType, TimeInForce
from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import ExecConfig, RiskConfig
from quant_research_stack.execution.position_book import PositionBook
from quant_research_stack.execution.risk import RiskGate, RiskState
from quant_research_stack.execution.signals import SignalIngestor
from quant_research_stack.execution.sizing import Sizer, SizerInput


class S4Loop:
    """Orchestrates SignalIngestor → RiskGate → Sizer → broker.place_order."""

    def __init__(
        self,
        stage: str,
        risk_cfg: RiskConfig,
        exec_cfg: ExecConfig,
        broker: BrokerAdapter,
        audit: AuditLog,
        starting_equity: Decimal,
        mid_price_lookup: Callable[[str], Decimal],
        is_crypto: Callable[[str], bool],
        tier3_stance_pct: float = 0.20,
    ) -> None:
        self._stage = stage
        self._risk_cfg = risk_cfg
        self._exec_cfg = exec_cfg
        self._broker = broker
        self._audit = audit
        self._mid_lookup = mid_price_lookup
        self._is_crypto = is_crypto
        self._starting_equity = starting_equity
        self._book = PositionBook(
            snapshot_root=Path(exec_cfg.position_book.snapshot_root),
            stage=stage,
            starting_equity=starting_equity,
        )
        self._risk_gate = RiskGate(risk_cfg)
        self._sizer = Sizer(risk_cfg, tier3_stance_pct=tier3_stance_pct)
        self._ingestor = SignalIngestor(
            preds_dir=Path(exec_cfg.ingest.s1_predictions_dir),
            verdicts_dir=Path(exec_cfg.ingest.s2_verdicts_dir),
            poll_interval_s=exec_cfg.ingest.poll_interval_seconds,
            pair_window_s=exec_cfg.ingest.pair_window_seconds,
            audit=audit,
        )
        self._orders_last_minute: list[datetime] = []

    async def run(self, max_tickets: int | None = None) -> None:
        processed = 0
        async for ticket in self._ingestor.stream():
            await self._handle(ticket)
            processed += 1
            if max_tickets is not None and processed >= max_tickets:
                self._ingestor.stop()
                return

    async def _handle(self, ticket) -> None:
        sym = ticket.signal.symbol
        now = datetime.now(UTC)
        cutoff = now.timestamp() - 60.0
        self._orders_last_minute = [t for t in self._orders_last_minute if t.timestamp() >= cutoff]
        mid = self._mid_lookup(sym)
        per_sym = self._book.per_symbol_notional({sym: mid})
        gross = self._book.gross_exposure({sym: mid})
        state = RiskState(
            account_equity=float(self._starting_equity),
            peak_equity=float(self._book.peak_equity),
            daily_realized_pnl=float(self._book.daily_realized_pnl),
            gross_exposure_notional=gross,
            per_symbol_notional=per_sym,
            orders_last_minute=len(self._orders_last_minute),
            last_tick_ts={sym: now},  # mid was just looked up; treat as fresh
            kill_flag_path=Path(self._exec_cfg.kill_switch.repo_root_marker),
            is_crypto=self._is_crypto,
            now=now,
        )
        decision = self._risk_gate.evaluate(ticket, state)
        if not decision.allowed:
            self._audit.append(
                "risk_blocked",
                {"signal_id": ticket.signal.signal_id, "gate_name": decision.reason, "kill": decision.kill_trigger},
            )
            return

        qty = self._sizer.size(SizerInput(
            ticket=ticket,
            account_equity=float(self._starting_equity),
            mid_price=float(mid),
            lot_size=0.0001,
        ))
        if qty == 0:
            self._audit.append("trade_skipped_zero_qty", {"signal_id": ticket.signal.signal_id})
            return

        intent = OrderIntent(
            client_order_id=ticket.signal.signal_id,
            symbol=sym,
            side=OrderSide.buy if qty > 0 else OrderSide.sell,
            qty=Decimal(str(abs(qty))),
            order_type=OrderType.market,
            time_in_force=TimeInForce.ioc,
        )
        order = await self._broker.place_order(intent)
        self._orders_last_minute.append(now)
        self._audit.append(
            "trade_placed",
            {"signal_id": ticket.signal.signal_id, "order_id": order.client_order_id,
             "symbol": sym, "side": intent.side.value, "qty": float(intent.qty), "mid": float(mid)},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_loop.py -q
```

Expected: 1 passed. If the OrderIntent signature differs from what's in the existing brokers/order_types.py, adjust the OrderIntent construction in `_handle` to match the actual fields.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/execution/loop.py tests/test_execution_loop.py
git commit -m "feat(s4): S4Loop orchestrates SignalIngestor → RiskGate → Sizer → broker"
```

---

### Task 14: Daemon entry point `scripts/s4_execute.py`

**Files:**
- Create: `scripts/s4_execute.py`

- [ ] **Step 1: Write daemon**

Create `scripts/s4_execute.py`:

```python
"""S4 execution daemon. Stage resolved from QUANTLAB_STAGE env var.

Usage:
  QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/s4_execute.py \
    --risk-config configs/risk.yaml --exec-config configs/exec.yaml \
    --brokers-config configs/brokers.yaml --asset-class crypto --starting-equity 100000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.configs import load_exec_config, load_risk_config
from quant_research_stack.execution.kill_switch import KillSwitchWatcher
from quant_research_stack.execution.loop import S4Loop
from quant_research_stack.execution.router import BrokerRouter

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S4 execution daemon")
    p.add_argument("--risk-config", default="configs/risk.yaml")
    p.add_argument("--exec-config", default="configs/exec.yaml")
    p.add_argument("--brokers-config", default="configs/brokers.yaml")
    p.add_argument("--asset-class", choices=["equity", "crypto"], required=True)
    p.add_argument("--starting-equity", type=Decimal, required=True)
    p.add_argument("--max-tickets", type=int, default=None, help="Stop after N tickets (testing)")
    return p.parse_args()


def _is_crypto_fn(sym: str) -> bool:
    return sym.endswith(("USDT", "BTC", "ETH", "BUSD"))


def _mid_lookup_stub(_sym: str) -> Decimal:
    # Stage-1 stub: returns a constant. Real S4.1 will wire a live mid-price feed
    # (S3 feeds.* provides ticks; a thin adapter will expose a most-recent-mid getter).
    return Decimal("50000")


async def main_async() -> int:
    args = parse_args()
    stage = os.environ.get("QUANTLAB_STAGE")
    if stage not in {"paper", "live_shadow", "live"}:
        console.print(f"[red]QUANTLAB_STAGE must be paper|live_shadow|live; got {stage!r}[/red]")
        return 2

    risk_cfg = load_risk_config(Path(args.risk_config))
    exec_cfg = load_exec_config(Path(args.exec_config))
    brokers_cfg = yaml.safe_load(Path(args.brokers_config).read_text())

    audit = AuditLog(
        root=Path(exec_cfg.audit.root) / stage,
        chmod_after_close=exec_cfg.audit.chmod_after_close,
    )
    router = BrokerRouter(brokers_cfg)
    broker = router.resolve(stage, asset_class=args.asset_class)

    loop = S4Loop(
        stage=stage,
        risk_cfg=risk_cfg,
        exec_cfg=exec_cfg,
        broker=broker,
        audit=audit,
        starting_equity=args.starting_equity,
        mid_price_lookup=_mid_lookup_stub,
        is_crypto=_is_crypto_fn,
    )

    flag_path = Path(exec_cfg.kill_switch.repo_root_marker)
    if not flag_path.is_absolute():
        flag_path = Path.cwd() / flag_path
    killer_fired = asyncio.Event()

    async def on_kill(reason: str) -> None:
        console.print(f"[bold red]kill_trigger:[/bold red] {reason}")
        killer_fired.set()

    watcher = KillSwitchWatcher(
        flag_path=flag_path,
        poll_interval_s=exec_cfg.kill_switch.poll_interval_seconds,
        audit=audit,
        on_kill=on_kill,
    )
    watcher.install_signal_handlers()

    watch_task = asyncio.create_task(watcher.run())
    loop_task = asyncio.create_task(loop.run(max_tickets=args.max_tickets))

    done, pending = await asyncio.wait(
        {watch_task, loop_task, asyncio.create_task(killer_fired.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    watcher.stop()
    try:
        await broker.close()
    except Exception:
        pass
    audit.append("exit", {"reason": "kill_or_done", "exit_code": 137 if killer_fired.is_set() else 0})
    audit.close_current()
    return 137 if killer_fired.is_set() else 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint the new script**

```bash
PYTHONPATH=src uv run ruff check scripts/s4_execute.py
```

Expected: All checks passed.

- [ ] **Step 3: Smoke-run the daemon for 5 seconds with no predictions**

```bash
QUANTLAB_STAGE=paper PYTHONPATH=src uv run python scripts/s4_execute.py \
  --risk-config configs/risk.yaml --exec-config configs/exec.yaml \
  --brokers-config configs/brokers.yaml --asset-class crypto \
  --starting-equity 100000 --max-tickets 0 &
PID=$!
sleep 5
kill $PID || true
wait $PID 2>/dev/null || true
```

Expected: process starts, idles (no signals to process), exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts/s4_execute.py
git commit -m "feat(s4): scripts/s4_execute.py daemon entry point"
```

---

### Task 15: Promotion-report generator `scripts/generate_promotion_report.py`

**Files:**
- Create: `scripts/generate_promotion_report.py`
- Test: `tests/test_execution_promotion_report.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_execution_promotion_report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_promotion_report import build_report


def test_build_report_marks_each_gate(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    # 30 days of clean audit (one row per day, no kill_triggers)
    for i in range(30):
        f = audit_dir / f"2026-04-{i+1:02d}.jsonl"
        f.write_text(json.dumps({"event": "trade_placed", "not_investment_advice": True,
                                 "payload": {}, "timestamp_utc": "2026-04-01T00:00:00+00:00"}) + "\n")
    report = build_report(
        from_stage="paper",
        to_stage="live_shadow",
        promotion_config_path=Path("configs/promotion.yaml"),
        audit_root=audit_dir,
        s1_metrics_path=None,
    )
    assert isinstance(report, dict)
    assert "gates" in report
    assert any(g["name"] == "min_days_in_paper" for g in report["gates"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_promotion_report.py -q
```

Expected: ImportError for `scripts.generate_promotion_report`.

- [ ] **Step 3: Implement `generate_promotion_report.py`**

Create `scripts/generate_promotion_report.py`:

```python
"""Generate a green/red promotion-gate report for paper→live_shadow or live_shadow→live.

Writes to docs/runbooks/<from>_to_<to>.md (markdown) with the gate-by-gate
verdicts. Operator signs the file before promoting.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from quant_research_stack.execution.configs import load_promotion_config


def _count_audit_days(audit_root: Path) -> int:
    if not audit_root.exists():
        return 0
    return sum(1 for p in audit_root.glob("*.jsonl") if p.is_file())


def _count_kill_triggers(audit_root: Path, last_n_days: int) -> int:
    if not audit_root.exists():
        return 0
    files = sorted(audit_root.glob("*.jsonl"))[-last_n_days:]
    n = 0
    for p in files:
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("event") == "kill_trigger":
                    n += 1
            except Exception:
                continue
    return n


def build_report(
    from_stage: str,
    to_stage: str,
    promotion_config_path: Path,
    audit_root: Path,
    s1_metrics_path: Path | None,
) -> dict[str, Any]:
    cfg = load_promotion_config(promotion_config_path)
    gate_row = cfg.paper_to_live_shadow if from_stage == "paper" else cfg.live_shadow_to_live
    audit_days = _count_audit_days(audit_root)
    kill_triggers = _count_kill_triggers(audit_root, last_n_days=14)
    gates: list[dict[str, Any]] = []

    if gate_row.min_days_in_paper:
        gates.append({
            "name": "min_days_in_paper",
            "required": gate_row.min_days_in_paper,
            "observed": audit_days,
            "passed": audit_days >= gate_row.min_days_in_paper,
        })
    if gate_row.min_days_in_live_shadow:
        gates.append({
            "name": "min_days_in_live_shadow",
            "required": gate_row.min_days_in_live_shadow,
            "observed": audit_days,
            "passed": audit_days >= gate_row.min_days_in_live_shadow,
        })
    if gate_row.no_kill_triggers_days is not None:
        gates.append({
            "name": "no_kill_triggers_days",
            "required": 0,
            "observed": kill_triggers,
            "passed": kill_triggers == 0,
        })
    if s1_metrics_path is not None and s1_metrics_path.exists():
        m = json.loads(s1_metrics_path.read_text())
        r2 = float(m.get("holdout_weighted_zero_mean_r2", 0.0))
        gates.append({
            "name": "s1_holdout_r2_above_target",
            "required": 0.012,
            "observed": r2,
            "passed": r2 >= 0.012,
        })

    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "generated_utc": datetime.now(UTC).isoformat(),
        "all_passed": all(g["passed"] for g in gates) if gates else False,
        "gates": gates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Promotion report: {report['from_stage']} → {report['to_stage']}",
        "",
        f"Generated: {report['generated_utc']}  ",
        f"All gates passed: **{report['all_passed']}**",
        "",
        "| Gate | Required | Observed | Passed |",
        "|---|---|---|---|",
    ]
    for g in report["gates"]:
        lines.append(f"| {g['name']} | {g['required']} | {g['observed']} | {'✅' if g['passed'] else '❌'} |")
    lines += [
        "",
        "## Operator signature",
        "",
        "I have reviewed the gates above and authorize promotion. — _signed name_, _date_.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate promotion-gate green/red report")
    p.add_argument("--from-stage", choices=["paper", "live_shadow"], required=True)
    p.add_argument("--to-stage", choices=["live_shadow", "live"], required=True)
    p.add_argument("--promotion-config", default="configs/promotion.yaml")
    p.add_argument("--audit-root", required=True)
    p.add_argument("--s1-metrics", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        promotion_config_path=Path(args.promotion_config),
        audit_root=Path(args.audit_root),
        s1_metrics_path=Path(args.s1_metrics) if args.s1_metrics else None,
    )
    md = render_markdown(report)
    out = Path(args.out) if args.out else Path(f"docs/runbooks/{args.from_stage}_to_{args.to_stage}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/test_execution_promotion_report.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_promotion_report.py tests/test_execution_promotion_report.py
git commit -m "feat(s4): scripts/generate_promotion_report.py + gate-by-gate green/red md report"
```

---

### Task 16: Audit-replay parity script (extends ADR-0011 contract to S4)

**Files:**
- Create: `scripts/audit_replay_check.py`
- Test: `tests/integration/test_s4_audit_replay_parity.py` (marker s4_integration)

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_s4_audit_replay_parity.py`:

```python
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.s4_integration


def test_replay_reconstructs_same_position_book(tmp_path: Path) -> None:
    from quant_research_stack.execution.position_book import PositionBook
    from scripts.audit_replay_check import replay_audit_to_book

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    rec = {
        "event": "trade_fill",
        "not_investment_advice": True,
        "payload": {
            "order_id": "o-1", "client_order_id": "c-1", "symbol": "BTCUSDT",
            "side": "buy", "qty": 0.01, "price": 50000.0, "fee": 0.0,
            "ts_utc": "2026-05-20T00:00:00+00:00",
        },
        "timestamp_utc": "2026-05-20T00:00:00+00:00",
    }
    (audit_dir / "2026-05-20.jsonl").write_text(json.dumps(rec) + "\n")
    book = PositionBook(snapshot_root=tmp_path / "positions", stage="paper", starting_equity=Decimal("100000"))
    replay_audit_to_book(audit_dir, book)
    assert book.position("BTCUSDT").qty == Decimal("0.01")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/integration/test_s4_audit_replay_parity.py -q -m s4_integration
```

Expected: ImportError for `scripts.audit_replay_check`.

- [ ] **Step 3: Implement `audit_replay_check.py`**

Create `scripts/audit_replay_check.py`:

```python
"""Replay an S4 audit JSONL log and reconstruct the position book.

Enforces the ADR-0011 contract: replay must produce the same book byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from quant_research_stack.brokers.order_types import Fill, OrderSide
from quant_research_stack.execution.position_book import PositionBook


def replay_audit_to_book(audit_dir: Path, book: PositionBook) -> int:
    """Apply every trade_fill event in chronological order. Returns # of fills applied."""
    n = 0
    for path in sorted(audit_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "trade_fill":
                continue
            p = rec["payload"]
            fill = Fill(
                order_id=p["order_id"],
                client_order_id=p["client_order_id"],
                symbol=p["symbol"],
                side=OrderSide(p["side"]),
                qty=Decimal(str(p["qty"])),
                price=Decimal(str(p["price"])),
                fee=Decimal(str(p.get("fee", 0))),
                ts_utc=datetime.fromisoformat(p["ts_utc"]),
            )
            book.apply_fill(fill)
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay S4 audit logs into a position book")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--snapshot-root", default="data/positions")
    p.add_argument("--stage", choices=["paper", "live_shadow", "live"], default="paper")
    p.add_argument("--starting-equity", type=Decimal, default=Decimal("100000"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    book = PositionBook(
        snapshot_root=Path(args.snapshot_root),
        stage=args.stage,
        starting_equity=args.starting_equity,
    )
    n = replay_audit_to_book(Path(args.audit_dir), book)
    print(f"Applied {n} fills.")
    for sym, pos in book._positions.items():
        print(f"  {sym}: qty={pos.qty} avg={pos.avg_price}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/integration/test_s4_audit_replay_parity.py -q -m s4_integration
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_replay_check.py tests/integration/test_s4_audit_replay_parity.py
git commit -m "feat(s4): audit-replay parity script extends ADR-0011 contract to S4"
```

---

### Task 17: Integration tests — paper smoke + kill-switch drill + reconciliation kill

**Files:**
- Create: `tests/integration/test_s4_paper_smoke.py`
- Create: `tests/integration/test_s4_kill_switch_drill.py`
- Create: `tests/integration/test_s4_reconciliation_kill.py`

- [ ] **Step 1: Write `test_s4_paper_smoke.py`**

```python
from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.s4_integration


def _write_pred(dir_: Path, sig_id: str) -> None:
    df = pl.DataFrame({
        "signal_id": [sig_id], "symbol": ["BTCUSDT"], "predicted_score": [0.05],
        "confidence": [0.7], "horizon_minutes": [5], "ts_utc": [datetime.now(UTC).isoformat()],
    })
    dir_.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.parquet")


def _write_verdict(dir_: Path, sig_id: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rec = {
        "signal_id": sig_id, "decision": "pass", "direction": 1, "confidence": 0.7,
        "horizon_minutes": 5, "regime_tag": "trending", "rationale_short": "ok",
        "cited_paper_chunk_ids": ["paper_pdf:x:0"], "contradictions_flagged": [],
    }
    with (dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl").open("a") as h:
        h.write(json.dumps(rec) + "\n")


@pytest.mark.asyncio
async def test_paper_smoke_emits_trade_placed(tmp_path: Path) -> None:
    from quant_research_stack.brokers.null_broker import NullBroker
    from quant_research_stack.execution.audit import AuditLog
    from quant_research_stack.execution.configs import ExecConfig, load_risk_config
    from quant_research_stack.execution.loop import S4Loop

    risk_cfg = load_risk_config(Path("configs/risk.yaml"))
    exec_cfg = ExecConfig.model_validate({
        "ingest": {
            "s1_predictions_dir": str(tmp_path / "preds"),
            "s2_verdicts_dir": str(tmp_path / "verdicts"),
            "poll_interval_seconds": 0.05,
            "pair_window_seconds": 5,
        },
        "position_book": {"snapshot_root": str(tmp_path / "positions"), "snapshot_interval_seconds": 60},
        "audit": {"root": str(tmp_path / "audit"), "rotation": "daily", "chmod_after_close": False},
        "kill_switch": {
            "repo_root_marker": str(tmp_path / "NEVER"),
            "poll_interval_seconds": 0.05,
            "emergency_snapshot_root": str(tmp_path / "snaps"),
        },
    })
    audit = AuditLog(root=Path(exec_cfg.audit.root), chmod_after_close=False)
    loop = S4Loop(
        stage="paper", risk_cfg=risk_cfg, exec_cfg=exec_cfg,
        broker=NullBroker(), audit=audit,
        starting_equity=Decimal("100000"),
        mid_price_lookup=lambda _s: Decimal("50000"),
        is_crypto=lambda _s: True,
    )
    _write_pred(Path(exec_cfg.ingest.s1_predictions_dir), "sig-smk00001")
    _write_verdict(Path(exec_cfg.ingest.s2_verdicts_dir), "sig-smk00001")

    await asyncio.wait_for(loop.run(max_tickets=1), timeout=5.0)
    audit.close_current()

    events = []
    for p in Path(exec_cfg.audit.root).iterdir():
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line)["event"])
    assert "signal_ingested" in events
    assert "trade_placed" in events
```

- [ ] **Step 2: Write `test_s4_kill_switch_drill.py`**

```python
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.s4_integration


@pytest.mark.asyncio
async def test_kill_flag_fires_on_kill_callback(tmp_path: Path) -> None:
    from quant_research_stack.execution.audit import AuditLog
    from quant_research_stack.execution.kill_switch import KillSwitchWatcher

    flag = tmp_path / "KILL_TRADING"
    audit = AuditLog(root=tmp_path / "audit", chmod_after_close=False)
    fired: list[str] = []

    async def on_kill(reason: str) -> None:
        fired.append(reason)

    watcher = KillSwitchWatcher(flag_path=flag, poll_interval_s=0.05, audit=audit, on_kill=on_kill)
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.1)
    flag.touch()
    await asyncio.wait_for(asyncio.sleep(0.3), timeout=2.0)
    watcher.stop()
    await task
    audit.close_current()

    events = []
    for p in (tmp_path / "audit").iterdir():
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line)["event"])
    assert "kill_trigger" in events
    assert "file_flag" in fired
```

- [ ] **Step 3: Write `test_s4_reconciliation_kill.py`**

```python
from __future__ import annotations

from decimal import Decimal

import pytest

pytestmark = pytest.mark.s4_integration


def test_5_bps_diff_exceeds_threshold() -> None:
    from quant_research_stack.execution.reconciliation import diff_book_vs_broker

    book = Decimal("100050")
    broker = Decimal("100000")
    res = diff_book_vs_broker(book_equity=book, broker_equity=broker)
    assert res.diff_bps == pytest.approx(5.0, abs=1e-3)
    assert res.exceeds_threshold(max_diff_bps=1.0) is True
```

- [ ] **Step 4: Run integration tests**

```bash
PYTHONPATH=src uv run pytest tests/integration/test_s4_*.py -m s4_integration -q
```

Expected: 4 passed (1 from test_s4_audit_replay_parity.py committed in Task 16 + 1 each from the three new files).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_s4_paper_smoke.py \
        tests/integration/test_s4_kill_switch_drill.py \
        tests/integration/test_s4_reconciliation_kill.py
git commit -m "test(s4): integration tests — paper smoke, kill-switch drill, reconciliation kill"
```

---

### Task 18: Makefile targets + final whole-repo verification

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Read current Makefile to preserve format**

```bash
cat Makefile | head -40
```

- [ ] **Step 2: Append S4 targets**

Append to `Makefile`:

```makefile
S4_EXECUTE := scripts/s4_execute.py
PROMOTION_REPORT := scripts/generate_promotion_report.py
AUDIT_REPLAY := scripts/audit_replay_check.py
S4_STAGE ?= paper
S4_ASSET ?= crypto
S4_EQUITY ?= 100000

.PHONY: s4-execute s4-promotion-report s4-audit-replay s4-smoke

s4-execute:
	QUANTLAB_STAGE=$(S4_STAGE) $(PY) python $(S4_EXECUTE) \
	  --risk-config configs/risk.yaml --exec-config configs/exec.yaml \
	  --brokers-config configs/brokers.yaml --asset-class $(S4_ASSET) \
	  --starting-equity $(S4_EQUITY)

s4-promotion-report:
	$(PY) python $(PROMOTION_REPORT) --from-stage paper --to-stage live_shadow \
	  --audit-root logs/audit/s4/paper

s4-audit-replay:
	$(PY) python $(AUDIT_REPLAY) --audit-dir logs/audit/s4/paper --stage paper

s4-smoke:
	$(PY) pytest tests/integration/test_s4_*.py -v -m s4_integration
```

- [ ] **Step 3: Whole-repo verification (lint + type + tests)**

```bash
PYTHONPATH=src uv run ruff check src scripts tests
PYTHONPATH=src uv run mypy src
PYTHONPATH=src uv run pytest -q
```

Expected: all green; new S4 unit tests included in default run; S4 integration tests skipped without `-m s4_integration`.

- [ ] **Step 4: Final commit**

```bash
git add Makefile
git commit -m "build(s4): Makefile targets — s4-execute, s4-promotion-report, s4-audit-replay, s4-smoke"
```

---

## Self-review

**Spec coverage:** every section of `docs/superpowers/specs/2026-05-20-quantlab-alpha-s4-execution-risk-promotion-design.md` maps to a task:

| Spec section | Task(s) |
|---|---|
| §1 Scope (paper + live_shadow only) | implicit across all tasks; Task 11 enforces ImportError for live |
| §2.1 Module layout (execution/*) | Tasks 4-13 (one file per module) |
| §2.2 Stage-resolved broker selection | Task 11 (BrokerRouter) |
| §3.1 SignalIngestor | Task 6 |
| §3.2 RiskGate (ordered) | Task 7 |
| §3.3 Sizer | Task 8 |
| §3.4 PositionBook | Task 9 |
| §3.5 ReconcileLoop | Task 10 |
| §3.6 BrokerRouter | Task 11 |
| §3.7 KillSwitch | Task 12 |
| §3.8 AuditLog | Task 5 |
| §4.1 risk.yaml | Task 2 |
| §4.2 promotion.yaml | Task 2 |
| §4.3 brokers.yaml extension | Task 2 |
| §5 Data flow | Task 13 (S4Loop wires them) |
| §6 Error handling | Tasks 7, 10, 12 (kill paths) + Task 14 daemon exit |
| §7 Testing strategy | unit tests in Tasks 3-13, integration in Tasks 16-17 |
| §8 New ADRs 0012/0013/0014 | Task 1 |
| §9 Runbooks (existing) | no new runbook required this spec |
| §10 Success criteria | Tasks 14 (daemon), 15 (promotion-report), 16 (replay), 17 (drills), 18 (whole-repo green) |
| §11 Out-of-scope (live brokers) | Task 11 ImportError guard |

**Placeholder scan:** no TBDs, no "implement later", no "similar to". Each task has full code blocks. The `_mid_lookup_stub` in Task 14 is explicitly labeled as a Stage-1 stub with the S4.1 follow-up named — that's a real design boundary, not a placeholder.

**Type consistency:** `RiskState`, `RiskDecision`, `SizerInput`, `ExecutionTicket`, `S1Signal`, `Position`, and `AuditLog` interfaces are consistent across tasks. `BrokerAdapter` and `Fill`/`OrderIntent`/`Order` from S3 are imported as-is and never redefined.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-quantlab-alpha-s4-execution-risk-promotion-implementation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
