# QuantLab Alpha — S4: Execution + Risk + Promotion Gates (Design)

**Date:** 2026-05-20
**Status:** approved (operator-approved via brainstorming, awaiting spec review)
**Predecessors:** S1 (predictions), S2 (verdicts), S3 (broker abstraction)
**Successor:** S4.1 (live broker implementations — requires two-person review per CLAUDE.md §1.13)

## 0. Goal

S4 is the trading layer. It consumes S1's numeric predictions and S2's structured verdicts, applies risk gates, sizes positions, routes orders to a stage-resolved broker, maintains an in-memory position book reconciled against the broker every minute, and writes every decision to an append-only audit log that can be replayed deterministically.

Three operating stages, set by the `QUANTLAB_STAGE` environment variable and never changed in-process (per CLAUDE.md §1.7 and §11):

```text
paper        → brokers/*_paper.py (Alpaca paper, Binance testnet)
live_shadow  → brokers/null_broker.py for writes + read-only account API for reconciliation
live         → brokers/*_live.py  (OUT OF SCOPE — see §11 below; ImportError-guarded)
```

S4 must reach the same operational discipline as S2 and S3: every decision is auditable, every gate is configured in YAML and checked on startup, no kill condition can be defeated by a separate check raising first.

## 1. Scope

**In scope:**
- Long-running execution daemon (`scripts/s4_execute.py`)
- Risk gate with ordered pre-trade checks (kill flag, freshness, drawdown, exposure, rate limit, governor decision)
- Confidence-scaled position sizing with hard caps
- In-memory position book with per-minute Parquet snapshots
- 60-second broker reconciliation with kill-on-divergence
- Kill switch (file flag + SIGTERM/SIGINT) with graceful close-out
- Promotion-gate logic (paper → live_shadow → live) and report generator
- `configs/risk.yaml` (caps + DD + freshness)
- `configs/promotion.yaml` (transition gates)
- Extension to `configs/brokers.yaml` (stage→broker routing)
- Audit log append (extends the JSONL+chmod-a-w pattern from S2/S3)

**Out of scope (deferred to S4.1):**
- `brokers/alpaca_live.py` and `brokers/binance_live.py` — CLAUDE.md §1.13 requires two-person review; these belong in a separate spec/plan with an operator-signed promotion report.
- L2 order-book backtesting (S3.3 — already filed as task #89).
- Multi-account portfolio routing.
- OCO / bracket order child-order accounting (S3 supports them via the `OrderType` enum, but S4 sizing only emits market and limit orders in this spec).
- Smart order routing across venues.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       scripts/s4_execute.py                          │
│                       (entry point, signal-loop)                     │
│                                                                       │
│   ┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐ │
│   │ SignalIngestor  │ ─→ │   RiskGate     │ ─→ │   Sizer         │ │
│   │ (tails S1 parq +│    │ (caps, DD,     │    │ (confidence ×   │ │
│   │  S2 verdict     │    │  freshness,    │    │  stance × cap)  │ │
│   │  JSONL files)   │    │  kill-flag)    │    │                 │ │
│   └─────────────────┘    └────────────────┘    └────────┬────────┘ │
│                                                          │           │
│   ┌─────────────────┐    ┌────────────────┐    ┌────────▼────────┐ │
│   │  PositionBook   │ ←─ │ ReconcileLoop  │    │  BrokerRouter   │ │
│   │ (in-mem + 1-min │    │ (every 60s,    │    │ (S3 BrokerAdapter│ │
│   │  parquet snap)  │    │  diff > 1bp →  │    │  resolved at    │ │
│   │                 │    │  kill)         │    │  startup by     │ │
│   └────────┬────────┘    └────────────────┘    │ QUANTLAB_STAGE) │ │
│            │                                    └────────┬────────┘ │
│            └────────────────┬───────────────────────────┘           │
│                             │                                        │
│                             ▼                                        │
│                  ┌────────────────────────┐                          │
│                  │ AuditLog (JSONL append,│                          │
│                  │ chmod a-w on rotate)   │                          │
│                  └────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Module layout

```text
src/quant_research_stack/execution/
  __init__.py
  signals.py         # SignalIngestor — tails predictions.parquet + verdict JSONL
  risk.py            # RiskGate — ordered pre-trade checks
  sizing.py          # Sizer — confidence × stance × cap → qty
  position_book.py   # In-memory book + per-minute Parquet snapshot + replay
  reconciliation.py  # 60s broker diff check; triggers kill on mismatch
  router.py          # BrokerRouter — resolves a BrokerAdapter via QUANTLAB_STAGE
  kill_switch.py     # File-flag watcher + SIGTERM/SIGINT handler + force-close
  loop.py            # Async orchestration glue
```

Test mirror: `tests/test_execution_*.py` for unit tests, `tests/integration/test_s4_*.py` for the smoke + drill tests under the `s4_integration` pytest marker.

### 2.2 Stage-resolved broker selection

`BrokerRouter.resolve(stage, asset_class)` reads `configs/brokers.yaml` and returns a concrete `BrokerAdapter` instance. The mapping (CLAUDE.md §11):

```text
paper         → alpaca_paper        (equity)
              → binance_testnet     (crypto)
live_shadow   → null_broker         (writes)
              + alpaca_paper        (read-only equity account for reconciliation)
              + binance_testnet     (read-only crypto account for reconciliation)
live          → alpaca_live         (OUT OF SCOPE; ImportError-guarded)
              → binance_live        (OUT OF SCOPE)
```

In `live_shadow`, every `place_order` goes to `null_broker` (no real order), but the reconciliation loop still queries the real (paper) account API to track what positions a live trader would have held. This produces the operational evidence required by the `paper_to_live_shadow_to_live` promotion gates.

## 3. Components (detailed)

### 3.1 SignalIngestor

**Responsibility:** Produce `(S1Signal, GovernorVerdict)` pairs.

**Inputs:**
- S1 predictions: tailed from a directory configured in `configs/exec.yaml` (default `data/live/s1_predictions/<YYYY-MM-DD>.parquet`). The current `experiments/alpha_s1/<run_id>/predictions.parquet` is a backtest artifact with `split ∈ {oof, holdout}`; the daemon does NOT trade off backtest predictions. Live-inference predictions are produced by a future S1-inference path (out of scope for S4 but planned in S4-adjacent work); for this spec we define the schema the daemon expects: `signal_id, symbol, predicted_score, confidence, horizon_minutes, ts_utc`.
- S2 verdicts: `experiments/s2_verdicts_<stage>/<date>.jsonl` (the `primary_verdicts_dir` from `configs/governor.yaml`).

**Algorithm:** poll both files every 100 ms (configurable). For each new prediction row, wait up to 60 s for the corresponding `GovernorVerdict` with matching `signal_id`. If the verdict doesn't arrive within the window, audit `event: "verdict_timeout"` and drop the signal (do not trade without a verdict).

**Output:** an `asyncio.Queue[ExecutionTicket]` where `ExecutionTicket = (S1Signal, GovernorVerdict, ingested_at)`.

### 3.2 RiskGate

**Responsibility:** Block the order, or let it pass to the Sizer.

**Ordered checks (sequence matters):**
1. `kill_flag_check()` — checks `KILL_TRADING` file presence at repo root. If present, immediately raises `KillSwitchTriggered`. **MUST BE FIRST** (ADR-0014).
2. `feed_freshness_check()` — confirms last tick on the symbol's venue is within `freshness.{crypto,equity}_max_gap_seconds`. Failing this is a kill condition (per CLAUDE.md §11), not just a skip.
3. `drawdown_check()` — daily realized < `drawdown.daily_realized_dd_kill_pct`; cumulative from peak < `drawdown.cumulative_dd_kill_pct`. Either breach is a kill.
4. `exposure_check()` — would this order push gross exposure > `limits.max_gross_exposure_pct` or per-symbol > `limits.max_per_symbol_pct`? If so, skip (audit `exposure_blocked`).
5. `rate_limit_check()` — orders in the last 60 s < `limits.max_orders_per_minute`? If not, skip (audit `rate_limit_blocked`).
6. `governor_decision_check()` — `verdict.decision == Decision.pass_`? If not (`veto` or `insufficient_evidence`), skip (audit `governor_blocked`).

All checks emit structured audit events. Kill checks also write a `kill_trigger` row and signal the daemon to shut down.

### 3.3 Sizer

The `GovernorVerdict` schema (defined in S2) does not carry a numeric `stance_modifier` field — instead, the governor config holds `stance.tier3_stance_modifier_pct` (default 0.20 per the existing `configs/governor.yaml`). The Sizer derives the modifier from the most recent Tier-3 verdict's relationship to the primary verdict:

```python
def _stance_modifier(primary: GovernorVerdict, tier3: GovernorVerdict | None, cfg_pct: float) -> float:
    if tier3 is None or tier3.decision == Decision.insufficient_evidence:
        return 0.0
    if tier3.decision == Decision.veto:
        return -cfg_pct        # tier 3 said no → shrink
    if tier3.direction == primary.direction:
        return +cfg_pct        # tier 3 agrees → boost
    return -cfg_pct            # tier 3 disagrees on direction → shrink
```

**Sizing math (ADR-0012):**

```python
stance_mod = _stance_modifier(primary, tier3, cfg.stance.tier3_stance_modifier_pct)
target_notional = (
    account.equity
    * cfg.limits.base_notional_per_trade_pct
    * primary.confidence                  # in [0, 1]
    * (1.0 + stance_mod)                  # in [1 - cfg_pct, 1 + cfg_pct]
)
target_notional = min(
    target_notional,
    account.equity * cfg.limits.max_per_symbol_pct,
)
qty = primary.direction * target_notional / mid_price   # direction ∈ {-1, 0, 1}
qty = round_to_lot(qty, symbol.lot_size)
```

`Decision.veto` on the primary verdict short-circuits the Sizer to `qty = 0` before any math runs. `direction == 0` (neutral primary) also yields `qty = 0`.

### 3.4 PositionBook

**State:** in-memory `dict[symbol, Position]` plus a running daily realized PnL and a peak equity high-water mark.

**Persistence:** every 60 s and on every fill, write `data/positions/<stage>/<YYYY-MM-DD>.parquet` (one row per symbol per snapshot, with a `snapshot_ts_utc` column). Files are chmod-a-w on date rotation so historical snapshots are immutable.

**Startup recovery:**
1. Load the latest snapshot from `data/positions/<stage>/`.
2. Query `broker.account()` + `broker.positions()`.
3. Diff. If diff > `reconciliation.max_diff_bps`, refuse to start (operator must intervene).
4. Otherwise, broker is authoritative for the initial state; subsequent writes go through the in-memory book first.

### 3.5 ReconcileLoop

Async task running every `reconciliation.interval_seconds` (default 60). Computes:

```python
diff_bps = abs(book_equity - broker_equity) / broker_equity * 10000
```

If `diff_bps > reconciliation.max_diff_bps` (default 1.0), emit a `kill_trigger` audit event and raise `ReconciliationDivergence`. The daemon's top-level handler cancels open orders, snapshots, and exits with code 137 (matches the kill-switch contract).

### 3.6 BrokerRouter

Reads `configs/brokers.yaml.stage_routes[QUANTLAB_STAGE]`. Resolves the equity and crypto routes, then imports the corresponding modules lazily. If `live` is configured but the live module isn't present, raises `ImportError("Live broker not installed; see S4.1")` at startup — `paper` and `live_shadow` are never affected.

### 3.7 KillSwitch

Three triggers (CLAUDE.md §11):
- Touch `KILL_TRADING` at the repo root — detected within 100 ms by the file watcher.
- SIGTERM / SIGINT — installed handler runs the same close-out path.
- Internal `kill_trigger` event raised by any RiskGate or ReconcileLoop check.

Close-out sequence (must be the same for all three triggers):
1. Cancel all open orders on the active broker (`broker.cancel_order` for each known order ID).
2. Take a final position snapshot to `data/snapshots/kill_<utc_iso>.parquet`.
3. Append `kill_trigger` audit row (reason + stage + UTC timestamp).
4. `sys.exit(137)`.

### 3.8 AuditLog

Reuses S2's transport (`logs/audit/s4/<YYYY-MM-DD>.jsonl`, chmod a-w on rotation). Event taxonomy:

```text
signal_ingested        (signal_id, symbol, prediction, confidence)
verdict_received       (signal_id, verdict)
verdict_timeout        (signal_id, waited_seconds)
risk_blocked           (signal_id, gate_name, reason)
trade_placed           (signal_id, verdict_id, order_id, intent, sized_qty, sized_notional)
trade_fill             (order_id, fill_qty, fill_price, fees)
position_snapshot      (snapshot_ts, equity, gross_exposure_pct)
reconciliation_diff    (diff_bps)
kill_trigger           (reason, stage, trigger_type)
exit                   (exit_code, reason)
```

Every event includes `not_investment_advice: true` (matches S2 transport).

## 4. Configs

### 4.1 `configs/risk.yaml` (new)

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

### 4.2 `configs/promotion.yaml` (new)

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

### 4.3 `configs/brokers.yaml` (extend)

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

## 5. Data flow (one signal end-to-end)

```text
1. S1 emits → predictions.parquet row appears (split=live)
2. S2 govern daemon emits → verdict JSONL row appears
3. SignalIngestor pairs them by signal_id within 60 s
4. RiskGate runs the ordered checks; any failure → audit + skip (or kill)
5. Sizer.size(signal, verdict, account) → OrderIntent
6. BrokerRouter.broker.place_order(intent) → Order
7. PositionBook.apply(order); on fill → apply(fill)
8. AuditLog.append(trade_placed); on fill → append(trade_fill)
9. Every 60 s: ReconcileLoop diffs book vs broker.account() + broker.positions()
   → diff > 1 bp → kill_trigger + sys.exit(137)
```

## 6. Error handling

| Failure | Response |
|---|---|
| `KILL_TRADING` file present | Cancel open orders → snapshot → audit `kill_trigger` → exit 137 |
| Broker API 5xx / network | S3 exponential backoff; max 3 retries; on exhaustion → kill_trigger + SIGTERM |
| Malformed verdict JSON | Audit `verdict_parse_error`; skip signal (do NOT trade) |
| Reconciliation diff > 1 bp | kill_trigger + SIGTERM; operator must investigate before restart |
| Position snapshot read fails on startup | Start empty, treat broker as source of truth, refuse to trade until first reconciliation passes |
| SIGTERM / SIGINT | Graceful close-out (cancel → snapshot → audit → exit 0) |
| Live broker module missing in `live` stage | `ImportError("Live broker not installed; see S4.1")` at startup; `paper` and `live_shadow` stages unaffected |

## 7. Testing strategy

### 7.1 Unit tests (default, in `tests/`)

- `test_execution_risk.py` — Every check fires/skips correctly; ordering proven (kill_flag must short-circuit before all others).
- `test_execution_sizing.py` — Confidence/stance/cap math; veto → qty 0; sign correctness; lot rounding; zero-price edge case.
- `test_execution_position_book.py` — `apply(order/fill)`, snapshot round-trip, equity computation, reconciliation diff math.
- `test_execution_kill_switch.py` — File flag detection, signal handlers, close-out ordering (cancel → snapshot → audit → exit).
- `test_execution_signals.py` — Pair-within-window, drop-stale, handle-malformed-verdict.
- `test_execution_router.py` — Stage → broker resolution; `live_shadow` wires null_broker for writes; missing live module raises ImportError at startup.

### 7.2 Integration tests (marker `s4_integration`, gated by env or `-m s4_integration`)

- `test_s4_paper_smoke.py` — Synthetic predictions.parquet + verdict JSONL → daemon for 60 s → assert orders on paper broker, position book matches broker, audit log contains expected events.
- `test_s4_kill_switch_drill.py` — Start daemon → touch `KILL_TRADING` → assert orders cancelled, snapshot written, audit `kill_trigger`, exit 137.
- `test_s4_reconciliation_kill.py` — Inject artificial 5 bp diff between book and broker → assert kill triggered.

### 7.3 Audit-replay parity (extends ADR 0011)

`scripts/audit_replay_check.py last-day` re-applies every audit event in order; the reconstructed final position book MUST equal the on-disk snapshot byte-for-byte. This is the contract that ties S4 to the rest of the platform's audit discipline.

## 8. New ADRs

- **ADR-0012** — Confidence-scaled position sizing with hard caps.
- **ADR-0013** — In-memory position book with 60 s broker reconciliation.
- **ADR-0014** — Kill-switch precedence (file flag check is the first gate in the RiskGate ordering).

## 9. Runbooks affected

Existing runbooks already cover S4's operational surface; this spec wires the code that implements them:

- `docs/runbooks/kill_switch.md` (existing) — file flag mechanism.
- `docs/runbooks/stage_promotion.md` (existing) — paper → live_shadow → live procedure; this spec adds `scripts/generate_promotion_report.py` which generates the green/red gate report.
- `docs/runbooks/incident_response.md` (existing) — DD breach, broker reconciliation mismatch, feed gap procedures.

No new runbooks are required for the in-scope work. **S4.1** (live brokers) will add `docs/runbooks/s4_1_live_broker_credentials.md` symmetric with the existing `s3_paper_broker_credentials.md`.

## 10. Success criteria

S4 is complete when:

1. `scripts/s4_execute.py` runs for 24 h in paper stage without unhandled exceptions.
2. `configs/risk.yaml` + `configs/promotion.yaml` validate via Pydantic on startup.
3. Reconciliation diff stays ≤ 1 bp for all 60 s windows in the 24 h run.
4. Kill-switch drill passes — operator touches `KILL_TRADING` → daemon exits 137 with open orders cancelled and a final snapshot persisted.
5. `scripts/audit_replay_check.py last-day` reconstructs the same position book.
6. `scripts/generate_promotion_report.py` produces a green/red row report and writes `docs/runbooks/<from>_to_<to>.md`.
7. `PYTHONPATH=src pytest -q` passes (including new S4 unit tests; s4_integration tests gated by marker).
8. `ruff check src scripts tests` passes.
9. `mypy src` passes.

## 11. Out-of-scope (S4.1)

The live broker implementations are explicitly deferred and protected by CLAUDE.md §1.13's two-person-review rule. They will arrive as:

- **`docs/superpowers/specs/<date>-quantlab-alpha-s4_1-live-brokers-design.md`** — full module-level design with credential handling, idempotency, rate-limit accounting, and an operator-signed promotion report template.
- **`docs/superpowers/plans/<date>-quantlab-alpha-s4_1-live-brokers-implementation.md`** — the implementation plan that produces `brokers/alpaca_live.py` and `brokers/binance_live.py`.

This spec does NOT include them, ImportError-guards the `live` route, and never imports them from any test or non-live code path (matches the existing rule in CLAUDE.md §4: "Any module that places real orders [...] is forbidden from being imported by tests or training code").
