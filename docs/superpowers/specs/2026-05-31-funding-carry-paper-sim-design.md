# Design — Funding-Carry Paper-Trading Simulation (S4 paper stage)

**Date:** 2026-05-31
**Status:** Approved design. Pre-implementation spec.
**Branch:** `s4-paper-trading-sim` (off `quant-llm-implementation` tip `97db912`). Do NOT
merge to `main` and do NOT open a PR until the operator says so.
**Author:** research engineer (Claude) under the QuantLab Alpha program discipline.

## 0. Goal & non-goals

**Goal.** Run the funding-carry delta-neutral strategy (long spot / short USDT-M perp,
BTC+ETH) forward through the existing S4 paper-execution engine, using the simulated
broker on **real public Binance mainnet data**, at `QUANTLAB_STAGE=paper` with the kill
switch armed. Produce (a) a tangible running paper bot + append-only audit log for the
team leader, and (b) a **live-vs-model reconciliation report** answering whether real
fills / funding / basis match our backtest cost-and-tail models.

**Non-goals / hard guardrails (program rules — absolute).**
- **Observation-only.** This does NOT validate or promote the strategy. Funding-carry is
  `DO_NOT_ADVANCE` (see `docs/research/2026-05-NEGATIVE-RESULT-FUNDING-CARRY.md`).
  Promotion `paper → live_shadow → live` stays gated and operator-only (CLAUDE.md §7,
  §11); in-process self-promotion is forbidden. No promotion language anywhere.
- **No real money, no live broker.** Simulated fills only (`NullBroker` + `FillModel`).
  The runner MUST refuse to start if `QUANTLAB_STAGE != paper` or if any `brokers/*_live.py`
  class would be imported/loaded.
- **No new research, no gate-weakening.** 1× unlevered only (leverage is what kills this
  strategy per the realism pass).
- Do NOT modify `configs/promotion.yaml` or any `brokers/*_live.py` (CLAUDE.md §13).

## 1. Architecture & data flow

```
binance_ws (real public spot + perp mark + funding)
  -> FundingCarryStrategy (computes target delta-neutral book, emits OrderIntent deltas)
  -> CarryLoop (runner.py): apply risk.yaml caps + kill conditions directly
  -> NullBroker + FillModel (simulated fills)
  -> PositionBook  <- FundingAccrual (credits/debits short-perp at 00/08/16 UTC)
  -> append-only JSONL audit ; KillSwitchWatcher arms the loop
```

**Orchestration decision (operator-approved 2026-05-31):** do NOT reuse `S4Loop`. It is
forecast-driven and single-leg — it ingests S1 prediction files + S2 verdicts and sizes
one market order per ticket from `predicted_score`/`confidence` (`Sizer`/`RiskGate` are
themselves coupled to the `ExecutionTicket(S1Signal + GovernorVerdict)` shape). A
delta-neutral carry is a *target-position rule with two legs + 8h funding*, which that
abstraction does not fit. Instead a small dedicated **CarryLoop** in `runner.py` reuses
the safety-critical **components** directly.

Reused as components (NOT `S4Loop`/`SignalIngestor`/`RiskGate`/`Sizer`):
`brokers/{null_broker,fill_model,order_types,capabilities,base}.py`,
`execution/{position_book,audit,kill_switch}.py`, `configs/risk.yaml` (caps as numbers),
`feeds/binance_ws.py` + `feeds/{recorder,replayer}.py`. New code is confined to §9.

## 2. Market data — real Binance public mainnet (no API key)

Source spot price, perp mark price / `premiumIndex`, and the realized 8h funding rate
from free public endpoints via `feeds/binance_ws.py` (extend only if a needed stream is
missing). No credentials → no secret-management surface. The reconciliation in §7 depends
on funding/basis being **real**, which is why public mainnet (not synthetic testnet) data
is used. A feed gap > 2 minutes (crypto) is a kill condition (§6).

## 3. Strategy → signals — `FundingCarryStrategy` (1× unlevered, fixed small notional)

Wraps the math in `src/quant_research_stack/crypto_research/funding/carry.py`. Each cycle:
compute the target delta-neutral book (long spot + short perp at equal notional per asset,
BTC + ETH), diff against the current `PositionBook`, and emit the rebalancing `OrderIntent`
deltas (spot buy/sell + perp sell/buy). Fixed total notional from `configs/paper_sim.yaml`;
**1× leverage, no margin stacking** (mirrors the unlevered backtest; avoids the
crash-liquidation tail the realism pass flagged). Rebalance cadence configurable
(default: each 8h settlement boundary + a drift band).

## 4. Funding accrual — `FundingAccrual`

`NullBroker`/`FillModel` model fills but not funding, and funding is the entire carry edge.
At each 00/08/16 UTC settlement, `FundingAccrual` credits/debits the short-perp leg by
`funding_rate * perp_notional` (short receives when rate > 0) using the **real realized
rate** from §2, and writes a `funding_settled` audit row. This is the strategy's only P&L
source besides basis drift and simulated cost.

## 5. Execution wiring — dedicated `CarryLoop` reusing S4 components (not `S4Loop`)

`runner.py`'s `CarryLoop` orchestrates: pull the latest spot/perp/funding from the feed →
`FundingCarryStrategy` computes the target book and rebalancing `OrderIntent` deltas →
apply the `configs/risk.yaml` caps + kill conditions directly (a small in-loop guard) →
`NullBroker.place_order` (simulated) → `PositionBook.apply_fill` (via `stream_fills`) →
`FundingAccrual` at settlements → `AuditLog`.

- `NullBroker` (`supports_shorting=True`) executes both legs in one simulated account;
  `FillModel` applies `half_spread_bps + slippage_bps` configured to ~match the backtest's
  5/10 bps taker assumptions.
- Risk caps reused as numbers from `configs/risk.yaml` (per-symbol %, gross %, base
  notional %, orders/minute) plus the drawdown + crypto feed-gap kill conditions — applied
  by a small `CarryLoop` guard rather than via `RiskGate(ticket)` (which needs the forecast
  ticket shape).
- **No S2 governor in the loop.** The governor exists to veto LLM-originated/predicted
  trades lacking paper-citations (CLAUDE.md §11–12); a mechanical delta-neutral carry never
  enters that pipeline, so there is nothing to veto. The audit records the carry's
  deterministic decisions directly. This is a deliberate, documented scoping choice.

## 6. Risk & safety

- `QUANTLAB_STAGE=paper` enforced at startup; refuse to run otherwise or if a `*_live.py`
  broker would load.
- Kill switch armed; honor all CLAUDE.md §11 hard-kill conditions: daily realized DD > 5%,
  cumulative DD > 15% from peak, **≥ 2-minute crypto data gap**, `KILL_TRADING` file in
  repo root, SIGTERM/SIGINT. Each kill writes a `kill_trigger` audit row.
- `configs/risk.yaml` caps apply (per-symbol %, gross exposure %, base notional %,
  orders/minute); first-30d half-size override (`StageOverrides.cap_multiplier_first_30d`).
- Conservative paper notional from `configs/paper_sim.yaml`.

## 7. Observation metrics — live-vs-model reconciliation (NOT promotion gates)

A report + audit stream tracking, per cycle / settlement:
- realized fill slippage vs the `FillModel` assumption,
- realized funding vs the expected/backtest funding,
- live perp-spot basis vs the daily-close basis model,
- equity curve (unlevered) and rolling Sharpe,
- liquidation-proximity diagnostic (worst adverse basis vs the §-stress thresholds).

These are **observation metrics only**. The report states explicitly that no promotion
decision follows from them.

## 8. Run modes

- **Continuous** async loop (the running bot), marking each cycle in real time, settling
  funding at 00/08/16 UTC.
- **Bounded demo:** `--max-cycles N` / `--duration` for a clean team-leader demo.
- **Deterministic replay:** drive the loop from a recorded public-data fixture via
  `feeds/replayer.py` (ADR 0011 record/replay parity) for tests.

## 9. File structure (new code only)

```
src/quant_research_stack/execution/paper_sim/
  __init__.py
  strategy.py          FundingCarryStrategy: target book -> OrderIntent deltas
  funding_accrual.py   FundingAccrual: 8h funding P&L on the short leg
  runner.py            CarryLoop: feed -> strategy -> risk/kill guard -> NullBroker ->
                       PositionBook -> FundingAccrual -> AuditLog (+ run modes)
  reconciliation.py    live-vs-model metrics + report (distinct from execution/reconciliation.py)
scripts/run_funding_carry_paper.py   CLI entry (stage guard, kill arming, run modes)
configs/paper_sim.yaml               notional, rebalance cadence, fill bps, symbols
```
Each file one responsibility. No edits to existing S4 modules beyond wiring imports.

## 10. Testing (≥ 80% on new code)

- Deterministic replay: recorded public-data fixture → assert the fill / funding / audit
  sequence is reproducible byte-for-byte (ADR 0011).
- Unit: `funding_accrual` sign + magnitude; `strategy` target-book → intent-delta mapping
  (including rebalance drift and the 1× cap); stage-guard refuses non-paper.
- Reuse existing S4 risk/kill tests where they cover the reused spine.
- `PYTHONPATH=src uv run pytest -q`, `ruff check src scripts tests`, `mypy src` all clean.

## 11. Honesty / safety guardrails (cross-cutting)

- Observation-only banner in both the audit header and the report.
- Startup stage guard (paper-only) + kill switch armed.
- No promotion language; the report cannot be read as validation or a step toward live.
- Audit log is append-only; `chmod a-w` on rotation; replay reproduces the decision
  sequence byte-for-byte (CLAUDE.md §12).

## 12. Acceptance

- `scripts/run_funding_carry_paper.py` starts only at `QUANTLAB_STAGE=paper`, arms the
  kill switch, runs the continuous loop on real public data, and the bounded/replay modes
  work.
- Funding accrues at 00/08/16 UTC from the real rate; fills are simulated; positions stay
  delta-neutral at 1×.
- The reconciliation report renders (fills/funding/basis vs model + equity) and is labeled
  observation-only.
- Audit log is append-only and replays deterministically.
- Tests + ruff + mypy clean. No PR, no merge.

## 13. Risks

- **Feed gaps / public-endpoint rate limits** — handled by the 2-min data-gap kill +
  backoff in the feed adapter.
- **Misreading paper as validation** — mitigated by §0/§11 guardrails and the report's
  explicit observation-only framing.
- **Scope creep** (adding S1 equity sleeve, leverage, real testnet orders) — explicitly
  out of scope for v1; revisit only on operator request.
