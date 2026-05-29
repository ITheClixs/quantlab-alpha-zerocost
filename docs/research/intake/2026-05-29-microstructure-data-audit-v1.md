# Intake — Microstructure Data Audit v1

**Date:** 2026-05-29
**Status:** PRE-REGISTRATION (audit submitted; data run pending)
**Audit name:** `microstructure_data_audit_v1`
**Proposer:** QuantLab research
**Type:** Data-feasibility audit. **Not a strategy proposal.**
**Promotion intent:** None. This intake authorizes investigation only.

## 0. One-paragraph summary

The signal-research program review (commit `adf1c47`) closed the OHLCV
mechanical-alpha search and ranked microstructure / order-book v1 as the
top remaining direction conditional on data availability. The binding
constraint is data quality, not methodology. This intake pre-registers a
data-feasibility audit that answers exactly one question: **can we build
a believable microstructure backtest from the available BTCUSDT data?**

The audit deliberately precedes any strategy work. The audit does not
introduce a new validation gate, does not amend the no-OHLCV rule, and
does not authorize live trading or paper trading. Its sole output is one
of six data-quality labels and a corresponding next-direction decision.

## 1. Scope

### 1.1 Primary instrument

- **BTCUSDT on Binance.** Most accessible high-quality public crypto
  microstructure data. The free historical archives at
  `data.binance.vision` carry aggregated trades, klines, and (for some
  intervals) depth snapshots back to 2017.

### 1.2 Fallback instrument

- **ETHUSDT on Binance.** Only audited if BTCUSDT fails one of the
  quality checks and ETHUSDT has materially more complete data. Reuses
  the same audit methodology with no additional pre-registration.

### 1.3 Explicitly excluded from this audit

- **SPY / QQQ microstructure.** Free L2 / tick equity data is widely
  known to be incomplete (consolidated tape is unsuitable; per-venue feeds
  cost), and the cost of paid feeds is out of scope for v1. The accepted
  exception policy already limits SPY/QQQ exception-path validation to
  daily bars.
- **ES / NQ futures.** Tier-2 per accepted exception policy §1; require a
  separate roll/financing audit before any microstructure work.
- **Any Tier-1 equity proxy at intraday timeframes.** Same reason.
- **Sentiment, options, fundamentals.** These are independent intakes
  under the default rule.

### 1.4 Strict shape constraint

This audit produces **no strategy** and **no signal generator**. The
audit ingests sample data, runs documented quality checks, and emits a
data-quality label plus a decision recommendation. Any strategy work
that follows is gated on the audit's result and requires a separate
strategy intake.

## 2. Binding question

> **Can we build a believable microstructure backtest from the available
> BTCUSDT data?**

A "believable backtest" in this context means:

- Fills happen at prices that were actually available at the time
- Bid/ask spreads are observable, not modeled
- Taker fees and maker rebates are realistic per Binance's published
  spec
- Same-event leakage is prevented (signal observed at time `t` cannot
  use trade data with timestamp `>= t`)
- Latency from signal observation to order placement can be modeled
- The strategy backtest can survive a one-event or one-second delay
  stress test

If any of these is not supportable from the available data, the audit
fails and microstructure v1 is rejected.

## 3. Data sources to inspect

The audit inspects the following Binance public sources only. **No paid
data sources are used.** No private API endpoints are used.

### 3.1 Binance public REST endpoints (low-volume, for sanity checks)

- `/api/v3/aggTrades` — recent aggregated trades
- `/api/v3/depth` — current order book snapshot (up to 5000 levels)
- `/api/v3/klines` — OHLCV at intervals from 1s to 1mo
- `/api/v3/ticker/bookTicker` — best bid/ask current

### 3.2 Binance Data Vision archives (primary source for the audit)

- `data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/*.zip`
- `data.binance.vision/data/spot/daily/trades/BTCUSDT/*.zip`
- `data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/*.zip`

The audit pulls a small representative sample (typical: 1-7 days across
calm and volatile periods) sufficient to characterize the data, not the
full history. Full-history ingestion is a downstream task gated on a
clean audit verdict.

### 3.3 What is NOT inspected in this v1 audit

- Per-level L2 order book reconstruction from WebSocket deltas. This
  requires running a live capture process which is not part of a static
  audit; will be added in a v2 audit if v1 passes.
- Funding rates (only relevant for perpetuals; this audit covers spot).
- Withdrawal / deposit flows.
- Cross-venue arbitrage (Binance only in v1).

## 4. Audit categories and pre-registered checks

Each category produces specific pass/fail or quantitative outputs that
combine into the §6 data-quality label.

### 4.1 What data exists

For each of {aggTrades, trades, klines 1s, depth snapshot}, the audit
reports:

- whether the source is accessible
- file format and schema
- date range with at least one full day of data
- gaps in the available date range (missing daily files)
- row count for at least one representative day

### 4.2 Timestamp quality

For the trade stream, the audit reports:

- timestamp resolution (seconds, milliseconds, microseconds)
- presence of separate exchange and local-receipt timestamps
- monotonicity: percentage of consecutive rows where `t_i >= t_{i-1}`
- duplicate event detection: percentage of exact-duplicate (timestamp,
  trade_id, price, qty) rows
- missing intervals: longest gap with no trade events
- timezone convention (must be UTC; document if not)

A pass requires: millisecond or better resolution; monotonicity ≥ 99.9%;
duplicate rate ≤ 0.01%; longest gap < 5 minutes on the audited day(s);
UTC timestamps.

### 4.3 Order book reconstructability

For the depth snapshot endpoint and any available delta source, the
audit reports:

- whether snapshots include a sequence ID or last-update ID
- whether deltas reference the snapshot's last-update ID for replay
- whether snapshots are dense at the top of the book (first 10 levels)
- whether crossed-book events (best bid >= best ask) are present
- if crossed: frequency and likely cause

A "microstructure_clean" verdict requires: sequence IDs allow gap-free
replay; no crossed book events on the audited day; snapshots have ≥ 10
levels on each side.

### 4.4 Trade data quality

For the trade stream, the audit reports:

- buyer/seller aggressor flag availability (Binance's `isBuyerMaker`)
- coverage rate of the flag (% of trades with the flag set)
- trade direction reconstructability (using aggressor flag + sequence)
- count of zero-volume events
- count of zero-price events
- outlier trades by price (more than ±5σ from rolling 1-minute mean)
- missing volume rate

A pass requires: aggressor flag present on ≥ 99% of rows; zero-volume
rate ≤ 0.01%; outlier rate ≤ 0.1%.

### 4.5 Backtest feasibility

The audit documents which of the following are supportable on the
audited data:

- bid/ask execution at observable prices
- taker fee modeling (Binance spot taker fee schedule is published)
- maker fill modeling at realistic queue priority
- one-event delay stress (the strategy must survive a one-event lookahead
  prohibition)
- one-second delay stress
- enforcement that fills do not occur at prices that were not available
  at the time

This section produces a yes/no per item.

### 4.6 Storage feasibility

The audit estimates and reports:

- rows per day for aggTrades (representative day from §4.1)
- compressed parquet size per day
- compressed parquet size projection per year
- replay speed estimate (rows / second for a representative analytical
  pass)
- whether the projected year of data exceeds 100 GB compressed (a
  practical hard limit on a single workstation)

### 4.7 Market coverage

The audit reports the longest contiguous available history for BTCUSDT
aggTrades. The decision rule below requires ≥ 6 months of available
data for any prototype, ≥ 24 months before any promotion claim.

### 4.8 Data-quality summary

A single label is assigned from the §6 list based on the §4.1-§4.7
findings.

## 5. Documented external references

The audit cross-references the following Binance documentation. If any
of these documents are unavailable or contradict the data observed, the
audit notes the discrepancy explicitly.

- Binance Spot API: <https://binance-docs.github.io/apidocs/spot/en/>
- Binance Data Vision: <https://data.binance.vision/>
- Binance fee schedule (spot, USDT pairs): documented at intake time
- Binance time synchronization: UTC convention

## 6. Data-quality labels (closed list)

Exactly one label is assigned at the end of the audit.

| Label | Meaning | Decision recommendation |
|---|---|---|
| `microstructure_clean` | aggTrades + L2 depth reconstructable; ≥ 6 months; passes all §4 checks | proceed to L2 order-book v1 strategy intake |
| `trade_only_clean` | aggTrades clean; ≥ 6 months; depth reconstruction not possible or not audited | proceed to trade-flow / imbalance / short-horizon momentum-reversal v1 (no L2 strategies) |
| `quotes_incomplete` | aggTrades clean; quote/depth data exists but is incomplete (gaps, missing sequence IDs, crossed events) | research-only; no strategy intake authorized |
| `book_not_reconstructable` | depth data exists but cannot be replayed gap-free | as `quotes_incomplete` |
| `research_only` | data exists but coverage / horizon / quality is below the threshold for a strategy proposal | research note only; no intake |
| `reject` | data unavailable, unreliable, or below the bar for any further work | reject microstructure v1 direction; pivot to event-conditioned macro/calendar |

## 7. Decision rule after audit

The label drives the next direction without further negotiation:

- **`microstructure_clean`** → open L2 order-book v1 strategy intake with
  `InformationSource.MICROSTRUCTURE_BOOK` declared. Strategy intake
  requires pre-registered variant grid, fill model, latency model, fee
  model.
- **`trade_only_clean`** → open trade-flow v1 strategy intake with
  `InformationSource.MICROSTRUCTURE_TICK` declared. Allowed signal
  families: trade imbalance, signed-volume momentum, short-horizon
  mean-reversion off trade prints. L2 strategies are explicitly not
  authorized.
- **`quotes_incomplete`** OR **`book_not_reconstructable`** → no strategy
  intake. Investigate paid feeds in a separate review; otherwise pivot to
  event-conditioned macro/calendar per program review §8.B.
- **`research_only`** → document the findings; no further investment in
  microstructure direction at v1; pivot to event-conditioned macro/calendar.
- **`reject`** → reject microstructure v1 direction entirely; pivot to
  event-conditioned macro/calendar.

## 8. Methodology

### 8.1 Sample selection

The audit pulls aggTrades for **at least three** representative days:

- One day during a calm low-vol regime (target: a typical 2024 weekday)
- One day during a high-vol regime (target: a known crypto event day,
  e.g. a notable rally or correction)
- One recent day (within 30 days of the audit)

The selection is documented in the report. Days are picked from the
available `data.binance.vision` archive at audit time.

### 8.2 Sample-period justification

Three days is sufficient to characterize:
- file format and schema
- timestamp quality across regimes
- aggressor flag coverage
- typical row count and projected storage cost
- presence of outlier events under volatility stress

Three days is NOT sufficient to:
- estimate longest gaps over the full history
- characterize seasonal microstructure changes
- detect rare data-quality regressions

Those are downstream tasks if the audit passes.

### 8.3 Quality checks are deterministic and reproducible

Each check is a function `(data) -> (pass/fail, quantitative value, notes)`.
The audit script records every check's output. The report tabulates them.
Re-running the audit on the same sample produces the same output.

### 8.4 Documentation cross-reference

For each Binance schema (aggTrades, klines, depth), the report
cross-references the documented schema vs the observed schema and notes
any deviation.

## 9. Explicit non-goals

9.1. **No strategy backtest is performed.** This audit produces no
trading signal, no portfolio, no PnL series. Any backtest reference
appears in §4.5 as a feasibility check on what would be required,
not as a result.

9.2. **No new validation gate is introduced.** The existing
`ValidationSpec` and `ValidationPipeline` are not amended.

9.3. **No promotion is authorized.** A passing audit authorizes a
strategy intake, not a strategy promotion. The strategy intake then
goes through the same validation pipeline as any other proposal.

9.4. **No live data capture is performed.** Only static archive downloads
and one snapshot of the depth endpoint at audit time.

9.5. **No paid data sources are evaluated.** If the free Binance sources
are insufficient, the audit fails and the recommendation is to pivot
direction, not to authorize paid-feed procurement.

9.6. **No exception-policy amendment is proposed.** BTCUSDT remains
Tier-2 per the accepted single-index risk-timing exception policy. This
audit is for data feasibility only; even if microstructure v1 produces
a strategy, that strategy invokes the default `STRATEGY_INTAKE.md`
protocol with `InformationSource.MICROSTRUCTURE_TICK` or
`MICROSTRUCTURE_BOOK`, not the exception path. The single-index
risk-timing exception covers daily-resolution single-index timing,
which is a different shape from microstructure.

9.7. **No SPY / QQQ microstructure work is authorized.** Free equity L2
data is excluded per §1.3.

## 10. Deliverables

| Artifact | Path |
|---|---|
| Intake document | `docs/research/intake/2026-05-29-microstructure-data-audit-v1.md` (this document) |
| Audit script | `scripts/audit_microstructure_data.py` |
| Audit raw outputs | `reports/signal_research/microstructure/audit_raw/` (sample summaries, schemas, per-check JSON) |
| Audit report | `reports/signal_research/microstructure/data_audit_report.md` |

The audit report is structured per the §4 categories and ends with the
§6 label assignment and §7 decision recommendation.

## 11. Reproducibility

The audit script must:

- Record the git SHA at run time in the report header
- Record the exact URLs and dates of the data files downloaded
- Record SHA256 checksums of the downloaded files in the raw output
  directory
- Use seeded random selection where any random choice is made
- Be re-runnable: a second run on the same sample produces a
  byte-identical report (modulo timestamps)

## 12. Sign-off

The proposer acknowledges:

- This audit produces a data-quality label and a decision recommendation,
  not a strategy.
- A passing audit authorizes a separate strategy intake; it does not
  authorize promotion or paper trading.
- A failing audit pivots the next direction to event-conditioned
  macro/calendar per the program review §8.B without further negotiation.
- The audit methodology in §4 is frozen at intake submission; checks
  may not be added or removed after seeing results.

**Proposer:** QuantLab research (Phase A microstructure audit)
**Intake submitted:** 2026-05-29
**Program review reference:** `docs/research/2026-05-PROGRAM-REVIEW-SIGNAL-RESEARCH.md` §8.A

## References

- `docs/research/2026-05-PROGRAM-REVIEW-SIGNAL-RESEARCH.md` — program review
- `docs/research/STRATEGY_INTAKE.md` — default strategy intake protocol
- `docs/research/VALIDATION_RUNBOOK.md` — validation pipeline runbook
- `docs/research/intake/2026-05-28-single-index-risk-timing-exception.md`
  — accepted exception policy (informational only; not invoked by this audit)
- Binance API documentation
- Binance Data Vision documentation
