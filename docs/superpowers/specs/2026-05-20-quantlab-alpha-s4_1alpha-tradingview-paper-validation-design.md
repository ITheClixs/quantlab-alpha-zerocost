# QuantLab Alpha — S4.1α: TradingView Paper Validation (Design)

**Date:** 2026-05-20
**Status:** approved (operator-approved via brainstorming, awaiting spec review)
**Predecessor:** S4 (execution + risk + promotion gates for paper + live_shadow stages)
**Successor:** S4.1 (live broker implementations — currently deferred until S4.1α gates pass)

## 0. Goal

Before any real-money trading, QuantLab's signals must be validated against a paper-trading environment that the operator can review visually in TradingView. This spec defines:

- The integration mode between QuantLab and TradingView (Mode A: TV as the chart viewer for Alpaca paper — the only path that has a programmatic, supported API at both ends).
- A daily validation report that captures hit rate, Sharpe, drawdown, governor block rate, position-book reconciliation, and a per-signal table.
- A validation-gate config (`configs/validation.yaml`) that defines what "signals validated" means.
- An extension to the existing promotion-report generator that includes the new hit-rate gate row.
- Operator runbooks for TradingView setup and daily review.

S4.1α is the precondition for opening the S4.1 (live broker) conversation. Until the validation gates are green, no work on `brokers/alpaca_live.py` or `brokers/binance_live.py` proceeds.

## 1. Scope

**In scope:**

- New package `src/quant_research_stack/validation/` with: hit-rate calculator, forward-return fetcher, daily-report builder, reconciliation cross-check.
- New script `scripts/tv_validation_report.py` invoked once per trading day (after close) to produce both a Markdown daily report and a per-signal Parquet table.
- New `configs/validation.yaml` holding duration + threshold settings, kept separate from `configs/promotion.yaml` so this work does not trigger CLAUDE.md §1.13's two-person-review requirement.
- Modification to `scripts/generate_promotion_report.py` to read the latest validation report and include a `hit_rate_min` row in the green/red table.
- Two new runbooks under `docs/runbooks/`: `tradingview_paper_setup.md` and `paper_validation_methodology.md`.
- A new Makefile target `tv-validation-report`.

**Out of scope (this spec):**

- `brokers/alpaca_live.py` and `brokers/binance_live.py` — explicitly deferred; live broker design re-opens only after the validation gates have passed.
- Touching `configs/promotion.yaml` (CLAUDE.md §1.13 — two-person review required).
- Pine Script ports of S1/S2 logic to TradingView's runtime.
- Browser automation against TradingView's web UI.
- Inbound integration with TradingView's internal Paper Trading account (TV does not expose a public order-placement API for it; see §2.1).
- Multi-broker, cross-venue, or multi-account validation.

## 2. Technical reality of TradingView integration

TradingView exposes two distinct "paper trading" features that the operator may confuse:

1. **Connected broker paper accounts** (Alpaca paper, OANDA demo, Tradovate demo, etc.). TV's Trading Panel connects to the broker's paper API; orders happen on the broker; TV displays them. The broker's API is the source of truth and is fully programmatic.

2. **TradingView's own internal Paper Trading broker**. A paper account that lives entirely inside TradingView's infrastructure, selectable in the Trading Panel alongside the external brokers. There is **no public outbound API** that lets an external Python process place an order on this account. The only routes IN are: Pine Script `strategy.entry()` inside TV's runtime, manual UI clicks, or brittle browser-automation (Playwright) which is ToS-fragile and breaks on every TV UI revision.

This spec uses option 1 (Alpaca paper, viewed in TV). The operator opens TradingView, connects their Alpaca paper account via TV's Trading Panel, and watches QuantLab's orders appear on the chart in real time. There is no QuantLab → TradingView code path — the validation tooling all targets QuantLab's own artifacts plus the Alpaca paper account's REST API.

## 3. Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ QuantLab daemon (scripts/s4_execute.py, stage=paper)                    │
│   ↓ place_order via AlpacaPaperBroker                                   │
│ Alpaca paper account ←──────────── operator-connected TradingView chart │
│   ↑ account/positions REST                                              │
│                                                                          │
│ Daily after close:                                                       │
│   scripts/tv_validation_report.py                                        │
│     reads:                                                               │
│       - experiments/alpha_s1/<run_id>/predictions.parquet (live rows)    │
│       - experiments/s2_verdicts_<stage>/<YYYY-MM-DD>.jsonl               │
│       - logs/audit/s4/paper/<YYYY-MM-DD>.jsonl  (trade_placed, trade_fill)│
│       - AlpacaPaperBroker.account() + .positions()                       │
│       - forward bars from configs/validation.yaml.data.forward_return_source│
│     uses:                                                                │
│       - validation/hit_rate.py        (predicted_dir vs realized_dir)    │
│       - validation/forward_returns.py (horizon → bar realization)        │
│       - validation/reconcile.py       (book_equity vs broker_equity bps) │
│       - validation/daily_report.py    (assemble Markdown + Parquet)      │
│     writes:                                                              │
│       - docs/validation/<YYYY-MM-DD>.md                                  │
│       - data/validation/<YYYY-MM-DD>.parquet                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Module layout

```text
src/quant_research_stack/validation/
  __init__.py
  hit_rate.py             # directional hit-rate calculator
  forward_returns.py      # fetches realized N-bar returns to match predictions
  reconcile.py            # cross-checks QuantLab's position book vs Alpaca paper
  daily_report.py         # assembles the markdown daily report

configs/
  validation.yaml         # new — duration + thresholds + report paths

scripts/
  tv_validation_report.py # entry point (called daily by operator/cron)

docs/runbooks/
  tradingview_paper_setup.md
  paper_validation_methodology.md

docs/validation/
  <YYYY-MM-DD>.md         # daily reports (generated, committed by operator)

data/validation/
  <YYYY-MM-DD>.parquet    # per-signal table (generated, gitignored)
```

### 3.2 Interaction with existing systems

- **`scripts/generate_promotion_report.py`** (S4): extended to read the most recent N validation Parquet files, compute rolling hit-rate, and add a `hit_rate_min` row to the green/red table. The threshold comes from `validation.yaml`. `promotion.yaml` is NOT modified.
- **`PositionBook`** (S4): unchanged. The reconciliation row in the daily report compares the latest snapshot under `data/positions/paper/<date>.parquet` with a fresh `AlpacaPaperBroker.account()` call.
- **`AlpacaPaperBroker`** (S3): unchanged. The validation script uses the existing `account()` and `positions()` methods.
- **S1 predictions schema**: assumed to include `signal_id, symbol, predicted_score, confidence, horizon_minutes, ts_utc` (per the S4 spec's documented schema for live predictions). This spec does not modify it.
- **S2 verdict schema**: unchanged (existing `GovernorVerdict` shape).

## 4. Component specifications

### 4.1 `validation/hit_rate.py`

```python
@dataclass(frozen=True)
class HitRateResult:
    hit_rate: float                   # in [0, 1], weighted
    n_signals: int                    # signals where predicted_direction != 0
    n_hits: int                       # subset where predicted_direction == realized_direction
    governor_block_rate: float        # vetos + insufficient_evidence over total signals


def compute_hit_rate(
    signals: list[ScoredSignal],
) -> HitRateResult:
    ...


@dataclass(frozen=True)
class ScoredSignal:
    signal_id: str
    predicted_direction: int    # in {-1, 0, 1}; 0 means "no trade per S2"
    realized_direction: int     # in {-1, 0, 1}; 0 means flat / no realized data
    weight: float               # positive; from configs/risk.yaml caps
    s2_decision: str            # "pass" | "veto" | "insufficient_evidence"
```

Math (already given in §2.2 of brainstorming):

```text
hit_rate = sum(hit * weight for s where predicted_direction != 0)
         / sum(weight for s where predicted_direction != 0)

governor_block_rate = count(s where s2_decision in {"veto", "insufficient_evidence"})
                    / count(all signals)
```

Vetoes and `insufficient_evidence` are excluded from the hit-rate numerator and denominator. They are counted separately in `governor_block_rate`.

### 4.2 `validation/forward_returns.py`

Fetches the realized return at `horizon_minutes` after each filled signal. Default source is the existing S3 `AlpacaRest` bars endpoint (already in the codebase from S3 Task 8). Alternatives `yfinance` and `polygon` are listed in `validation.yaml.data.forward_return_source` but not wired in the first pass.

Horizon-to-bar alignment uses `ceil_to_next_bar`: if a fill happens at 09:35:42 with `horizon_minutes=5`, the realized return is measured from the 09:35 bar close to the 09:40 bar close. Missing bars yield `realized_direction=0` and the signal is excluded from hit-rate.

### 4.3 `validation/reconcile.py`

A thin wrapper that loads the latest PositionBook snapshot, calls `AlpacaPaperBroker.account()`, and reuses S4's `execution/reconciliation.py:diff_book_vs_broker` to compute a basis-point diff. The threshold (`reconciliation.max_diff_bps`, currently 1.0) comes from `configs/risk.yaml`. A diff > 1 bp in the daily report flags red.

### 4.4 `validation/daily_report.py`

Assembles the Markdown report described in §5 below. Pure function: takes the result of the other three modules + the source predictions/audit/verdicts as input, returns a string. Side-effect-free for testability.

### 4.5 `scripts/tv_validation_report.py`

Entry point. Glues data loading, the four modules, and writing the two artifacts.

```python
def main() -> int:
    args = parse_args()  # --date YYYY-MM-DD (default: today) --config configs/validation.yaml
    cfg = load_validation_config(Path(args.config))

    predictions = _load_live_predictions(args.date)
    verdicts    = _load_verdicts(args.date)
    audit       = _load_audit(args.date)
    fwd_returns = forward_returns.fetch(predictions, source=cfg.data.forward_return_source)
    broker      = AlpacaPaperBroker()  # paper account, existing S3 adapter
    reconc      = reconcile.run(broker, snapshot_date=args.date)
    hit         = hit_rate.compute(predictions, verdicts, audit.fills, fwd_returns)

    md  = daily_report.render(args.date, predictions, verdicts, audit, hit, reconc, cfg)
    pq  = daily_report.per_signal_table(predictions, verdicts, audit, fwd_returns)

    md_path = Path(cfg.artifacts.daily_report_dir) / f"{args.date}.md"
    pq_path = Path(cfg.artifacts.per_signal_parquet_dir) / f"{args.date}.parquet"
    md_path.write_text(md)
    pq.write_parquet(pq_path, compression="zstd")
    print(f"Wrote {md_path}\nWrote {pq_path}")
    return 0
```

### 4.6 `configs/validation.yaml`

```yaml
window:
  min_trading_days: 30
  rolling_window_days: 14

thresholds:
  hit_rate_min: 0.53
  sharpe_min: 1.0
  max_daily_dd_pct: 0.05
  governor_block_rate_max: 0.50

data:
  forward_return_source: alpaca_bars
  horizon_alignment: ceil_to_next_bar

artifacts:
  daily_report_dir: docs/validation
  per_signal_parquet_dir: data/validation
```

Validated by a new `ValidationConfig` Pydantic model in `validation/__init__.py`. All thresholds are required; sane-range validators (`0 < hit_rate_min < 1`, `0 < max_daily_dd_pct < 1`, etc.) enforce on load.

### 4.7 `scripts/generate_promotion_report.py` extension

Adds a single new row to the green/red table, sourced from `validation.yaml` and the most recent N=`min_trading_days` per-signal parquets:

```text
| hit_rate_min | 0.53 | <rolling 30d weighted hit rate> | ✅/❌ |
```

No changes to `promotion.yaml` — the threshold lives entirely in `validation.yaml`. The promotion script reads BOTH config files at evaluation time.

## 5. Daily report format

Markdown structure (`docs/validation/<YYYY-MM-DD>.md`):

```markdown
# QuantLab paper validation — 2026-05-21

Stage: paper · Broker: alpaca_paper · TV chart account: Alpaca paper (operator-connected)

## Headline
- n_signals: <int>   (passed-S2: <int> · vetoed: <int> · insufficient_evidence: <int>)
- n_trades: <int>
- hit_rate (weighted): <float, 3dp>
- daily_pnl_pct: <signed float, 2dp>
- daily_dd_pct: <float, 2dp>
- Sharpe (rolling 14d): <float, 2dp>
- governor_block_rate: <float, 2dp>

## Per-signal table
| signal_id | symbol | predicted_score | confidence | s2_decision | fill_price | horizon_min | realized_return | hit |
| ... | ... | ... | ... | ... | ... | ... | ... | ✅/❌/— |

## Position-book reconciliation
QuantLab book equity:    <decimal>
Alpaca paper equity:     <decimal>
Diff bps:                <float, 2dp>     (>1.0 → ⚠)

## TV chart cross-check (operator-filled)
- [ ] I reviewed today's trades on the TV chart with Alpaca connected.
- [ ] Any signal looked obviously wrong on the chart (please annotate):
- Operator initials + date:

## Promotion gate status (informational)
- hit_rate_min (0.53):                 ✅/❌ <observed>
- sharpe_min (1.0 rolling):            ✅/❌ <observed>
- max_daily_dd (0.05):                 ✅/❌ <observed>
- governor_block_rate_max (0.50):      ✅/❌ <observed>
- min_trading_days (30):               ✅/🟡/❌ <day count>
```

Per-signal Parquet schema (`data/validation/<YYYY-MM-DD>.parquet`):

```text
signal_id        str
symbol           str
predicted_score  float32
confidence       float32
predicted_dir    int8           in {-1, 0, 1}
s2_decision      str            "pass" | "veto" | "insufficient_evidence"
fill_price       float64        NaN if no fill
horizon_minutes  int16
realized_return  float64        NaN if no bar data
realized_dir     int8           in {-1, 0, 1}; 0 if NaN
hit              boolean        true iff predicted_dir == realized_dir != 0
weight           float32        position weight at trade time
fill_ts_utc      datetime64[us, UTC]   NaN if no fill
```

## 6. Operator workflow

```text
Morning (pre-open):
  1. git pull
  2. make s4-execute S4_STAGE=paper S4_ASSET=equity S4_EQUITY=100000
  3. Open TradingView, watchlist with the symbols S4 is trading.
  4. In TV's Trading Panel, confirm Alpaca paper is connected.

Intraday:
  - QuantLab places orders on Alpaca paper.
  - TV chart shows orders/fills in real time.
  - Operator does NOT manually trade; pure observation.

End of day:
  5. make tv-validation-report
  6. Open docs/validation/<today>.md, review:
       a. Headline metrics
       b. Per-signal table — scan for outliers
       c. TV chart cross-check — eyeball flagged signals
       d. Position-book reconciliation row — must be < 1 bp
  7. Fill in operator-checklist section (initials + observations).
  8. git add docs/validation/<today>.md && git commit -m "validation: <today> review"

After 30 trading days:
  9. make s4-promotion-report
       → docs/runbooks/paper_to_live_shadow.md
 10. If all gates green, sign the report. Live broker design (S4.1) reopens
     ONLY after this point.
```

## 7. Testing

### 7.1 Unit tests (default, in `tests/`, ≥80% coverage)

- `test_validation_hit_rate.py`: pass-with-correct-dir, pass-with-wrong-dir, veto-excluded, insufficient_evidence-excluded, zero-prediction-excluded, zero-return-edge-case, zero-weight-edge-case, all-veto edge case (n_signals=0, governor_block_rate=1.0).
- `test_validation_forward_returns.py`: horizon-to-bar alignment, missing-bar handling, multi-symbol batch.
- `test_validation_reconcile.py`: 0 bps for matched book, 5 bps for injected diff, zero broker equity treated as divergence.
- `test_validation_daily_report.py`: given fixture predictions + audit + mocked broker, the report contains every section; the Parquet companion has the expected schema and row count.
- `test_validation_configs.py`: Pydantic validation of `validation.yaml` (sane ranges, missing-key errors).

### 7.2 Integration tests (marker `validation_integration`, skipped by default)

- `test_validation_against_alpaca_paper.py`: 60-second end-to-end. Requires `~/.alpaca/paper_keys.json`. Runs the S4 paper daemon, then runs `tv_validation_report.py`, asserts the report is non-empty and includes the expected sections.

### 7.3 Tooling

- Add `validation_integration` marker to `pyproject.toml`'s `[tool.pytest.ini_options]`.

## 8. Success criteria

1. `scripts/tv_validation_report.py --date <today>` produces a valid Markdown report at `docs/validation/<today>.md` and a Parquet table at `data/validation/<today>.parquet`.
2. `configs/validation.yaml` is loaded via Pydantic with range validation.
3. The two runbooks are committed and the operator can complete the daily workflow without further questions.
4. `scripts/generate_promotion_report.py` includes the `hit_rate_min` row sourced from `validation.yaml`. `configs/promotion.yaml` is unchanged.
5. `PYTHONPATH=src pytest -q` passes including new validation unit tests; `validation_integration` tests gated by marker.
6. `ruff check src scripts tests` passes.
7. `mypy src` passes.
8. The operator has run the daily workflow end-to-end at least once (≥1 day in `docs/validation/`) before declaring S4.1α complete.

## 9. Decision gate for reopening S4.1

S4.1 (live broker design + implementation) opens only when ALL of the following hold:

- ≥ 30 trading days of `docs/validation/*.md` reports committed.
- Latest promotion report (`docs/runbooks/paper_to_live_shadow.md`) shows all 5 gates green.
- Operator signs the promotion report.

Until then, no spec, no plan, no implementation work touching `brokers/*_live.py` proceeds. The S4 spec's §11 ("out-of-scope") guards stand: the `live` route in `BrokerRouter` continues to raise `ImportError`.
