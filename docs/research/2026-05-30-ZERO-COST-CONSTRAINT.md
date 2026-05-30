# Operating Constraint — Zero-Cost Data Only (2026-05-30)

**Status:** Binding program constraint (operator decision). Supersedes the paid-data
acquisition path for now.

## Constraint

No paid data: no Sharadar, no Massive/Polygon upgrade, no futures-curve vendor, no
PIT fundamentals, no TAQ, no entitled microstructure feed. **Build within zero-cost
data only**, with a directly-tradable, paper-trading-ready deliverable as the goal.

## Frozen branches (do not spend more effort unless the operator supplies data)

| Branch | Why frozen | State |
|---|---|---|
| Sharadar / paid equity acquisition | needs purchase | scaffold retained dormant (`data/sharadar/`); feasibility doc → FROZEN |
| EDGAR 10-Q equity selection | needs survivorship-safe labels (paid) | frozen (audit `REJECT_ON_DATA`) |
| Options-IV cross-sectional selection | needs survivorship-safe returns (paid) | closed (equity-return audit) |
| Native futures carry | needs front/next/expiry curve (paid) | closed (data audit) |
| Paid microstructure / maker-queue modeling | needs entitled event data | closed (negative-result note) |

The Sharadar ingestion+audit scaffold (`88cf14d`) stays in the repo as **dormant
infrastructure** — it runs only if the operator later provides a free sample or a
purchased dataset. No further paid-data integration work.

## Valid zero-cost universe (new rules)

- Directly-traded instruments only.
- No hidden constituent universe.
- No cross-sectional stock selection unless the label source is already
  survivorship-safe (it is not, under zero cost).
- No paid data.
- No promotion language until paper trading.

## Allowed instruments / sources

- **SPY, QQQ** (free daily OHLCV; on disk 2010-2026 via `data/processed/vrp/bars/`).
- **BTCUSDT, ETHUSDT** (free Binance public archives / yfinance).
- Other liquid ETFs only if free OHLCV is clean.
- **FRED / public macro** features **only if timestamp-safe** (prefer market-priced
  daily series with no revision; avoid revised aggregates without ALFRED vintages).
- Public SEC filing data **only if** the tradable target is directly traded or the
  labels are already survivorship-safe.

## Consequence

Cross-sectional stock alpha discovery is **blocked** under zero cost (no
survivorship-safe label source). The program pivots to a **single-instrument /
small-basket directly-tradable** strategy path → see
`docs/research/intake/2026-05-30-zero-cost-deployable-v1.md`.
