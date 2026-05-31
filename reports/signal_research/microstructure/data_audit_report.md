# Microstructure Data Audit Report — BTCUSDT (Binance)

**Audit date:** 2026-05-29T10:15:23.949566+00:00
**Instrument:** BTCUSDT
**Git SHA:** 1e3c2758567b
**Intake reference:** `docs/research/intake/2026-05-29-microstructure-data-audit-v1.md`

## §0 Binding question

> **Can we build a believable microstructure backtest from the available BTCUSDT data?**

Answer (per §6 label assigned below): **see §7 Decision** at the end.

## §4.1 What data exists

Days audited:

| Day | Rows | Duration (h) | URL | SHA256 |
|---|---:|---:|---|---|
| 2024-04-01 | 1,489,577 | 24.00 | `https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-04-01.zip` | `d266bb2e03a42525...` |
| 2024-08-05 | 5,280,704 | 24.00 | `https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-08-05.zip` | `ba052cf0c7e00794...` |
| 2026-05-22 | 670,571 | 24.00 | `https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-05-22.zip` | `590753b52e0ec8c6...` |

Schema observed: `agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp_raw, is_buyer_maker, is_best_match, timestamp_ms`

## §4.2 - §4.7 Check outcomes

| Check | Pass | Value | Notes |
|---|:---:|---|---|
| `timestamp_resolution_ms_2024-04-01` | YES | 1 | smallest positive inter-event interval in ms |
| `monotonicity_pct_2024-04-01` | YES | 100.0 | % of consecutive rows where t_i >= t_{i-1} |
| `duplicate_event_pct_2024-04-01` | YES | 0.0 | % of exact (agg_trade_id, timestamp_ms, price, qty) duplicates |
| `longest_gap_seconds_2024-04-01` | YES | 3.974 | longest inter-trade gap on the audited day |
| `timezone_utc_2024-04-01` | YES | UTC (Unix epoch ms) | Binance aggTrades timestamps are Unix ms by spec |
| `aggressor_flag_coverage_pct_2024-04-01` | YES | 100.0 | % of rows with is_buyer_maker not null |
| `zero_volume_pct_2024-04-01` | YES | 0.0 | % of rows with quantity <= 0 |
| `zero_price_pct_2024-04-01` | YES | 0.0 | % of rows with price <= 0 |
| `outlier_5sigma_pct_2024-04-01` | YES | 0.0 | % of trades > 5σ from 1-min rolling mean |
| `timestamp_resolution_ms_2024-08-05` | YES | 1 | smallest positive inter-event interval in ms |
| `monotonicity_pct_2024-08-05` | YES | 100.0 | % of consecutive rows where t_i >= t_{i-1} |
| `duplicate_event_pct_2024-08-05` | YES | 0.0 | % of exact (agg_trade_id, timestamp_ms, price, qty) duplicates |
| `longest_gap_seconds_2024-08-05` | YES | 2.576 | longest inter-trade gap on the audited day |
| `timezone_utc_2024-08-05` | YES | UTC (Unix epoch ms) | Binance aggTrades timestamps are Unix ms by spec |
| `aggressor_flag_coverage_pct_2024-08-05` | YES | 100.0 | % of rows with is_buyer_maker not null |
| `zero_volume_pct_2024-08-05` | YES | 0.0 | % of rows with quantity <= 0 |
| `zero_price_pct_2024-08-05` | YES | 0.0 | % of rows with price <= 0 |
| `outlier_5sigma_pct_2024-08-05` | YES | 0.0 | % of trades > 5σ from 1-min rolling mean |
| `timestamp_resolution_ms_2026-05-22` | YES | 1 | smallest positive inter-event interval in ms |
| `monotonicity_pct_2026-05-22` | YES | 100.0 | % of consecutive rows where t_i >= t_{i-1} |
| `duplicate_event_pct_2026-05-22` | YES | 0.0 | % of exact (agg_trade_id, timestamp_ms, price, qty) duplicates |
| `longest_gap_seconds_2026-05-22` | YES | 9.359 | longest inter-trade gap on the audited day |
| `timezone_utc_2026-05-22` | YES | UTC (Unix epoch ms) | Binance aggTrades timestamps are Unix ms by spec |
| `aggressor_flag_coverage_pct_2026-05-22` | YES | 100.0 | % of rows with is_buyer_maker not null |
| `zero_volume_pct_2026-05-22` | YES | 0.0 | % of rows with quantity <= 0 |
| `zero_price_pct_2026-05-22` | YES | 0.0 | % of rows with price <= 0 |
| `outlier_5sigma_pct_2026-05-22` | YES | 0.004921179114515838 | % of trades > 5σ from 1-min rolling mean |
| `depth_snapshot_accessible` | YES | bids=5000 asks=5000 lastUpdateId=94440504989 | REST snapshot fetched successfully |
| `depth_has_sequence_id` | YES | 94440504989 | lastUpdateId enables snapshot/delta replay |
| `depth_bids_10_levels` | YES | 5000 | bid-side level count |
| `depth_asks_10_levels` | YES | 5000 | ask-side level count |
| `depth_crossed_book` | YES | best_bid=73703.86 best_ask=73703.87 | best bid must be strictly < best ask |
| `book_ticker_accessible` | YES | {'symbol': 'BTCUSDT', 'bidPrice': '73703.86000000', 'bidQty': '3.06594000', '... | REST best-bid/ask fetched |
| `avg_rows_per_day` | YES | 2480284.0 | averaged across audited days |
| `projected_yearly_gb_compressed` | YES | 50.58778412640095 | rough estimate at 60 bytes/row in zstd parquet |
| `archive_2018_01_01_available` | YES | https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrad... | HEAD on data.binance.vision aggTrades archive |
| `archive_recent_30d_available` | YES | https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrad... | HEAD on recent aggTrades archive |

## §6 Data-quality label

**`trade_only_clean`**

## §7 Decision recommendation

aggTrades stream is clean and history exceeds 6 months. Depth REST snapshot is accessible with sequence IDs, but full L2 replay requires WebSocket delta capture which is OUT OF SCOPE for this v1 audit. Recommend: proceed to trade-flow v1 (InformationSource.MICROSTRUCTURE_TICK). A future v2 audit may upgrade to microstructure_clean after a live delta-capture pass.

## Reproducibility

- Audit run at: 2026-05-29T10:15:23.949566+00:00
- Git SHA: 1e3c2758567b
- All downloaded files SHA256-recorded in §4.1 table
- Raw aggTrades zip archives stored under `reports/signal_research/microstructure/audit_raw/`
- Re-running this script on the same sample dates produces a byte-identical report modulo `audit_timestamp_utc`.

## What this audit does NOT cover (per intake §3.3, §9)

- Per-level L2 order book reconstruction from WebSocket deltas. The REST snapshot endpoint was sanity-checked but full gap-free replay requires a live capture pass (deferred to a v2 audit if applicable).
- Funding rates (perpetuals only; this audit covers spot).
- Cross-venue arbitrage (Binance only).
- Paid data feeds.
- Any strategy backtest. **No PnL series was produced.**

## Next steps tied to this label

- `microstructure_clean` → open L2 order-book v1 strategy intake.
- `trade_only_clean` → open trade-flow v1 strategy intake (`InformationSource.MICROSTRUCTURE_TICK`).
- `quotes_incomplete` / `book_not_reconstructable` → consider paid feeds in a separate review.
- `research_only` → document; no strategy intake; pivot to event-conditioned macro/calendar.
- `reject` → reject microstructure v1; pivot to event-conditioned macro/calendar.
