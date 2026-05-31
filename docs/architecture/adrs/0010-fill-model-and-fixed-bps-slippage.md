# ADR 0010: Fill model uses fixed-bps slippage and commission; no L2 modeling in S3

## Status
Accepted, 2026-05-17.

## Context
The S3 backtester needs a fill simulator. Two ends of the spectrum:
- "Optimistic": fill at the next event's price with zero cost. Misleading.
- "Realistic": L2 order-book modeling with queue position and impact. Heavy; needs L2 data.

S3 ships paper brokers + a backtester; L2 simulation is deferred to S3.3 which uses
the already-downloaded CryptoLOB-2025 + HFT LOB datasets (35 GB total on disk).

## Decision
The S3 FillModel uses a fixed-bps approximation:

  fill_px = next_event_mid + half_spread_bps * 1e-4 * mid + slippage_bps * 1e-4 * mid
  commission = fill_px * qty * commission_bps * 1e-4
  fill is delayed by fill_latency_ms

Default config errs conservative: 1 bps commission + 2 bps slippage + 1 bps half-spread
+ 50 ms latency = 4 bps total per-side. This is realistic-to-pessimistic for retail
equity and crypto on the venues we target.

The model is deterministic. No randomness in fill price or timing.

## Consequences
+ Backtests reproducible byte-identical across runs with the same config.
+ Default config does not over-promise; strategies that look good here have margin.
+ Per-strategy configs can tighten (e.g. for a market-making study).
- Underestimates impact for large orders. Mitigated by reject_if_notional_above_pct_adv
  cap that refuses to backtest sizes that would have required L2 modeling.
- Future L2 modeling (S3.3) will likely show some strategies lose more than S3's
  fill model says. Documented limitation in every backtest report.
