# Multi-Model Comparison — Same Fixture

## Fixture
- universe: top 30 SP500 by ADV
- history: 2018-01-01 → 2026-05-26
- dev:     2018-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- costs: 4.0 bps commission + 50.0 bps spread
- cost-stress multiplier: 2.0×

## Data quality banner

DATA QUALITY: data_quality_label=public_snapshot_not_pit, constituent_survivorship_applicable=False. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## Side-by-side results

| Model | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x Sharpe | research_pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `raw_avellaneda_lee` | -4.801 | -99.19% | -5.490 | -4.183 | -6.791 | -97.26% | -9.049 | no |
| `crossectional_momentum_12_1` | +0.050 | -33.41% | -0.782 | +0.909 | +0.211 | -20.71% | -0.273 | no |
| `gkx_lightgbm` | -1.649 | -81.32% | -2.328 | -0.986 | -2.243 | -72.40% | -3.274 | no |
| `triple_barrier_meta_av_lee` | -3.208 | -94.75% | -4.140 | -2.352 | -4.783 | -90.90% | -6.268 | no |

## Promotion gates
All four models tested against the same gates:
- dev Sharpe ≥ 1.0
- holdout Sharpe ≥ 0.5
- cost-stress 2× Sharpe > 0
- bootstrap 95% lower-CI Sharpe > 0

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5 and the QuantLab promotion runbook).

## Cross-strategy multiple-testing controls

- **PBO raw_global**: 0.000  (gate: ≤ 0.25)
- **Best strategy**: `crossectional_momentum_12_1`
- **DSR for best**: 0.000  (gate: ≥ 0.50)
- **PSR_zero for best**: 0.553
- **n_strategies in DSR deflation**: 4
