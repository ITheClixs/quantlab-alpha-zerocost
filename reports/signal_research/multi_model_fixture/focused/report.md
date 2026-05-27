# Multi-Model Comparison — Same Fixture

## Fixture
- universe: top 50 SP500 by ADV
- history: 2015-01-01 → 2026-05-26
- dev:     2015-01-01 → 2022-12-31
- holdout: 2023-01-01 → 2026-05-26
- costs: 0.5 bps commission + 10.0 bps spread
- cost-stress multiplier: 2.0×

## Data quality banner

DATA QUALITY: data_quality_label=survivorship_prototype_only, constituent_survivorship_applicable=True. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## Side-by-side results

| Model | dev Sharpe | dev DD | dev CI_lo | dev CI_hi | holdout Sharpe | holdout DD | cost-2x Sharpe | research_pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `raw_avellaneda_lee` | -2.471 | -76.57% | -3.222 | -1.775 | -2.759 | -50.03% | -5.243 | no |
| `crossectional_momentum_12_1` | +0.260 | -21.47% | -0.314 | +0.890 | +0.576 | -30.73% | +0.117 | no |
| `gkx_lightgbm` | -0.490 | -40.04% | -1.133 | +0.179 | -1.383 | -39.33% | -1.158 | no |
| `triple_barrier_meta_av_lee` | -1.509 | -61.34% | -2.332 | -0.779 | -1.686 | -36.70% | -3.595 | no |

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
