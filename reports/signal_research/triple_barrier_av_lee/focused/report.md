# Triple-Barrier Meta-Labeled Avellaneda-Lee — Backtest Report

## Data quality banner

DATA QUALITY: data_quality_label=survivorship_prototype_only, constituent_survivorship_applicable=True. Universe = current SP500 snapshot from Wikipedia, NOT a point-in-time membership feed. Results may overstate alpha due to survivorship bias. Institutional-grade labels (per spec §5.4) are NOT allowed for this run.

## Universe
- initial universe size: 50
- used in M4 backtest: 50

## Dev metrics (commission 0.5 bps, spread 1.0 bps)
- Sharpe (annualized): 11.463
- Max drawdown: -4.56%
- Cumulative return: 176141.46%
- Trading days: 1994
- Bootstrap 95% CI for Sharpe: [10.001, 13.085]

## Cost stress (2x)
- Sharpe (annualized): 9.465
- Max drawdown: -5.18%

## Holdout metrics (touched once)
- Sharpe (annualized): -1.545
- Max drawdown: -34.78%
- Cumulative return: -31.44%
- Trading days: 850

## Selection funnel
- **universe_initial**: 50
- **after_feature_engineering**: 50
- **after_primary_signal**: 50
- **after_meta_labeler_trained**: 1
- **universe_used_in_m4**: 50
- **dev_sharpe_positive**: 1
- **holdout_sharpe_positive**: 0
- **cost_stress_sharpe_positive**: 1
- **research_pass**: 0
- **promotion_eligible**: 0
- **paper_trade_candidate**: 0
- **production_candidate**: 0

## Disclaimer
Research output only. Past performance does not guarantee future results. 
No promotion to capital deployment occurs without an explicit promotion record 
(spec §6.5 and the QuantLab promotion runbook).
