# Fingerprint-VWAP Meta-Labeling v1 — Result

**Status:** research_only. Not investment advice. No paper. No live.

**Verdict:** DO_NOT_ADVANCE

## Eligibility (primary VWAP entry)
- eligible: True  reason: n/a
- primary net Sharpe: 1.117  events: 4869

## Meta-labeling (net of cost)
- meta net Sharpe: 0.929
- baseline (take-every-entry) net Sharpe: 1.117
- **lift**: -0.189
- deflated Sharpe: {'status': 'computed_approximation', 'probability': 0.3432054829096619, 'z_score': -0.4037304222524274, 'observations': 1027, 'trials': 50, 'annual_sharpe': 0.9288462107411167, 'benchmark_annual_sharpe': 1.1281235843057504, 'sample_skew': 0.4261712073459054, 'sample_kurtosis': 20.70230800572206}
- failed gates: ['deflated_sharpe', 'lift']

## Spec
`FingerprintVwapSpec(windows=(20, 60, 120, 252), vwap_window=5, band=0.0, horizon_days=3, cost_bps_one_way=1.0, train_window_days=252, test_window_days=63, step_days=63, min_train_events=200)`