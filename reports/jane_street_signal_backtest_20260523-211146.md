# Jane Street S1 Signal Backtest `20260523-211146`

## Scope

This benchmark tests the serious S1 stack on its native Jane Street competition-style market.
The data is anonymized: rows have `date_id`, `time_id`, `symbol_id`, `weight`, `feature_00..feature_78`, and `responder_6`; there are no exchange prices or OHLCV bars, so PnL is reported in weighted responder units rather than dollars.

## Configuration

- Git SHA: `38fe611fac072542fa53ce95e7c809f45083501e`
- Source run: `experiments/alpha_s1/20260523-160541`
- Training source: `data/raw/huggingface/TnnnT0326__Jane_Street_Competition`
- Training row cap: `5000000`
- Target: `responder_6`
- Weight: `weight`
- Holdout date range: `1672` to `1698`
- Holdout rows: `1,008,656`
- Holdout dates: `27`
- Holdout symbols: `39`
- Holdout time buckets: `968`
- Long/short pseudo-backtest: top/bottom `10%` predictions per `date_id`, weighted by Jane Street `weight`.

## Model Metrics

| model | rows | weighted R2 | weighted directional acc. | positive precision | negative precision | weighted sign capture | weighted IC | positive signal share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stacked` | 1,008,656 | 0.00548937 | 0.533109 | 0.509493 | 0.551809 | 0.0413188 | 0.0786741 | 0.441912 |
| `ridge` | 1,008,656 | 0.00391526 | 0.52152 | 0.496683 | 0.544645 | 0.0317581 | 0.0656958 | 0.482141 |
| `lgb` | 1,008,656 | 0.00246686 | 0.531101 | 0.506627 | 0.553831 | 0.0408865 | 0.0735496 | 0.481529 |
| `xgb` | 1,008,656 | -0.00931082 | 0.528986 | 0.504546 | 0.550605 | 0.0365694 | 0.0603805 | 0.469364 |
| `cat` | 1,008,656 | 0.00617079 | 0.526785 | 0.502107 | 0.550534 | 0.0392475 | 0.0789462 | 0.490402 |
| `mlp` | 1,008,656 | -0.00942149 | 0.528836 | 0.505672 | 0.542029 | 0.0279685 | 0.0458685 | 0.362882 |
| `seq` | 1,008,656 | -0.000658894 | 0.51968 | 0.478852 | 0.525202 | 0.00144468 | 0.0065049 | 0.119139 |

## Long/Short Pseudo-Backtest

| model | dates | total pnl units | mean pnl units | mean long-short spread | sharpe-like | hit rate | max drawdown units |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stacked` | 27 | 2.5653 | 0.0950109 | 0.190022 | 30.3879 | 0.962963 | -0.00849656 |
| `ridge` | 27 | 1.94084 | 0.0718828 | 0.143766 | 31.0911 | 0.962963 | -0.0431106 |
| `lgb` | 27 | 2.50648 | 0.0928326 | 0.185665 | 29.8972 | 1 | 0 |
| `xgb` | 27 | 1.96499 | 0.0727774 | 0.145555 | 23.9306 | 0.888889 | -0.0170156 |
| `cat` | 27 | 2.29906 | 0.0851505 | 0.170301 | 27.5742 | 0.925926 | -0.0232957 |
| `mlp` | 27 | 1.69804 | 0.0628905 | 0.125781 | 19.0253 | 0.888889 | -0.0500847 |
| `seq` | 27 | 0.52709 | 0.0195219 | 0.0390437 | 7.83092 | 0.703704 | -0.0535507 |

## Artifacts

- Output dir: `experiments/jane_street_signal_backtests/20260523-211146`
- `summary.json`: all metrics and run metadata
- `stacked_daily_curve.parquet`: date-level pseudo-PnL curve for the stacked model

## Interpretation Notes

- `weighted_zero_mean_r2` is the native Jane Street competition-style score.
- Pseudo-PnL is not dollar PnL because the dataset does not expose tradable prices, spreads, or fills.
- This is the correct native benchmark for the S1 stack; testing this model on S&P/Nasdaq/NYSE OHLCV would require an invalid feature mapping from OHLCV to opaque `feature_00..feature_78`.
- `not_investment_advice: true`
