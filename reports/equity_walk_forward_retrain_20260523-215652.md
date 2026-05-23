# Equity Walk-Forward Retrain Report `20260523-215652`

## Scope

This benchmark trains dedicated OHLCV signal heads separately for each selected US equity universe.
Every fold trains only on dates before the test window, then evaluates out-of-sample predictions with the same cost-aware daily dollar-neutral long/short backtest used for the generic OHLCV benchmark.

## Configuration

- Git SHA: `3ec990450e4a597198c4267721df63090da28f4a`
- Target: `future_return_1`
- Features: past/current OHLCV-derived features only
- Models: `ridge, hist_gradient, ensemble_mean`
- Minimum training dates: `504`
- Test window dates: `63`
- Step dates: `63`
- Max folds: `4`
- Max train rows per fold: `150000`
- Selection fraction: `10.00%`
- Cost model: `5.0` bps one-way, two gross turns per daily rebalance
- Starting equity: `100000.0`
- Max rows per dataset: `None`
- Tail dates per dataset: `1000`
- Save predictions: `true`

## Results

### `sp500` - S&P 500 daily equities

- Source: `data/raw/huggingface/jwigginton__timeseries-daily-sp500/data/train-00000-of-00001.parquet`
- Raw rows: `4,026,556`
- Normalized rows: `499,986`
- Symbols: `503`
- Dates: `1,000`
- Walk-forward folds: `4`
- Feature count: `13`
- Artifacts: `experiments/equity_walk_forward/20260523-215652/sp500`

| model | rows | directional acc. | rank IC | top-bottom spread | net return | gross return | cost drag | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 122,064 | 50.610% | -0.00611192 | 0.020% | -19.944% | 2.087% | 22.031% | -2.77034 | -21.598% |
| `hist_gradient` | 122,064 | 50.136% | -0.00534498 | 0.012% | -20.556% | 1.307% | 21.864% | -3.63832 | -20.714% |
| `ensemble_mean` | 122,064 | 50.652% | -0.00597673 | 0.008% | -21.064% | 0.660% | 21.725% | -3.0011 | -21.897% |

Fold windows:

| fold | train dates | train window | test dates | test window | train rows | test rows |
|---:|---:|---|---:|---|---:|---:|
| 0 | 756 | `2020-03-20` to `2023-03-21` | 63 | `2023-03-22` to `2023-06-21` | 150,000 | 31,596 |
| 1 | 819 | `2020-03-20` to `2023-06-21` | 63 | `2023-06-22` to `2023-09-20` | 150,000 | 31,626 |
| 2 | 882 | `2020-03-20` to `2023-09-20` | 63 | `2023-09-21` to `2023-12-19` | 150,000 | 31,680 |
| 3 | 945 | `2020-03-20` to `2023-12-19` | 54 | `2023-12-20` to `2024-03-08` | 150,000 | 27,162 |

### `nasdaq` - NASDAQ daily equities

- Source: `data/raw/huggingface/benstaf__nasdaq_2013_2023/trade_data_2019_2023.csv`
- Raw rows: `105,588`
- Normalized rows: `84,000`
- Symbols: `84`
- Dates: `1,000`
- Walk-forward folds: `4`
- Feature count: `13`
- Artifacts: `experiments/equity_walk_forward/20260523-215652/nasdaq`

| model | rows | directional acc. | rank IC | top-bottom spread | net return | gross return | cost drag | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 20,412 | 50.691% | 0.0111351 | -0.006% | -22.588% | -1.280% | 21.308% | -2.28616 | -25.309% |
| `hist_gradient` | 20,412 | 52.445% | -0.00609363 | -0.110% | -31.749% | -12.952% | 18.797% | -3.83095 | -32.484% |
| `ensemble_mean` | 20,412 | 51.122% | 0.00679219 | -0.016% | -23.529% | -2.479% | 21.050% | -2.44779 | -25.673% |

Fold windows:

| fold | train dates | train window | test dates | test window | train rows | test rows |
|---:|---:|---|---:|---|---:|---:|
| 0 | 756 | `2020-01-09` to `2023-01-09` | 63 | `2023-01-10` to `2023-04-11` | 63,504 | 5,292 |
| 1 | 819 | `2020-01-09` to `2023-04-11` | 63 | `2023-04-12` to `2023-07-12` | 68,796 | 5,292 |
| 2 | 882 | `2020-01-09` to `2023-07-12` | 63 | `2023-07-13` to `2023-10-10` | 74,088 | 5,292 |
| 3 | 945 | `2020-01-09` to `2023-10-10` | 54 | `2023-10-11` to `2023-12-27` | 79,380 | 4,536 |

### `nyse` - NYSE daily equities

- Source: `data/raw/kaggle/datasets/dgawlik__nyse/prices-split-adjusted.csv`
- Raw rows: `851,264`
- Normalized rows: `493,322`
- Symbols: `501`
- Dates: `1,000`
- Walk-forward folds: `4`
- Feature count: `13`
- Artifacts: `experiments/equity_walk_forward/20260523-215652/nyse`

| model | rows | directional acc. | rank IC | top-bottom spread | net return | gross return | cost drag | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 121,625 | 48.436% | -0.00482473 | 0.045% | -17.514% | 5.183% | 22.697% | -2.10655 | -17.556% |
| `hist_gradient` | 121,625 | 49.168% | 0.0109718 | 0.090% | -12.714% | 11.297% | 24.011% | -2.19673 | -14.173% |
| `ensemble_mean` | 121,625 | 48.557% | 0.000887015 | 0.024% | -19.475% | 2.686% | 22.160% | -2.69361 | -19.843% |

Fold windows:

| fold | train dates | train window | test dates | test window | train rows | test rows |
|---:|---:|---|---:|---|---:|---:|
| 0 | 756 | `2013-01-14` to `2016-01-13` | 63 | `2016-01-14` to `2016-04-14` | 150,000 | 31,500 |
| 1 | 819 | `2013-01-14` to `2016-04-14` | 63 | `2016-04-15` to `2016-07-14` | 150,000 | 31,508 |
| 2 | 882 | `2013-01-14` to `2016-07-14` | 63 | `2016-07-15` to `2016-10-12` | 150,000 | 31,563 |
| 3 | 945 | `2013-01-14` to `2016-10-12` | 54 | `2016-10-13` to `2016-12-29` | 150,000 | 27,054 |

## Interpretation Notes

- This is still daily OHLCV, not futures tick/order-book data, so it cannot prove HFT viability.
- The benchmark is leakage-aware at the date level but does not model short borrow, exchange-specific fees, queue position, or intraday fills.
- `ensemble_mean` is a fixed average of ridge and histogram-gradient predictions inside each fold; it is cooperative but intentionally simple.
- `not_investment_advice: true`
