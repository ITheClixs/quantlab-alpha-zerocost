# Equity Signal Backtest Report `20260523-202904`

## Scope

This report evaluates the locally persisted market signal head on historical S&P 500 daily equities, NASDAQ daily equities, and NYSE daily equities data.
The Jane Street S1 stack is not applied here because its inference contract is opaque `feature_00..feature_78`; mapping listed-equity OHLCV into those features would be out-of-domain.

## Configuration

- Git SHA: `caf6396f99963ab2ceb01050105add63820a449a`
- Signal model artifact: `experiments/local_signal_training/market/ridge.joblib`
- Model feature count: `11`
- Model features: `close, high, high_low_range, log_return_1, low, open, realized_vol_20, realized_vol_5, realized_vol_60, return_1, volume`
- Strategy: daily dollar-neutral long/short, top/bottom `10.00%` by predicted next-day return
- Cost model: `5.0` bps per one-way notional, two gross notional turns per daily rebalance
- Starting equity: `100000.0`
- Max rows per dataset: `None`
- Max symbols per side: `None`
- Save signal parquet: `true`
- Target: `future_return_1`

## Headline Results

| dataset | rows | symbols | dates | directional acc. | rank IC | top-bottom spread | net return | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sp500` | 3,822,786 | 503 | 11,141 | 51.004% | 0.00178749 | -0.011% | -100.000% | -1.55946 | -100.000% |
| `nasdaq` | 105,504 | 84 | 1,257 | 49.283% | 0.0033041 | 0.051% | -62.154% | -1.62866 | -62.219% |
| `nyse` | 850,763 | 501 | 1,762 | 49.193% | 0.00287559 | 0.009% | -81.453% | -5.52615 | -81.656% |

## Dataset Details

### `sp500` - S&P 500 daily equities

- Source: `data/raw/huggingface/jwigginton__timeseries-daily-sp500/data/train-00000-of-00001.parquet`
- Date range: `1980-01-02` to `2024-03-11`
- Normalized rows: `3,823,289`
- Evaluated signal rows: `3,822,786`
- Symbols: `503`

| metric | value |
|---|---:|
| `directional_accuracy` | 51.004% |
| `positive_precision` | 50.305% |
| `negative_precision` | 46.407% |
| `zero_mean_r2` | -643.255 |
| `information_coefficient` | 0.000896771 |
| `rank_ic_mean` | 0.00178749 |
| `rank_ic_std` | 0.103507 |
| `top_mean_forward_return` | 0.135% |
| `bottom_mean_forward_return` | 0.147% |
| `top_bottom_spread_return` | -0.011% |
| `positive_signal_share` | 19.250% |

| backtest metric | value |
|---|---:|
| `n_days` | 11,140 |
| `total_return` | -100.000% |
| `gross_total_return` | -68.009% |
| `cost_drag_return` | 31.991% |
| `annualized_return` | -24.266% |
| `sharpe_ratio` | -1.55946 |
| `max_drawdown` | -100.000% |
| `hit_rate` | 38.043% |
| `avg_daily_turnover` | 2 |
| `avg_daily_net_return` | -0.106% |
| `avg_daily_gross_return` | -0.006% |

Artifacts: `experiments/equity_signal_backtests/20260523-202904/sp500`

### `nasdaq` - NASDAQ daily equities

- Source: `data/raw/huggingface/benstaf__nasdaq_2013_2023/trade_data_2019_2023.csv`
- Date range: `2019-01-02` to `2023-12-28`
- Normalized rows: `105,588`
- Evaluated signal rows: `105,504`
- Symbols: `84`

| metric | value |
|---|---:|
| `directional_accuracy` | 49.283% |
| `positive_precision` | 52.953% |
| `negative_precision` | 47.061% |
| `zero_mean_r2` | -3980.09 |
| `information_coefficient` | 0.0102426 |
| `rank_ic_mean` | 0.0033041 |
| `rank_ic_std` | 0.152936 |
| `top_mean_forward_return` | 0.161% |
| `bottom_mean_forward_return` | 0.110% |
| `top_bottom_spread_return` | 0.051% |
| `positive_signal_share` | 34.279% |

| backtest metric | value |
|---|---:|
| `n_days` | 1,256 |
| `total_return` | -62.154% |
| `gross_total_return` | 32.942% |
| `cost_drag_return` | 95.095% |
| `annualized_return` | -17.712% |
| `sharpe_ratio` | -1.62866 |
| `max_drawdown` | -62.219% |
| `hit_rate` | 44.108% |
| `avg_daily_turnover` | 2 |
| `avg_daily_net_return` | -0.075% |
| `avg_daily_gross_return` | 0.025% |

Artifacts: `experiments/equity_signal_backtests/20260523-202904/nasdaq`

### `nyse` - NYSE daily equities

- Source: `data/raw/kaggle/datasets/dgawlik__nyse/prices-split-adjusted.csv`
- Date range: `2010-01-04` to `2016-12-30`
- Normalized rows: `851,264`
- Evaluated signal rows: `850,763`
- Symbols: `501`

| metric | value |
|---|---:|
| `directional_accuracy` | 49.193% |
| `positive_precision` | 51.463% |
| `negative_precision` | 47.632% |
| `zero_mean_r2` | -311.37 |
| `information_coefficient` | 0.00117941 |
| `rank_ic_mean` | 0.00287559 |
| `rank_ic_std` | 0.0971225 |
| `top_mean_forward_return` | 0.068% |
| `bottom_mean_forward_return` | 0.058% |
| `top_bottom_spread_return` | 0.009% |
| `positive_signal_share` | 26.851% |

| backtest metric | value |
|---|---:|
| `n_days` | 1,761 |
| `total_return` | -81.453% |
| `gross_total_return` | 7.997% |
| `cost_drag_return` | 89.450% |
| `annualized_return` | -21.424% |
| `sharpe_ratio` | -5.52615 |
| `max_drawdown` | -81.656% |
| `hit_rate` | 32.879% |
| `avg_daily_turnover` | 2 |
| `avg_daily_net_return` | -0.095% |
| `avg_daily_gross_return` | 0.005% |

Artifacts: `experiments/equity_signal_backtests/20260523-202904/nyse`

## Interpretation

- Treat these as out-of-sample transfer diagnostics for the locally trained generic OHLCV signal head, not as production trading results.
- A useful signal should show positive rank IC, positive top-bottom forward-return spread, and survive the explicit cost model.
- Negative zero-mean R2 can coexist with useful rank spread; for trading, ranking quality and costed portfolio returns matter more than raw point-forecast MSE.
- The current model artifact was not trained specifically on these US equity universes. A dedicated walk-forward retrain per universe should be the next upgrade before promotion decisions.

## Limitations

- Daily close-to-close simulation only; no intraday fill timing, opening auction, borrow fees, locate constraints, dividends, delistings, or point-in-time constituent membership are modeled.
- The long/short book uses equal weights and a simple daily full-rebalance cost approximation.
- The S&P and NYSE historical datasets may contain survivorship and vendor-adjustment bias. Use point-in-time universe files before treating results as capacity evidence.
- `not_investment_advice: true`
