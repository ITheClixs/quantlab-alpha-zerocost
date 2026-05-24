# Order-Book Microstructure Benchmark `20260524-110854`

## Scope

This run trains order-book signal heads on local Binance futures L2 depth snapshots/updates and backtests future-midprice signals with explicit spread and fee costs.

## Configuration

- Git SHA: `c5493ed150073f455a1f2fe6c69bed3bd54daed8`
- Raw root: `data/raw/huggingface/predict-quant__binance-future-orderbook`
- Symbols: `BTCUSDT`
- Max files per symbol: `1`
- Max rows per file: `50000`
- Max feature rows: `50000`
- Horizons: `(1, 5, 15, 60)`
- Depth levels: `(1, 5, 10, 20)`
- Target column: `future_mid_return_5`
- Min train rows: `20000`
- Test rows: `10000`
- Max folds: `2`
- Fee: `1.0` bps per side
- Min signal abs sweep: `(0.0, 2e-05, 5e-05, 0.0001, 0.0002)`

## Data

- Feature files produced: `1`
- Feature rows loaded: `50,000`
- Symbols loaded: `1`
- Prediction rows: `19,995`
- Prediction symbols: `1`
- Event time range: `1772710958099` to `1772713035695`

## Feature Columns

`ask_depth_1`, `ask_depth_10`, `ask_depth_20`, `ask_depth_5`, `best_ask`, `best_ask_qty`, `best_bid`, `best_bid_qty`, `bid_depth_1`, `bid_depth_10`, `bid_depth_20`, `bid_depth_5`, `imbalance_depth_1`, `imbalance_depth_10`, `imbalance_depth_20`, `imbalance_depth_5`, `imbalance_l1`, `microprice_l1`, `mid_price`, `relative_spread`, `row_index`, `spread`

## Model Accuracy

| model | rows | directional acc. | zero-mean R2 | IC |
|---|---:|---:|---:|---:|
| `ridge` | 19,995 | 65.976% | 0.101292 | 0.327074 |
| `hist_gradient` | 19,995 | 63.146% | 0.0731988 | 0.310573 |
| `ensemble_mean` | 19,995 | 64.966% | 0.101167 | 0.336138 |

## Costed Backtest Sweep

| model | min signal abs | trades | trade rate | hit rate | avg gross/trade | avg cost/trade | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 0 | 19,995 | 100.000% | 1.320% | 0.001% | 0.020% | -0.019% | -97.650% | 31.937% |
| `ridge` | 2e-05 | 7,127 | 35.644% | 2.119% | 0.003% | 0.020% | -0.017% | -70.794% | 22.803% |
| `ridge` | 5e-05 | 195 | 0.975% | 10.256% | 0.008% | 0.020% | -0.012% | -2.375% | 1.595% |
| `ridge` | 0.0001 | 10 | 0.050% | 20.000% | 0.013% | 0.021% | -0.008% | -0.084% | 0.131% |
| `ridge` | 0.0002 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 19,995 | 100.000% | 1.265% | 0.001% | 0.020% | -0.019% | -97.700% | 29.136% |
| `hist_gradient` | 2e-05 | 1,574 | 7.872% | 4.638% | 0.004% | 0.020% | -0.016% | -22.123% | 7.012% |
| `hist_gradient` | 5e-05 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 19,995 | 100.000% | 1.320% | 0.001% | 0.020% | -0.019% | -97.651% | 31.886% |
| `ensemble_mean` | 2e-05 | 3,152 | 15.764% | 3.395% | 0.004% | 0.020% | -0.016% | -39.903% | 13.476% |
| `ensemble_mean` | 5e-05 | 53 | 0.265% | 13.208% | 0.010% | 0.021% | -0.011% | -0.566% | 0.545% |
| `ensemble_mean` | 0.0001 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0002 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |

## Best Candidate

- Model: `ridge`
- Min signal abs: `0.0001`
- Trades: `10`
- Total net return: `-0.084%`
- Gross total return: `0.131%`
- Hit rate: `20.000%`
- Avg net/trade: `-0.008%`

## Limitations

- This is a research benchmark, not a live HFT simulator.
- PnL is based on future midprice markout minus spread crossing and fee assumptions; it does not model queue position, partial fills, latency, exchange rebates, funding, or market impact.
- Raw Binance depth updates are parsed as observed top-of-book snapshots; this benchmark does not rebuild a full exchange book from deltas.
- `not_investment_advice: true`
