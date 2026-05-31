# Order-Book Microstructure Benchmark `20260525-162557`

## Scope

This run trains order-book signal heads on local Binance futures L2 depth snapshots/updates and backtests future-midprice signals with explicit spread and fee costs.

## Configuration

- Git SHA: `85abb9f5ed2c6a641c1db056d8a922648ac1773b`
- Raw root: `data/raw/huggingface/predict-quant__binance-future-orderbook`
- Symbols: `BTCUSDT, ETHUSDT, SOLUSDT`
- Max files per symbol: `12`
- Max rows per file: `30000`
- Max feature rows: `750000`
- Horizons: `(1, 5, 15, 60)`
- Depth levels: `(1, 5, 10, 20)`
- Target column: `future_mid_return_5`
- Min train rows: `120000`
- Test rows: `40000`
- Max folds: `4`
- Fee: `1.0` bps per side
- Slippage: `0.5` bps per side
- Min signal abs sweep: `(0.0, 2e-05, 5e-05, 0.0001, 0.0002)`
- Min edge over cost sweep: `(0.0, 5e-05, 0.0001, 0.0002)`
- Edge-to-cost k sweep: `(1.0, 1.5, 2.0, 2.5, 3.0, 4.0)`
- Target horizon sweep: `()`
- Max relative spread: `None`
- Min entry depth: `None`

## Data

- Feature files produced: `36`
- Feature source date coverage: `2026-03-05` to `2026-03-24` across `16` file dates
- Feature rows loaded: `750,000`
- Symbols loaded: `3`
- Prediction rows: `149,875`
- Prediction symbols: `2`
- Event time range: `1772707847547` to `1773795272757`

## Feature Columns

`ask_depth_1`, `ask_depth_10`, `ask_depth_20`, `ask_depth_5`, `best_ask`, `best_ask_qty`, `best_bid`, `best_bid_qty`, `bid_depth_1`, `bid_depth_10`, `bid_depth_20`, `bid_depth_5`, `imbalance_depth_1`, `imbalance_depth_10`, `imbalance_depth_20`, `imbalance_depth_5`, `imbalance_l1`, `microprice_l1`, `mid_price`, `relative_spread`, `row_index`, `spread`

## Model Accuracy

| model | rows | directional acc. | zero-mean R2 | IC |
|---|---:|---:|---:|---:|
| `ridge` | 149,875 | 72.724% | -3.54737 | 0.0239614 |
| `hist_gradient` | 149,875 | 72.505% | 0.0350827 | 0.187428 |
| `ensemble_mean` | 149,875 | 72.913% | -0.859145 | 0.041336 |

## Costed Backtest Sweep

Selection prioritizes annualized daily Sharpe when at least two daily returns are available, then event-trade Sharpe, then net return.

| model | min signal abs | min edge > cost | candidates | filtered | trades | daily Sharpe | trade Sharpe | trade rate | hit rate | avg gross/trade | avg cost/trade | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 0 | 0 | 149,875 | 141,497 | 8,378 | 0 | -321.131 | 5.590% | 0.847% | -0.000% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 0 | 5e-05 | 149,875 | 141,504 | 8,371 | 0 | -321.353 | 5.585% | 0.848% | -0.000% | 0.041% | -0.041% | -96.763% | -1.387% |
| `ridge` | 0 | 0.0001 | 149,875 | 141,561 | 8,314 | 0 | -321.765 | 5.547% | 0.830% | -0.000% | 0.041% | -0.041% | -96.693% | -1.552% |
| `ridge` | 0 | 0.0002 | 149,875 | 142,476 | 7,399 | 0 | -306.787 | 4.937% | 0.797% | -0.000% | 0.041% | -0.041% | -95.236% | -2.378% |
| `ridge` | 2e-05 | 0 | 71,133 | 62,755 | 8,378 | 0 | -321.131 | 5.590% | 0.847% | -0.000% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 2e-05 | 5e-05 | 71,133 | 62,762 | 8,371 | 0 | -321.353 | 5.585% | 0.848% | -0.000% | 0.041% | -0.041% | -96.763% | -1.387% |
| `ridge` | 2e-05 | 0.0001 | 71,133 | 62,819 | 8,314 | 0 | -321.765 | 5.547% | 0.830% | -0.000% | 0.041% | -0.041% | -96.693% | -1.552% |
| `ridge` | 2e-05 | 0.0002 | 71,133 | 63,734 | 7,399 | 0 | -306.787 | 4.937% | 0.797% | -0.000% | 0.041% | -0.041% | -95.236% | -2.378% |
| `ridge` | 5e-05 | 0 | 13,848 | 5,470 | 8,378 | 0 | -321.131 | 5.590% | 0.847% | -0.000% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 5e-05 | 5e-05 | 13,848 | 5,477 | 8,371 | 0 | -321.353 | 5.585% | 0.848% | -0.000% | 0.041% | -0.041% | -96.763% | -1.387% |
| `ridge` | 5e-05 | 0.0001 | 13,848 | 5,534 | 8,314 | 0 | -321.765 | 5.547% | 0.830% | -0.000% | 0.041% | -0.041% | -96.693% | -1.552% |
| `ridge` | 5e-05 | 0.0002 | 13,848 | 6,449 | 7,399 | 0 | -306.787 | 4.937% | 0.797% | -0.000% | 0.041% | -0.041% | -95.236% | -2.378% |
| `ridge` | 0.0001 | 0 | 8,711 | 333 | 8,378 | 0 | -321.131 | 5.590% | 0.847% | -0.000% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 0.0001 | 5e-05 | 8,711 | 340 | 8,371 | 0 | -321.353 | 5.585% | 0.848% | -0.000% | 0.041% | -0.041% | -96.763% | -1.387% |
| `ridge` | 0.0001 | 0.0001 | 8,711 | 397 | 8,314 | 0 | -321.765 | 5.547% | 0.830% | -0.000% | 0.041% | -0.041% | -96.693% | -1.552% |
| `ridge` | 0.0001 | 0.0002 | 8,711 | 1,312 | 7,399 | 0 | -306.787 | 4.937% | 0.797% | -0.000% | 0.041% | -0.041% | -95.236% | -2.378% |
| `ridge` | 0.0002 | 0 | 8,398 | 20 | 8,378 | 0 | -321.131 | 5.590% | 0.847% | -0.000% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 0.0002 | 5e-05 | 8,398 | 27 | 8,371 | 0 | -321.353 | 5.585% | 0.848% | -0.000% | 0.041% | -0.041% | -96.763% | -1.387% |
| `ridge` | 0.0002 | 0.0001 | 8,398 | 84 | 8,314 | 0 | -321.765 | 5.547% | 0.830% | -0.000% | 0.041% | -0.041% | -96.693% | -1.552% |
| `ridge` | 0.0002 | 0.0002 | 8,398 | 999 | 7,399 | 0 | -306.787 | 4.937% | 0.797% | -0.000% | 0.041% | -0.041% | -95.236% | -2.378% |
| `hist_gradient` | 0 | 0 | 149,875 | 149,875 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 5e-05 | 149,875 | 149,875 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 0.0001 | 149,875 | 149,875 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 0.0002 | 149,875 | 149,875 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0 | 22,903 | 22,903 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 5e-05 | 22,903 | 22,903 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0.0001 | 22,903 | 22,903 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0.0002 | 22,903 | 22,903 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0 | 1,535 | 1,535 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 5e-05 | 1,535 | 1,535 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0.0001 | 1,535 | 1,535 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0.0002 | 1,535 | 1,535 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0 | 525 | 525 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 5e-05 | 525 | 525 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0.0001 | 525 | 525 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0.0002 | 525 | 525 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 5e-05 | 0 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0.0001 | 0 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0.0002 | 0 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 0 | 149,875 | 148,096 | 1,779 | 0 | -145.413 | 1.187% | 0.562% | 0.000% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 0 | 5e-05 | 149,875 | 149,405 | 470 | 0 | -86.3732 | 0.314% | 0.426% | 0.001% | 0.041% | -0.040% | -17.271% | 0.249% |
| `ensemble_mean` | 0 | 0.0001 | 149,875 | 149,762 | 113 | 0 | -41.6486 | 0.075% | 1.770% | 0.003% | 0.041% | -0.038% | -4.217% | 0.319% |
| `ensemble_mean` | 0 | 0.0002 | 149,875 | 149,873 | 2 | 0 | -0.389825 | 0.001% | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ensemble_mean` | 2e-05 | 0 | 46,992 | 45,213 | 1,779 | 0 | -145.413 | 1.187% | 0.562% | 0.000% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 2e-05 | 5e-05 | 46,992 | 46,522 | 470 | 0 | -86.3732 | 0.314% | 0.426% | 0.001% | 0.041% | -0.040% | -17.271% | 0.249% |
| `ensemble_mean` | 2e-05 | 0.0001 | 46,992 | 46,879 | 113 | 0 | -41.6486 | 0.075% | 1.770% | 0.003% | 0.041% | -0.038% | -4.217% | 0.319% |
| `ensemble_mean` | 2e-05 | 0.0002 | 46,992 | 46,990 | 2 | 0 | -0.389825 | 0.001% | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ensemble_mean` | 5e-05 | 0 | 10,423 | 8,644 | 1,779 | 0 | -145.413 | 1.187% | 0.562% | 0.000% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 5e-05 | 5e-05 | 10,423 | 9,953 | 470 | 0 | -86.3732 | 0.314% | 0.426% | 0.001% | 0.041% | -0.040% | -17.271% | 0.249% |
| `ensemble_mean` | 5e-05 | 0.0001 | 10,423 | 10,310 | 113 | 0 | -41.6486 | 0.075% | 1.770% | 0.003% | 0.041% | -0.038% | -4.217% | 0.319% |
| `ensemble_mean` | 5e-05 | 0.0002 | 10,423 | 10,421 | 2 | 0 | -0.389825 | 0.001% | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ensemble_mean` | 0.0001 | 0 | 8,398 | 6,619 | 1,779 | 0 | -145.413 | 1.187% | 0.562% | 0.000% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 0.0001 | 5e-05 | 8,398 | 7,928 | 470 | 0 | -86.3732 | 0.314% | 0.426% | 0.001% | 0.041% | -0.040% | -17.271% | 0.249% |
| `ensemble_mean` | 0.0001 | 0.0001 | 8,398 | 8,285 | 113 | 0 | -41.6486 | 0.075% | 1.770% | 0.003% | 0.041% | -0.038% | -4.217% | 0.319% |
| `ensemble_mean` | 0.0001 | 0.0002 | 8,398 | 8,396 | 2 | 0 | -0.389825 | 0.001% | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ensemble_mean` | 0.0002 | 0 | 8,386 | 6,607 | 1,779 | 0 | -145.413 | 1.187% | 0.562% | 0.000% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 0.0002 | 5e-05 | 8,386 | 7,916 | 470 | 0 | -86.3732 | 0.314% | 0.426% | 0.001% | 0.041% | -0.040% | -17.271% | 0.249% |
| `ensemble_mean` | 0.0002 | 0.0001 | 8,386 | 8,273 | 113 | 0 | -41.6486 | 0.075% | 1.770% | 0.003% | 0.041% | -0.038% | -4.217% | 0.319% |
| `ensemble_mean` | 0.0002 | 0.0002 | 8,386 | 8,384 | 2 | 0 | -0.389825 | 0.001% | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% | 0.081% |

## Cost-Aware Threshold Sweep

This sweep trades only when `abs(prediction) > k * estimated_round_trip_cost`.

| model | k | candidates | filtered | trades | gross hit | net hit | daily Sharpe | max drawdown | avg cost/trade | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 1 | 149,875 | 141,497 | 8,378 | 17.713% | 0.847% | 0 | -96.773% | 0.041% | -0.041% | -96.774% | -1.435% |
| `ridge` | 1.5 | 149,875 | 142,562 | 7,313 | 17.079% | 0.807% | 0 | -95.064% | 0.041% | -0.041% | -95.066% | -2.388% |
| `ridge` | 2 | 149,875 | 148,138 | 1,737 | 18.538% | 0.576% | 0 | -50.553% | 0.041% | -0.041% | -50.573% | 0.432% |
| `ridge` | 2.5 | 149,875 | 149,765 | 110 | 14.545% | 1.818% | 0 | -4.074% | 0.041% | -0.038% | -4.100% | 0.319% |
| `ridge` | 3 | 149,875 | 149,873 | 2 | 100.000% | 50.000% | 0 | 0.000% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ridge` | 4 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 1 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 1.5 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2.5 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 3 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 4 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 1 | 149,875 | 148,096 | 1,779 | 18.606% | 0.562% | 0 | -51.356% | 0.041% | -0.041% | -51.376% | 0.507% |
| `ensemble_mean` | 1.5 | 149,875 | 149,873 | 2 | 100.000% | 50.000% | 0 | 0.000% | 0.049% | -0.008% | -0.017% | 0.081% |
| `ensemble_mean` | 2 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 2.5 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 3 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 4 | 149,875 | 149,875 | 0 | 0.000% | 0.000% | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |

## Best Candidate

- Model: `ensemble_mean`
- Min signal abs: `0.0`
- Min edge over cost: `0.0002`
- Edge-to-cost k: `None`
- Trades: `2`
- Candidates: `149,875`
- Filtered: `149,873`
- Total net return: `-0.017%`
- Gross total return: `0.081%`
- Gross hit rate: `100.000%`
- Net hit rate: `50.000%`
- Daily Sharpe: `0`
- Event-trade Sharpe: `-0.389825`
- Max drawdown: `0.000%`
- Daily return count: `1`
- Hit rate: `50.000%`
- Avg net/trade: `-0.008%`
- Avg spread cost/trade: `0.019%`
- Avg fee cost/trade: `0.020%`
- Long net return: `-0.017%` on `2` trades
- Short net return: `0.000%` on `0` trades

## Failure Classification

- Labels: `genuinely predictive but untradable signal`
- Evidence: inverted signal did not improve gross or net performance

## Trading Conversion Debug

- Best-trade audit parquet: `experiments/orderbook_microstructure_multisymbol/20260525-162557/best_trade_pnl_audit.parquet`
- Best-trade audit CSV: `experiments/orderbook_microstructure_multisymbol/20260525-162557/best_trade_pnl_audit.csv`
- All-row prediction metrics: `{'rows': 149875, 'ic': 0.04133601860158968, 'zero_mean_r2': -0.859145323530992, 'directional_accuracy': 0.7291331617883564}`
- Traded-row prediction metrics: `{'rows': 2, 'ic': 1.0, 'zero_mean_r2': 0.005491337910585425, 'directional_accuracy': 1.0}`

### Cost Regimes

The first table holds the selected trade set fixed and recomputes PnL under each cost component.

| regime | trades | gross hit | net hit | avg gross/trade | avg net/trade | total net |
|---|---:|---:|---:|---:|---:|---:|
| `selected_no_cost` | 2 | 100.000% | 100.000% | 0.040% | 0.040% | 0.081% |
| `selected_spread_only` | 2 | 100.000% | 50.000% | 0.040% | 0.022% | 0.043% |
| `selected_fee_only` | 2 | 100.000% | 50.000% | 0.040% | 0.020% | 0.041% |
| `selected_full_cost` | 2 | 100.000% | 50.000% | 0.040% | -0.008% | -0.017% |

The second table reruns the backtest under each cost regime, so the trade set may change as gates change.

| regime | trades | gross hit | net hit | avg gross/trade | avg spread cost | avg fee cost | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cost` | 8,398 | 17.790% | 17.766% | -0.000% | 0.000% | 0.000% | -0.010% | -57.397% | -1.332% |
| `spread_only` | 8,374 | 17.710% | 5.660% | -0.000% | 0.011% | 0.000% | -0.021% | -82.746% | -1.440% |
| `fee_only` | 7,609 | 17.098% | 1.892% | -0.000% | 0.000% | 0.020% | -0.030% | -90.066% | -2.578% |
| `spread_plus_fee` | 1,779 | 18.606% | 0.562% | 0.000% | 0.011% | 0.020% | -0.041% | -51.376% | 0.507% |

### Inverted Signal

- Trades: `1,779`
- Gross hit rate: `15.852%`
- Net hit rate: `0.731%`
- Total net return: `-51.867%`
- Gross total return: `-0.507%`

### Null Baselines

| baseline | trades | gross hit | net hit | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|
| `random_same_trade_count` | 2 | 50.000% | 0.000% | -0.030% | -0.060% | 0.011% |
| `shuffled_predictions` | 6,048 | 16.782% | 0.579% | -0.032% | -85.191% | -2.043% |
| `always_long` | 2 | 0.000% | 0.000% | -0.030% | -0.061% | 0.000% |
| `always_short` | 2 | 0.000% | 0.000% | -0.030% | -0.061% | 0.000% |

### Edge-To-Cost Buckets

| bucket | trades | net hit | avg gross | avg cost | avg net | total net |
|---|---:|---:|---:|---:|---:|---:|
| `1.5-2x` | 2 | 50.000% | 0.040% | 0.049% | -0.008% | -0.017% |

### Spread Regime

| bucket | trades | net hit | avg gross | avg cost | avg net | total net |
|---|---:|---:|---:|---:|---:|---:|
| `medium` | 1 | 100.000% | 0.065% | 0.052% | 0.013% | 0.013% |
| `tight` | 1 | 0.000% | 0.016% | 0.046% | -0.030% | -0.030% |

### Volatility Regime

| bucket | trades | net hit | avg gross | avg cost | avg net | total net |
|---|---:|---:|---:|---:|---:|---:|
| `low_vol` | 1 | 0.000% | 0.016% | 0.046% | -0.030% | -0.030% |
| `mid_vol` | 1 | 100.000% | 0.065% | 0.052% | 0.013% | 0.013% |

### Liquidity Regime

| bucket | trades | net hit | avg gross | avg cost | avg net | total net |
|---|---:|---:|---:|---:|---:|---:|
| `normal` | 1 | 100.000% | 0.065% | 0.052% | 0.013% | 0.013% |
| `thin` | 1 | 0.000% | 0.016% | 0.046% | -0.030% | -0.030% |

## Horizon Sweep

| horizon | best model | k | IC | zero-mean R2 | directional acc. | trades | gross hit | net hit | avg gross/trade | avg net/trade | turnover | net PnL | gross PnL |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Gate Diagnostic

- Backtest variants with raw candidates: `56` / `60`
- Backtest variants with executed trades: `40` / `60`
- Max raw candidates in one variant: `149,875`
- Max executed trades in one variant: `8,378`

## Limitations

- This is a research benchmark, not a live HFT simulator.
- PnL is based on future midprice markout minus spread crossing and fee assumptions; it does not model queue position, partial fills, latency, exchange rebates, funding, or market impact.
- Daily Sharpe is annualized from grouped UTC event dates when at least two daily returns are available; event-trade Sharpe is not a calendar annualized live Sharpe.
- Raw Binance depth updates are parsed as observed top-of-book snapshots; this benchmark does not rebuild a full exchange book from deltas.
- `not_investment_advice: true`
