# Order-Book Microstructure Benchmark `20260524-201858`

## Scope

This run trains order-book signal heads on local Binance futures L2 depth snapshots/updates and backtests future-midprice signals with explicit spread and fee costs.

## Configuration

- Git SHA: `8ef3f423b343e33c2b8b7d5ff8fac371d503013b`
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
- Min edge over cost sweep: `(0.0, 5e-05, 0.0001, 0.0002)`
- Max relative spread: `5e-06`
- Min entry depth: `0.5`

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

| model | min signal abs | min edge > cost | candidates | filtered | trades | trade rate | hit rate | avg gross/trade | avg cost/trade | avg net/trade | total net | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ridge` | 0 | 0 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0 | 5e-05 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0 | 0.0001 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0 | 0.0002 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 2e-05 | 0 | 7,127 | 7,127 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 2e-05 | 5e-05 | 7,127 | 7,127 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 2e-05 | 0.0001 | 7,127 | 7,127 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 2e-05 | 0.0002 | 7,127 | 7,127 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 5e-05 | 0 | 195 | 195 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 5e-05 | 5e-05 | 195 | 195 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 5e-05 | 0.0001 | 195 | 195 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 5e-05 | 0.0002 | 195 | 195 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0001 | 0 | 10 | 10 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0001 | 5e-05 | 10 | 10 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0001 | 0.0001 | 10 | 10 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0001 | 0.0002 | 10 | 10 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0002 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0002 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0002 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ridge` | 0.0002 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 0 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 5e-05 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 0.0001 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0 | 0.0002 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0 | 1,574 | 1,574 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 5e-05 | 1,574 | 1,574 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0.0001 | 1,574 | 1,574 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 2e-05 | 0.0002 | 1,574 | 1,574 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 5e-05 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0001 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `hist_gradient` | 0.0002 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 0 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 5e-05 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 0.0001 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0 | 0.0002 | 19,995 | 19,995 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 2e-05 | 0 | 3,152 | 3,152 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 2e-05 | 5e-05 | 3,152 | 3,152 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 2e-05 | 0.0001 | 3,152 | 3,152 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 2e-05 | 0.0002 | 3,152 | 3,152 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 5e-05 | 0 | 53 | 53 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 5e-05 | 5e-05 | 53 | 53 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 5e-05 | 0.0001 | 53 | 53 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 5e-05 | 0.0002 | 53 | 53 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0001 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0001 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0001 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0001 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0002 | 0 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0002 | 5e-05 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0002 | 0.0001 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `ensemble_mean` | 0.0002 | 0.0002 | 0 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |

## Best Candidate

No costed backtest candidate was produced.

## Gate Diagnostic

- Backtest variants with raw candidates: `36` / `60`
- Backtest variants with executed trades: `0` / `60`
- Max raw candidates in one variant: `19,995`
- Max executed trades in one variant: `0`
- All raw candidates were filtered after applying predicted-edge-over-cost, spread, and depth gates. This indicates directional predictability without enough predicted markout magnitude to pay aggressive execution costs.

## Limitations

- This is a research benchmark, not a live HFT simulator.
- PnL is based on future midprice markout minus spread crossing and fee assumptions; it does not model queue position, partial fills, latency, exchange rebates, funding, or market impact.
- Raw Binance depth updates are parsed as observed top-of-book snapshots; this benchmark does not rebuild a full exchange book from deltas.
- `not_investment_advice: true`
