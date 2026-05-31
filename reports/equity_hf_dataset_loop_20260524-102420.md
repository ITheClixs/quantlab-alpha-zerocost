# HF Equity Dataset Loop `20260524-102420`

## Scope

This run probes Hugging Face equity datasets, fetches missing configured datasets when enabled, rejects non-OHLCV schemas, trains dedicated OHLCV signal heads, and backtests out-of-sample predictions by month.

## Configuration

- Git SHA: `4a7f52255099f55c2b81f0a95309c9d85f8d5ced`
- Target column: `future_return_1`
- Min train dates: `504`
- Test window dates: `63`
- Max folds: `4`
- Max rows per dataset: `None`
- Tail dates per dataset: `1000`
- Min close: `5.0`
- Min dollar volume: `1000000.0`
- Max abs future return: `0.25`
- Cost: `10.0` bps one-way
- Rebalance every N days: `5`
- Selection fraction: `10.00%`
- Target avg monthly net return: `0.0`
- Target total return: `0.0`
- Downloaded this run: `none`

## Leaderboard

| dataset | status | best model | months | avg monthly net | positive months | total net | gross total | rank IC | dir. acc. | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf_hexquant_stocks_daily` | `ok` | `ensemble_mean` | 13 | 1.631% | 61.538% | 22.828% | 35.460% | 0.0295731 | 49.636% |  |

## Best Candidate

- Dataset: `hf_hexquant_stocks_daily`
- Repo: `HexQuant/Stocks-Daily-Price`
- Model: `ensemble_mean`
- Avg monthly net return: `0.0163101`
- Avg monthly net income: `1756`
- Total net return: `0.228281`
- Gross total return: `0.354602`
- Rank IC: `0.0295731`
- Target passed: `true`

## Interpretation Notes

- The controller searches a finite configured dataset list; it does not run unbounded retries until profitability appears.
- Poor net return with positive gross return usually means the signal is too high-turnover for the cost model.
- Rejected datasets remain in the report so adapters can be added deliberately.
- `not_investment_advice: true`
