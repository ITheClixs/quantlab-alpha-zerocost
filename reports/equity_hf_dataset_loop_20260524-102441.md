# HF Equity Dataset Loop `20260524-102441`

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
- Rebalance every N days: `21`
- Selection fraction: `10.00%`
- Target avg monthly net return: `0.0`
- Target total return: `0.0`
- Downloaded this run: `none`

## Leaderboard

| dataset | status | best model | months | avg monthly net | positive months | total net | gross total | rank IC | dir. acc. | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf_hexquant_stocks_daily` | `ok` | `hist_gradient` | 13 | 1.313% | 76.923% | 18.193% | 21.064% | 0.0152049 | 49.204% |  |

## Best Candidate

- Dataset: `hf_hexquant_stocks_daily`
- Repo: `HexQuant/Stocks-Daily-Price`
- Model: `hist_gradient`
- Avg monthly net return: `0.0131306`
- Avg monthly net income: `1399.47`
- Total net return: `0.181932`
- Gross total return: `0.210644`
- Rank IC: `0.0152049`
- Target passed: `true`

## Interpretation Notes

- The controller searches a finite configured dataset list; it does not run unbounded retries until profitability appears.
- Poor net return with positive gross return usually means the signal is too high-turnover for the cost model.
- Rejected datasets remain in the report so adapters can be added deliberately.
- `not_investment_advice: true`
