# HF Equity Dataset Loop `20260524-101306`

## Scope

This run probes Hugging Face equity datasets, fetches missing configured datasets when enabled, rejects non-OHLCV schemas, trains dedicated OHLCV signal heads, and backtests out-of-sample predictions by month.

## Configuration

- Git SHA: `0f9b98e7f4c7412c183032ad6d40a6ac1bf105cb`
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
- Selection fraction: `10.00%`
- Target avg monthly net return: `0.0`
- Target total return: `0.0`
- Downloaded this run: `none`

## Leaderboard

| dataset | status | best model | months | avg monthly net | positive months | total net | gross total | rank IC | dir. acc. | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf_hexquant_stocks_daily` | `ok` | `ensemble_mean` | 13 | 0.381% | 61.538% | 4.149% | 69.231% | 0.0295731 | 49.636% |  |

## Best Candidate

- Dataset: `hf_hexquant_stocks_daily`
- Repo: `HexQuant/Stocks-Daily-Price`
- Model: `ensemble_mean`
- Avg monthly net return: `0.00381095`
- Avg monthly net income: `319.131`
- Total net return: `0.041487`
- Gross total return: `0.692314`
- Rank IC: `0.0295731`
- Target passed: `true`

## Interpretation Notes

- The controller searches a finite configured dataset list; it does not run unbounded retries until profitability appears.
- Poor net return with positive gross return usually means the signal is too high-turnover for the cost model.
- Rejected datasets remain in the report so adapters can be added deliberately.
- `not_investment_advice: true`
