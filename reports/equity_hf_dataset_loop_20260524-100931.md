# HF Equity Dataset Loop `20260524-100931`

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
- Cost: `5.0` bps one-way
- Selection fraction: `10.00%`
- Target avg monthly net return: `0.0`
- Target total return: `0.0`
- Downloaded this run: `none`

## Leaderboard

| dataset | status | best model | months | avg monthly net | positive months | total net | gross total | rank IC | dir. acc. | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf_sp500_daily` | `ok` | `ridge` | 13 | -1.667% | 23.077% | -19.944% | 2.087% | -0.00611192 | 50.610% |  |
| `hf_nasdaq_2013_2023` | `ok` | `ridge` | 12 | -2.078% | 25.000% | -22.588% | -1.280% | 0.0111351 | 50.691% |  |
| `hf_hexquant_stocks_daily` | `ok` | `hist_gradient` | 13 | 17.055% | 92.308% | 569.015% | 751.430% | -0.0111249 | 47.167% |  |
| `hf_sp500_2025` | `error` | `` | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0 | 0.000% | ValueError: not enough dates for walk-forward split: dates=246 min_train_dates=504 |
| `hf_nasdaq_shifted_reject_probe` | `rejected` | `` | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0 | 0.000% | missing OHLCV columns: ['high', 'low', 'symbol', 'volume'] |
| `hf_nasdaq_close_sequence_reject_probe` | `rejected` | `` | 0 | 0.000% | 0.000% | 0.000% | 0.000% | 0 | 0.000% | missing OHLCV columns: ['close', 'date', 'high', 'low', 'open', 'symbol', 'volume'] |

## Best Candidate

- Dataset: `hf_hexquant_stocks_daily`
- Repo: `HexQuant/Stocks-Daily-Price`
- Model: `hist_gradient`
- Avg monthly net return: `0.170554`
- Avg monthly net income: `43770.3`
- Total net return: `5.69015`
- Gross total return: `7.5143`
- Rank IC: `-0.0111249`
- Target passed: `true`

## Interpretation Notes

- The controller searches a finite configured dataset list; it does not run unbounded retries until profitability appears.
- Poor net return with positive gross return usually means the signal is too high-turnover for the cost model.
- Rejected datasets remain in the report so adapters can be added deliberately.
- `not_investment_advice: true`
