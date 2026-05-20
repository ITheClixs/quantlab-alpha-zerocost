# Runbook: TradingView paper-trading setup

## One-time setup
1. Create a TradingView account (free tier works).
2. In the chart, open the Trading Panel (bottom of the chart, "Trading" tab).
3. Click "Connect" → "Alpaca" → log in with your Alpaca paper credentials
   (the same `~/.alpaca/paper_keys.json` QuantLab uses).
4. After connecting, set the account selector to "Paper Trading" (Alpaca's paper account).
5. Build a watchlist that mirrors `configs/exec.yaml`'s symbol universe.
6. Pin the watchlist to the chart sidebar for quick switching.

## Daily review
1. Open today's chart in TV with the connected paper account selected.
2. Each QuantLab order appears as a labeled order on the chart in real time.
3. After close, open `docs/validation/<today>.md` and use the per-signal table
   to spot-check any signal that looked wrong on the TV chart.
4. Annotate the operator-checklist section of the report with observations.

Before review, generate the report:

```bash
make tv-validation-report VALIDATION_DATE=<YYYY-MM-DD>
```

The report uses S4 fills, Alpaca 1-minute bars, and Alpaca paper account equity
to compute realized forward returns, hit rate, net daily PnL percentage, daily
drawdown percentage, rolling Sharpe from prior `data/validation/*.parquet`
files, and book-vs-broker reconciliation.

## Limitations
- TV's chart can lag the broker by 0-2 seconds. Trust the QuantLab audit log
  timestamps for forensic work, not the TV chart visual.
- TV's Trading Panel does NOT show partial fills as separate events; it shows
  the most recent fill state. Use `logs/audit/s4/paper/<date>.jsonl` for the
  full fill stream.
- TV is purely a viewer in this setup. Do not place orders manually in TV
  during a paper-validation run; the report will not know about them and the
  reconciliation row will turn red.
- If Alpaca data credentials are missing, the report still renders but realized
  returns, PnL gates, and broker reconciliation are incomplete; fix credentials
  before treating the report as promotion evidence.
