# Runbook: Paper-stage signal validation methodology

## Why
Per operator decision 2026-05-20: no real-money trading until QuantLab's signals
are demonstrated to be accurate on a paper-trading account, with daily review
through TradingView's chart UI connected to Alpaca paper.

## Gates (configs/validation.yaml)
- ≥ 30 trading days in paper stage
- Rolling 14-day Sharpe ≥ 1.0
- Max daily realized DD ≤ 5%
- Hit rate ≥ 0.53 on filled trades (weighted by position weight)
- Governor block rate ≤ 0.50 (S2 is allowed to veto, but if it vetos > half the
  time, the gate fails on signal coverage; revisit S2 calibration before
  promoting)

## Operator daily workflow
See `tradingview_paper_setup.md`.

## What "signals validated" means
After 30 trading days, the promotion report at
`docs/runbooks/paper_to_live_shadow.md` shows all 5 gates green. At that point
the live broker design (S4.1, currently deferred) becomes the next conversation.
Two-person review of any `brokers/*_live.py` change remains required per
CLAUDE.md §1.13.
