# Zero-Cost Feature Availability (P0)

**Built:** 2026-05-30T10:43:16.571480+00:00

## Per-instrument OHLCV features (always available from instrument bars)
- trailing realized vol (20/60d), downside vol, trend (12-1 momentum, SMA50/200 state),
  drawdown-from-peak, return autocorrelation. Computed at close t, used t+1.

## Macro features available now
- vix: available (market_price_clean)
- vix3m: available (market_price_clean)
- bonds_tlt: available (market_price_clean)
- gold_gld: available (market_price_clean)
- credit_hyg: available (market_price_clean)
- usd_uup: available (market_price_clean)
- ust10y: available (daily_next_day_only)
- ust2y: available (daily_next_day_only)

## Derived features (built in P1)
- `vix_term_structure`: vix / vix3m (>1 backwardation = stress)
- `yield_slope`: ust10y - ust2y (inversion = stress)
- `credit_trend`: HYG trend / drawdown state
- `usd_trend`: UUP trend state
- `gold_trend`: GLD trend state

## Regime features
- HMM risk-on/off (reuse `signal_research/strategies/hmm_single_index.py`), past-data-only;
  vol-regime fallback (trailing vol vs rolling median).
