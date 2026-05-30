# Zero-Cost Instrument Coverage (P0)

**Built:** 2026-05-30T10:43:16.571480+00:00

| instrument | source | available | rows | date range |
|---|---|:---:|---:|---|
| SPY | disk | True | 4122 | 2010-01-04..2026-05-22 |
| QQQ | disk | True | 4122 | 2010-01-04..2026-05-22 |
| BTCUSDT | yfinance | True | 4274 | 2014-09-17..2026-05-30 |
| ETHUSDT | yfinance | True | 3125 | 2017-11-09..2026-05-30 |

**Basket common window (intersection):** `2017-11-09` .. `2026-05-22`
- ETH-USD (~2017 start) is the binding constraint on the 4-instrument basket window;
  SPY/QQQ alone cover 2010-2026. The basket validation runs on the common window.
- Long-flat, weekly rebalance, equal-risk; decisions at close t, execution t+1.
