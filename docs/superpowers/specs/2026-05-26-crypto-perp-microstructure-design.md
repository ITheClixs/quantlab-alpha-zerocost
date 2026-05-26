# Crypto Perp Microstructure Research Design

## Purpose

Build a free-data-first crypto perpetual futures research path that can ingest Binance/Hugging Face public microstructure data, train BTCUSDT and ETHUSDT perpetual models, run event-driven bid/ask backtests, estimate overfitting risk, and practice real-time feed ingestion without live-money trading.

This design does not weaken the platform's production-intent constraints. It creates research and paper-trading infrastructure that can reject bad strategies. A strategy remains `research_validation_only` unless it passes chronological validation, cost stress, delay stress, PBO, DSR, bootstrap confidence intervals, concentration checks, and audited paper-trading gates.

## Scope

The first implementation targets Binance USDT perpetuals only:

- `BTCUSDT`
- `ETHUSDT`

Spot pairs are out of scope for the first slice. Nasdaq/S&P futures and equities are out of scope because venue-grade CME/Nasdaq order-book data is usually paid/licensed.

## Public Data Sources

The implementation should support two free or mostly free source classes:

1. Binance public historical archives and public WebSocket market-data streams.
   - Historical archive entry point: <https://data.binance.vision/>
   - WebSocket stream docs: <https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams>

2. Local or downloadable Hugging Face Binance futures/order-book datasets.
   - Existing repo scripts already reference `data/raw/huggingface/predict-quant__binance-future-orderbook`.
   - Any HF dataset used must get a manifest with source, row count, hash, schema, symbols, date range, timestamp convention, and quality label.

The data layer must not assume every source has every field. Trades, book ticker, depth snapshots, funding, mark price, and liquidations are optional source capabilities. Missing optional fields must be recorded in the manifest and report.

## Architecture

The design reuses existing repo surfaces instead of creating a parallel stack:

- `src/quant_research_stack/feeds/binance_ws.py` for public Binance stream ingestion.
- `src/quant_research_stack/backtest/orderbook_signal.py` for existing order-book feature and model primitives.
- `src/quant_research_stack/crypto_research/` for crypto-specific datasets, candidate registries, validation, reports, and strategy search.
- `src/quant_research_stack/brokers/null_broker.py` and `src/quant_research_stack/brokers/binance_testnet.py` for paper/testnet-only execution practice.
- `src/quant_research_stack/execution/` for risk gates, audit logs, kill switch, and future paper-trading integration.

New crypto perpetual modules should live under `src/quant_research_stack/crypto_research/perps/` so the implementation is isolated from G-Research minute-bar experiments and the older daily OHLCV work.

## Data Model

Normalized historical and replayed live records should converge to a small set of typed event tables:

- `trades`: event time, receive time if known, symbol, price, size, side/aggressor flag when available, trade id.
- `book_ticker`: event time, receive time if known, symbol, best bid, best bid size, best ask, best ask size, update id.
- `depth_l2`: event time, receive time if known, symbol, bid levels, ask levels, first/last update id.
- `mark_price`: event time, symbol, mark price, index price, next funding time, funding rate when available.
- `liquidations`: event time, symbol, side, price, quantity when available.

Every normalized table must include:

- `source`
- `dataset_id`
- `symbol`
- `event_time`
- `ingested_at_utc`

The first implementation can start with `trades` plus `book_ticker` or existing `depth_l2` files, because those are enough to build bid/ask executable features and event-driven backtests.

## Feature Layer

Features are computed only from events available at or before the signal timestamp. Required first-slice features:

- midprice
- relative spread
- best bid/ask size
- L1 imbalance
- microprice
- microprice deviation from mid
- recent mid returns over short horizons
- rolling realized volatility
- trade imbalance if trade data is available
- update intensity or event count over rolling windows

Optional later features:

- depth imbalance at L5/L10/L20
- order-flow imbalance from depth deltas
- funding rate and mark/index basis
- liquidation shock features
- liquidity replenishment features

## Labels

Labels must be timestamp-safe and aligned with execution:

- future mid return over event horizons
- future executable taker return using future bid/ask
- binary direction over event horizons
- triple-barrier microstructure label

Initial horizons:

- `1`
- `5`
- `15`
- `60`
- `300` events

The plan can later add time-based horizons once event timestamps are regular enough after normalization.

## Model Families

Start with tabular models before neural nets:

- Ridge regression
- ElasticNet / SGDRegressor
- LogisticRegression for directional labels
- RandomForest or ExtraTrees where feasible
- HistGradientBoosting
- ensemble mean over calibrated tabular baselines

No external pretrained Hugging Face model may be promoted unless its inputs, outputs, timestamp assumptions, and leakage risks are audited. The first implementation should avoid external pretrained trading models.

## Event-Driven Backtest

The backtest must not use close-to-close proxy returns. It must use available bid/ask fields:

- long entry: buy at ask
- long exit: sell at bid
- short entry: sell at bid
- short exit: buy at ask

Costs and execution controls:

- maker/taker fee config, defaulting to taker-only for first slice
- slippage bps
- latency delay in events
- max relative spread filter
- min top-of-book depth filter
- edge-to-cost threshold sweep
- 1x, 2x, and 3x cost stress
- inverted-signal and random-signal baselines
- no-cost, spread-only, fee-only diagnostics

Outputs:

- strategy registry parquet
- all backtests parquet
- per-trade audit parquet
- cost sensitivity report
- holdout report
- failure report if no strategy passes

## Validation

All validation is chronological:

1. Development period for feature/model exploration.
2. Validation period for model and parameter selection.
3. Permanent holdout period touched once.
4. Optional walk-forward and combinatorial purged cross-validation.

Required diagnostics:

- IC and R2 on all prediction rows
- IC and R2 on traded rows
- gross and net PnL
- daily and event-trade Sharpe
- max drawdown and drawdown duration
- hit rate before and after costs
- profit factor
- turnover and trade count
- long/short PnL split
- PnL by spread, volatility, and liquidity regimes
- concentration by day, asset, parameter, and regime
- stationary bootstrap confidence interval
- Deflated Sharpe Ratio
- Probability of Backtest Overfitting across the full candidate registry

PBO and DSR must count every tested candidate variant. Failed trials must remain logged.

## Real-Time Ingestion Practice

Real-time work is paper-only and free-data-first:

- Subscribe to Binance public streams for BTCUSDT and ETHUSDT.
- Record raw events under `data/live/crypto/binance/`.
- Write a recorder manifest with stream names, start/end timestamps, event counts, and schema.
- Replay recorded events through the same normalization and feature builder used by historical data.
- Compare recorder/replayer parity.

The first stream set should be:

- `aggTrade`
- `bookTicker`
- optionally `depth@100ms` once the parser/replayer contract is stable

## Paper Trading

Paper trading should start with local/null execution:

- model emits signal
- signal is converted into intended side/size
- risk gate validates exposure and kill-switch state
- paper broker simulates fill from current bid/ask and configured latency/slippage
- audit log records signal, order intent, fill, risk decision, and market snapshot

Binance testnet integration can be practiced only after local/null paper mode passes replay parity and audit tests. No live-money path is part of this design.

## Status Semantics

Reports must distinguish:

- `research_validation_only`
- `research_pass`
- `promotion_eligible`
- `paper_trade_candidate`
- `production_candidate`

The first implementation is expected to produce `research_validation_only` or `research_pass` at most. `production_candidate` requires data/vendor/legal/compliance review outside this free-data slice.

## Non-Goals

- No live-money trading.
- No spot trading in the first slice.
- No CME/Nasdaq futures/equities integration.
- No co-location or ultra-low-latency HFT claims.
- No promotion from in-sample backtest results.
- No hidden notebook-only logic.

## Acceptance Criteria

The implementation is accepted when:

1. BTCUSDT and ETHUSDT perpetual historical microstructure data can be normalized into parquet with manifests.
2. The event-driven backtest uses bid/ask executable returns and explicit cost stress.
3. At least one tabular model family trains walk-forward on normalized microstructure features.
4. The run writes strategy registry, all backtests, PBO/DSR/bootstrap diagnostics, per-trade audit, and reports.
5. Binance public WebSocket recorder can capture a short paper-only session and replay it through the feature path.
6. Tests cover parsers, manifests, feature timestamp safety, backtest cost math, validation gates, recorder/replayer parity, and report status semantics.
7. Full verification passes: ruff, mypy, targeted tests, and the relevant integration smoke tests.
